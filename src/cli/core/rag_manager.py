"""
RAG manager for the Adaptrix CLI.

This module provides functionality for managing RAG document collections and vector stores.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import RAG components with error handling
try:
    from src.rag.vector_store import FAISSVectorStore
    from src.rag.document_processor import DocumentProcessor
    RAG_AVAILABLE = True
except ImportError:
    logger = logging.getLogger("rag_manager")
    logger.warning("RAG components not available, using mock implementations")
    RAG_AVAILABLE = False

    # Mock implementations for when RAG components are not available
    class MockDocumentProcessor:
        def __init__(self, **kwargs):
            pass

        def process_document(self, *args, **kwargs):
            return []

        def process_text(self, text, **kwargs):
            return [text]

    class MockFAISSVectorStore:
        @classmethod
        def load(cls, path):
            return cls()

        def __init__(self, **kwargs):
            pass

        def search(self, *args, **kwargs):
            return []

    # Use mock implementations
    FAISSVectorStore = MockFAISSVectorStore
    DocumentProcessor = MockDocumentProcessor
from src.cli.utils.logging import get_logger
from src.cli.utils.progress import ProgressBar

logger = get_logger("rag_manager")

class RAGManager:
    """
    Manages RAG document collections and vector stores for the CLI.
    """
    
    def __init__(self, config_manager):
        """
        Initialize RAG manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.rag_dir = Path(self.config.get("rag.directory"))
        self.rag_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize document processor
        self.document_processor = DocumentProcessor(
            chunk_size=self.config.get("rag.chunk_size", 512),
            chunk_overlap=self.config.get("rag.chunk_overlap", 50)
        )
        
        # Cache for vector stores
        self._vector_store_cache: Dict[str, FAISSVectorStore] = {}
        
        logger.info(f"RAGManager initialized with directory: {self.rag_dir}")
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all RAG collections.
        
        Returns:
            List of collection information dictionaries
        """
        collections = []
        
        for collection_dir in self.rag_dir.iterdir():
            if collection_dir.is_dir():
                collection_name = collection_dir.name
                
                # Load collection metadata
                metadata_path = collection_dir / "metadata.json"
                metadata = {}
                
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        logger.warning(f"Error loading metadata for {collection_name}: {e}")
                
                # Count documents and chunks
                documents_count = len(list(collection_dir.glob("*.txt")))
                chunks_count = metadata.get("total_chunks", 0)
                
                collection_info = {
                    "name": collection_name,
                    "documents": documents_count,
                    "chunks": chunks_count,
                    "size": self._get_collection_size(collection_dir),
                    "embedding_model": metadata.get("embedding_model", "unknown"),
                    "created_date": metadata.get("created_date", "unknown")
                }
                collections.append(collection_info)
        
        return collections
    
    def list_documents(self, collection_name: str) -> List[Dict[str, Any]]:
        """
        List documents in a specific collection.
        
        Args:
            collection_name: Collection name
        
        Returns:
            List of document information dictionaries
        """
        collection_dir = self.rag_dir / collection_name
        
        if not collection_dir.exists():
            return []
        
        documents = []
        
        # Load collection metadata
        metadata_path = collection_dir / "metadata.json"
        metadata = {}
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading metadata: {e}")
        
        # Get document information
        document_metadata = metadata.get("documents", {})
        
        for doc_file in collection_dir.glob("*.txt"):
            doc_name = doc_file.stem
            doc_info = document_metadata.get(doc_name, {})
            
            document_info = {
                "filename": doc_name,
                "chunks": doc_info.get("chunks", 0),
                "size": self._format_size(doc_file.stat().st_size),
                "added_date": doc_info.get("added_date", "unknown"),
                "original_path": doc_info.get("original_path", "unknown")
            }
            documents.append(document_info)
        
        return documents
    
    def add_documents(self, document_path: str, collection_name: str = "default", recursive: bool = False) -> int:
        """
        Add documents to a RAG collection.
        
        Args:
            document_path: Path to document or directory
            collection_name: Collection name
            recursive: Process directories recursively
        
        Returns:
            Number of documents added
        """
        try:
            # Create collection directory
            collection_dir = self.rag_dir / collection_name
            collection_dir.mkdir(parents=True, exist_ok=True)
            
            # Load or create collection metadata
            metadata_path = collection_dir / "metadata.json"
            metadata = {}
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading metadata: {e}")
            
            if "documents" not in metadata:
                metadata["documents"] = {}
            
            # Get document files
            document_files = self._get_document_files(document_path, recursive)
            
            if not document_files:
                logger.warning(f"No supported documents found in {document_path}")
                return 0
            
            # Process documents
            total_added = 0
            total_chunks = 0
            
            with ProgressBar(f"Processing documents", len(document_files)) as progress:
                for doc_file in document_files:
                    try:
                        # Process document
                        chunks = self.document_processor.process_document(str(doc_file))
                        
                        if chunks:
                            # Save processed document
                            doc_name = doc_file.stem
                            output_file = collection_dir / f"{doc_name}.txt"
                            
                            with open(output_file, 'w', encoding='utf-8') as f:
                                for chunk in chunks:
                                    f.write(chunk + "\n\n")
                            
                            # Update metadata
                            metadata["documents"][doc_name] = {
                                "chunks": len(chunks),
                                "original_path": str(doc_file),
                                "added_date": str(Path().cwd())
                            }
                            
                            total_added += 1
                            total_chunks += len(chunks)
                        
                        progress.update(1, f"Processed {doc_file.name}")
                        
                    except Exception as e:
                        logger.warning(f"Error processing {doc_file}: {e}")
                        progress.update(1, f"Failed {doc_file.name}")
                        continue
            
            # Update collection metadata
            metadata["total_documents"] = len(metadata["documents"])
            metadata["total_chunks"] = total_chunks
            metadata["embedding_model"] = self.config.get("rag.embedding_model", "all-MiniLM-L6-v2")
            metadata["last_updated"] = str(Path().cwd())
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Added {total_added} documents to collection {collection_name}")
            return total_added
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return 0
    
    def create_vector_store(self, collection_name: str, embedding_model: Optional[str] = None, 
                          chunk_size: Optional[int] = None) -> bool:
        """
        Create a vector store for a collection.
        
        Args:
            collection_name: Collection name
            embedding_model: Embedding model to use
            chunk_size: Chunk size for processing
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create collection directory
            collection_dir = self.rag_dir / collection_name
            collection_dir.mkdir(parents=True, exist_ok=True)
            
            # Create metadata
            metadata = {
                "name": collection_name,
                "embedding_model": embedding_model or self.config.get("rag.embedding_model", "all-MiniLM-L6-v2"),
                "chunk_size": chunk_size or self.config.get("rag.chunk_size", 512),
                "created_date": str(Path().cwd()),
                "documents": {},
                "total_documents": 0,
                "total_chunks": 0
            }
            
            # Save metadata
            metadata_path = collection_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Created vector store for collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Collection name
        
        Returns:
            Collection information dictionary or None if not found
        """
        collection_dir = self.rag_dir / collection_name
        
        if not collection_dir.exists():
            return None
        
        # Load metadata
        metadata_path = collection_dir / "metadata.json"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading collection metadata: {e}")
        
        # Return basic info if no metadata
        return {
            "name": collection_name,
            "documents": len(list(collection_dir.glob("*.txt"))),
            "size": self._get_collection_size(collection_dir)
        }
    
    def remove_document(self, collection_name: str, document_name: str) -> bool:
        """
        Remove a document from a collection.
        
        Args:
            collection_name: Collection name
            document_name: Document name
        
        Returns:
            True if successful, False otherwise
        """
        try:
            collection_dir = self.rag_dir / collection_name
            doc_file = collection_dir / f"{document_name}.txt"
            
            if not doc_file.exists():
                logger.warning(f"Document {document_name} not found in collection {collection_name}")
                return True
            
            # Remove document file
            doc_file.unlink()
            
            # Update metadata
            metadata_path = collection_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if document_name in metadata.get("documents", {}):
                        del metadata["documents"][document_name]
                        metadata["total_documents"] = len(metadata["documents"])
                        
                        # Recalculate total chunks
                        total_chunks = sum(doc.get("chunks", 0) for doc in metadata["documents"].values())
                        metadata["total_chunks"] = total_chunks
                        
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                
                except Exception as e:
                    logger.warning(f"Error updating metadata: {e}")
            
            logger.info(f"Removed document {document_name} from collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing document: {e}")
            return False
    
    def remove_collection(self, collection_name: str) -> bool:
        """
        Remove an entire collection.
        
        Args:
            collection_name: Collection name
        
        Returns:
            True if successful, False otherwise
        """
        try:
            collection_dir = self.rag_dir / collection_name
            
            if not collection_dir.exists():
                logger.warning(f"Collection {collection_name} does not exist")
                return True
            
            # Remove collection directory
            shutil.rmtree(collection_dir)
            
            logger.info(f"Removed collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing collection: {e}")
            return False
    
    def search_documents(self, query: str, collection_name: str = "default", top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search documents in a collection.
        
        Args:
            query: Search query
            collection_name: Collection name
            top_k: Number of results to return
        
        Returns:
            List of search results
        """
        try:
            # For now, implement simple text search
            # In a full implementation, this would use vector similarity search
            
            collection_dir = self.rag_dir / collection_name
            
            if not collection_dir.exists():
                return []
            
            results = []
            
            for doc_file in collection_dir.glob("*.txt"):
                try:
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Simple keyword matching (in real implementation, use vector search)
                    if query.lower() in content.lower():
                        # Calculate simple score based on frequency
                        score = content.lower().count(query.lower()) / len(content.split())
                        
                        results.append({
                            "document": doc_file.stem,
                            "content": content[:500],  # First 500 characters
                            "score": score
                        })
                
                except Exception as e:
                    logger.warning(f"Error searching {doc_file}: {e}")
                    continue
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def _get_document_files(self, path: str, recursive: bool = False) -> List[Path]:
        """Get list of supported document files."""
        path_obj = Path(path)
        supported_extensions = self.config.get("rag.supported_formats", [".txt", ".pdf", ".docx", ".md"])
        
        files = []
        
        if path_obj.is_file():
            if path_obj.suffix.lower() in supported_extensions:
                files.append(path_obj)
        elif path_obj.is_dir():
            if recursive:
                for ext in supported_extensions:
                    files.extend(path_obj.rglob(f"*{ext}"))
            else:
                for ext in supported_extensions:
                    files.extend(path_obj.glob(f"*{ext}"))
        
        return files
    
    def _get_collection_size(self, collection_dir: Path) -> str:
        """Get formatted size of a collection directory."""
        total_size = sum(f.stat().st_size for f in collection_dir.rglob('*') if f.is_file())
        return self._format_size(total_size)
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
