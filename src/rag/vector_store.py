"""
FAISS Vector Store for Adaptrix RAG System.

This module implements a FAISS-based vector store for efficient
document embedding storage and retrieval.
"""

import faiss
import numpy as np
import pickle
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store for document embeddings.
    
    Supports efficient similarity search and document retrieval
    for RAG applications.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_type: str = "flat",
        dimension: Optional[int] = None
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_model: Sentence transformer model name
            index_type: FAISS index type ('flat', 'ivf', 'hnsw')
            dimension: Embedding dimension (auto-detected if None)
        """
        self.embedding_model_name = embedding_model
        self.index_type = index_type
        self.dimension = dimension
        
        # Components
        self.encoder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        
        # Document storage
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.doc_id_to_index: Dict[str, int] = {}
        
        # State
        self.is_initialized = False
        self.num_documents = 0
        
    def initialize(self) -> bool:
        """Initialize the vector store."""
        try:
            logger.info(f"🚀 Initializing FAISS vector store with {self.embedding_model_name}")
            
            # Load sentence transformer
            self.encoder = SentenceTransformer(self.embedding_model_name)
            
            # Get embedding dimension
            if self.dimension is None:
                test_embedding = self.encoder.encode(["test"])
                self.dimension = test_embedding.shape[1]
            
            # Initialize FAISS index
            self._create_index()
            
            self.is_initialized = True
            logger.info(f"✅ Vector store initialized (dimension: {self.dimension})")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            return False
    
    def _create_index(self):
        """Create FAISS index based on specified type."""
        if self.index_type == "flat":
            # Flat index for exact search (good for small datasets)
            self.index = faiss.IndexFlatL2(self.dimension)
            
        elif self.index_type == "ivf":
            # IVF index for faster approximate search
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
        elif self.index_type == "hnsw":
            # HNSW index for very fast approximate search
            M = 16  # Number of connections
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        logger.info(f"Created {self.index_type} index with dimension {self.dimension}")
    
    def add_documents(
        self, 
        documents: List[str], 
        metadata: Optional[List[Dict[str, Any]]] = None,
        doc_ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            doc_ids: Optional document IDs
            
        Returns:
            Success status
        """
        if not self.is_initialized:
            logger.error("Vector store not initialized")
            return False
        
        try:
            logger.info(f"📝 Adding {len(documents)} documents to vector store")
            
            # Generate embeddings
            embeddings = self.encoder.encode(
                documents, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Prepare metadata
            if metadata is None:
                metadata = [{}] * len(documents)
            
            if doc_ids is None:
                doc_ids = [f"doc_{self.num_documents + i}" for i in range(len(documents))]
            
            # Add to storage
            start_idx = len(self.documents)
            self.documents.extend(documents)
            self.metadata.extend(metadata)
            
            # Update document ID mapping
            for i, doc_id in enumerate(doc_ids):
                self.doc_id_to_index[doc_id] = start_idx + i
            
            # Add to FAISS index
            if self.index_type == "ivf" and not self.index.is_trained:
                # Train IVF index if not already trained
                if len(embeddings) >= 100:  # Need enough data to train
                    logger.info("Training IVF index...")
                    self.index.train(embeddings.astype(np.float32))
                else:
                    logger.warning("Not enough data to train IVF index, using flat index")
                    self.index = faiss.IndexFlatL2(self.dimension)
            
            self.index.add(embeddings.astype(np.float32))
            self.num_documents = len(self.documents)
            
            logger.info(f"✅ Added {len(documents)} documents (total: {self.num_documents})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def search(
        self, 
        query: str, 
        k: int = 5,
        return_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            return_metadata: Whether to include metadata in results
            
        Returns:
            List of search results with documents, scores, and metadata
        """
        if not self.is_initialized:
            logger.error("Vector store not initialized")
            return []
        
        if self.num_documents == 0:
            logger.warning("No documents in vector store")
            return []
        
        try:
            # Encode query
            query_embedding = self.encoder.encode([query])
            
            # Search FAISS index
            scores, indices = self.index.search(
                query_embedding.astype(np.float32), 
                min(k, self.num_documents)
            )
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                result = {
                    "rank": i + 1,
                    "document": self.documents[idx],
                    "score": float(score),
                    "index": int(idx)
                }
                
                if return_metadata:
                    result["metadata"] = self.metadata[idx]
                
                results.append(result)
            
            logger.debug(f"Found {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        if doc_id not in self.doc_id_to_index:
            return None
        
        idx = self.doc_id_to_index[doc_id]
        return {
            "document": self.documents[idx],
            "metadata": self.metadata[idx],
            "index": idx
        }
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID (not supported by FAISS, requires rebuild)."""
        logger.warning("Document deletion requires rebuilding the index")
        return False
    
    def save(self, save_path: str) -> bool:
        """Save vector store to disk."""
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_path = save_path / "faiss.index"
            faiss.write_index(self.index, str(index_path))
            
            # Save documents and metadata
            data = {
                "documents": self.documents,
                "metadata": self.metadata,
                "doc_id_to_index": self.doc_id_to_index,
                "num_documents": self.num_documents
            }
            
            data_path = save_path / "documents.pkl"
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Save configuration
            config = {
                "embedding_model": self.embedding_model_name,
                "index_type": self.index_type,
                "dimension": self.dimension,
                "num_documents": self.num_documents
            }
            
            config_path = save_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"✅ Vector store saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            return False
    
    def load(self, load_path: str) -> bool:
        """Load vector store from disk."""
        try:
            load_path = Path(load_path)
            
            if not load_path.exists():
                logger.error(f"Vector store path does not exist: {load_path}")
                return False
            
            # Load configuration
            config_path = load_path / "config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.embedding_model_name = config["embedding_model"]
            self.index_type = config["index_type"]
            self.dimension = config["dimension"]
            
            # Initialize encoder
            if not self.encoder:
                self.encoder = SentenceTransformer(self.embedding_model_name)
            
            # Load FAISS index
            index_path = load_path / "faiss.index"
            self.index = faiss.read_index(str(index_path))
            
            # Load documents and metadata
            data_path = load_path / "documents.pkl"
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.doc_id_to_index = data["doc_id_to_index"]
            self.num_documents = data["num_documents"]
            
            self.is_initialized = True
            
            logger.info(f"✅ Vector store loaded from {load_path}")
            logger.info(f"   📊 Documents: {self.num_documents}")
            logger.info(f"   🧠 Model: {self.embedding_model_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "initialized": self.is_initialized,
            "embedding_model": self.embedding_model_name,
            "index_type": self.index_type,
            "dimension": self.dimension,
            "num_documents": self.num_documents,
            "index_size": self.index.ntotal if self.index else 0
        }
