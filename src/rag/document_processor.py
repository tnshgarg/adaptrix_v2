"""
Document Processing Pipeline for Adaptrix RAG System.

This module implements document chunking, preprocessing, and indexing
for the RAG system.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata."""
    text: str
    chunk_id: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


class DocumentProcessor:
    """
    Document processing pipeline for RAG system.
    
    Handles document loading, chunking, and preprocessing
    for vector store indexing.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target size for document chunks (in characters)
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Supported file types
        self.supported_extensions = {'.txt', '.md', '.py', '.js', '.html', '.json', '.csv'}
        
    def load_document(self, file_path: str) -> Optional[str]:
        """
        Load document from file.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Document text or None if failed
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File does not exist: {file_path}")
                return None
            
            if file_path.suffix.lower() not in self.supported_extensions:
                logger.warning(f"Unsupported file type: {file_path.suffix}")
                return None
            
            # Read file with encoding detection
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            logger.info(f"📄 Loaded document: {file_path} ({len(content)} chars)")
            return content
            
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            return None
    
    def load_documents_from_directory(
        self, 
        directory: str,
        recursive: bool = True
    ) -> Dict[str, str]:
        """
        Load all supported documents from directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search recursively
            
        Returns:
            Dictionary mapping file paths to content
        """
        directory = Path(directory)
        documents = {}
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory}")
            return documents
        
        # Get file pattern
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                content = self.load_document(str(file_path))
                if content:
                    documents[str(file_path)] = content
        
        logger.info(f"📚 Loaded {len(documents)} documents from {directory}")
        return documents
    
    def process_text(self, text: str, chunk_size: int = None) -> List[str]:
        """
        Process raw text into chunks.

        Args:
            text: Raw text to process
            chunk_size: Size of chunks (uses default if None)

        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Create chunks
        chunks = self.chunk_text(processed_text, source_file="text_input")

        # Extract just the text content
        return [chunk.content for chunk in chunks]

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess document text.
        
        Args:
            text: Raw document text
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\]', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[\.]{3,}', '...', text)
        text = re.sub(r'[\!\?]{2,}', '!', text)
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([\.,:;!?])', r'\1', text)
        text = re.sub(r'([\.,:;!?])\s+', r'\1 ', text)
        
        # Strip and normalize
        text = text.strip()
        
        return text
    
    def chunk_text(
        self, 
        text: str, 
        source_file: str = "unknown"
    ) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            source_file: Source file path
            
        Returns:
            List of document chunks
        """
        if len(text) < self.min_chunk_size:
            logger.warning(f"Text too short to chunk: {len(text)} chars")
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                sentence_end = -1
                
                for i in range(end - 1, search_start - 1, -1):
                    if text[i] in '.!?':
                        # Check if this is likely a sentence end (not abbreviation)
                        if i + 1 < len(text) and text[i + 1].isspace():
                            sentence_end = i + 1
                            break
                
                if sentence_end != -1:
                    end = sentence_end
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            # Skip chunks that are too small
            if len(chunk_text) < self.min_chunk_size:
                start = end - self.chunk_overlap
                continue
            
            # Preprocess chunk
            chunk_text = self.preprocess_text(chunk_text)
            
            if len(chunk_text) >= self.min_chunk_size:
                # Create chunk
                chunk = DocumentChunk(
                    text=chunk_text,
                    chunk_id=f"{Path(source_file).stem}_chunk_{chunk_index}",
                    source_file=source_file,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata={
                        "file_type": Path(source_file).suffix.lower(),
                        "chunk_size": len(chunk_text),
                        "original_size": len(text)
                    }
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= end:
                break
        
        logger.info(f"📄 Created {len(chunks)} chunks from {source_file}")
        return chunks
    
    def process_document(
        self, 
        file_path: str,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Process a single document into chunks.
        
        Args:
            file_path: Path to document
            custom_metadata: Additional metadata to add to chunks
            
        Returns:
            List of document chunks
        """
        # Load document
        content = self.load_document(file_path)
        if not content:
            return []
        
        # Create chunks
        chunks = self.chunk_text(content, file_path)
        
        # Add custom metadata
        if custom_metadata:
            for chunk in chunks:
                chunk.metadata.update(custom_metadata)
        
        return chunks
    
    def process_documents(
        self, 
        file_paths: List[str],
        custom_metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[DocumentChunk]:
        """
        Process multiple documents into chunks.
        
        Args:
            file_paths: List of document paths
            custom_metadata: Dictionary mapping file paths to metadata
            
        Returns:
            List of all document chunks
        """
        all_chunks = []
        
        for file_path in file_paths:
            metadata = custom_metadata.get(file_path, {}) if custom_metadata else {}
            chunks = self.process_document(file_path, metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"📚 Processed {len(file_paths)} documents into {len(all_chunks)} chunks")
        return all_chunks
    
    def process_directory(
        self, 
        directory: str,
        recursive: bool = True,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Process all documents in a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search recursively
            custom_metadata: Metadata to add to all chunks
            
        Returns:
            List of all document chunks
        """
        # Load all documents
        documents = self.load_documents_from_directory(directory, recursive)
        
        # Process each document
        all_chunks = []
        for file_path, content in documents.items():
            chunks = self.chunk_text(content, file_path)
            
            # Add custom metadata
            if custom_metadata:
                for chunk in chunks:
                    chunk.metadata.update(custom_metadata)
            
            all_chunks.extend(chunks)
        
        logger.info(f"📁 Processed directory {directory}: {len(all_chunks)} chunks from {len(documents)} files")
        return all_chunks
    
    def chunks_to_documents(self, chunks: List[DocumentChunk]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """
        Convert chunks to format suitable for vector store.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Tuple of (texts, metadata, doc_ids)
        """
        texts = []
        metadata = []
        doc_ids = []
        
        for chunk in chunks:
            texts.append(chunk.text)
            
            # Combine chunk metadata with chunk info
            chunk_metadata = chunk.metadata.copy()
            chunk_metadata.update({
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char
            })
            
            metadata.append(chunk_metadata)
            doc_ids.append(chunk.chunk_id)
        
        return texts, metadata, doc_ids
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor configuration."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "supported_extensions": list(self.supported_extensions)
        }
