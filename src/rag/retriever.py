"""
Document Retrieval System for Adaptrix RAG.

This module implements the retrieval component that finds relevant
documents for query augmentation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .vector_store import FAISSVectorStore
from .document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a document retrieval result."""
    document: str
    score: float
    metadata: Dict[str, Any]
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document": self.document,
            "score": self.score,
            "metadata": self.metadata,
            "rank": self.rank
        }


class DocumentRetriever:
    """
    Document retrieval system for RAG.
    
    Combines vector store search with query processing and
    result ranking for optimal document retrieval.
    """
    
    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        vector_store_path: Optional[str] = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        rerank: bool = True
    ):
        """
        Initialize document retriever.

        Args:
            vector_store: FAISS vector store instance
            vector_store_path: Path to load vector store from
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            rerank: Whether to apply reranking to results
        """
        if vector_store is not None:
            self.vector_store = vector_store
        elif vector_store_path is not None:
            self.vector_store = FAISSVectorStore.load(vector_store_path)
        else:
            # Create empty vector store for testing
            self.vector_store = FAISSVectorStore(dimension=384)
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.rerank = rerank
        
        # Query processing
        self.query_processor = DocumentProcessor(
            chunk_size=512,
            chunk_overlap=0,
            min_chunk_size=10
        )
        
        # Statistics
        self.retrieval_stats = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "average_results": 0.0,
            "score_distribution": []
        }
    
    def retrieve(
        self, 
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return (overrides default)
            score_threshold: Score threshold (overrides default)
            filter_metadata: Metadata filters to apply
            
        Returns:
            List of retrieval results
        """
        if not self.vector_store.is_initialized:
            logger.error("Vector store not initialized")
            return []
        
        try:
            # Use provided parameters or defaults
            k = top_k or self.top_k
            threshold = score_threshold or self.score_threshold
            
            # Preprocess query
            processed_query = self.query_processor.preprocess_text(query)
            
            logger.debug(f"Retrieving documents for query: {processed_query[:100]}...")
            
            # Search vector store
            raw_results = self.vector_store.search(
                processed_query,
                k=k * 2,  # Get more results for filtering/reranking
                return_metadata=True
            )
            
            # Convert to RetrievalResult objects
            results = []
            for result in raw_results:
                retrieval_result = RetrievalResult(
                    document=result["document"],
                    score=result["score"],
                    metadata=result["metadata"],
                    rank=result["rank"]
                )
                results.append(retrieval_result)
            
            # Apply score threshold
            if threshold > 0:
                results = [r for r in results if r.score >= threshold]
            
            # Apply metadata filters
            if filter_metadata:
                results = self._apply_metadata_filters(results, filter_metadata)
            
            # Rerank if enabled
            if self.rerank and len(results) > 1:
                results = self._rerank_results(results, processed_query)
            
            # Limit to requested number
            results = results[:k]
            
            # Update ranks
            for i, result in enumerate(results):
                result.rank = i + 1
            
            # Update statistics
            self._update_retrieval_stats(results)
            
            logger.debug(f"Retrieved {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def _apply_metadata_filters(
        self, 
        results: List[RetrievalResult], 
        filters: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """Apply metadata filters to results."""
        filtered_results = []
        
        for result in results:
            match = True
            
            for key, value in filters.items():
                if key not in result.metadata:
                    match = False
                    break
                
                if isinstance(value, list):
                    # Check if metadata value is in the list
                    if result.metadata[key] not in value:
                        match = False
                        break
                else:
                    # Exact match
                    if result.metadata[key] != value:
                        match = False
                        break
            
            if match:
                filtered_results.append(result)
        
        logger.debug(f"Filtered {len(results)} -> {len(filtered_results)} results")
        return filtered_results
    
    def _rerank_results(
        self, 
        results: List[RetrievalResult], 
        query: str
    ) -> List[RetrievalResult]:
        """
        Rerank results using additional scoring methods.
        
        This is a simple implementation - could be enhanced with
        more sophisticated reranking models.
        """
        try:
            # Simple keyword-based reranking
            query_words = set(query.lower().split())
            
            for result in results:
                doc_words = set(result.document.lower().split())
                
                # Calculate keyword overlap
                overlap = len(query_words.intersection(doc_words))
                total_query_words = len(query_words)
                
                if total_query_words > 0:
                    keyword_score = overlap / total_query_words
                else:
                    keyword_score = 0.0
                
                # Combine with original score (weighted average)
                # Lower FAISS scores are better (L2 distance), so invert
                original_score = 1.0 / (1.0 + result.score)
                combined_score = 0.7 * original_score + 0.3 * keyword_score
                
                result.score = combined_score
            
            # Sort by combined score (higher is better)
            results.sort(key=lambda x: x.score, reverse=True)
            
            logger.debug("Applied reranking to results")
            return results
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results
    
    def _update_retrieval_stats(self, results: List[RetrievalResult]):
        """Update retrieval statistics."""
        try:
            self.retrieval_stats["total_queries"] += 1
            
            if results:
                self.retrieval_stats["successful_retrievals"] += 1
                
                # Update average results
                total_results = self.retrieval_stats["average_results"] * (self.retrieval_stats["total_queries"] - 1)
                total_results += len(results)
                self.retrieval_stats["average_results"] = total_results / self.retrieval_stats["total_queries"]
                
                # Update score distribution
                scores = [r.score for r in results]
                self.retrieval_stats["score_distribution"].extend(scores)
                
                # Keep only recent scores (last 1000)
                if len(self.retrieval_stats["score_distribution"]) > 1000:
                    self.retrieval_stats["score_distribution"] = self.retrieval_stats["score_distribution"][-1000:]
            
        except Exception as e:
            logger.error(f"Failed to update retrieval stats: {e}")
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        stats = self.retrieval_stats.copy()
        
        # Calculate additional statistics
        if stats["score_distribution"]:
            scores = stats["score_distribution"]
            stats["min_score"] = min(scores)
            stats["max_score"] = max(scores)
            stats["avg_score"] = sum(scores) / len(scores)
        
        if stats["total_queries"] > 0:
            stats["success_rate"] = stats["successful_retrievals"] / stats["total_queries"]
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    def create_context(
        self, 
        results: List[RetrievalResult],
        max_context_length: int = 2000,
        include_metadata: bool = False
    ) -> str:
        """
        Create context string from retrieval results.
        
        Args:
            results: List of retrieval results
            max_context_length: Maximum context length in characters
            include_metadata: Whether to include metadata in context
            
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            # Format document
            doc_text = result.document.strip()
            
            # Add metadata if requested
            if include_metadata and result.metadata:
                source = result.metadata.get("source_file", "unknown")
                doc_header = f"[Source: {source}]\n"
                doc_text = doc_header + doc_text
            
            # Check if adding this document would exceed limit
            if current_length + len(doc_text) > max_context_length and context_parts:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
            
            # Add separator between documents
            if i < len(results) - 1:
                separator = "\n\n---\n\n"
                if current_length + len(separator) <= max_context_length:
                    context_parts.append(separator)
                    current_length += len(separator)
        
        context = "".join(context_parts)
        
        logger.debug(f"Created context with {len(context)} characters from {len(context_parts)} documents")
        return context
    
    def reset_stats(self):
        """Reset retrieval statistics."""
        self.retrieval_stats = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "average_results": 0.0,
            "score_distribution": []
        }
        logger.info("Retrieval statistics reset")
    
    def get_config(self) -> Dict[str, Any]:
        """Get retriever configuration."""
        return {
            "top_k": self.top_k,
            "score_threshold": self.score_threshold,
            "rerank": self.rerank,
            "vector_store_stats": self.vector_store.get_stats()
        }
