"""
RAG API endpoints.
"""

import time
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from ..models import DocumentRetrievalRequest, DocumentRetrievalResponse, DocumentInfo
from ..dependencies import rag_dependencies

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/retrieve", response_model=DocumentRetrievalResponse)
async def retrieve_documents(request: DocumentRetrievalRequest, deps: Dict[str, Any] = Depends(rag_dependencies)):
    """Retrieve relevant documents for a query."""
    engine = deps["engine"]
    try:
        start_time = time.time()
        results = engine.retrieve_documents(request.query, top_k=request.top_k)
        retrieval_time = time.time() - start_time
        
        documents = [
            DocumentInfo(
                document=doc["document"],
                score=doc["score"],
                rank=doc["rank"],
                metadata=doc.get("metadata") if request.include_metadata else None
            )
            for doc in results
        ]
        
        return DocumentRetrievalResponse(
            query=request.query,
            documents=documents,
            retrieval_time=retrieval_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
