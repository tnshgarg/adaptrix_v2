"""
Text generation API endpoints.

This module provides endpoints for text generation functionality.
"""

import time
import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from ..models import (
    GenerationRequest, GenerationResponse,
    BatchGenerationRequest, BatchGenerationResponse,
    RAGGenerationRequest, RAGGenerationResponse
)
from ..dependencies import generation_dependencies, add_response_headers
from ...core.base_model_interface import GenerationConfig

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    deps: Dict[str, Any] = Depends(generation_dependencies)
):
    """
    Generate text from a prompt.
    
    This endpoint provides text generation with automatic adapter selection,
    RAG integration, and caching support.
    """
    engine = deps["engine"]
    request_id = deps["request_id"]
    
    try:
        start_time = time.time()
        
        logger.info(f"🎯 Generation request {request_id}: {request.prompt[:100]}...")
        
        # Generate text
        response_text = engine.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            task_type=request.task_type.value if request.task_type else "auto",
            adapter_name=request.adapter_name,
            use_rag=request.use_rag,
            rag_top_k=request.rag_top_k,
            use_cache=request.use_cache,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_sequences=request.stop_sequences
        )
        
        processing_time = time.time() - start_time
        
        # Get generation info if available
        generation_info = {}
        if hasattr(engine, '_last_generation_info'):
            generation_info = getattr(engine, '_last_generation_info', {})
        
        logger.info(f"✅ Generation completed {request_id} in {processing_time:.3f}s")
        
        response = GenerationResponse(
            generated_text=response_text,
            prompt=request.prompt,
            generation_info=generation_info,
            processing_time=processing_time
        )
        
        # Add response headers
        headers = add_response_headers(deps.get("request"))
        
        return JSONResponse(
            content=response.dict(),
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"❌ Generation failed {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {str(e)}"
        )


@router.post("/generate/batch", response_model=BatchGenerationResponse)
async def batch_generate_text(
    request: BatchGenerationRequest,
    background_tasks: BackgroundTasks,
    deps: Dict[str, Any] = Depends(generation_dependencies)
):
    """
    Generate text for multiple prompts in batch.
    
    This endpoint provides efficient batch processing for multiple prompts.
    """
    engine = deps["engine"]
    request_id = deps["request_id"]
    
    try:
        start_time = time.time()
        
        logger.info(f"🎯 Batch generation request {request_id}: {len(request.prompts)} prompts")
        
        # Check if engine supports batch generation
        if hasattr(engine, 'batch_generate'):
            responses = engine.batch_generate(
                prompts=request.prompts,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )
        else:
            # Fall back to sequential generation
            responses = []
            for prompt in request.prompts:
                response = engine.generate(
                    prompt=prompt,
                    max_length=request.max_length,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k
                )
                responses.append(response)
        
        processing_time = time.time() - start_time
        
        # Format results
        results = []
        for i, (prompt, response) in enumerate(zip(request.prompts, responses)):
            results.append({
                "index": i,
                "prompt": prompt,
                "generated_text": response,
                "success": True
            })
        
        logger.info(f"✅ Batch generation completed {request_id} in {processing_time:.3f}s")
        
        response = BatchGenerationResponse(
            results=results,
            processing_time=processing_time
        )
        
        # Add response headers
        headers = add_response_headers(deps["request"])
        
        return JSONResponse(
            content=response.dict(),
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"❌ Batch generation failed {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch text generation failed: {str(e)}"
        )


@router.post("/generate/rag", response_model=RAGGenerationResponse)
async def rag_generate_text(
    request: RAGGenerationRequest,
    background_tasks: BackgroundTasks,
    deps: Dict[str, Any] = Depends(generation_dependencies)
):
    """
    Generate text with RAG (Retrieval Augmented Generation).
    
    This endpoint provides text generation enhanced with document retrieval.
    """
    engine = deps["engine"]
    request_id = deps["request_id"]
    
    try:
        start_time = time.time()
        
        logger.info(f"🎯 RAG generation request {request_id}: {request.prompt[:100]}...")
        
        # Check if engine supports RAG
        if not hasattr(engine, 'retrieve_documents'):
            raise HTTPException(
                status_code=501,
                detail="RAG functionality not supported"
            )
        
        # Retrieve documents if requested
        sources = []
        if request.include_sources:
            retrieved_docs = engine.retrieve_documents(
                request.prompt,
                top_k=request.rag_top_k
            )
            sources = [
                {
                    "document": doc["document"],
                    "score": doc["score"],
                    "rank": doc["rank"],
                    "metadata": doc.get("metadata")
                }
                for doc in retrieved_docs
            ]
        
        # Generate text with RAG
        response_text = engine.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            task_type=request.task_type.value if request.task_type else "auto",
            adapter_name=request.adapter_name,
            use_rag=True,
            rag_top_k=request.rag_top_k,
            use_cache=request.use_cache,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_sequences=request.stop_sequences
        )
        
        processing_time = time.time() - start_time
        
        # Get generation info
        generation_info = {}
        rag_info = {}
        if hasattr(engine, '_last_generation_info'):
            last_info = getattr(engine, '_last_generation_info', {})
            generation_info = last_info.get('selection', {})
            rag_info = last_info.get('rag', {})
        
        logger.info(f"✅ RAG generation completed {request_id} in {processing_time:.3f}s")
        
        response = RAGGenerationResponse(
            generated_text=response_text,
            prompt=request.prompt,
            generation_info=generation_info,
            processing_time=processing_time,
            sources=sources if request.include_sources else None,
            rag_info=rag_info
        )
        
        # Add response headers
        headers = add_response_headers(deps["request"])
        
        return JSONResponse(
            content=response.dict(),
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"❌ RAG generation failed {request_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"RAG text generation failed: {str(e)}"
        )


@router.get("/models/current")
async def get_current_model(
    deps: Dict[str, Any] = Depends(generation_dependencies)
):
    """Get information about the current model."""
    engine = deps["engine"]
    
    try:
        if hasattr(engine, 'get_optimization_status'):
            status = engine.get_optimization_status()
        elif hasattr(engine, 'get_moe_status'):
            status = engine.get_moe_status()
        else:
            status = engine.get_system_status()
        
        model_info = status.get("model_info", {})
        
        return {
            "success": True,
            "model_info": model_info,
            "capabilities": {
                "moe": "moe" in status,
                "rag": status.get("moe", {}).get("rag_initialized", False),
                "optimization": "optimization" in status,
                "adapters": len(status.get("adapters", [])) > 0
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model information: {str(e)}"
        )


@router.get("/generation/config")
async def get_generation_config():
    """Get default generation configuration."""
    return {
        "success": True,
        "default_config": {
            "max_length": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "task_type": "auto"
        },
        "limits": {
            "max_length": {"min": 1, "max": 2048},
            "temperature": {"min": 0.0, "max": 2.0},
            "top_p": {"min": 0.0, "max": 1.0},
            "top_k": {"min": 1, "max": 100},
            "batch_size": {"max": 32}
        }
    }
