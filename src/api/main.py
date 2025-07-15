"""
FastAPI Application for Adaptrix.

This module provides a comprehensive REST API for the Adaptrix system,
including text generation, adapter management, RAG, and MoE functionality.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .routers import generation, adapters, rag, moe, system
from .dependencies import get_engine, get_rate_limiter
from .models import ErrorResponse
from .config import APIConfig, get_api_config

logger = logging.getLogger(__name__)

# Global engine instance
engine_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global engine_instance
    
    try:
        # Initialize engine on startup
        logger.info("🚀 Starting Adaptrix API server")
        
        config = get_api_config()
        
        # Import and initialize the appropriate engine
        if config.use_optimized_engine:
            from ..inference.optimized_engine import OptimizedAdaptrixEngine
            from ..inference.quantization import create_int4_config
            from ..inference.caching import create_default_cache_manager
            
            # Create quantization config if enabled
            quant_config = None
            if config.enable_quantization:
                quant_config = create_int4_config()
            
            # Create cache manager if enabled
            cache_manager = None
            if config.enable_caching:
                cache_manager = create_default_cache_manager(
                    enable_persistence=True
                )
            
            engine_instance = OptimizedAdaptrixEngine(
                model_id=config.model_id,
                device=config.device,
                adapters_dir=config.adapters_dir,
                classifier_path=config.classifier_path,
                enable_auto_selection=config.enable_auto_selection,
                rag_vector_store_path=config.rag_vector_store_path,
                enable_rag=config.enable_rag,
                use_vllm=config.use_vllm,
                quantization_config=quant_config,
                enable_caching=config.enable_caching,
                cache_manager=cache_manager,
                max_batch_size=config.max_batch_size,
                enable_async=config.enable_async
            )
        else:
            from ..moe.moe_engine import MoEAdaptrixEngine
            
            engine_instance = MoEAdaptrixEngine(
                model_id=config.model_id,
                device=config.device,
                adapters_dir=config.adapters_dir,
                classifier_path=config.classifier_path,
                enable_auto_selection=config.enable_auto_selection,
                rag_vector_store_path=config.rag_vector_store_path,
                enable_rag=config.enable_rag
            )
        
        # Initialize engine
        if not engine_instance.initialize():
            raise RuntimeError("Failed to initialize Adaptrix engine")
        
        logger.info("✅ Adaptrix API server started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start Adaptrix API server: {e}")
        raise
    finally:
        # Cleanup on shutdown
        if engine_instance:
            logger.info("🧹 Shutting down Adaptrix API server")
            engine_instance.cleanup()
            engine_instance = None
        logger.info("✅ Adaptrix API server shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Adaptrix API",
    description="REST API for the Adaptrix Modular AI System",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"📥 {request.method} {request.url.path} - {request.client.host}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"📤 {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code,
            path=str(request.url.path)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            status_code=500,
            path=str(request.url.path),
            details=str(exc) if app.debug else None
        ).dict()
    )


# Include routers
app.include_router(
    generation.router,
    prefix="/api/v1/generation",
    tags=["Text Generation"]
)

app.include_router(
    adapters.router,
    prefix="/api/v1/adapters",
    tags=["Adapter Management"]
)

app.include_router(
    rag.router,
    prefix="/api/v1/rag",
    tags=["RAG (Retrieval Augmented Generation)"]
)

app.include_router(
    moe.router,
    prefix="/api/v1/moe",
    tags=["MoE (Mixture of Experts)"]
)

app.include_router(
    system.router,
    prefix="/api/v1/system",
    tags=["System Information"]
)


# Root endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Adaptrix API",
        "version": "2.0.0",
        "description": "REST API for the Adaptrix Modular AI System",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        engine = get_engine()
        
        if hasattr(engine, 'get_optimization_status'):
            status = engine.get_optimization_status()
        elif hasattr(engine, 'get_moe_status'):
            status = engine.get_moe_status()
        else:
            status = engine.get_system_status()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "engine_initialized": engine._initialized if hasattr(engine, '_initialized') else True,
            "model_info": status.get("model_info", {}),
            "components": {
                "moe": status.get("moe", {}).get("classifier_initialized", False),
                "rag": status.get("moe", {}).get("rag_initialized", False),
                "optimization": status.get("optimization", {}).get("optimization_enabled", False)
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    try:
        engine = get_engine()
        
        if hasattr(engine, 'get_optimization_status'):
            status = engine.get_optimization_status()
        elif hasattr(engine, 'get_moe_status'):
            status = engine.get_moe_status()
        else:
            status = engine.get_system_status()
        
        metrics = {
            "timestamp": time.time(),
            "model_info": status.get("model_info", {}),
            "performance": {}
        }
        
        # Add MoE metrics
        if "moe" in status:
            moe_info = status["moe"]
            metrics["moe"] = {
                "selection_stats": moe_info.get("selection_stats", {}),
                "retrieval_stats": moe_info.get("retrieval_stats", {}),
                "vector_store_stats": moe_info.get("vector_store_stats", {})
            }
        
        # Add optimization metrics
        if "optimization" in status:
            opt_info = status["optimization"]
            metrics["optimization"] = {
                "vllm_stats": opt_info.get("vllm_stats", {}),
                "cache_stats": opt_info.get("cache_stats", {}),
                "quantization_config": opt_info.get("quantization_config", {})
            }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


def create_app(config_override: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Create FastAPI application with optional config override."""
    if config_override:
        # Update global config
        api_config = get_api_config()
        for key, value in config_override.items():
            if hasattr(api_config, key):
                setattr(api_config, key, value)
    
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
    log_level: str = "info"
):
    """Run the FastAPI server."""
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    run_server()
