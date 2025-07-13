"""
System API endpoints.
"""

import time
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from ..models import SystemStatus, SystemMetrics, ModelInfo, HealthResponse, APIConfigResponse
from ..dependencies import system_dependencies, get_config
from ..config import get_environment_info, validate_config

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/status", response_model=SystemStatus)
async def get_system_status(deps: Dict[str, Any] = Depends(system_dependencies)):
    """Get comprehensive system status."""
    engine = deps["engine"]
    try:
        if hasattr(engine, 'get_optimization_status'):
            status = engine.get_optimization_status()
        elif hasattr(engine, 'get_moe_status'):
            status = engine.get_moe_status()
        else:
            status = engine.get_system_status()
        
        model_info_dict = status.get("model_info", {})
        model_info = ModelInfo(
            model_id=model_info_dict.get("model_id", "unknown"),
            model_family=model_info_dict.get("model_family", "unknown"),
            context_length=model_info_dict.get("context_length", 0),
            total_parameters=model_info_dict.get("total_parameters"),
            device=model_info_dict.get("device", "unknown"),
            torch_dtype=model_info_dict.get("torch_dtype")
        )
        
        components = {
            "moe": status.get("moe", {}).get("classifier_initialized", False),
            "rag": status.get("moe", {}).get("rag_initialized", False),
            "optimization": status.get("optimization", {}).get("optimization_enabled", False)
        }
        
        return SystemStatus(
            status="healthy",
            uptime=time.time() - deps.get("start_time", time.time()),
            model_info=model_info,
            components=components,
            performance=status.get("performance", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config", response_model=APIConfigResponse)
async def get_api_config(config = Depends(get_config)):
    """Get API configuration and environment info."""
    try:
        config_dict = config.dict()
        # Remove sensitive information
        sensitive_keys = ["api_key", "jwt_secret"]
        for key in sensitive_keys:
            if key in config_dict:
                config_dict[key] = "***" if config_dict[key] else None
        
        return APIConfigResponse(
            config=config_dict,
            environment=get_environment_info(),
            warnings=validate_config(config)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
