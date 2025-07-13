"""
Adapter management API endpoints.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from ..models import AdapterListResponse, AdapterLoadRequest, AdapterLoadResponse, AdapterInfo, AdapterStatus
from ..dependencies import adapter_dependencies

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", response_model=AdapterListResponse)
async def list_adapters(deps: Dict[str, Any] = Depends(adapter_dependencies)):
    """List all available adapters."""
    engine = deps["engine"]
    try:
        adapters = engine.list_adapters()
        adapter_infos = [
            AdapterInfo(
                name=name,
                path=info.get("path", ""),
                status=AdapterStatus.AVAILABLE,
                metadata=info
            )
            for name, info in adapters.items()
        ]
        return AdapterListResponse(adapters=adapter_infos, total_count=len(adapter_infos))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/load", response_model=AdapterLoadResponse)
async def load_adapter(request: AdapterLoadRequest, deps: Dict[str, Any] = Depends(adapter_dependencies)):
    """Load a specific adapter."""
    engine = deps["engine"]
    try:
        success = engine.switch_adapter(request.adapter_name)
        return AdapterLoadResponse(
            adapter_name=request.adapter_name,
            status=AdapterStatus.LOADED if success else AdapterStatus.ERROR,
            message="Adapter loaded successfully" if success else "Failed to load adapter"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
