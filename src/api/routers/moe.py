"""
MoE API endpoints.
"""

import time
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from ..models import AdapterPredictionRequest, AdapterPredictionResponse, MoEStatsResponse
from ..dependencies import moe_dependencies

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict", response_model=AdapterPredictionResponse)
async def predict_adapter(request: AdapterPredictionRequest, deps: Dict[str, Any] = Depends(moe_dependencies)):
    """Predict the best adapter for a given prompt."""
    engine = deps["engine"]
    try:
        start_time = time.time()
        result = engine.predict_adapter(request.prompt)
        prediction_time = time.time() - start_time
        
        return AdapterPredictionResponse(
            prompt=request.prompt,
            predicted_adapter=result.get("adapter_name", "unknown"),
            confidence=result.get("confidence", 0.0),
            probabilities=result.get("probabilities") if request.return_probabilities else None,
            prediction_time=prediction_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=MoEStatsResponse)
async def get_moe_stats(deps: Dict[str, Any] = Depends(moe_dependencies)):
    """Get MoE system statistics."""
    engine = deps["engine"]
    try:
        stats = engine.get_selection_stats()
        status = engine.get_moe_status()
        
        return MoEStatsResponse(
            selection_stats=stats,
            classifier_status=status.get("moe", {}).get("classifier_status", {}),
            total_predictions=stats.get("total_selections", 0),
            average_confidence=stats.get("average_confidence", 0.0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
