"""
Pydantic models for Adaptrix API.

This module defines request and response models for the FastAPI application.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


# Enums
class TaskType(str, Enum):
    """Task types for generation."""
    AUTO = "auto"
    GENERAL = "general"
    CODE = "code"
    LEGAL = "legal"
    MATH = "math"


class AdapterStatus(str, Enum):
    """Adapter status."""
    AVAILABLE = "available"
    LOADED = "loaded"
    ERROR = "error"


# Base models
class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = True
    timestamp: float = Field(default_factory=lambda: __import__('time').time())


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    status_code: int
    path: Optional[str] = None
    details: Optional[str] = None
    timestamp: float = Field(default_factory=lambda: __import__('time').time())


# Generation models
class GenerationRequest(BaseModel):
    """Text generation request."""
    prompt: str = Field(..., description="Input prompt for text generation", min_length=1, max_length=10000)
    max_length: Optional[int] = Field(default=150, description="Maximum tokens to generate", ge=1, le=2048)
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, description="Top-p sampling", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, description="Top-k sampling", ge=1, le=100)
    task_type: Optional[TaskType] = Field(default=TaskType.AUTO, description="Task type for generation")
    adapter_name: Optional[str] = Field(default=None, description="Specific adapter to use")
    use_rag: Optional[bool] = Field(default=None, description="Whether to use RAG")
    rag_top_k: Optional[int] = Field(default=3, description="Number of documents for RAG", ge=1, le=10)
    use_cache: Optional[bool] = Field(default=None, description="Whether to use response caching")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Stop sequences for generation")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()


class GenerationResponse(BaseResponse):
    """Text generation response."""
    generated_text: str
    prompt: str
    generation_info: Dict[str, Any] = Field(default_factory=dict)
    processing_time: float


class BatchGenerationRequest(BaseModel):
    """Batch text generation request."""
    prompts: List[str] = Field(..., description="List of prompts", min_items=1, max_items=32)
    max_length: Optional[int] = Field(default=150, description="Maximum tokens to generate", ge=1, le=2048)
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, description="Top-p sampling", ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=50, description="Top-k sampling", ge=1, le=100)
    
    @validator('prompts')
    def validate_prompts(cls, v):
        if not all(prompt.strip() for prompt in v):
            raise ValueError('All prompts must be non-empty')
        return [prompt.strip() for prompt in v]


class BatchGenerationResponse(BaseResponse):
    """Batch text generation response."""
    results: List[Dict[str, Any]]
    processing_time: float


# Adapter models
class AdapterInfo(BaseModel):
    """Adapter information."""
    name: str
    path: str
    status: AdapterStatus
    metadata: Dict[str, Any] = Field(default_factory=dict)
    size_mb: Optional[float] = None
    last_modified: Optional[float] = None


class AdapterListResponse(BaseResponse):
    """Adapter list response."""
    adapters: List[AdapterInfo]
    total_count: int


class AdapterLoadRequest(BaseModel):
    """Adapter load request."""
    adapter_name: str = Field(..., description="Name of adapter to load")
    force_reload: bool = Field(default=False, description="Force reload if already loaded")


class AdapterLoadResponse(BaseResponse):
    """Adapter load response."""
    adapter_name: str
    status: AdapterStatus
    message: str


# RAG models
class DocumentRetrievalRequest(BaseModel):
    """Document retrieval request."""
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    top_k: Optional[int] = Field(default=5, description="Number of documents to retrieve", ge=1, le=20)
    score_threshold: Optional[float] = Field(default=0.0, description="Minimum similarity score", ge=0.0, le=1.0)
    include_metadata: bool = Field(default=True, description="Include document metadata")


class DocumentInfo(BaseModel):
    """Document information."""
    document: str
    score: float
    rank: int
    metadata: Optional[Dict[str, Any]] = None


class DocumentRetrievalResponse(BaseResponse):
    """Document retrieval response."""
    query: str
    documents: List[DocumentInfo]
    retrieval_time: float


class RAGGenerationRequest(GenerationRequest):
    """RAG-enhanced generation request."""
    use_rag: bool = Field(default=True, description="Enable RAG (overrides base class)")
    rag_top_k: int = Field(default=3, description="Number of documents for RAG", ge=1, le=10)
    include_sources: bool = Field(default=True, description="Include source documents in response")


class RAGGenerationResponse(GenerationResponse):
    """RAG-enhanced generation response."""
    sources: Optional[List[DocumentInfo]] = None
    rag_info: Dict[str, Any] = Field(default_factory=dict)


# MoE models
class AdapterPredictionRequest(BaseModel):
    """Adapter prediction request."""
    prompt: str = Field(..., description="Input prompt for adapter prediction", min_length=1, max_length=1000)
    return_probabilities: bool = Field(default=True, description="Return prediction probabilities")


class AdapterPredictionResponse(BaseResponse):
    """Adapter prediction response."""
    prompt: str
    predicted_adapter: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    prediction_time: float


class MoEStatsResponse(BaseResponse):
    """MoE statistics response."""
    selection_stats: Dict[str, Any]
    classifier_status: Dict[str, Any]
    total_predictions: int
    average_confidence: float


# System models
class ModelInfo(BaseModel):
    """Model information."""
    model_id: str
    model_family: str
    context_length: int
    total_parameters: Optional[int] = None
    device: str
    torch_dtype: Optional[str] = None


class SystemStatus(BaseModel):
    """System status information."""
    status: str
    uptime: float
    model_info: ModelInfo
    components: Dict[str, bool]
    performance: Dict[str, Any] = Field(default_factory=dict)


class SystemMetrics(BaseModel):
    """System metrics."""
    timestamp: float
    model_info: ModelInfo
    performance: Dict[str, Any]
    moe: Optional[Dict[str, Any]] = None
    optimization: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: float
    engine_initialized: bool
    model_info: Dict[str, Any]
    components: Dict[str, bool]


# Configuration models
class APIConfigResponse(BaseResponse):
    """API configuration response."""
    config: Dict[str, Any]
    environment: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)


# Utility models
class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, description="Page number", ge=1)
    page_size: int = Field(default=20, description="Items per page", ge=1, le=100)


class SortParams(BaseModel):
    """Sort parameters."""
    sort_by: str = Field(default="name", description="Field to sort by")
    sort_order: str = Field(default="asc", description="Sort order (asc/desc)")
    
    @validator('sort_order')
    def validate_sort_order(cls, v):
        if v.lower() not in ['asc', 'desc']:
            raise ValueError('Sort order must be "asc" or "desc"')
        return v.lower()


# Response wrappers
class PaginatedResponse(BaseResponse):
    """Paginated response wrapper."""
    items: List[Any]
    total_count: int
    page: int
    page_size: int
    total_pages: int


def create_paginated_response(
    items: List[Any],
    total_count: int,
    page: int,
    page_size: int
) -> PaginatedResponse:
    """Create a paginated response."""
    total_pages = (total_count + page_size - 1) // page_size
    
    return PaginatedResponse(
        items=items,
        total_count=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )


# Validation helpers
def validate_generation_params(
    max_length: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None
) -> Dict[str, Any]:
    """Validate and normalize generation parameters."""
    params = {}
    
    if max_length is not None:
        if max_length < 1 or max_length > 2048:
            raise ValueError("max_length must be between 1 and 2048")
        params["max_length"] = max_length
    
    if temperature is not None:
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        params["temperature"] = temperature
    
    if top_p is not None:
        if top_p < 0.0 or top_p > 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        params["top_p"] = top_p
    
    if top_k is not None:
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")
        params["top_k"] = top_k
    
    return params
