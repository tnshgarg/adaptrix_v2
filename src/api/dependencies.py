"""
Dependencies for Adaptrix API.

This module provides dependency injection for FastAPI endpoints.
"""

import time
import logging
from typing import Optional, Dict, Any
from collections import defaultdict, deque

from fastapi import HTTPException, Depends, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .config import get_api_config, APIConfig

logger = logging.getLogger(__name__)

# Global engine instance (set by main.py)
_engine_instance = None

# Rate limiting storage
_rate_limit_storage: Dict[str, deque] = defaultdict(deque)

# Security
security = HTTPBearer(auto_error=False)


def set_engine_instance(engine):
    """Set the global engine instance."""
    global _engine_instance
    _engine_instance = engine


def get_engine():
    """Get the global engine instance."""
    global _engine_instance
    if _engine_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Engine not initialized"
        )
    return _engine_instance


def get_config() -> APIConfig:
    """Get API configuration."""
    return get_api_config()


class RateLimiter:
    """Rate limiter for API endpoints."""
    
    def __init__(self, requests_per_minute: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            window_seconds: Time window in seconds
        """
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        current_time = time.time()
        client_requests = _rate_limit_storage[client_id]
        
        # Remove old requests outside the window
        while client_requests and client_requests[0] < current_time - self.window_seconds:
            client_requests.popleft()
        
        # Check if limit exceeded
        if len(client_requests) >= self.requests_per_minute:
            return False
        
        # Add current request
        client_requests.append(current_time)
        return True
    
    def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client."""
        current_time = time.time()
        client_requests = _rate_limit_storage[client_id]
        
        # Remove old requests
        while client_requests and client_requests[0] < current_time - self.window_seconds:
            client_requests.popleft()
        
        return max(0, self.requests_per_minute - len(client_requests))
    
    def get_reset_time(self, client_id: str) -> float:
        """Get time when rate limit resets for client."""
        client_requests = _rate_limit_storage[client_id]
        if not client_requests:
            return time.time()
        
        return client_requests[0] + self.window_seconds


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        config = get_api_config()
        _rate_limiter = RateLimiter(
            requests_per_minute=config.rate_limit_requests,
            window_seconds=config.rate_limit_window
        )
    return _rate_limiter


def check_rate_limit(request: Request, config: APIConfig = Depends(get_config)):
    """Check rate limit for request."""
    if not config.enable_rate_limiting:
        return
    
    # Get client identifier
    client_id = request.client.host
    if hasattr(request.state, 'user_id'):
        client_id = f"user_{request.state.user_id}"
    
    rate_limiter = get_rate_limiter()
    
    if not rate_limiter.is_allowed(client_id):
        remaining = rate_limiter.get_remaining_requests(client_id)
        reset_time = rate_limiter.get_reset_time(client_id)
        
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(rate_limiter.requests_per_minute),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(int(reset_time)),
                "Retry-After": str(int(reset_time - time.time()))
            }
        )


def verify_api_key(
    request: Request,
    authorization: Optional[str] = Header(None),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    config: APIConfig = Depends(get_config)
):
    """Verify API key authentication."""
    if not config.enable_auth:
        return
    
    api_key = None
    
    # Check Authorization header
    if credentials:
        api_key = credentials.credentials
    elif authorization:
        if authorization.startswith("Bearer "):
            api_key = authorization[7:]
        elif authorization.startswith("ApiKey "):
            api_key = authorization[7:]
        else:
            api_key = authorization
    
    # Check X-API-Key header
    if not api_key:
        api_key = request.headers.get("X-API-Key")
    
    # Verify API key
    if not api_key or api_key != config.api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Store user info in request state
    request.state.authenticated = True
    request.state.api_key = api_key


def get_client_info(request: Request) -> Dict[str, Any]:
    """Get client information from request."""
    return {
        "ip": request.client.host,
        "port": request.client.port,
        "user_agent": request.headers.get("user-agent"),
        "referer": request.headers.get("referer"),
        "authenticated": getattr(request.state, 'authenticated', False),
        "timestamp": time.time()
    }


def validate_content_type(request: Request):
    """Validate request content type."""
    if request.method in ["POST", "PUT", "PATCH"]:
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            raise HTTPException(
                status_code=415,
                detail="Content-Type must be application/json"
            )


def check_request_size(
    request: Request,
    config: APIConfig = Depends(get_config)
):
    """Check request size limits."""
    content_length = request.headers.get("content-length")
    if content_length:
        size = int(content_length)
        if size > config.max_request_size:
            raise HTTPException(
                status_code=413,
                detail=f"Request too large. Maximum size: {config.max_request_size} bytes"
            )


def add_response_headers(request: Request) -> Dict[str, str]:
    """Add standard response headers."""
    headers = {
        "X-API-Version": "2.0.0",
        "X-Powered-By": "Adaptrix",
        "X-Request-ID": getattr(request.state, 'request_id', 'unknown')
    }
    
    # Add rate limit headers if enabled
    config = get_api_config()
    if config.enable_rate_limiting:
        client_id = request.client.host
        if hasattr(request.state, 'user_id'):
            client_id = f"user_{request.state.user_id}"
        
        rate_limiter = get_rate_limiter()
        remaining = rate_limiter.get_remaining_requests(client_id)
        reset_time = rate_limiter.get_reset_time(client_id)
        
        headers.update({
            "X-RateLimit-Limit": str(rate_limiter.requests_per_minute),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(reset_time))
        })
    
    return headers


# Common dependency combinations
def common_dependencies(
    request: Request,
    config: APIConfig = Depends(get_config)
):
    """Common dependencies for all endpoints."""
    # Generate request ID
    import uuid
    request.state.request_id = str(uuid.uuid4())
    
    # Validate request
    validate_content_type(request)
    check_request_size(request, config)
    
    # Check authentication
    verify_api_key(request, config=config)
    
    # Check rate limiting
    check_rate_limit(request, config)
    
    return {
        "request_id": request.state.request_id,
        "client_info": get_client_info(request),
        "config": config
    }


def generation_dependencies(
    request: Request,
    engine = Depends(get_engine),
    deps = Depends(common_dependencies)
):
    """Dependencies for generation endpoints."""
    return {
        "engine": engine,
        **deps
    }


def adapter_dependencies(
    request: Request,
    engine = Depends(get_engine),
    deps = Depends(common_dependencies)
):
    """Dependencies for adapter endpoints."""
    # Check if engine supports adapter management
    if not hasattr(engine, 'list_adapters'):
        raise HTTPException(
            status_code=501,
            detail="Adapter management not supported by current engine"
        )
    
    return {
        "engine": engine,
        **deps
    }


def rag_dependencies(
    request: Request,
    engine = Depends(get_engine),
    deps = Depends(common_dependencies)
):
    """Dependencies for RAG endpoints."""
    # Check if engine supports RAG
    if not hasattr(engine, 'retrieve_documents'):
        raise HTTPException(
            status_code=501,
            detail="RAG functionality not supported by current engine"
        )
    
    return {
        "engine": engine,
        **deps
    }


def moe_dependencies(
    request: Request,
    engine = Depends(get_engine),
    deps = Depends(common_dependencies)
):
    """Dependencies for MoE endpoints."""
    # Check if engine supports MoE
    if not hasattr(engine, 'predict_adapter'):
        raise HTTPException(
            status_code=501,
            detail="MoE functionality not supported by current engine"
        )
    
    return {
        "engine": engine,
        **deps
    }


def system_dependencies(
    request: Request,
    engine = Depends(get_engine),
    deps = Depends(common_dependencies)
):
    """Dependencies for system endpoints."""
    return {
        "engine": engine,
        **deps
    }
