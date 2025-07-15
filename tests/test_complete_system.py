#!/usr/bin/env python3
"""
Comprehensive Test Suite for Adaptrix System.

This test suite validates all components of the Adaptrix system including:
- Base model functionality
- MoE adapter selection
- RAG document retrieval
- vLLM optimization
- API endpoints
- System integration
"""

import pytest
import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Test configuration
TEST_CONFIG = {
    "model_id": "Qwen/Qwen3-1.7B",
    "device": "cpu",
    "adapters_dir": "adapters",
    "test_timeout": 300,  # 5 minutes
    "skip_model_loading": False,  # Set to True for CI/CD
    "skip_vllm_tests": True,  # Set to False if vLLM is available
}


class TestAdaptrixSystem:
    """Comprehensive system tests for Adaptrix."""
    
    @pytest.fixture(scope="class")
    def engine(self):
        """Create engine instance for testing."""
        if TEST_CONFIG["skip_model_loading"]:
            pytest.skip("Model loading disabled for testing")
        
        from src.moe.moe_engine import MoEAdaptrixEngine
        
        engine = MoEAdaptrixEngine(
            model_id=TEST_CONFIG["model_id"],
            device=TEST_CONFIG["device"],
            adapters_dir=TEST_CONFIG["adapters_dir"],
            classifier_path="models/classifier",
            enable_auto_selection=True,
            rag_vector_store_path="models/rag_vector_store",
            enable_rag=True
        )
        
        # Initialize with timeout
        start_time = time.time()
        success = engine.initialize()
        
        if not success:
            pytest.fail("Failed to initialize engine")
        
        if time.time() - start_time > TEST_CONFIG["test_timeout"]:
            pytest.fail("Engine initialization timeout")
        
        yield engine
        
        # Cleanup
        engine.cleanup()
    
    def test_basic_generation(self, engine):
        """Test basic text generation."""
        prompt = "What is machine learning?"
        response = engine.generate(prompt, max_length=100)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert response != prompt
        assert "error" not in response.lower()
    
    def test_moe_adapter_selection(self, engine):
        """Test MoE adapter selection."""
        test_prompts = [
            ("Write a Python function to sort a list", "code"),
            ("Analyze this contract clause", "legal"),
            ("What is the weather like?", "general"),
            ("Solve this equation: 2x + 5 = 15", "math")
        ]
        
        for prompt, expected_domain in test_prompts:
            if hasattr(engine, 'predict_adapter'):
                prediction = engine.predict_adapter(prompt)
                assert "adapter_name" in prediction
                assert "confidence" in prediction
                assert prediction["confidence"] > 0.0
    
    def test_rag_functionality(self, engine):
        """Test RAG document retrieval."""
        if not hasattr(engine, 'retrieve_documents'):
            pytest.skip("RAG not available")
        
        query = "machine learning algorithms"
        results = engine.retrieve_documents(query, top_k=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        
        for result in results:
            assert "document" in result
            assert "score" in result
            assert "rank" in result
            assert isinstance(result["score"], float)
    
    def test_adapter_management(self, engine):
        """Test adapter loading and switching."""
        if not hasattr(engine, 'list_adapters'):
            pytest.skip("Adapter management not available")
        
        # List adapters
        adapters = engine.list_adapters()
        assert isinstance(adapters, dict)
        
        # Test switching if adapters available
        if adapters:
            adapter_name = list(adapters.keys())[0]
            success = engine.switch_adapter(adapter_name)
            assert isinstance(success, bool)
    
    def test_system_status(self, engine):
        """Test system status reporting."""
        if hasattr(engine, 'get_optimization_status'):
            status = engine.get_optimization_status()
        elif hasattr(engine, 'get_moe_status'):
            status = engine.get_moe_status()
        else:
            status = engine.get_system_status()
        
        assert isinstance(status, dict)
        assert "model_info" in status
        
        model_info = status["model_info"]
        assert "model_id" in model_info
        assert "model_family" in model_info
        assert "device" in model_info
    
    def test_performance_metrics(self, engine):
        """Test performance and timing."""
        prompt = "Explain quantum computing"
        
        start_time = time.time()
        response = engine.generate(prompt, max_length=50)
        generation_time = time.time() - start_time
        
        assert generation_time < 30.0  # Should complete within 30 seconds
        assert len(response) > 0
    
    def test_error_handling(self, engine):
        """Test error handling and edge cases."""
        # Empty prompt
        response = engine.generate("", max_length=10)
        assert isinstance(response, str)
        
        # Very long prompt
        long_prompt = "test " * 1000
        response = engine.generate(long_prompt, max_length=10)
        assert isinstance(response, str)
        
        # Invalid parameters
        response = engine.generate("test", max_length=-1)
        assert isinstance(response, str)


class TestOptimizedEngine:
    """Tests for optimized engine with vLLM."""
    
    @pytest.fixture(scope="class")
    def optimized_engine(self):
        """Create optimized engine for testing."""
        if TEST_CONFIG["skip_vllm_tests"]:
            pytest.skip("vLLM tests disabled")
        
        from src.inference.optimized_engine import OptimizedAdaptrixEngine
        from src.inference.quantization import create_int4_config
        
        engine = OptimizedAdaptrixEngine(
            model_id=TEST_CONFIG["model_id"],
            device=TEST_CONFIG["device"],
            use_vllm=True,
            enable_quantization=False,  # Disable for testing
            enable_caching=True
        )
        
        success = engine.initialize()
        if not success:
            pytest.fail("Failed to initialize optimized engine")
        
        yield engine
        engine.cleanup()
    
    def test_vllm_generation(self, optimized_engine):
        """Test vLLM-based generation."""
        prompt = "Write a hello world program"
        response = optimized_engine.generate(prompt, max_length=100)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "hello" in response.lower() or "world" in response.lower()
    
    def test_batch_generation(self, optimized_engine):
        """Test batch generation."""
        prompts = [
            "What is AI?",
            "Explain Python",
            "Define machine learning"
        ]
        
        responses = optimized_engine.batch_generate(prompts, max_length=50)
        
        assert isinstance(responses, list)
        assert len(responses) == len(prompts)
        
        for response in responses:
            assert isinstance(response, str)
            assert len(response) > 0
    
    def test_caching(self, optimized_engine):
        """Test response caching."""
        prompt = "What is 2+2?"
        
        # First generation
        start_time = time.time()
        response1 = optimized_engine.generate(prompt, max_length=20, use_cache=True)
        time1 = time.time() - start_time
        
        # Second generation (should be cached)
        start_time = time.time()
        response2 = optimized_engine.generate(prompt, max_length=20, use_cache=True)
        time2 = time.time() - start_time
        
        assert response1 == response2
        assert time2 < time1  # Cached response should be faster


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from src.api.main import app
        
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
    
    def test_generation_endpoint(self, client):
        """Test text generation endpoint."""
        if TEST_CONFIG["skip_model_loading"]:
            pytest.skip("Model loading disabled")
        
        payload = {
            "prompt": "What is Python?",
            "max_length": 50,
            "temperature": 0.7
        }
        
        response = client.post("/api/v1/generation/generate", json=payload)
        
        if response.status_code == 503:
            pytest.skip("Engine not initialized")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "generated_text" in data
        assert "processing_time" in data
    
    def test_adapter_endpoints(self, client):
        """Test adapter management endpoints."""
        if TEST_CONFIG["skip_model_loading"]:
            pytest.skip("Model loading disabled")
        
        # List adapters
        response = client.get("/api/v1/adapters/")
        
        if response.status_code == 503:
            pytest.skip("Engine not initialized")
        
        assert response.status_code in [200, 501]  # 501 if not supported
    
    def test_system_endpoints(self, client):
        """Test system information endpoints."""
        # System status
        response = client.get("/api/v1/system/status")
        
        if response.status_code == 503:
            pytest.skip("Engine not initialized")
        
        assert response.status_code == 200
        
        # API config
        response = client.get("/api/v1/system/config")
        assert response.status_code == 200
        
        data = response.json()
        assert "config" in data
        assert "environment" in data


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        if TEST_CONFIG["skip_model_loading"]:
            pytest.skip("Model loading disabled")
        
        from src.moe.moe_engine import MoEAdaptrixEngine
        
        # Initialize engine
        engine = MoEAdaptrixEngine(
            model_id=TEST_CONFIG["model_id"],
            device=TEST_CONFIG["device"],
            adapters_dir=TEST_CONFIG["adapters_dir"]
        )
        
        success = engine.initialize()
        if not success:
            pytest.skip("Engine initialization failed")
        
        try:
            # Test workflow: prediction -> generation -> cleanup
            prompt = "Write a Python function"
            
            # 1. Predict adapter
            if hasattr(engine, 'predict_adapter'):
                prediction = engine.predict_adapter(prompt)
                assert "adapter_name" in prediction
            
            # 2. Generate text
            response = engine.generate(prompt, max_length=100)
            assert isinstance(response, str)
            assert len(response) > 0
            
            # 3. Check system status
            status = engine.get_system_status()
            assert isinstance(status, dict)
            
        finally:
            engine.cleanup()


# Test utilities
def run_performance_benchmark():
    """Run performance benchmark tests."""
    if TEST_CONFIG["skip_model_loading"]:
        print("Performance benchmark skipped - model loading disabled")
        return
    
    from src.moe.moe_engine import MoEAdaptrixEngine
    
    engine = MoEAdaptrixEngine(
        model_id=TEST_CONFIG["model_id"],
        device=TEST_CONFIG["device"]
    )
    
    if not engine.initialize():
        print("Failed to initialize engine for benchmark")
        return
    
    try:
        test_prompts = [
            "What is artificial intelligence?",
            "Write a Python function to calculate fibonacci",
            "Explain contract law basics",
            "Solve: 2x + 5 = 15"
        ]
        
        total_time = 0
        total_tokens = 0
        
        for prompt in test_prompts:
            start_time = time.time()
            response = engine.generate(prompt, max_length=100)
            generation_time = time.time() - start_time
            
            total_time += generation_time
            total_tokens += len(response.split())
            
            print(f"Prompt: {prompt[:30]}...")
            print(f"Time: {generation_time:.2f}s")
            print(f"Tokens: {len(response.split())}")
            print("-" * 50)
        
        avg_time = total_time / len(test_prompts)
        tokens_per_second = total_tokens / total_time
        
        print(f"Average generation time: {avg_time:.2f}s")
        print(f"Tokens per second: {tokens_per_second:.2f}")
        
    finally:
        engine.cleanup()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Run benchmark if requested
    import sys
    if "--benchmark" in sys.argv:
        run_performance_benchmark()
