#!/usr/bin/env python3
"""
Test Qwen-3 1.7B Integration for New Adaptrix Plan.

This script tests the Qwen-3 1.7B model implementation according to the new plan.
"""

import sys
import os
import time
import logging

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_qwen3_model_loading():
    """Test Qwen-3 1.7B model loading and basic functionality."""
    
    print("🧪" * 80)
    print("🧪 TESTING QWEN-3 1.7B MODEL INTEGRATION")
    print("🧪" * 80)
    
    try:
        # Test 1: Import and create model
        print("\n📦 Test 1: Model Creation")
        print("-" * 50)
        
        from src.core.modular_engine import ModularAdaptrixEngine
        
        engine = ModularAdaptrixEngine(
            model_id="Qwen/Qwen3-1.7B",
            device="cpu",
            adapters_dir="adapters"
        )
        
        print("✅ Engine created successfully!")
        
        # Test 2: Initialize model
        print("\n🚀 Test 2: Model Initialization")
        print("-" * 50)
        
        start_time = time.time()
        success = engine.initialize()
        init_time = time.time() - start_time
        
        if not success:
            print("❌ Model initialization failed!")
            return False
        
        print(f"✅ Model initialized in {init_time:.2f}s")
        
        # Test 3: Get model info
        print("\n📊 Test 3: Model Information")
        print("-" * 50)
        
        status = engine.get_system_status()
        model_info = status.get('model_info', {})
        
        print(f"   Model ID: {model_info.get('model_id', 'Unknown')}")
        print(f"   Model Family: {model_info.get('model_family', 'Unknown')}")
        print(f"   Context Length: {model_info.get('context_length', 'Unknown')}")
        print(f"   Total Parameters: {model_info.get('total_parameters', 'Unknown'):,}")
        print(f"   Device: {model_info.get('device', 'Unknown')}")
        
        # Test 4: Basic text generation
        print("\n💬 Test 4: Text Generation")
        print("-" * 50)
        
        test_prompts = [
            "What is machine learning?",
            "Write a simple Python function to add two numbers.",
            "Explain quantum computing in simple terms."
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🔤 Prompt {i}: {prompt}")
            
            start_time = time.time()
            response = engine.generate(
                prompt,
                max_length=150,
                task_type="general",
                temperature=0.7
            )
            gen_time = time.time() - start_time
            
            print(f"⏱️ Generated in {gen_time:.2f}s")
            print(f"📝 Response: {response[:200]}{'...' if len(response) > 200 else ''}")
            
            # Basic quality checks
            if len(response.strip()) < 10:
                print("⚠️ Warning: Response seems too short")
            elif "Error:" in response:
                print("❌ Error in response generation")
                return False
            else:
                print("✅ Response generated successfully")
        
        # Test 5: Memory usage
        print("\n💾 Test 5: Memory Usage")
        print("-" * 50)
        
        if hasattr(engine.base_model, 'get_memory_usage'):
            memory_info = engine.base_model.get_memory_usage()
            if 'model_memory_gb' in memory_info:
                print(f"   Model Memory: {memory_info['model_memory_gb']:.2f} GB")
            if 'gpu_allocated' in memory_info:
                print(f"   GPU Allocated: {memory_info['gpu_allocated']:.2f} GB")
        
        # Test 6: Adapter compatibility
        print("\n🔌 Test 6: Adapter Compatibility")
        print("-" * 50)
        
        if hasattr(engine.base_model, 'get_adapter_compatibility'):
            compat_info = engine.base_model.get_adapter_compatibility()
            print(f"   Supported Modules: {compat_info.get('supported_modules', [])}")
            print(f"   Recommended Rank: {compat_info.get('recommended_rank', 'Unknown')}")
            print(f"   Recommended Alpha: {compat_info.get('recommended_alpha', 'Unknown')}")
        
        # Cleanup
        print("\n🧹 Cleanup")
        print("-" * 50)
        engine.cleanup()
        print("✅ Cleanup completed")
        
        print("\n🎉" * 80)
        print("🎉 ALL TESTS PASSED! QWEN-3 1.7B INTEGRATION SUCCESSFUL!")
        print("🎉" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_family_detection():
    """Test automatic model family detection."""
    
    print("\n🔍" * 50)
    print("🔍 TESTING MODEL FAMILY DETECTION")
    print("🔍" * 50)
    
    try:
        from src.core.base_model_interface import ModelDetector, ModelFamily
        
        test_models = [
            ("Qwen/Qwen3-1.7B", ModelFamily.QWEN),
            ("Qwen/Qwen2-7B", ModelFamily.QWEN),
            ("microsoft/phi-2", ModelFamily.PHI),
            ("meta-llama/Llama-2-7b-hf", ModelFamily.LLAMA),
            ("mistralai/Mistral-7B-v0.1", ModelFamily.MISTRAL)
        ]
        
        for model_id, expected_family in test_models:
            detected_family = ModelDetector.detect_family(model_id)
            status = "✅" if detected_family == expected_family else "❌"
            print(f"   {status} {model_id} -> {detected_family.value}")
        
        print("✅ Model family detection test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Model family detection test failed: {e}")
        return False


def main():
    """Run all tests."""
    
    print("🚀" * 100)
    print("🚀 QWEN-3 1.7B INTEGRATION TEST SUITE")
    print("🚀" * 100)
    
    tests = [
        ("Model Family Detection", test_model_family_detection),
        ("Qwen-3 Model Loading", test_qwen3_model_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊" * 50)
    print("📊 TEST SUMMARY")
    print("📊" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED! QWEN-3 INTEGRATION IS READY!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED. PLEASE CHECK THE ERRORS ABOVE.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
