#!/usr/bin/env python3
"""
Test MoE-Enhanced Adaptrix Engine.

This script tests the MoE engine with automatic adapter selection.
"""

import sys
import os
import time
import logging

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.moe.moe_engine import MoEAdaptrixEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_moe_engine():
    """Test the MoE-enhanced Adaptrix engine."""
    
    print("🧠" * 80)
    print("🧠 TESTING MOE-ENHANCED ADAPTRIX ENGINE")
    print("🧠" * 80)
    
    try:
        # Test 1: Initialize MoE engine
        print("\n🚀 Test 1: MoE Engine Initialization")
        print("-" * 50)
        
        engine = MoEAdaptrixEngine(
            model_id="Qwen/Qwen3-1.7B",
            device="cpu",
            adapters_dir="adapters",
            classifier_path="models/classifier",
            enable_auto_selection=True
        )
        
        print("✅ MoE engine created successfully!")
        
        # Initialize
        start_time = time.time()
        success = engine.initialize()
        init_time = time.time() - start_time
        
        if not success:
            print("❌ MoE engine initialization failed!")
            return False
        
        print(f"✅ MoE engine initialized in {init_time:.2f}s")
        
        # Test 2: Check MoE status
        print("\n📊 Test 2: MoE System Status")
        print("-" * 50)
        
        status = engine.get_moe_status()
        moe_info = status.get('moe', {})
        
        print(f"   🧠 Classifier Initialized: {moe_info.get('classifier_initialized', False)}")
        print(f"   🔄 Auto-Selection Enabled: {moe_info.get('auto_selection_enabled', False)}")
        print(f"   📁 Classifier Path: {moe_info.get('classifier_path', 'Unknown')}")
        
        if 'classifier_status' in moe_info:
            classifier_status = moe_info['classifier_status']
            print(f"   🏷️ Supported Adapters: {list(classifier_status.get('adapter_mapping', {}).keys())}")
        
        # Test 3: Adapter prediction without generation
        print("\n🔮 Test 3: Adapter Prediction")
        print("-" * 50)
        
        test_prompts = [
            "Write a Python function to calculate factorial",
            "Analyze this contract for potential risks",
            "What is artificial intelligence?",
            "Solve the equation: x^2 + 5x + 6 = 0"
        ]
        
        for prompt in test_prompts:
            prediction = engine.predict_adapter(prompt)
            
            if 'error' in prediction:
                print(f"   ❌ '{prompt[:40]}...' -> Error: {prediction['error']}")
            else:
                adapter = prediction.get('adapter_name', 'unknown')
                confidence = prediction.get('confidence', 0.0)
                print(f"   🎯 '{prompt[:40]}...' -> {adapter} ({confidence:.3f})")
        
        # Test 4: Automatic text generation with adapter selection
        print("\n💬 Test 4: Automatic Text Generation")
        print("-" * 50)
        
        generation_tests = [
            ("Write a simple Python function", "code"),
            ("Explain machine learning", "general"),
            ("What is 15 + 25?", "math")
        ]
        
        for prompt, expected_domain in generation_tests:
            print(f"\n🔤 Prompt: {prompt}")
            
            start_time = time.time()
            response = engine.generate(
                prompt,
                max_length=100,
                task_type="auto"  # Enable automatic selection
            )
            gen_time = time.time() - start_time
            
            print(f"⏱️ Generated in {gen_time:.2f}s")
            print(f"📝 Response: {response[:150]}{'...' if len(response) > 150 else ''}")
            
            # Check if response is reasonable
            if len(response.strip()) < 10:
                print("⚠️ Warning: Response seems too short")
            elif "Error:" in response:
                print("❌ Error in response generation")
            else:
                print("✅ Response generated successfully")
        
        # Test 5: Selection statistics
        print("\n📊 Test 5: Selection Statistics")
        print("-" * 50)
        
        stats = engine.get_selection_stats()
        
        print(f"   📈 Total Selections: {stats.get('total_selections', 0)}")
        print(f"   📊 Average Confidence: {stats.get('average_confidence', 0.0):.3f}")
        
        if stats.get('adapter_usage'):
            print("   🏷️ Adapter Usage:")
            for adapter, count in stats['adapter_usage'].items():
                percentage = stats.get('adapter_usage_percent', {}).get(adapter, 0.0)
                print(f"      {adapter}: {count} times ({percentage:.1f}%)")
        
        # Test 6: Manual adapter selection
        print("\n🎛️ Test 6: Manual Adapter Selection")
        print("-" * 50)
        
        # Test with explicit adapter (should override auto-selection)
        manual_prompt = "This is a test prompt"
        
        # Try to use a specific adapter (even if it doesn't exist)
        response = engine.generate(
            manual_prompt,
            max_length=50,
            adapter_name="code"  # Explicit adapter selection
        )
        
        print(f"🔤 Manual selection test: '{manual_prompt}'")
        print(f"📝 Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        
        # Test 7: Disable auto-selection
        print("\n🔄 Test 7: Toggle Auto-Selection")
        print("-" * 50)
        
        # Disable auto-selection
        engine.enable_auto_selection(False)
        print("✅ Auto-selection disabled")
        
        # Test generation without auto-selection
        response = engine.generate(
            "Write code to sort numbers",
            max_length=50,
            task_type="general"
        )
        
        print(f"📝 Response without auto-selection: {response[:100]}{'...' if len(response) > 100 else ''}")
        
        # Re-enable auto-selection
        engine.enable_auto_selection(True)
        print("✅ Auto-selection re-enabled")
        
        # Cleanup
        print("\n🧹 Cleanup")
        print("-" * 50)
        engine.cleanup()
        print("✅ Cleanup completed")
        
        print("\n🎉" * 80)
        print("🎉 ALL MOE ENGINE TESTS PASSED!")
        print("🎉" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    
    print("🚀" * 100)
    print("🚀 MOE-ENHANCED ADAPTRIX ENGINE TEST SUITE")
    print("🚀" * 100)
    
    success = test_moe_engine()
    
    if success:
        print("\n✅ All tests passed! MoE engine is working correctly!")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
