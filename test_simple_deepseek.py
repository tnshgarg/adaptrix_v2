"""
Simple test for DeepSeek-R1 without context preservation to isolate issues.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.core.engine import AdaptrixEngine


def test_base_model_only():
    """Test just the base model without any adapters."""
    print("🔍 Testing Base Model Only")
    print("=" * 50)
    
    try:
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        # Disable context preservation
        if hasattr(engine.layer_injector, 'enable_context_preservation'):
            engine.layer_injector.enable_context_preservation = False
            print("✅ Context preservation disabled")
        
        test_queries = [
            "Hello, how are you?",
            "What is 2 + 2?",
            "My name is Alice."
        ]
        
        print("\n💬 Base Model Responses:")
        for i, query in enumerate(test_queries, 1):
            try:
                response = engine.query(query, max_length=20)
                print(f"   {i}. '{query}' → '{response}'")
            except Exception as e:
                print(f"   {i}. '{query}' → ERROR: {e}")
        
        engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_adapter():
    """Test with a simple adapter with correct dimensions."""
    print("\n🔧 Testing Simple Adapter")
    print("=" * 50)
    
    try:
        import torch
        import json
        import shutil
        
        # Create a simple test adapter with correct dimensions
        adapter_dir = "adapters/simple_test"
        if os.path.exists(adapter_dir):
            shutil.rmtree(adapter_dir)
        os.makedirs(adapter_dir)
        
        # Create metadata
        metadata = {
            'name': 'simple_test',
            'version': '1.0.0',
            'description': 'Simple test adapter for DeepSeek-R1',
            'source': 'manual_creation',
            'base_model': 'deepseek-ai/deepseek-r1-distill-qwen-1.5b',
            'target_layers': [14],  # Just one layer
            'target_modules': ['self_attn.q_proj'],  # Just one module
            'rank': 4,  # Small rank
            'alpha': 8
        }
        
        with open(os.path.join(adapter_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create weights for layer 14, q_proj only
        # q_proj: 1536 -> 1536 (from architecture analysis)
        layer_weights = {
            'self_attn.q_proj': {
                'lora_A': torch.randn(4, 1536) * 0.01,  # rank=4, input=1536
                'lora_B': torch.randn(1536, 4) * 0.01,  # output=1536, rank=4
                'rank': 4,
                'alpha': 8
            }
        }
        
        layer_file = os.path.join(adapter_dir, "layer_14.pt")
        torch.save(layer_weights, layer_file)
        
        print("✅ Created simple test adapter")
        
        # Test with the adapter
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        # Disable context preservation
        if hasattr(engine.layer_injector, 'enable_context_preservation'):
            engine.layer_injector.enable_context_preservation = False
            print("✅ Context preservation disabled")
        
        print("\n💬 Testing without adapter:")
        response1 = engine.query("Hello, how are you?", max_length=15)
        print(f"   Without adapter: '{response1}'")
        
        print("\n💬 Testing with adapter:")
        success = engine.load_adapter("simple_test")
        if success:
            print("   ✅ Simple adapter loaded")
            response2 = engine.query("Hello, how are you?", max_length=15)
            print(f"   With adapter: '{response2}'")
            
            # Check if responses are different (indicating adapter is working)
            if response1 != response2:
                print("   ✅ Adapter is affecting output (responses differ)")
            else:
                print("   ⚠️  Adapter may not be working (responses identical)")
        else:
            print("   ❌ Failed to load adapter")
        
        engine.cleanup()
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run simple tests to isolate issues."""
    print("🚀 Simple DeepSeek-R1 Testing")
    print("=" * 60)
    print("Testing base model and simple adapter without context preservation")
    print("=" * 60)
    
    # Test 1: Base model only
    base_working = test_base_model_only()
    
    # Test 2: Simple adapter
    if base_working:
        adapter_working = test_simple_adapter()
    else:
        adapter_working = False
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"🎉 Simple Test Results")
    print(f"=" * 60)
    print(f"💬 Base model: {'✅ WORKING' if base_working else '❌ FAILED'}")
    print(f"🔧 Simple adapter: {'✅ WORKING' if adapter_working else '❌ FAILED'}")
    
    if base_working and adapter_working:
        print(f"\n🎊 BASIC FUNCTIONALITY WORKING!")
        print(f"✅ Ready to debug context preservation and complex adapters")
    else:
        print(f"\n⚠️  Basic issues need to be resolved first")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
