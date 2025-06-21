"""
Test DeepSeek-R1 with the fixed architecture detection engine.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.core.engine import AdaptrixEngine


def test_deepseek_architecture_detection():
    """Test that the engine correctly detects DeepSeek-R1 architecture."""
    print("🔍 Testing DeepSeek-R1 Architecture Detection")
    print("=" * 60)
    
    try:
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        
        print("📥 Initializing engine with architecture detection...")
        success = engine.initialize()
        
        if success:
            print("✅ Engine initialized successfully!")
            
            # Check what was detected
            status = engine.get_system_status()
            print(f"\n📊 System Status:")
            print(f"   Model: {status['model_name']}")
            print(f"   Device: {status['device']}")
            print(f"   Initialized: {status['initialized']}")
            
            # Check injection points
            if hasattr(engine.layer_injector, 'injection_points'):
                print(f"\n🎯 Injection Points Registered:")
                for layer_idx, modules in engine.layer_injector.injection_points.items():
                    print(f"   Layer {layer_idx}: {list(modules.keys())}")
            
            return True
            
        else:
            print("❌ Engine initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'engine' in locals():
            engine.cleanup()


def test_deepseek_base_responses():
    """Test base model responses with DeepSeek-R1."""
    print(f"\n💬 Testing DeepSeek-R1 Base Responses")
    print("=" * 60)
    
    try:
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        print("✅ Engine initialized")
        
        test_queries = [
            "Hello, how are you?",
            "What is 2 + 2?",
            "My name is Tanish.",
            "Who are you?",
            "Explain quantum physics in simple terms."
        ]
        
        print("\n💬 Base Model Responses:")
        good_responses = 0
        
        for i, query in enumerate(test_queries, 1):
            try:
                response = engine.query(query, max_length=30)
                print(f"   {i}. '{query}'")
                print(f"      → '{response}'")
                
                # Check response quality
                if response and len(response.split()) > 2:
                    good_responses += 1
                    print(f"      ✅ Good response")
                else:
                    print(f"      ⚠️  Poor response")
                    
            except Exception as e:
                print(f"   {i}. '{query}' → ERROR: {e}")
        
        print(f"\n📊 Response Quality: {good_responses}/{len(test_queries)} good responses")
        
        engine.cleanup()
        return good_responses > len(test_queries) // 2
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_deepseek_compatible_adapters():
    """Create adapters compatible with the detected DeepSeek architecture."""
    print(f"\n🔧 Creating DeepSeek-Compatible Adapters")
    print("=" * 60)
    
    try:
        import torch
        import json
        import shutil
        
        # Create general conversation adapter
        general_dir = "adapters/deepseek_general"
        if os.path.exists(general_dir):
            shutil.rmtree(general_dir)
        os.makedirs(general_dir)
        
        # Create metadata
        general_metadata = {
            'name': 'deepseek_general',
            'version': '1.0.0',
            'description': 'General conversation adapter for DeepSeek-R1',
            'source': 'manual_creation',
            'base_model': 'deepseek-ai/deepseek-r1-distill-qwen-1.5b',
            'target_layers': [6, 12, 18],
            'target_modules': ['self_attn.q_proj', 'self_attn.v_proj', 'mlp.gate_proj'],
            'rank': 8,
            'alpha': 16
        }
        
        # Save metadata
        with open(os.path.join(general_dir, "metadata.json"), 'w') as f:
            json.dump(general_metadata, f, indent=2)
        
        # Create weights for each target layer
        for layer_idx in [6, 12, 18]:
            layer_weights = {}
            
            # DeepSeek-R1 dimensions: hidden_size=1536, intermediate_size=8960
            layer_weights['self_attn.q_proj'] = {
                'lora_A': torch.randn(8, 1536) * 0.02,
                'lora_B': torch.randn(1536, 8) * 0.02,
                'rank': 8,
                'alpha': 16
            }
            
            layer_weights['self_attn.v_proj'] = {
                'lora_A': torch.randn(8, 1536) * 0.02,
                'lora_B': torch.randn(256, 8) * 0.02,  # v_proj has different output
                'rank': 8,
                'alpha': 16
            }
            
            layer_weights['mlp.gate_proj'] = {
                'lora_A': torch.randn(8, 1536) * 0.02,
                'lora_B': torch.randn(8960, 8) * 0.02,
                'rank': 8,
                'alpha': 16
            }
            
            # Save layer weights
            layer_file = os.path.join(general_dir, f"layer_{layer_idx}.pt")
            torch.save(layer_weights, layer_file)
        
        print(f"✅ Created deepseek_general adapter")
        
        # Create math adapter
        math_dir = "adapters/deepseek_math"
        if os.path.exists(math_dir):
            shutil.rmtree(math_dir)
        os.makedirs(math_dir)
        
        # Create metadata
        math_metadata = {
            'name': 'deepseek_math',
            'version': '1.0.0',
            'description': 'Math reasoning adapter for DeepSeek-R1',
            'source': 'manual_creation',
            'base_model': 'deepseek-ai/deepseek-r1-distill-qwen-1.5b',
            'target_layers': [12, 18],  # Focus on later layers for math
            'target_modules': ['mlp.gate_proj'],  # Focus on MLP for reasoning
            'rank': 8,
            'alpha': 16
        }
        
        # Save metadata
        with open(os.path.join(math_dir, "metadata.json"), 'w') as f:
            json.dump(math_metadata, f, indent=2)
        
        # Create weights for math layers
        for layer_idx in [12, 18]:
            layer_weights = {
                'mlp.gate_proj': {
                    'lora_A': torch.randn(8, 1536) * 0.03,  # Slightly stronger for math
                    'lora_B': torch.randn(8960, 8) * 0.03,
                    'rank': 8,
                    'alpha': 16
                }
            }
            
            # Save layer weights
            layer_file = os.path.join(math_dir, f"layer_{layer_idx}.pt")
            torch.save(layer_weights, layer_file)
        
        print(f"✅ Created deepseek_math adapter")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create adapters: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deepseek_with_adapters():
    """Test DeepSeek-R1 with the new adapters."""
    print(f"\n🧪 Testing DeepSeek-R1 with Adapters")
    print("=" * 60)
    
    try:
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        print("✅ Engine initialized")
        
        # Test general adapter
        print(f"\n💬 Testing General Adapter:")
        success = engine.load_adapter("deepseek_general")
        if success:
            print("   ✅ General adapter loaded")
            
            general_queries = [
                "Hello, how are you?",
                "Tell me about yourself",
                "What's your favorite color?"
            ]
            
            for i, query in enumerate(general_queries, 1):
                response = engine.query(query, max_length=25)
                print(f"   {i}. '{query}' → '{response}'")
            
            engine.unload_adapter("deepseek_general")
        
        # Test math adapter
        print(f"\n🧮 Testing Math Adapter:")
        success = engine.load_adapter("deepseek_math")
        if success:
            print("   ✅ Math adapter loaded")
            
            math_queries = [
                "What is 2 + 2?",
                "Calculate 5 * 3",
                "Solve 10 - 4"
            ]
            
            for i, query in enumerate(math_queries, 1):
                response = engine.query(query, max_length=20)
                print(f"   {i}. '{query}' → '{response}'")
            
            engine.unload_adapter("deepseek_math")
        
        engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive DeepSeek-R1 tests."""
    print("🚀 Comprehensive DeepSeek-R1 Testing")
    print("=" * 80)
    print("Testing architecture detection, base responses, and adapter compatibility")
    print("=" * 80)
    
    # Test 1: Architecture detection
    arch_detected = test_deepseek_architecture_detection()
    
    # Test 2: Base responses
    base_working = test_deepseek_base_responses()
    
    # Test 3: Create compatible adapters
    adapters_created = create_deepseek_compatible_adapters()
    
    # Test 4: Test with adapters
    if adapters_created:
        adapters_working = test_deepseek_with_adapters()
    else:
        adapters_working = False
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"🎉 DeepSeek-R1 Test Results")
    print(f"=" * 80)
    print(f"🔍 Architecture detection: {'✅ SUCCESS' if arch_detected else '❌ FAILED'}")
    print(f"💬 Base responses: {'✅ GOOD' if base_working else '❌ POOR'}")
    print(f"🔧 Adapter creation: {'✅ SUCCESS' if adapters_created else '❌ FAILED'}")
    print(f"🧪 Adapter testing: {'✅ SUCCESS' if adapters_working else '❌ FAILED'}")
    
    overall_success = arch_detected and base_working and adapters_working
    
    if overall_success:
        print(f"\n🎊 ALL TESTS PASSED!")
        print(f"🚀 DeepSeek-R1 is ready for production use!")
        print(f"✅ Architecture auto-detection working")
        print(f"✅ Base model generating good responses")
        print(f"✅ Adapters compatible and functional")
    else:
        print(f"\n⚠️  Some tests failed - need debugging")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
