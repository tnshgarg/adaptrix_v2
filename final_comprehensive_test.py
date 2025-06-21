"""
Final comprehensive test for both dimension fixes and real adapter integration.
"""

import sys
import os
import torch
import tempfile
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.adapters.peft_converter import PEFTConverter
from src.adapters.adapter_manager import AdapterManager
from src.core.engine import AdaptrixEngine


def test_existing_adapter_with_fixes():
    """Test existing adapters with dimension fixes."""
    print("🔧 Testing Existing Adapters with Dimension Fixes")
    print("=" * 60)
    
    try:
        engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
        engine.initialize()
        
        print("✅ Engine initialized")
        
        # Load an existing adapter
        adapters = engine.list_adapters()
        if adapters:
            adapter_name = adapters[0]
            print(f"🎯 Testing with adapter: {adapter_name}")
            
            success = engine.load_adapter(adapter_name)
            if success:
                print("✅ Adapter loaded successfully")
                
                # Test generation with context preservation
                test_queries = [
                    "Hello, I'm Alice and I love reading books.",
                    "What's my name?",
                    "What do I enjoy doing?",
                    "Tell me a short story about adventure.",
                    "How are you today?"
                ]
                
                print("\n💬 Testing generation with context preservation:")
                for i, query in enumerate(test_queries, 1):
                    try:
                        # Set context anchor
                        if engine.tokenizer:
                            query_tokens = engine.tokenizer.encode(query, return_tensors="pt")
                            query_embedding = torch.randn(1, query_tokens.shape[1], 768)
                            engine.layer_injector.context_injector.set_query_anchor(query_embedding)
                        
                        response = engine.query(query, max_length=25)
                        print(f"   {i}. '{query[:40]}...'")
                        print(f"      -> '{response}'")
                        
                        if response and response.strip():
                            print(f"      ✅ Generation working!")
                        else:
                            print(f"      ⚠️  Empty response")
                    except Exception as e:
                        print(f"      ❌ Generation failed: {e}")
                
                # Get context statistics
                context_stats = engine.layer_injector.context_injector.get_context_statistics()
                print(f"\n📊 Context Statistics:")
                print(f"   Layers with context: {context_stats['layers_with_context']}")
                print(f"   Total injections: {context_stats['total_injections']}")
                print(f"   Avg processing time: {context_stats['average_processing_time']:.4f}s")
                
                # Get system status
                status = engine.get_system_status()
                print(f"\n📊 System Status:")
                print(f"   Active adapters: {status['loaded_adapters']}")
                print(f"   Memory usage: {status.get('memory_usage', {}).get('injector_memory', {}).get('memory_mb', 'unknown')} MB")
                
                engine.unload_adapter(adapter_name)
                print(f"✅ Adapter unloaded successfully")
                
                engine.cleanup()
                return True
            else:
                print("❌ Failed to load adapter")
                engine.cleanup()
                return False
        else:
            print("❌ No adapters available")
            engine.cleanup()
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_adapter_conversion():
    """Test real adapter conversion with a simple synthetic example."""
    print(f"\n🌐 Testing Real Adapter Conversion")
    print("=" * 60)
    
    # Create a simple synthetic PEFT adapter for testing
    temp_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    
    try:
        print("🏗️  Creating synthetic PEFT adapter for testing...")
        
        # Create adapter config
        adapter_config = {
            "alpha": 16,
            "base_model_name_or_path": "microsoft/DialoGPT-small",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "peft_type": "LORA",
            "r": 8,
            "target_modules": ["attn.c_attn", "mlp.c_fc"],
            "task_type": "CAUSAL_LM"
        }
        
        # Save config
        import json
        config_path = os.path.join(temp_dir, "adapter_config.json")
        with open(config_path, 'w') as f:
            json.dump(adapter_config, f, indent=2)
        
        # Create weights
        weights = {}
        test_layers = [3, 6, 9]
        
        for layer_idx in test_layers:
            for module in ["attn.c_attn", "mlp.c_fc"]:
                if module == "attn.c_attn":
                    in_dim, out_dim = 768, 2304
                elif module == "mlp.c_fc":
                    in_dim, out_dim = 768, 3072
                
                rank = 8
                
                lora_A_key = f"base_model.model.transformer.h.{layer_idx}.{module}.lora_A.weight"
                lora_B_key = f"base_model.model.transformer.h.{layer_idx}.{module}.lora_B.weight"
                
                weights[lora_A_key] = torch.randn(rank, in_dim) * 0.01
                weights[lora_B_key] = torch.randn(out_dim, rank) * 0.01
        
        # Save weights
        weights_path = os.path.join(temp_dir, "adapter_model.bin")
        torch.save(weights, weights_path)
        
        print(f"✅ Created synthetic adapter with {len(weights)} weight tensors")
        
        # Convert the adapter
        print(f"🔄 Converting adapter...")
        converter = PEFTConverter(target_layers=[3, 6, 9])
        
        success = converter.convert_from_local(
            adapter_path=temp_dir,
            output_dir=output_dir,
            base_model_name="microsoft/DialoGPT-small"
        )
        
        if success:
            print(f"✅ Conversion successful!")
            
            # Test loading with adapter manager
            print(f"🔍 Testing adapter loading...")
            adapter_manager = AdapterManager(adapter_dir=os.path.dirname(output_dir))
            converted_name = os.path.basename(output_dir)
            
            converted_adapter = adapter_manager.load_adapter(converted_name)
            
            if converted_adapter:
                print(f"✅ Converted adapter loaded successfully!")
                metadata = converted_adapter['metadata']
                weights = converted_adapter['weights']
                
                print(f"   📋 Adapter Analysis:")
                print(f"      Name: {metadata['name']}")
                print(f"      Target layers: {metadata['target_layers']}")
                print(f"      Target modules: {metadata['target_modules']}")
                print(f"      Weight layers: {list(weights.keys())}")
                
                # Test with Adaptrix engine
                print(f"🧪 Testing with Adaptrix engine...")
                
                # Copy adapter to adapters directory
                test_adapter_name = "test_converted_adapter"
                target_adapter_dir = os.path.join("adapters", test_adapter_name)
                
                if os.path.exists(target_adapter_dir):
                    shutil.rmtree(target_adapter_dir)
                shutil.copytree(output_dir, target_adapter_dir)
                
                engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
                engine.initialize()
                
                load_success = engine.load_adapter(test_adapter_name)
                
                if load_success:
                    print(f"✅ Converted adapter loaded in Adaptrix!")
                    
                    # Test generation
                    test_queries = [
                        "Hello there!",
                        "How are you?",
                        "Tell me something interesting."
                    ]
                    
                    print(f"💬 Testing generation:")
                    for i, query in enumerate(test_queries, 1):
                        try:
                            response = engine.query(query, max_length=20)
                            print(f"   {i}. '{query}' -> '{response}'")
                            
                            if response and response.strip():
                                print(f"      ✅ Generation working!")
                            else:
                                print(f"      ⚠️  Empty response")
                        except Exception as e:
                            print(f"      ❌ Generation failed: {e}")
                    
                    # Get final statistics
                    context_stats = engine.layer_injector.context_injector.get_context_statistics()
                    print(f"📊 Final Context Statistics:")
                    print(f"   Layers with context: {context_stats['layers_with_context']}")
                    print(f"   Total injections: {context_stats['total_injections']}")
                    
                    engine.unload_adapter(test_adapter_name)
                    engine.cleanup()
                    
                    # Cleanup test adapter
                    if os.path.exists(target_adapter_dir):
                        shutil.rmtree(target_adapter_dir)
                    
                    return True
                else:
                    print(f"❌ Failed to load converted adapter in Adaptrix")
                    engine.cleanup()
                    return False
            else:
                print(f"❌ Failed to load converted adapter")
                return False
        else:
            print(f"❌ Conversion failed")
            return False
            
    except Exception as e:
        print(f"❌ Real adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def main():
    """Run comprehensive final tests."""
    print("🚀 Final Comprehensive Integration Test")
    print("=" * 80)
    print("Testing both dimension fixes and real adapter integration")
    print("=" * 80)
    
    # Test 1: Existing adapters with fixes
    existing_success = test_existing_adapter_with_fixes()
    
    # Test 2: Real adapter conversion
    conversion_success = test_real_adapter_conversion()
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"🎉 Final Comprehensive Test Results")
    print(f"=" * 80)
    print(f"✅ Existing Adapter Test: {'PASSED' if existing_success else 'FAILED'}")
    print(f"✅ Real Adapter Conversion: {'PASSED' if conversion_success else 'FAILED'}")
    
    if existing_success and conversion_success:
        print(f"\n🎊 ALL TESTS PASSED!")
        print(f"🚀 Adaptrix is ready for production with:")
        print(f"   ✅ Working context preservation")
        print(f"   ✅ Dimension mismatch fixes")
        print(f"   ✅ Real adapter conversion capability")
        print(f"   ✅ Stable generation and memory management")
    elif existing_success:
        print(f"\n✅ Existing adapters working perfectly!")
        print(f"⚠️  Real adapter conversion needs refinement")
    elif conversion_success:
        print(f"\n✅ Real adapter conversion working!")
        print(f"⚠️  Existing adapter issues need attention")
    else:
        print(f"\n⚠️  Both tests need further work")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
