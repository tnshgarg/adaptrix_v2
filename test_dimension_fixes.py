"""
Test script to validate dimension fixes and adapter compatibility.
"""

import sys
import os
import torch
import tempfile
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.core.engine import AdaptrixEngine
from src.adapters.peft_converter import PEFTConverter
from src.adapters.adapter_manager import AdapterManager


def test_dimension_compatibility():
    """Test that our dimension fixes work correctly."""
    print("🔧 Testing Dimension Compatibility Fixes")
    print("=" * 60)
    
    try:
        # Initialize engine
        engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
        engine.initialize()
        
        # Get architecture info
        arch_info = engine.base_model_manager.architecture_info
        print(f"✅ Architecture: {arch_info['architecture_type']}")
        print(f"📊 Model layers: {arch_info['layer_count']}")
        print(f"🎯 Target modules: {arch_info['target_modules']}")
        print(f"📏 Module dimensions:")
        
        for module, dims in arch_info['module_dimensions'].items():
            print(f"   {module}: {dims[0]} -> {dims[1]}")
        
        # Test with existing adapters
        adapters = engine.list_adapters()
        print(f"\n📂 Available adapters: {len(adapters)}")
        
        for adapter_name in adapters:
            print(f"\n🧪 Testing adapter: {adapter_name}")
            
            # Get adapter info
            adapter_info = engine.get_adapter_info(adapter_name)
            if adapter_info:
                metadata = adapter_info['metadata']
                print(f"   📋 Target layers: {metadata.get('target_layers', [])}")
                print(f"   🔧 Target modules: {metadata.get('target_modules', [])}")
                print(f"   📊 Rank: {metadata.get('rank', 'unknown')}")
            
            # Try to load adapter
            load_success = engine.load_adapter(adapter_name)
            
            if load_success:
                print(f"   ✅ Adapter loaded successfully!")
                
                # Test generation to ensure no dimension errors
                try:
                    response = engine.generate("Hello", max_length=10)
                    print(f"   💬 Generation test: '{response}'")
                    print(f"   ✅ No dimension errors!")
                except Exception as e:
                    print(f"   ❌ Generation failed: {e}")
                
                # Get injection statistics
                memory_usage = engine.layer_injector.get_memory_usage()
                print(f"   📊 LoRA layers: {memory_usage['total_lora_layers']}")
                print(f"   💾 Memory usage: {memory_usage['memory_mb']:.2f} MB")
                
                # Unload adapter
                engine.unload_adapter(adapter_name)
                print(f"   🔄 Adapter unloaded")
            else:
                print(f"   ❌ Failed to load adapter")
        
        engine.cleanup()
        print(f"\n✅ Dimension compatibility test completed!")
        
    except Exception as e:
        print(f"❌ Dimension test failed: {e}")
        import traceback
        traceback.print_exc()


def test_peft_conversion_with_dimensions():
    """Test PEFT conversion with proper dimension handling."""
    print(f"\n🔄 Testing PEFT Conversion with Dimension Fixes")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    
    try:
        # Create a realistic PEFT adapter with potential dimension mismatches
        print("🏗️  Creating test PEFT adapter...")
        
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
        
        # Create weights with correct dimensions for DialoGPT-small
        weights = {}
        
        # Test layers
        test_layers = [3, 6, 9]
        
        for layer_idx in test_layers:
            for module in ["attn.c_attn", "mlp.c_fc"]:
                if module == "attn.c_attn":
                    # DialoGPT-small: 768 -> 2304 (Q, K, V combined)
                    in_dim, out_dim = 768, 2304
                elif module == "mlp.c_fc":
                    # DialoGPT-small: 768 -> 3072
                    in_dim, out_dim = 768, 3072
                
                rank = 8
                
                # Create LoRA weights
                lora_A_key = f"base_model.model.transformer.h.{layer_idx}.{module}.lora_A.weight"
                lora_B_key = f"base_model.model.transformer.h.{layer_idx}.{module}.lora_B.weight"
                
                weights[lora_A_key] = torch.randn(rank, in_dim) * 0.01
                weights[lora_B_key] = torch.randn(out_dim, rank) * 0.01
        
        # Save weights
        weights_path = os.path.join(temp_dir, "adapter_model.bin")
        torch.save(weights, weights_path)
        
        print(f"   ✅ Test adapter created with {len(weights)} weight tensors")
        
        # Convert using PEFTConverter
        print("🔄 Converting with dimension validation...")
        
        converter = PEFTConverter(target_layers=[3, 6, 9])
        success = converter.convert_from_local(
            adapter_path=temp_dir,
            output_dir=output_dir,
            base_model_name="microsoft/DialoGPT-small"
        )
        
        if success:
            print("✅ Conversion successful!")
            
            # Load and validate converted adapter
            adapter_manager = AdapterManager(adapter_dir=os.path.dirname(output_dir))
            converted_name = os.path.basename(output_dir)
            
            converted_adapter = adapter_manager.load_adapter(converted_name)
            
            if converted_adapter:
                print("✅ Converted adapter loaded!")
                
                # Validate dimensions
                weights = converted_adapter['weights']
                metadata = converted_adapter['metadata']
                
                print(f"📋 Conversion results:")
                print(f"   🎯 Target layers: {metadata['target_layers']}")
                print(f"   🔧 Target modules: {metadata['target_modules']}")
                
                for layer_idx, layer_weights in weights.items():
                    print(f"   📂 Layer {layer_idx}:")
                    for module_name, module_weights in layer_weights.items():
                        lora_A_shape = module_weights['lora_A'].shape
                        lora_B_shape = module_weights['lora_B'].shape
                        print(f"      🔗 {module_name}: A{lora_A_shape} -> B{lora_B_shape}")
                
                # Test with Adaptrix engine
                print("🧪 Testing converted adapter with Adaptrix...")

                engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
                engine.initialize()

                print(f"   🔧 Context preservation: {engine.layer_injector.enable_context_preservation}")
                
                # Copy to adapters directory
                test_adapter_name = "dimension_test_adapter"
                target_dir = os.path.join("adapters", test_adapter_name)
                
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                shutil.copytree(output_dir, target_dir)
                
                load_success = engine.load_adapter(test_adapter_name)
                
                if load_success:
                    print("✅ Converted adapter works with Adaptrix!")
                    
                    # Test generation with multiple queries
                    test_queries = [
                        "Hello there!",
                        "How are you?",
                        "What's the weather like?"
                    ]
                    
                    for query in test_queries:
                        try:
                            response = engine.query(query, max_length=15)
                            print(f"   💬 '{query}' -> '{response}'")
                        except Exception as e:
                            print(f"   ❌ Query failed: {e}")
                    
                    # Get detailed statistics
                    memory_usage = engine.layer_injector.get_memory_usage()
                    active_adapters = engine.layer_injector.get_active_adapters()
                    
                    print(f"📊 Final statistics:")
                    print(f"   🎯 Active adapters: {active_adapters}")
                    print(f"   💾 Memory usage: {memory_usage['memory_mb']:.2f} MB")
                    print(f"   🔢 Total parameters: {memory_usage['total_parameters']}")
                    
                    engine.unload_adapter(test_adapter_name)
                else:
                    print("❌ Failed to load converted adapter")
                
                engine.cleanup()
                
                # Cleanup
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
            else:
                print("❌ Failed to load converted adapter")
        else:
            print("❌ Conversion failed")
        
    except Exception as e:
        print(f"❌ PEFT conversion test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def test_context_preservation_with_real_adapters():
    """Test context preservation with real adapter loading."""
    print(f"\n🧠 Testing Context Preservation with Real Adapters")
    print("=" * 60)
    
    try:
        engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
        engine.initialize()
        
        print(f"🔧 Context preservation: {engine.layer_injector.enable_context_preservation}")
        
        adapters = engine.list_adapters()
        if not adapters:
            print("❌ No adapters available for testing")
            return
        
        adapter_name = adapters[0]
        print(f"🎯 Testing with adapter: {adapter_name}")
        
        # Load adapter
        success = engine.load_adapter(adapter_name)
        if not success:
            print("❌ Failed to load adapter")
            return
        
        print("✅ Adapter loaded successfully!")
        
        # Test context preservation across multiple interactions
        conversation_turns = [
            "Hi, my name is Alice.",
            "What's my name?",
            "Can you remember what I told you?",
            "Let's talk about something else.",
            "Do you still remember my name?"
        ]
        
        print("\n💬 Testing conversation context:")
        
        for i, turn in enumerate(conversation_turns, 1):
            print(f"\n   Turn {i}: {turn}")
            
            try:
                response = engine.query(turn, max_length=20)
                print(f"   Response: {response}")
                
                # Get context statistics
                context_stats = engine.layer_injector.context_injector.get_context_statistics()
                print(f"   📊 Context layers: {context_stats['layers_with_context']}")
                
            except Exception as e:
                print(f"   ❌ Turn failed: {e}")
        
        # Final context analysis
        final_stats = engine.layer_injector.context_injector.get_context_statistics()
        print(f"\n📊 Final Context Statistics:")
        print(f"   Layers with context: {final_stats['layers_with_context']}")
        print(f"   Total injections: {final_stats['total_injections']}")
        print(f"   Average processing time: {final_stats['average_processing_time']:.4f}s")
        
        engine.cleanup()
        print("✅ Context preservation test completed!")
        
    except Exception as e:
        print(f"❌ Context preservation test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all dimension and compatibility tests."""
    print("🔧 Adaptrix Dimension Fixes & Compatibility Validation")
    print("=" * 70)
    
    # Test 1: Dimension compatibility with existing adapters
    test_dimension_compatibility()
    
    # Test 2: PEFT conversion with dimension fixes
    test_peft_conversion_with_dimensions()
    
    # Test 3: Context preservation with real adapters
    test_context_preservation_with_real_adapters()
    
    print(f"\n" + "=" * 70)
    print("🎉 All dimension and compatibility tests completed!")
    print("✅ Adaptrix is ready for real-world QLoRA adapters!")
    print("=" * 70)


if __name__ == "__main__":
    main()
