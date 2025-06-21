"""
Test script for PEFT integration and adapter conversion.
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
from src.models.architecture_registry import architecture_registry


def test_architecture_detection():
    """Test architecture detection for different models."""
    print("Testing Architecture Detection")
    print("=" * 50)
    
    test_models = [
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium", 
        "gpt2",
        "gpt2-medium"
    ]
    
    for model_name in test_models:
        try:
            arch_info = architecture_registry.get_architecture_info(model_name)
            print(f"\nModel: {model_name}")
            print(f"  Architecture: {arch_info['architecture_type']}")
            print(f"  Layers: {arch_info['layer_count']}")
            print(f"  Hidden Size: {arch_info['hidden_size']}")
            print(f"  Target Modules: {arch_info['target_modules'][:3]}...")  # Show first 3
            print(f"  Recommended Middle Layers: {arch_info['recommended_middle_layers']}")
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    
    print("\n✅ Architecture detection test completed")


def test_context_preservation():
    """Test context preservation functionality."""
    print("\nTesting Context Preservation")
    print("=" * 50)
    
    try:
        # Initialize engine with context preservation
        engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
        engine.initialize()
        
        # Check if context preservation is enabled
        context_enabled = engine.layer_injector.enable_context_preservation
        print(f"Context preservation enabled: {context_enabled}")
        
        # Get context statistics
        context_stats = engine.layer_injector.context_injector.get_context_statistics()
        print(f"Context statistics: {context_stats}")
        
        # Test with an adapter if available
        adapters = engine.list_adapters()
        if adapters:
            print(f"\nTesting with adapter: {adapters[0]}")
            
            # Set query anchor for context preservation
            test_query = "What is the capital of France?"
            
            # Generate with context preservation
            response1 = engine.query(test_query, adapter_name=adapters[0], max_length=20)
            print(f"Response 1: {response1}")
            
            # Generate again to test context consistency
            response2 = engine.query(test_query, adapter_name=adapters[0], max_length=20)
            print(f"Response 2: {response2}")
            
            # Get updated context statistics
            final_stats = engine.layer_injector.context_injector.get_context_statistics()
            print(f"Final context statistics: {final_stats}")
        
        engine.cleanup()
        print("✅ Context preservation test completed")
        
    except Exception as e:
        print(f"❌ Context preservation test failed: {e}")
        import traceback
        traceback.print_exc()


def create_mock_peft_adapter():
    """Create a mock PEFT adapter for testing conversion."""
    print("\nCreating Mock PEFT Adapter")
    print("=" * 50)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create adapter config
        adapter_config = {
            "base_model_name_or_path": "microsoft/DialoGPT-small",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": 16,
            "revision": None,
            "target_modules": ["attn.c_attn", "mlp.c_fc"],
            "task_type": "CAUSAL_LM"
        }
        
        # Save config
        import json
        config_path = os.path.join(temp_dir, "adapter_config.json")
        with open(config_path, 'w') as f:
            json.dump(adapter_config, f, indent=2)
        
        # Create mock weights
        mock_weights = {}
        
        # Create weights for a few layers
        for layer_idx in [3, 6, 9]:
            for module in ["attn.c_attn", "mlp.c_fc"]:
                if module == "attn.c_attn":
                    # DialoGPT c_attn: 768 -> 2304 (Q, K, V combined)
                    in_dim, out_dim = 768, 2304
                else:  # mlp.c_fc
                    # DialoGPT mlp.c_fc: 768 -> 3072
                    in_dim, out_dim = 768, 3072
                
                rank = 16
                
                # Create LoRA A and B weights
                lora_A_key = f"base_model.model.transformer.h.{layer_idx}.{module}.lora_A.weight"
                lora_B_key = f"base_model.model.transformer.h.{layer_idx}.{module}.lora_B.weight"
                
                mock_weights[lora_A_key] = torch.randn(rank, in_dim) * 0.01
                mock_weights[lora_B_key] = torch.zeros(out_dim, rank)  # B initialized to zero
        
        # Save weights
        weights_path = os.path.join(temp_dir, "adapter_model.bin")
        torch.save(mock_weights, weights_path)
        
        print(f"✅ Mock PEFT adapter created at: {temp_dir}")
        print(f"   Config keys: {list(adapter_config.keys())}")
        print(f"   Weight keys: {len(mock_weights)} tensors")
        
        return temp_dir
        
    except Exception as e:
        print(f"❌ Failed to create mock adapter: {e}")
        shutil.rmtree(temp_dir)
        return None


def test_peft_conversion():
    """Test PEFT adapter conversion."""
    print("\nTesting PEFT Conversion")
    print("=" * 50)
    
    # Create mock PEFT adapter
    mock_adapter_path = create_mock_peft_adapter()
    if not mock_adapter_path:
        return
    
    output_dir = tempfile.mkdtemp()
    
    try:
        # Initialize converter
        converter = PEFTConverter(target_layers=[3, 6, 9])
        
        # Convert mock adapter
        success = converter.convert_from_local(
            adapter_path=mock_adapter_path,
            output_dir=output_dir,
            base_model_name="microsoft/DialoGPT-small"
        )

        if success:
            print("✅ PEFT conversion successful")

            # Test loading converted adapter
            adapter_manager = AdapterManager(adapter_dir="./")  # Use current directory
            converted_adapter = adapter_manager.load_adapter(os.path.basename(output_dir))
            
            if converted_adapter:
                print("✅ Converted adapter loaded successfully")
                print(f"   Metadata: {converted_adapter['metadata']['name']}")
                print(f"   Target layers: {converted_adapter['metadata']['target_layers']}")
                print(f"   Weight layers: {list(converted_adapter['weights'].keys())}")
                
                # Test with Adaptrix engine
                engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
                engine.initialize()
                
                # Try to load the converted adapter
                load_success = engine.load_adapter(os.path.basename(output_dir))
                if load_success:
                    print("✅ Converted adapter works with Adaptrix engine")
                    
                    # Test generation
                    response = engine.generate("Hello", max_length=10)
                    print(f"   Generated: '{response}'")
                else:
                    print("❌ Failed to load converted adapter in engine")
                
                engine.cleanup()
            else:
                print("❌ Failed to load converted adapter")
        else:
            print("❌ PEFT conversion failed")
    
    except Exception as e:
        print(f"❌ PEFT conversion test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if mock_adapter_path:
            shutil.rmtree(mock_adapter_path)
        if output_dir:
            shutil.rmtree(output_dir)


def test_multi_layer_injection():
    """Test multi-layer injection with context preservation."""
    print("\nTesting Multi-Layer Injection")
    print("=" * 50)
    
    try:
        engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
        engine.initialize()
        
        # Get architecture info
        arch_info = engine.base_model_manager.architecture_info
        recommended_layers = arch_info['recommended_middle_layers']
        
        print(f"Model layers: {arch_info['layer_count']}")
        print(f"Recommended injection layers: {recommended_layers}")
        
        # Test with available adapters
        adapters = engine.list_adapters()
        if adapters:
            adapter_name = adapters[0]
            print(f"Testing with adapter: {adapter_name}")
            
            # Load adapter with specific layers
            success = engine.load_adapter(adapter_name, layer_indices=recommended_layers[:2])
            
            if success:
                print("✅ Multi-layer injection successful")
                
                # Test generation with context preservation
                test_queries = [
                    "What is 2 + 2?",
                    "Explain quantum physics",
                    "Write a short poem"
                ]
                
                for query in test_queries:
                    response = engine.query(query, max_length=15)
                    print(f"   Query: {query}")
                    print(f"   Response: {response}")
                
                # Get injection statistics
                context_stats = engine.layer_injector.context_injector.get_context_statistics()
                print(f"Context statistics: {context_stats}")
                
            else:
                print("❌ Multi-layer injection failed")
        
        engine.cleanup()
        print("✅ Multi-layer injection test completed")
        
    except Exception as e:
        print(f"❌ Multi-layer injection test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all integration tests."""
    print("Adaptrix PEFT Integration Tests")
    print("=" * 60)
    
    # Test 1: Architecture detection
    test_architecture_detection()
    
    # Test 2: Context preservation
    test_context_preservation()
    
    # Test 3: PEFT conversion
    test_peft_conversion()
    
    # Test 4: Multi-layer injection
    test_multi_layer_injection()
    
    print("\n" + "=" * 60)
    print("🎉 All integration tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
