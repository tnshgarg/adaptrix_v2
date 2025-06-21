"""
Debug script to examine adapter conversion process in detail.
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


def debug_peft_conversion():
    """Debug the PEFT conversion process step by step."""
    print("🔍 Debugging PEFT Conversion Process")
    print("=" * 60)
    
    # Test with a simple synthetic adapter first
    temp_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    
    try:
        print("🏗️  Creating debug synthetic adapter...")
        
        # Create a simple PEFT adapter config
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
        
        # Create weights with proper naming for DialoGPT
        weights = {}
        
        # Create weights for specific layers that should be found
        test_layers = [3, 6, 9]
        
        for layer_idx in test_layers:
            for module in ["attn.c_attn", "mlp.c_fc"]:
                if module == "attn.c_attn":
                    in_dim, out_dim = 768, 2304
                elif module == "mlp.c_fc":
                    in_dim, out_dim = 768, 3072
                
                rank = 8
                
                # Use the exact key format that the converter expects
                lora_A_key = f"base_model.model.transformer.h.{layer_idx}.{module}.lora_A.weight"
                lora_B_key = f"base_model.model.transformer.h.{layer_idx}.{module}.lora_B.weight"
                
                weights[lora_A_key] = torch.randn(rank, in_dim) * 0.01
                weights[lora_B_key] = torch.randn(out_dim, rank) * 0.01
                
                print(f"   Created weights: {lora_A_key} -> {weights[lora_A_key].shape}")
                print(f"   Created weights: {lora_B_key} -> {weights[lora_B_key].shape}")
        
        # Save weights
        weights_path = os.path.join(temp_dir, "adapter_model.bin")
        torch.save(weights, weights_path)
        
        print(f"✅ Created synthetic adapter with {len(weights)} weight tensors")
        print(f"   Weight keys: {list(weights.keys())[:3]}...")
        
        # Now test the conversion process
        print(f"\n🔄 Testing conversion process...")
        
        converter = PEFTConverter(target_layers=[3, 6, 9])
        
        # Debug: Load and examine the weights before conversion
        print(f"\n🔍 Examining weights before conversion:")
        loaded_weights = torch.load(weights_path, map_location='cpu')
        print(f"   Loaded {len(loaded_weights)} weight tensors")
        for key in list(loaded_weights.keys())[:5]:
            print(f"   {key}: {loaded_weights[key].shape}")
        
        # Test the conversion
        success = converter.convert_from_local(
            adapter_path=temp_dir,
            output_dir=output_dir,
            base_model_name="microsoft/DialoGPT-small"
        )
        
        if success:
            print(f"✅ Conversion reported success!")
            
            # Examine the output directory
            print(f"\n📁 Examining output directory: {output_dir}")
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    print(f"   {item}: {size:,} bytes")
            
            # Try to load with adapter manager
            print(f"\n🔍 Testing adapter manager loading...")
            adapter_manager = AdapterManager(adapter_dir=os.path.dirname(output_dir))
            converted_name = os.path.basename(output_dir)
            
            # Debug: Check if metadata exists and is valid
            metadata_path = os.path.join(output_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"   ✅ Metadata found:")
                print(f"      Target layers: {metadata.get('target_layers', [])}")
                print(f"      Target modules: {metadata.get('target_modules', [])}")
                
                # Check for layer files
                for layer_idx in metadata.get('target_layers', []):
                    layer_file = os.path.join(output_dir, f"layer_{layer_idx}.pt")
                    if os.path.exists(layer_file):
                        layer_weights = torch.load(layer_file, map_location='cpu')
                        print(f"   ✅ Layer {layer_idx} file exists with {len(layer_weights)} modules")
                        for module_name, module_weights in layer_weights.items():
                            print(f"      {module_name}: lora_A {module_weights['lora_A'].shape}, lora_B {module_weights['lora_B'].shape}")
                    else:
                        print(f"   ❌ Layer {layer_idx} file missing: {layer_file}")
            else:
                print(f"   ❌ Metadata file missing: {metadata_path}")
            
            # Try to load the adapter
            converted_adapter = adapter_manager.load_adapter(converted_name)
            
            if converted_adapter:
                print(f"   ✅ Adapter loaded successfully!")
                metadata = converted_adapter['metadata']
                weights = converted_adapter['weights']
                print(f"      Metadata: {metadata['name']}")
                print(f"      Weight layers: {list(weights.keys())}")
            else:
                print(f"   ❌ Failed to load converted adapter")
        else:
            print(f"❌ Conversion failed")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def test_dimension_fixes():
    """Test the dimension mismatch fixes."""
    print(f"\n🔧 Testing Dimension Mismatch Fixes")
    print("=" * 60)
    
    try:
        from src.core.engine import AdaptrixEngine
        
        engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
        engine.initialize()
        
        print(f"✅ Engine initialized")
        
        # Load an existing adapter
        adapters = engine.list_adapters()
        if adapters:
            adapter_name = adapters[0]
            print(f"🎯 Testing with adapter: {adapter_name}")
            
            success = engine.load_adapter(adapter_name)
            if success:
                print(f"✅ Adapter loaded successfully")
                
                # Test generation
                test_queries = [
                    "Hello there!",
                    "How are you today?",
                    "Tell me a story."
                ]
                
                for query in test_queries:
                    try:
                        response = engine.query(query, max_length=15)
                        print(f"   💬 '{query}' -> '{response}'")
                        
                        if response and response.strip():
                            print(f"   ✅ Generation working!")
                        else:
                            print(f"   ⚠️  Empty response")
                    except Exception as e:
                        print(f"   ❌ Generation failed: {e}")
                
                # Get context statistics
                context_stats = engine.layer_injector.context_injector.get_context_statistics()
                print(f"📊 Context Statistics:")
                print(f"   Layers with context: {context_stats['layers_with_context']}")
                print(f"   Total injections: {context_stats['total_injections']}")
                
                engine.unload_adapter(adapter_name)
            else:
                print(f"❌ Failed to load adapter")
        else:
            print(f"❌ No adapters available")
        
        engine.cleanup()
        
    except Exception as e:
        print(f"❌ Dimension test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run debugging tests."""
    print("🔍 Adaptrix Conversion & Dimension Debug Suite")
    print("=" * 70)
    
    # Test 1: Debug PEFT conversion
    debug_peft_conversion()
    
    # Test 2: Test dimension fixes
    test_dimension_fixes()
    
    print(f"\n" + "=" * 70)
    print("🎉 Debug testing completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
