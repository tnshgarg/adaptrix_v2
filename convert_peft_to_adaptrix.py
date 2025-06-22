"""
Convert PEFT LoRA adapter to Adaptrix format.
"""

import sys
import os
import torch
import json
from safetensors import safe_open
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))


def convert_peft_adapter_to_adaptrix(peft_adapter_path: str, output_path: str = None):
    """
    Convert a PEFT LoRA adapter to Adaptrix format.
    
    Args:
        peft_adapter_path: Path to PEFT adapter directory
        output_path: Output path (defaults to same directory)
    """
    print(f"🔄 Converting PEFT adapter to Adaptrix format")
    print(f"Source: {peft_adapter_path}")
    
    if output_path is None:
        output_path = peft_adapter_path
    
    try:
        # Load PEFT adapter config
        adapter_config_path = os.path.join(peft_adapter_path, "adapter_config.json")
        with open(adapter_config_path, 'r') as f:
            peft_config = json.load(f)
        
        print(f"✅ Loaded PEFT config: {peft_config}")
        
        # Load PEFT weights
        adapter_weights_path = os.path.join(peft_adapter_path, "adapter_model.safetensors")
        
        if not os.path.exists(adapter_weights_path):
            print(f"❌ Adapter weights not found: {adapter_weights_path}")
            return False
        
        # Load weights using safetensors
        weights = {}
        with safe_open(adapter_weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        
        print(f"✅ Loaded {len(weights)} weight tensors")
        
        # Print weight structure for debugging
        print("Weight structure:")
        for key, tensor in weights.items():
            print(f"  {key}: {tensor.shape}")
        
        # Extract layer information and create Adaptrix format
        # PEFT format: "base_model.model.model.layers.{layer}.{module}.lora_{A/B}.weight"
        # Adaptrix format: layer files with module weights
        
        layer_weights = {}
        target_layers = set()
        
        for key, tensor in weights.items():
            # Parse the key to extract layer and module info
            parts = key.split('.')
            
            # Find layer number
            layer_idx = None
            module_path = []
            lora_type = None
            
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        # Get module path after layer index
                        module_parts = parts[i + 2:]
                        if module_parts[-1] == "weight":
                            module_parts = module_parts[:-1]
                        if module_parts[-1].startswith("lora_"):
                            lora_type = module_parts[-1]
                            module_parts = module_parts[:-1]
                        module_path = '.'.join(module_parts)
                        break
                    except ValueError:
                        continue
            
            if layer_idx is not None and module_path and lora_type:
                target_layers.add(layer_idx)
                
                if layer_idx not in layer_weights:
                    layer_weights[layer_idx] = {}
                
                if module_path not in layer_weights[layer_idx]:
                    layer_weights[layer_idx][module_path] = {}
                
                layer_weights[layer_idx][module_path][lora_type] = tensor
                
                print(f"  Mapped: Layer {layer_idx}, Module {module_path}, Type {lora_type}, Shape {tensor.shape}")
        
        print(f"✅ Found weights for layers: {sorted(target_layers)}")
        
        # Save layer files in Adaptrix format
        for layer_idx in sorted(target_layers):
            layer_file = os.path.join(output_path, f"layer_{layer_idx}.pt")
            
            # Convert to Adaptrix format
            adaptrix_layer_data = {}
            
            for module_name, module_weights in layer_weights[layer_idx].items():
                if 'lora_A' in module_weights and 'lora_B' in module_weights:
                    adaptrix_layer_data[module_name] = {
                        'lora_A': module_weights['lora_A'],
                        'lora_B': module_weights['lora_B'],
                        'rank': peft_config.get('r', 16),
                        'alpha': peft_config.get('lora_alpha', 32)
                    }
                    print(f"  Layer {layer_idx} {module_name}: A{module_weights['lora_A'].shape} B{module_weights['lora_B'].shape}")
            
            # Save layer file
            torch.save(adaptrix_layer_data, layer_file)
            print(f"✅ Saved layer {layer_idx} to {layer_file}")
        
        print(f"✅ Conversion completed successfully!")
        print(f"Created {len(target_layers)} layer files")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_converted_adapter(adapter_path: str):
    """Test the converted adapter with Adaptrix system."""
    print(f"\n🧪 Testing converted adapter")
    
    try:
        from src.core.engine import AdaptrixEngine
        
        # Initialize engine
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        # Get adapter name from path
        adapter_name = os.path.basename(adapter_path)
        
        # Test loading
        success = engine.load_adapter(adapter_name)
        
        if success:
            print(f"✅ Adapter loaded successfully!")
            
            # Test generation
            response = engine.generate(
                "Solve this math problem step by step.\n\nProblem: What is 4 + 5?\n\nSolution:",
                max_length=100,
                temperature=0.7
            )
            print(f"Test generation: {response}")
            
            engine.unload_adapter(adapter_name)
            print(f"✅ Adapter unloaded successfully!")
            
        else:
            print(f"❌ Failed to load converted adapter")
        
        engine.cleanup()
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main conversion function."""
    print("🎯 PEFT TO ADAPTRIX CONVERTER")
    print("=" * 60)
    
    adapter_path = "adapters/simple_math_test"
    
    if not os.path.exists(adapter_path):
        print(f"❌ Adapter path not found: {adapter_path}")
        return
    
    # Convert the adapter
    success = convert_peft_adapter_to_adaptrix(adapter_path)
    
    if success:
        # Test the converted adapter
        test_success = test_converted_adapter(adapter_path)
        
        print(f"\n" + "=" * 60)
        print(f"🎊 CONVERSION RESULTS")
        print(f"=" * 60)
        print(f"✅ Conversion: {'SUCCESS' if success else 'FAILED'}")
        print(f"✅ Integration: {'SUCCESS' if test_success else 'FAILED'}")
        
        if success and test_success:
            print(f"\n🎊 MATH ADAPTER FULLY CONVERTED AND WORKING!")
            print(f"✅ PEFT adapter converted to Adaptrix format")
            print(f"✅ Adapter loads and works with Adaptrix system")
            print(f"✅ Custom training pipeline complete!")
        else:
            print(f"\n⚠️  Conversion completed but integration issues remain")
    else:
        print(f"\n❌ Conversion failed")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
