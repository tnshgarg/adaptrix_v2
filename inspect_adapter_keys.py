#!/usr/bin/env python3
"""
🔍 ADAPTER KEY INSPECTOR 🔍

This script inspects the actual keys in your trained adapter to fix the loading issue.
"""

import os
from safetensors import safe_open

def inspect_adapter_keys():
    """Inspect the actual keys in the trained adapter."""
    print("🔍 INSPECTING YOUR TRAINED ADAPTER KEYS")
    print("=" * 60)
    
    adapter_file = "adapters/code_adapter/adapter_model.safetensors"
    
    if not os.path.exists(adapter_file):
        print(f"❌ Adapter file not found: {adapter_file}")
        return
    
    print(f"📁 Loading: {adapter_file}")
    
    with safe_open(adapter_file, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"🔢 Total keys found: {len(keys)}")
        
        # Group keys by layer
        layer_groups = {}
        attention_keys = []
        mlp_keys = []
        other_keys = []
        
        for key in keys:
            print(f"   📋 {key}")
            
            # Try to extract layer number
            if "layers." in key:
                parts = key.split("layers.")
                if len(parts) > 1:
                    layer_part = parts[1].split(".")[0]
                    try:
                        layer_num = int(layer_part)
                        if layer_num not in layer_groups:
                            layer_groups[layer_num] = []
                        layer_groups[layer_num].append(key)
                    except:
                        other_keys.append(key)
                else:
                    other_keys.append(key)
            else:
                other_keys.append(key)
            
            # Check module types
            if "self_attn" in key:
                attention_keys.append(key)
            elif "mlp" in key:
                mlp_keys.append(key)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Layers found: {sorted(layer_groups.keys())}")
        print(f"   Attention keys: {len(attention_keys)}")
        print(f"   MLP keys: {len(mlp_keys)}")
        print(f"   Other keys: {len(other_keys)}")
        
        # Show layer breakdown
        print(f"\n🗂️  LAYER BREAKDOWN:")
        for layer_num in sorted(layer_groups.keys())[:5]:  # Show first 5 layers
            print(f"   Layer {layer_num}: {len(layer_groups[layer_num])} keys")
            for key in layer_groups[layer_num][:3]:  # Show first 3 keys per layer
                print(f"      • {key}")
            if len(layer_groups[layer_num]) > 3:
                print(f"      ... and {len(layer_groups[layer_num]) - 3} more")
        
        # Show sample attention keys
        print(f"\n🎯 ATTENTION KEY SAMPLES:")
        for key in attention_keys[:10]:  # Show first 10
            print(f"   • {key}")
        
        # Check for lora_A and lora_B patterns
        print(f"\n🔍 CHECKING LORA PATTERNS:")
        lora_a_keys = [k for k in keys if "lora_A" in k]
        lora_b_keys = [k for k in keys if "lora_B" in k]
        
        print(f"   LoRA A keys: {len(lora_a_keys)}")
        if lora_a_keys:
            print(f"      Sample: {lora_a_keys[0]}")
        
        print(f"   LoRA B keys: {len(lora_b_keys)}")
        if lora_b_keys:
            print(f"      Sample: {lora_b_keys[0]}")
        
        # Generate the correct key pattern
        if lora_a_keys:
            sample_key = lora_a_keys[0]
            print(f"\n🎯 CORRECT KEY PATTERN DETECTED:")
            print(f"   Sample A key: {sample_key}")
            
            # Extract the pattern
            if "layers." in sample_key:
                before_layer = sample_key.split("layers.")[0]
                after_layer_part = sample_key.split("layers.")[1]
                layer_and_rest = after_layer_part.split(".", 1)
                if len(layer_and_rest) > 1:
                    after_layer_num = layer_and_rest[1]
                    pattern = f"{before_layer}layers.{{layer}}.{after_layer_num}"
                    print(f"   Pattern: {pattern}")
                    
                    # Test the pattern
                    print(f"\n🧪 TESTING PATTERN:")
                    for test_layer in [0, 1, 2]:
                        test_key_a = pattern.replace("{layer}", str(test_layer)).replace("lora_B", "lora_A")
                        test_key_b = pattern.replace("{layer}", str(test_layer)).replace("lora_A", "lora_B")
                        
                        try:
                            tensor_a = f.get_tensor(test_key_a)
                            tensor_b = f.get_tensor(test_key_b)
                            print(f"      Layer {test_layer}: ✅ A:{tensor_a.shape} B:{tensor_b.shape}")
                        except Exception as e:
                            print(f"      Layer {test_layer}: ❌ {e}")

if __name__ == "__main__":
    inspect_adapter_keys() 