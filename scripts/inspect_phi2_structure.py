#!/usr/bin/env python3
"""
Inspect Phi-2 model structure to find correct module names.
"""

import sys
import os
from transformers import AutoModelForCausalLM

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def inspect_phi2_structure():
    """Inspect the actual Phi-2 model structure."""
    
    print("🔍 Inspecting Phi-2 model structure...")
    
    # Load Phi-2 model
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    
    print(f"\nModel type: {type(model)}")
    print(f"Config: {model.config}")
    
    # Print model structure
    print("\n📋 Model structure:")
    for name, module in model.named_modules():
        if len(name.split('.')) <= 3:  # Only show top-level structure
            print(f"  {name}: {type(module).__name__}")
    
    # Look specifically at the first layer
    print("\n🔍 First layer structure:")
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        first_layer = model.transformer.h[0]
        print(f"First layer type: {type(first_layer)}")
        for name, module in first_layer.named_children():
            print(f"  {name}: {type(module).__name__}")
            
            # Look deeper into attention and MLP
            if 'attn' in name.lower():
                print(f"    Attention submodules:")
                for sub_name, sub_module in module.named_children():
                    print(f"      {sub_name}: {type(sub_module).__name__}")
            elif 'mlp' in name.lower():
                print(f"    MLP submodules:")
                for sub_name, sub_module in module.named_children():
                    print(f"      {sub_name}: {type(sub_module).__name__}")
    
    # Check if it's a different structure
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        first_layer = model.model.layers[0]
        print(f"First layer type: {type(first_layer)}")
        for name, module in first_layer.named_children():
            print(f"  {name}: {type(module).__name__}")
            
            # Look deeper
            if 'attn' in name.lower():
                print(f"    Attention submodules:")
                for sub_name, sub_module in module.named_children():
                    print(f"      {sub_name}: {type(sub_module).__name__}")
            elif 'mlp' in name.lower():
                print(f"    MLP submodules:")
                for sub_name, sub_module in module.named_children():
                    print(f"      {sub_name}: {type(sub_module).__name__}")
    
    # Print all module names that contain common LoRA targets
    print("\n🎯 Potential LoRA target modules:")
    target_keywords = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'dense', 'fc', 'gate', 'up', 'down']
    
    for name, module in model.named_modules():
        for keyword in target_keywords:
            if keyword in name.lower():
                print(f"  {name}: {type(module).__name__}")
                break
    
    print("\n✅ Inspection complete!")


def inspect_downloaded_adapter():
    """Inspect the downloaded HuggingFace adapter structure."""
    
    print("\n🔍 Inspecting downloaded adapter structure...")
    
    adapter_dir = "adapters/phi2_gsm8k_hf"
    
    # List all files
    print(f"\n📁 Files in {adapter_dir}:")
    for file in os.listdir(adapter_dir):
        print(f"  {file}")
    
    # Read adapter config
    import json
    config_file = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"\n📋 Adapter config:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    # Check if there's a safetensors file
    import safetensors
    safetensors_file = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.exists(safetensors_file):
        print(f"\n🔍 Adapter weights structure:")
        with safetensors.safe_open(safetensors_file, framework="pt") as f:
            for key in f.keys():
                print(f"  {key}: {f.get_tensor(key).shape}")
    
    print("\n✅ Adapter inspection complete!")


def main():
    """Main function."""
    print("🔍" * 60)
    print("🔍 PHI-2 MODEL STRUCTURE INSPECTOR 🔍")
    print("🔍" * 60)
    
    # Inspect model structure
    inspect_phi2_structure()
    
    # Inspect adapter structure
    inspect_downloaded_adapter()
    
    print("\n🎯 Use this information to fix module names in Adaptrix!")


if __name__ == "__main__":
    main()
