#!/usr/bin/env python3
"""
Debug script to understand Qwen3-1.7B model structure for proper injection.
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def inspect_qwen3_structure():
    """Inspect the Qwen3-1.7B model structure."""
    print("🔍 Inspecting Qwen3-1.7B Model Structure")
    print("=" * 60)
    
    model_name = "Qwen/Qwen3-1.7B"
    
    # Load model
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    
    print(f"Model type: {type(model)}")
    print(f"Model config: {model.config}")
    
    # Print model structure
    print("\n📐 Model Architecture:")
    print("-" * 40)
    
    for name, module in model.named_modules():
        if any(keyword in name for keyword in ["layers", "attn", "mlp", "proj"]):
            print(f"{name}: {type(module)}")
    
    # Focus on a specific layer
    print("\n🎯 Detailed Layer 7 Structure:")
    print("-" * 40)
    
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layer_7 = model.model.layers[7]
        print(f"Layer 7 type: {type(layer_7)}")
        
        for name, module in layer_7.named_modules():
            print(f"  {name}: {type(module)}")
    
    # Check what modules exist in the LoRA adapter
    print("\n🔧 LoRA Adapter Modules:")
    print("-" * 40)
    
    adapter_path = "adapters/code_adapter"
    try:
        lora_model = PeftModel.from_pretrained(model, adapter_path)
        
        # Check what modules are actually in the adapter
        for name, param in lora_model.named_parameters():
            if "lora" in name.lower():
                print(f"  {name}: {param.shape}")
    except Exception as e:
        print(f"Failed to load LoRA: {e}")
    
    # Test proper module names
    print("\n✅ Suggested Target Modules for Qwen3:")
    print("-" * 40)
    
    suggested_modules = []
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layer = model.model.layers[0]  # Check first layer
        
        for name, module in layer.named_modules():
            if any(target in name for target in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
                suggested_modules.append(name)
                print(f"  ✓ {name}")
    
    return suggested_modules

if __name__ == "__main__":
    modules = inspect_qwen3_structure()
    print(f"\n📋 Target modules to use: {modules}") 