"""
Debug script to examine DialoGPT model structure.
"""

import torch
from transformers import AutoModelForCausalLM

# Load the model
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

print("Model structure:")
print(model)

print("\n" + "="*60)
print("Transformer layers structure:")

# Check transformer layers
if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
    layers = model.transformer.h
    print(f"Number of layers: {len(layers)}")
    
    # Examine first layer structure
    first_layer = layers[0]
    print(f"\nFirst layer structure:")
    print(first_layer)
    
    print(f"\nFirst layer modules:")
    for name, module in first_layer.named_modules():
        if name:  # Skip the root module
            print(f"  {name}: {type(module).__name__}")
    
    print(f"\nFirst layer parameters:")
    for name, param in first_layer.named_parameters():
        print(f"  {name}: {param.shape}")

print("\n" + "="*60)
print("All model modules (first few levels):")
for name, module in model.named_modules():
    if name.count('.') <= 2:  # Only show first few levels
        print(f"{name}: {type(module).__name__}")
