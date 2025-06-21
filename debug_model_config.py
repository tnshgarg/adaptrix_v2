"""
Debug script to check DialoGPT model configuration.
"""

from transformers import AutoConfig

# Load DialoGPT-small config
config = AutoConfig.from_pretrained("microsoft/DialoGPT-small")

print("DialoGPT-small Configuration:")
print("=" * 40)

# Print all config attributes
for attr in dir(config):
    if not attr.startswith('_'):
        value = getattr(config, attr)
        if not callable(value):
            print(f"{attr}: {value}")

print("\n" + "=" * 40)
print("Key dimensions:")
print(f"n_embd: {getattr(config, 'n_embd', 'NOT_FOUND')}")
print(f"n_inner: {getattr(config, 'n_inner', 'NOT_FOUND')}")
print(f"n_layer: {getattr(config, 'n_layer', 'NOT_FOUND')}")
print(f"hidden_size: {getattr(config, 'hidden_size', 'NOT_FOUND')}")
print(f"intermediate_size: {getattr(config, 'intermediate_size', 'NOT_FOUND')}")

# Calculate expected dimensions
hidden_size = getattr(config, 'n_embd', 768)
intermediate_size = getattr(config, 'n_inner', None)

if intermediate_size is None:
    intermediate_size = 4 * hidden_size
    print(f"\nCalculated intermediate_size: {intermediate_size}")
else:
    print(f"\nFound intermediate_size: {intermediate_size}")

print(f"\nExpected module dimensions:")
print(f"attn.c_attn: {hidden_size} -> {3 * hidden_size}")
print(f"attn.c_proj: {hidden_size} -> {hidden_size}")
print(f"mlp.c_fc: {hidden_size} -> {intermediate_size}")
print(f"mlp.c_proj: {intermediate_size} -> {hidden_size}")
