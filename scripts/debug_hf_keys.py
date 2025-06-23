#!/usr/bin/env python3
"""
Debug HuggingFace adapter key format.
"""

import sys
import os
import safetensors

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def debug_hf_keys():
    """Debug the HuggingFace adapter key format."""
    
    print("🔍 Debugging HuggingFace adapter keys...")
    
    # Load HuggingFace weights
    safetensors_file = "adapters/phi2_gsm8k_hf/adapter_model.safetensors"
    
    with safetensors.safe_open(safetensors_file, framework="pt") as f:
        keys = list(f.keys())
    
    print(f"📊 Total keys: {len(keys)}")
    print("\n🔍 First 10 keys:")
    for i, key in enumerate(keys[:10]):
        parts = key.split('.')
        print(f"  {i+1}. {key}")
        print(f"     Parts: {parts}")
        print(f"     Length: {len(parts)}")
        print()
    
    # Analyze key patterns
    print("🔍 Key patterns:")
    patterns = {}
    for key in keys:
        parts = key.split('.')
        if len(parts) >= 6:
            pattern = '.'.join(parts[:6])
            if pattern not in patterns:
                patterns[pattern] = []
            patterns[pattern].append(key)
    
    for pattern, keys_list in patterns.items():
        print(f"  Pattern: {pattern}")
        print(f"  Count: {len(keys_list)}")
        print(f"  Example: {keys_list[0]}")
        print()


if __name__ == "__main__":
    debug_hf_keys()
