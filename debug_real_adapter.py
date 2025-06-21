"""
Debug script to examine real HuggingFace adapter structure.
"""

import torch
import tempfile
import os
from huggingface_hub import snapshot_download

def examine_adapter_structure(adapter_id: str):
    """Examine the structure of a real adapter."""
    print(f"🔍 Examining adapter: {adapter_id}")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Download adapter
        print("📥 Downloading adapter...")
        snapshot_download(repo_id=adapter_id, local_dir=temp_dir)
        
        # List all files
        print(f"\n📁 Files in adapter directory:")
        for root, dirs, files in os.walk(temp_dir):
            level = root.replace(temp_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"{subindent}{file} ({file_size:,} bytes)")
        
        # Check for config file
        config_path = os.path.join(temp_dir, "adapter_config.json")
        if os.path.exists(config_path):
            print(f"\n📋 Adapter config:")
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                for key, value in config.items():
                    print(f"   {key}: {value}")
        else:
            print(f"\n❌ No adapter_config.json found")
        
        # Check for weights
        weights_files = []
        for file in ["adapter_model.bin", "adapter_model.safetensors", "pytorch_model.bin"]:
            weights_path = os.path.join(temp_dir, file)
            if os.path.exists(weights_path):
                weights_files.append(file)
        
        if weights_files:
            print(f"\n💾 Weight files found: {weights_files}")
            
            # Load and examine weights
            for weights_file in weights_files:
                weights_path = os.path.join(temp_dir, weights_file)
                print(f"\n🔍 Examining {weights_file}:")
                
                try:
                    if weights_file.endswith('.safetensors'):
                        import safetensors.torch
                        weights = safetensors.torch.load_file(weights_path)
                    else:
                        weights = torch.load(weights_path, map_location='cpu')
                    
                    print(f"   📊 Total tensors: {len(weights)}")
                    print(f"   🔑 Weight keys (first 10):")
                    for i, key in enumerate(list(weights.keys())[:10]):
                        tensor = weights[key]
                        if isinstance(tensor, torch.Tensor):
                            print(f"      {i+1}. {key}: {tensor.shape}")
                        else:
                            print(f"      {i+1}. {key}: {type(tensor)}")
                    
                    if len(weights) > 10:
                        print(f"      ... and {len(weights) - 10} more")
                    
                    # Look for LoRA patterns
                    lora_keys = [k for k in weights.keys() if 'lora' in k.lower()]
                    if lora_keys:
                        print(f"\n🎯 LoRA keys found: {len(lora_keys)}")
                        for key in lora_keys[:5]:
                            tensor = weights[key]
                            if isinstance(tensor, torch.Tensor):
                                print(f"      {key}: {tensor.shape}")
                    else:
                        print(f"\n❌ No LoRA keys found")
                    
                except Exception as e:
                    print(f"   ❌ Failed to load {weights_file}: {e}")
        else:
            print(f"\n❌ No weight files found")
        
    except Exception as e:
        print(f"❌ Failed to examine adapter: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    """Examine multiple real adapters."""
    adapters_to_examine = [
        "tloen/alpaca-lora-7b",
        "chavinlo/gpt4-x-alpaca",
    ]
    
    for adapter_id in adapters_to_examine:
        examine_adapter_structure(adapter_id)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
