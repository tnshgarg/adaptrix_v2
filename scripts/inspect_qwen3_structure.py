#!/usr/bin/env python3
"""
🔍 QWEN3-1.7B MODEL STRUCTURE INSPECTOR

Inspects the Qwen3-1.7B model architecture to show all available target modules
for LoRA adapter training and configuration.
"""

import sys
import os
from typing import Dict, List, Set

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def inspect_qwen3_structure():
    """Inspect Qwen3-1.7B model structure and print target modules."""
    
    print("🔍" * 80)
    print("🔍 QWEN3-1.7B MODEL STRUCTURE INSPECTION 🔍")
    print("🔍" * 80)
    
    try:
        print("\n📦 Loading Qwen3-1.7B model...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_id = "Qwen/Qwen3-1.7B"
        
        # Load model with minimal resources
        print("   🔧 Loading model architecture...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        print("✅ Model loaded successfully!")
        
        # Inspect model structure
        print(f"\n🏗️ MODEL ARCHITECTURE OVERVIEW:")
        print("=" * 60)
        
        print(f"   Model Type: {type(model).__name__}")
        print(f"   Model Config: {type(model.config).__name__}")
        
        # Get basic model info
        if hasattr(model.config, 'num_hidden_layers'):
            print(f"   Number of Layers: {model.config.num_hidden_layers}")
        if hasattr(model.config, 'hidden_size'):
            print(f"   Hidden Size: {model.config.hidden_size}")
        if hasattr(model.config, 'num_attention_heads'):
            print(f"   Attention Heads: {model.config.num_attention_heads}")
        if hasattr(model.config, 'vocab_size'):
            print(f"   Vocabulary Size: {model.config.vocab_size}")
        
        # Collect all module names
        print(f"\n🔍 COLLECTING ALL MODULE NAMES...")
        all_modules = {}
        attention_modules = set()
        mlp_modules = set()
        embedding_modules = set()
        other_modules = set()
        
        for name, module in model.named_modules():
            if name:  # Skip empty names
                all_modules[name] = type(module).__name__
                
                # Categorize modules
                if any(attn_key in name.lower() for attn_key in ['attn', 'attention']):
                    if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                        attention_modules.add(name)
                elif any(mlp_key in name.lower() for mlp_key in ['mlp', 'feed_forward', 'ffn']):
                    if any(proj in name for proj in ['gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2']):
                        mlp_modules.add(name)
                elif any(emb_key in name.lower() for emb_key in ['embed', 'embedding', 'lm_head']):
                    embedding_modules.add(name)
                else:
                    if any(proj in name for proj in ['proj', 'linear', 'dense']):
                        other_modules.add(name)
        
        print(f"✅ Found {len(all_modules)} total modules")
        
        # Print attention modules
        print(f"\n🧠 ATTENTION MODULES:")
        print("=" * 60)
        attention_list = sorted(list(attention_modules))
        if attention_list:
            for module in attention_list[:10]:  # Show first 10
                module_type = all_modules.get(module, "Unknown")
                print(f"   {module} ({module_type})")
            if len(attention_list) > 10:
                print(f"   ... and {len(attention_list) - 10} more attention modules")
        else:
            print("   No standard attention modules found")
        
        # Print MLP modules  
        print(f"\n🔧 MLP/FEED-FORWARD MODULES:")
        print("=" * 60)
        mlp_list = sorted(list(mlp_modules))
        if mlp_list:
            for module in mlp_list[:10]:  # Show first 10
                module_type = all_modules.get(module, "Unknown")
                print(f"   {module} ({module_type})")
            if len(mlp_list) > 10:
                print(f"   ... and {len(mlp_list) - 10} more MLP modules")
        else:
            print("   No standard MLP modules found")
        
        # Print embedding modules
        print(f"\n📚 EMBEDDING MODULES:")
        print("=" * 60)
        embedding_list = sorted(list(embedding_modules))
        if embedding_list:
            for module in embedding_list:
                module_type = all_modules.get(module, "Unknown")
                print(f"   {module} ({module_type})")
        else:
            print("   No embedding modules found")
        
        # Find actual target modules by pattern matching
        print(f"\n🎯 LORA TARGET MODULE ANALYSIS:")
        print("=" * 60)
        
        # Common LoRA target patterns
        target_patterns = {
            'q_proj': [],
            'k_proj': [],
            'v_proj': [],
            'o_proj': [],
            'gate_proj': [],
            'up_proj': [],
            'down_proj': [],
            'fc1': [],
            'fc2': [],
            'dense': [],
            'linear': []
        }
        
        # Find modules matching each pattern
        for name in all_modules.keys():
            for pattern in target_patterns.keys():
                if pattern in name:
                    target_patterns[pattern].append(name)
        
        # Print found target modules
        for pattern, modules in target_patterns.items():
            if modules:
                print(f"\n   🎯 {pattern.upper()} modules:")
                for module in sorted(modules)[:5]:  # Show first 5
                    layer_info = ""
                    if 'layers.' in module:
                        layer_num = module.split('layers.')[1].split('.')[0]
                        layer_info = f" (Layer {layer_num})"
                    print(f"      {module}{layer_info}")
                if len(modules) > 5:
                    print(f"      ... and {len(modules) - 5} more")
        
        # Extract unique module suffixes for LoRA config
        print(f"\n📋 RECOMMENDED LORA TARGET MODULES:")
        print("=" * 60)
        
        unique_suffixes = set()
        for name in all_modules.keys():
            if any(pattern in name for pattern in target_patterns.keys()):
                # Extract the suffix (e.g., 'q_proj' from 'model.layers.0.self_attn.q_proj')
                parts = name.split('.')
                if len(parts) > 0:
                    suffix = parts[-1]
                    unique_suffixes.add(suffix)
        
        recommended_modules = sorted(list(unique_suffixes))
        
        print("   For LoRA adapter configuration, use these target_modules:")
        print(f"   {recommended_modules}")
        
        # Create example configuration
        print(f"\n📝 EXAMPLE LORA CONFIGURATION:")
        print("=" * 60)
        
        # Categorize recommended modules
        attention_targets = [m for m in recommended_modules if any(p in m for p in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])]
        mlp_targets = [m for m in recommended_modules if any(p in m for p in ['gate_proj', 'up_proj', 'down_proj', 'fc1', 'fc2'])]
        
        print("   # Basic attention-only configuration:")
        print(f'   "target_modules": {attention_targets}')
        
        if mlp_targets:
            print("\n   # Extended configuration (attention + MLP):")
            extended_targets = attention_targets + mlp_targets
            print(f'   "target_modules": {extended_targets}')
        
        # Show layer structure
        print(f"\n🏗️ LAYER STRUCTURE EXAMPLE:")
        print("=" * 60)
        
        # Find a sample layer
        sample_layer = None
        for name in all_modules.keys():
            if 'layers.0.' in name and 'q_proj' in name:
                sample_layer = name
                break
        
        if sample_layer:
            layer_prefix = '.'.join(sample_layer.split('.')[:-1])
            print(f"   Sample layer structure (Layer 0):")
            print(f"   Base path: {layer_prefix}")
            
            layer_modules = [name for name in all_modules.keys() if name.startswith(layer_prefix)]
            for module in sorted(layer_modules):
                suffix = module.replace(layer_prefix + '.', '')
                module_type = all_modules[module]
                print(f"      .{suffix} ({module_type})")
        
        # Memory cleanup
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"\n✅ INSPECTION COMPLETE!")
        return recommended_modules
        
    except Exception as e:
        print(f"❌ Error during inspection: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_common_lora_patterns():
    """Print common LoRA target module patterns for different model families."""
    
    print(f"\n📚 COMMON LORA TARGET PATTERNS BY MODEL FAMILY:")
    print("=" * 80)
    
    patterns = {
        "Qwen/Qwen3": {
            "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"],
            "embeddings": ["embed_tokens", "lm_head"]
        },
        "LLaMA/LLaMA2": {
            "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"],
            "embeddings": ["embed_tokens", "lm_head"]
        },
        "Phi/Phi-2": {
            "attention": ["Wqkv", "out_proj"],
            "mlp": ["fc1", "fc2"],
            "embeddings": ["embed_tokens", "lm_head"]
        },
        "Mistral": {
            "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "mlp": ["gate_proj", "up_proj", "down_proj"],
            "embeddings": ["embed_tokens", "lm_head"]
        }
    }
    
    for model_family, modules in patterns.items():
        print(f"\n🔧 {model_family}:")
        for category, module_list in modules.items():
            print(f"   {category.title()}: {module_list}")


def main():
    """Main inspection function."""
    
    # Inspect Qwen3 structure
    target_modules = inspect_qwen3_structure()
    
    # Show common patterns
    print_common_lora_patterns()
    
    if target_modules:
        print(f"\n🎯 SUMMARY FOR QWEN3-1.7B:")
        print("=" * 50)
        print(f"   Recommended target_modules: {target_modules}")
        print(f"   Total unique modules: {len(target_modules)}")
        
        print(f"\n📋 COPY-PASTE READY CONFIG:")
        print(f'   "target_modules": {target_modules}')


if __name__ == "__main__":
    main()
