"""
Check DeepSeek-R1 model architecture to understand the correct dimensions.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from transformers import AutoConfig, AutoModel, AutoTokenizer


def check_deepseek_architecture():
    """Check the DeepSeek-R1 model architecture."""
    print("🔍 Checking DeepSeek-R1 Model Architecture")
    print("=" * 60)
    
    try:
        model_name = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
        
        print(f"📋 Loading config for {model_name}...")
        config = AutoConfig.from_pretrained(model_name)
        
        print(f"\n📊 Model Configuration:")
        print(f"   Model type: {getattr(config, 'model_type', 'unknown')}")
        print(f"   Architecture: {getattr(config, 'architectures', 'unknown')}")
        print(f"   Hidden size: {getattr(config, 'hidden_size', 'unknown')}")
        print(f"   Intermediate size: {getattr(config, 'intermediate_size', 'unknown')}")
        print(f"   Num layers: {getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 'unknown'))}")
        print(f"   Num attention heads: {getattr(config, 'num_attention_heads', 'unknown')}")
        print(f"   Vocab size: {getattr(config, 'vocab_size', 'unknown')}")
        print(f"   Max position embeddings: {getattr(config, 'max_position_embeddings', 'unknown')}")
        
        # Check if it's a Qwen-based model
        if hasattr(config, 'model_type'):
            print(f"\n🏗️  Architecture Details:")
            if 'qwen' in config.model_type.lower():
                print(f"   This is a Qwen-based model")
                print(f"   Expected attention module: self_attn.q_proj, k_proj, v_proj, o_proj")
                print(f"   Expected MLP modules: mlp.gate_proj, up_proj, down_proj")
            elif 'gpt' in config.model_type.lower():
                print(f"   This is a GPT-based model")
                print(f"   Expected attention module: attn.c_attn, c_proj")
                print(f"   Expected MLP modules: mlp.c_fc, c_proj")
        
        # Try to load the model to check actual layer structure
        print(f"\n🔍 Loading model to check layer structure...")
        try:
            model = AutoModel.from_pretrained(model_name)
            
            print(f"✅ Model loaded successfully")
            
            # Check the first layer structure
            if hasattr(model, 'layers') and len(model.layers) > 0:
                first_layer = model.layers[0]
                print(f"\n📊 First Layer Structure:")
                for name, module in first_layer.named_children():
                    print(f"   {name}: {type(module).__name__}")
                    
                    # Check attention structure
                    if 'attn' in name:
                        print(f"      Attention submodules:")
                        for sub_name, sub_module in module.named_children():
                            if hasattr(sub_module, 'weight'):
                                print(f"         {sub_name}: {sub_module.weight.shape}")
                            else:
                                print(f"         {sub_name}: {type(sub_module).__name__}")
                    
                    # Check MLP structure
                    elif 'mlp' in name:
                        print(f"      MLP submodules:")
                        for sub_name, sub_module in module.named_children():
                            if hasattr(sub_module, 'weight'):
                                print(f"         {sub_name}: {sub_module.weight.shape}")
                            else:
                                print(f"         {sub_name}: {type(sub_module).__name__}")
            
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                # GPT-style model
                first_layer = model.transformer.h[0]
                print(f"\n📊 First Layer Structure (GPT-style):")
                for name, module in first_layer.named_children():
                    print(f"   {name}: {type(module).__name__}")
                    
                    if hasattr(module, 'weight'):
                        print(f"      Weight shape: {module.weight.shape}")
            
            else:
                print(f"⚠️  Unknown model structure")
                print(f"   Model attributes: {dir(model)}")
        
        except Exception as e:
            print(f"⚠️  Could not load full model: {e}")
            print(f"   Will use config information only")
        
        return config
        
    except Exception as e:
        print(f"❌ Failed to check DeepSeek architecture: {e}")
        import traceback
        traceback.print_exc()
        return None


def suggest_adapter_dimensions(config):
    """Suggest correct adapter dimensions based on model config."""
    print(f"\n🔧 Suggested Adapter Dimensions")
    print("=" * 60)
    
    if config is None:
        print("❌ No config available")
        return None
    
    hidden_size = getattr(config, 'hidden_size', 768)
    intermediate_size = getattr(config, 'intermediate_size', hidden_size * 4)
    num_attention_heads = getattr(config, 'num_attention_heads', 12)
    
    print(f"📊 Based on model config:")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Intermediate size: {intermediate_size}")
    print(f"   Attention heads: {num_attention_heads}")
    
    # Calculate attention dimensions
    head_dim = hidden_size // num_attention_heads
    total_attention_dim = hidden_size * 3  # q, k, v combined
    
    print(f"\n🎯 Recommended LoRA dimensions:")
    print(f"   For attention (q_proj/k_proj/v_proj): {hidden_size} -> {hidden_size}")
    print(f"   For attention output (o_proj): {hidden_size} -> {hidden_size}")
    print(f"   For MLP gate/up: {hidden_size} -> {intermediate_size}")
    print(f"   For MLP down: {intermediate_size} -> {hidden_size}")
    
    return {
        'hidden_size': hidden_size,
        'intermediate_size': intermediate_size,
        'attention_dim': hidden_size,
        'mlp_in_dim': hidden_size,
        'mlp_out_dim': intermediate_size
    }


def create_deepseek_compatible_adapter(dimensions):
    """Create an adapter compatible with DeepSeek-R1."""
    print(f"\n🔧 Creating DeepSeek-R1 Compatible Adapter")
    print("=" * 60)
    
    if dimensions is None:
        print("❌ No dimensions available")
        return False
    
    try:
        import torch
        import json
        import shutil
        
        # Create adapter directory
        adapter_dir = "adapters/deepseek_general"
        if os.path.exists(adapter_dir):
            shutil.rmtree(adapter_dir)
        os.makedirs(adapter_dir)
        
        # Create metadata
        metadata = {
            'name': 'deepseek_general',
            'version': '1.0.0',
            'description': 'General conversation adapter for DeepSeek-R1',
            'source': 'manual_creation',
            'base_model': 'deepseek-ai/deepseek-r1-distill-qwen-1.5b',
            'target_layers': [6, 12, 18],  # Spread across the model
            'target_modules': ['self_attn.q_proj', 'self_attn.v_proj', 'mlp.gate_proj'],
            'rank': 16,
            'alpha': 32
        }
        
        # Save metadata
        with open(os.path.join(adapter_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        hidden_size = dimensions['hidden_size']
        intermediate_size = dimensions['intermediate_size']
        
        # Create weights for each target layer
        for layer_idx in [6, 12, 18]:
            layer_weights = {}
            
            # Attention weights
            layer_weights['self_attn.q_proj'] = {
                'lora_A': torch.randn(16, hidden_size) * 0.02,
                'lora_B': torch.randn(hidden_size, 16) * 0.02,
                'rank': 16,
                'alpha': 32
            }
            
            layer_weights['self_attn.v_proj'] = {
                'lora_A': torch.randn(16, hidden_size) * 0.02,
                'lora_B': torch.randn(hidden_size, 16) * 0.02,
                'rank': 16,
                'alpha': 32
            }
            
            # MLP weights
            layer_weights['mlp.gate_proj'] = {
                'lora_A': torch.randn(16, hidden_size) * 0.02,
                'lora_B': torch.randn(intermediate_size, 16) * 0.02,
                'rank': 16,
                'alpha': 32
            }
            
            # Save layer weights
            layer_file = os.path.join(adapter_dir, f"layer_{layer_idx}.pt")
            torch.save(layer_weights, layer_file)
        
        print(f"✅ Created DeepSeek-R1 compatible adapter")
        print(f"   Target layers: {metadata['target_layers']}")
        print(f"   Target modules: {metadata['target_modules']}")
        print(f"   Dimensions: {hidden_size} hidden, {intermediate_size} intermediate")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create DeepSeek adapter: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Check DeepSeek architecture and create compatible adapters."""
    print("🔍 DeepSeek-R1 Architecture Analysis")
    print("=" * 80)
    print("Analyzing model structure to create proper adapters")
    print("=" * 80)
    
    # Check architecture
    config = check_deepseek_architecture()
    
    # Suggest dimensions
    dimensions = suggest_adapter_dimensions(config)
    
    # Create compatible adapter
    if dimensions:
        adapter_created = create_deepseek_compatible_adapter(dimensions)
    else:
        adapter_created = False
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"🎉 DeepSeek-R1 Analysis Results")
    print(f"=" * 80)
    print(f"🔍 Architecture analysis: {'✅ SUCCESS' if config else '❌ FAILED'}")
    print(f"📊 Dimension calculation: {'✅ SUCCESS' if dimensions else '❌ FAILED'}")
    print(f"🔧 Adapter creation: {'✅ SUCCESS' if adapter_created else '❌ FAILED'}")
    
    if adapter_created:
        print(f"\n🎊 DEEPSEEK-R1 ADAPTER READY!")
        print(f"🚀 Ready to test with proper model architecture!")
    else:
        print(f"\n⚠️  Need to debug architecture compatibility")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
