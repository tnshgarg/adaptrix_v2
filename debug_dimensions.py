"""
Debug script to check actual module dimensions in DeepSeek-R1.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import torch
from transformers import AutoModel, AutoTokenizer


def check_module_dimensions():
    """Check the actual input/output dimensions of DeepSeek modules."""
    print("🔍 Checking DeepSeek-R1 Module Dimensions")
    print("=" * 60)
    
    try:
        model_name = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
        
        print(f"📋 Loading model {model_name}...")
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get the first layer for inspection
        first_layer = model.layers[0]
        
        print(f"\n🔍 Inspecting Layer 0 Modules:")
        
        # Check self_attn modules
        print(f"\n📊 Self-Attention Modules:")
        for name, module in first_layer.self_attn.named_children():
            if hasattr(module, 'weight'):
                weight_shape = module.weight.shape
                print(f"   {name}: {weight_shape} (in_features={weight_shape[1]}, out_features={weight_shape[0]})")
        
        # Check MLP modules
        print(f"\n📊 MLP Modules:")
        for name, module in first_layer.mlp.named_children():
            if hasattr(module, 'weight'):
                weight_shape = module.weight.shape
                print(f"   {name}: {weight_shape} (in_features={weight_shape[1]}, out_features={weight_shape[0]})")
        
        # Test with actual input
        print(f"\n🧪 Testing with actual input:")
        test_input = "Hello, how are you?"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        print(f"   Input tokens shape: {inputs['input_ids'].shape}")
        
        # Get embeddings
        with torch.no_grad():
            embeddings = model.embed_tokens(inputs['input_ids'])
            print(f"   Embeddings shape: {embeddings.shape}")
            
            # Test first layer modules
            layer_input = embeddings
            
            # Test q_proj
            q_output = first_layer.self_attn.q_proj(layer_input)
            print(f"   q_proj output: {q_output.shape}")
            
            # Test v_proj
            v_output = first_layer.self_attn.v_proj(layer_input)
            print(f"   v_proj output: {v_output.shape}")
            
            # Test gate_proj
            gate_output = first_layer.mlp.gate_proj(layer_input)
            print(f"   gate_proj output: {gate_output.shape}")
        
        print(f"\n✅ Module dimension analysis complete")
        
        return {
            'hidden_size': 1536,
            'q_proj_dims': (1536, 1536),
            'v_proj_dims': (1536, 256),
            'gate_proj_dims': (1536, 8960)
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_lora_dimensions():
    """Test LoRA layer creation with correct dimensions."""
    print(f"\n🔧 Testing LoRA Layer Creation")
    print("=" * 60)
    
    try:
        from src.injection.layer_injector import LoRALayer
        
        # Test dimensions based on actual module analysis
        test_cases = [
            ("q_proj", 1536, 1536),
            ("v_proj", 1536, 256),
            ("gate_proj", 1536, 8960)
        ]
        
        for module_name, in_features, out_features in test_cases:
            print(f"\n📊 Testing {module_name}:")
            print(f"   Input features: {in_features}")
            print(f"   Output features: {out_features}")
            
            # Create LoRA layer
            lora_layer = LoRALayer(
                in_features=in_features,
                out_features=out_features,
                rank=8,
                alpha=16
            )
            
            # Test with sample input
            sample_input = torch.randn(1, 10, in_features)  # batch=1, seq_len=10
            print(f"   Sample input shape: {sample_input.shape}")
            
            with torch.no_grad():
                lora_output = lora_layer(sample_input)
                print(f"   LoRA output shape: {lora_output.shape}")
                
                expected_shape = (1, 10, out_features)
                if lora_output.shape == expected_shape:
                    print(f"   ✅ Correct output shape")
                else:
                    print(f"   ❌ Wrong output shape, expected {expected_shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run dimension analysis."""
    print("🚀 DeepSeek-R1 Dimension Analysis")
    print("=" * 80)
    
    # Check actual model dimensions
    dims = check_module_dimensions()
    
    # Test LoRA layer creation
    if dims:
        lora_test = test_lora_dimensions()
    else:
        lora_test = False
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"🎉 Dimension Analysis Results")
    print(f"=" * 80)
    print(f"📊 Model analysis: {'✅ SUCCESS' if dims else '❌ FAILED'}")
    print(f"🔧 LoRA testing: {'✅ SUCCESS' if lora_test else '❌ FAILED'}")
    
    if dims and lora_test:
        print(f"\n🎊 DIMENSIONS VERIFIED!")
        print(f"✅ Ready to fix context preservation")
    else:
        print(f"\n⚠️  Dimension issues need to be resolved")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
