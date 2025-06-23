#!/usr/bin/env python3
"""
Fix the dtype mismatch issue between model and LoRA weights.
"""

import sys
import os
import torch
import safetensors

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def check_dtypes():
    """Check the dtypes of model and LoRA weights."""
    
    print("🔍 Checking data types...")
    
    # Check HuggingFace adapter weights
    hf_adapter_dir = "adapters/phi2_gsm8k_hf"
    safetensors_file = os.path.join(hf_adapter_dir, "adapter_model.safetensors")
    
    print("\n📊 HuggingFace adapter weights:")
    with safetensors.safe_open(safetensors_file, framework="pt") as f:
        for i, key in enumerate(list(f.keys())[:5]):  # Check first 5
            tensor = f.get_tensor(key)
            print(f"  {key}: {tensor.dtype}")
    
    # Check our converted weights
    converted_adapter_dir = "adapters/phi2_gsm8k_converted"
    layer_file = os.path.join(converted_adapter_dir, "layer_0.pt")
    
    if os.path.exists(layer_file):
        print("\n📊 Our converted weights:")
        weights = torch.load(layer_file, map_location="cpu")
        for module_name, module_data in weights.items():
            for weight_name, weight_tensor in module_data.items():
                if isinstance(weight_tensor, torch.Tensor):
                    print(f"  {module_name}.{weight_name}: {weight_tensor.dtype}")
    
    # Check model weights
    print("\n📊 Model weights (loading Phi-2):")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    
    # Check a few model parameters
    for name, param in list(model.named_parameters())[:5]:
        print(f"  {name}: {param.dtype}")


def fix_converted_weights():
    """Fix the converted weights to match model dtype."""
    
    print("\n🔧 Fixing converted weights...")
    
    converted_adapter_dir = "adapters/phi2_gsm8k_converted"
    
    # Process each layer file
    for layer_file in os.listdir(converted_adapter_dir):
        if layer_file.startswith("layer_") and layer_file.endswith(".pt"):
            layer_path = os.path.join(converted_adapter_dir, layer_file)
            
            # Load weights
            weights = torch.load(layer_path, map_location="cpu")
            
            # Convert all tensors to float32
            fixed_weights = {}
            for module_name, module_data in weights.items():
                fixed_weights[module_name] = {}
                for key, value in module_data.items():
                    if isinstance(value, torch.Tensor):
                        fixed_weights[module_name][key] = value.float()  # Convert to float32
                    else:
                        fixed_weights[module_name][key] = value
            
            # Save fixed weights
            torch.save(fixed_weights, layer_path)
            print(f"  ✅ Fixed {layer_file}")
    
    print("✅ All weights converted to float32")


def test_fixed_adapter():
    """Test the adapter with fixed dtypes."""
    
    print("\n🧪 Testing fixed adapter...")
    
    try:
        from src.core.engine import AdaptrixEngine
        
        # Initialize engine
        engine = AdaptrixEngine("microsoft/phi-2", "cpu")
        if not engine.initialize():
            print("❌ Failed to initialize engine")
            return False
        
        # Load the fixed adapter
        print("📥 Loading fixed adapter...")
        if not engine.load_adapter("phi2_gsm8k_converted"):
            print("❌ Failed to load adapter")
            return False
        
        print("✅ Adapter loaded successfully!")
        
        # Test with a simple math problem
        print("\n📝 Testing math problem:")
        problem = "What is 25 * 4?"
        print(f"❓ {problem}")
        
        response = engine.generate(problem, max_length=50, do_sample=False)
        print(f"🤖 Response: {response}")
        
        # Check if we get LoRA computation errors
        if "LoRA computation failed" in response:
            print("❌ Still getting LoRA computation errors")
            return False
        else:
            print("✅ No LoRA computation errors!")
            return True
        
        engine.cleanup()
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False


def main():
    """Main function."""
    print("🔧" * 60)
    print("🔧 FIXING DTYPE MISMATCH ISSUE 🔧")
    print("🔧" * 60)
    
    # Step 1: Check current dtypes
    check_dtypes()
    
    # Step 2: Fix converted weights
    fix_converted_weights()
    
    # Step 3: Test fixed adapter
    if test_fixed_adapter():
        print("\n🎊 DTYPE ISSUE FIXED! 🎊")
        print("✅ LoRA adapters now working properly")
    else:
        print("\n❌ Issue still persists")


if __name__ == "__main__":
    main()
