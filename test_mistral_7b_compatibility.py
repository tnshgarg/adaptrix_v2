"""
Test Mistral 7B compatibility and memory requirements on MacBook Air 16GB.
"""

import sys
import os
import psutil
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))


def check_system_resources():
    """Check available system resources."""
    print("🖥️  System Resource Check")
    print("=" * 50)
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"💾 Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"💾 Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"💾 Used RAM: {memory.used / (1024**3):.1f} GB")
    print(f"💾 RAM Usage: {memory.percent:.1f}%")
    
    # Check if MPS (Metal Performance Shaders) is available on Mac
    if torch.backends.mps.is_available():
        print("🚀 MPS (Metal) acceleration available!")
        device = "mps"
    elif torch.cuda.is_available():
        print("🚀 CUDA acceleration available!")
        device = "cuda"
    else:
        print("⚠️  CPU only - no GPU acceleration")
        device = "cpu"
    
    print(f"🎯 Recommended device: {device}")
    
    return {
        'total_ram_gb': memory.total / (1024**3),
        'available_ram_gb': memory.available / (1024**3),
        'device': device
    }


def estimate_model_memory_requirements():
    """Estimate memory requirements for different model configurations."""
    print(f"\n📊 Mistral 7B Memory Requirements")
    print("=" * 50)
    
    # Model parameters: ~7B parameters
    # Each parameter in fp16 = 2 bytes, fp32 = 4 bytes, int8 = 1 byte, int4 = 0.5 bytes
    
    model_params = 7e9  # 7 billion parameters
    
    configs = {
        "FP32 (Full Precision)": 4,
        "FP16 (Half Precision)": 2,
        "INT8 (8-bit Quantization)": 1,
        "INT4 (4-bit Quantization)": 0.5
    }
    
    print("Memory requirements for model weights only:")
    for config_name, bytes_per_param in configs.items():
        memory_gb = (model_params * bytes_per_param) / (1024**3)
        print(f"   {config_name}: {memory_gb:.1f} GB")
    
    print(f"\nNote: Add ~2-4 GB for:")
    print(f"   - Model activations and gradients")
    print(f"   - System overhead")
    print(f"   - LoRA adapter weights (~10-100 MB each)")
    
    # Recommendations
    print(f"\n💡 Recommendations for 16GB MacBook Air:")
    print(f"   ✅ INT4 Quantization: ~4-6 GB total (RECOMMENDED)")
    print(f"   ✅ INT8 Quantization: ~8-10 GB total (GOOD)")
    print(f"   ⚠️  FP16: ~14-16 GB total (TIGHT)")
    print(f"   ❌ FP32: ~28-32 GB total (IMPOSSIBLE)")


def test_quantized_mistral_loading():
    """Test loading Mistral 7B with quantization."""
    print(f"\n🧪 Testing Quantized Mistral 7B Loading")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        
        # Try different Mistral models (open access)
        model_options = [
            "teknium/OpenHermes-2.5-Mistral-7B",  # Open Mistral-based model
            "HuggingFaceH4/zephyr-7b-beta",       # Open Mistral-based model
            "mistralai/Mistral-7B-v0.1"           # Older open Mistral
        ]

        model_name = None
        for candidate in model_options:
            try:
                # Test if we can access the model
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(candidate)
                model_name = candidate
                print(f"✅ Found accessible model: {candidate}")
                break
            except Exception as e:
                print(f"❌ Cannot access {candidate}: {str(e)[:100]}...")
                continue

        if model_name is None:
            print("❌ No accessible Mistral models found")
            return False, None
        
        print(f"📋 Model: {model_name}")
        
        # Check if bitsandbytes is available for quantization
        try:
            import bitsandbytes
            quantization_available = True
            print("✅ BitsAndBytes available for quantization")
        except ImportError:
            quantization_available = False
            print("❌ BitsAndBytes not available - will try without quantization")
        
        # Test tokenizer first (lightweight)
        print(f"\n1️⃣ Testing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded successfully")
        
        # Test model loading with quantization
        print(f"\n2️⃣ Testing model loading...")
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        if quantization_available:
            # Try 4-bit quantization first
            print("   Attempting 4-bit quantization...")
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                
                print("✅ 4-bit quantized model loaded successfully!")
                quantization_used = "4-bit"
                
            except Exception as e:
                print(f"❌ 4-bit quantization failed: {e}")
                
                # Try 8-bit quantization
                print("   Attempting 8-bit quantization...")
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
                    
                    print("✅ 8-bit quantized model loaded successfully!")
                    quantization_used = "8-bit"
                    
                except Exception as e:
                    print(f"❌ 8-bit quantization failed: {e}")
                    model = None
                    quantization_used = None
        else:
            # Try loading without quantization
            print("   Attempting FP16 loading...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto" if device != "mps" else None,
                    trust_remote_code=True
                )
                
                if device == "mps":
                    model = model.to(device)
                
                print("✅ FP16 model loaded successfully!")
                quantization_used = "FP16"
                
            except Exception as e:
                print(f"❌ FP16 loading failed: {e}")
                model = None
                quantization_used = None
        
        if model is not None:
            # Test generation
            print(f"\n3️⃣ Testing generation...")
            
            test_prompt = "What is 2 + 2?"
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            if device == "mps":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"✅ Generation test successful!")
            print(f"   Prompt: '{test_prompt}'")
            print(f"   Response: '{response.strip()}'")
            
            # Check memory usage
            memory = psutil.virtual_memory()
            print(f"\n📊 Memory usage after loading:")
            print(f"   Used RAM: {memory.used / (1024**3):.1f} GB")
            print(f"   Available RAM: {memory.available / (1024**3):.1f} GB")
            print(f"   Quantization: {quantization_used}")
            
            return True, quantization_used
        else:
            print("❌ Failed to load model with any configuration")
            return False, None
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Run Mistral 7B compatibility test."""
    print("🚀 Mistral 7B MacBook Air Compatibility Test")
    print("=" * 80)
    print("Testing if Mistral 7B can run on 16GB MacBook Air")
    print("=" * 80)
    
    # Check system resources
    resources = check_system_resources()
    
    # Estimate memory requirements
    estimate_model_memory_requirements()
    
    # Check if we have enough RAM
    if resources['available_ram_gb'] < 8:
        print(f"\n⚠️  WARNING: Only {resources['available_ram_gb']:.1f} GB RAM available")
        print("   Consider closing other applications before testing")
    
    # Test actual loading
    success, quantization = test_quantized_mistral_loading()
    
    # Final assessment
    print(f"\n" + "=" * 80)
    print(f"🎯 COMPATIBILITY ASSESSMENT")
    print(f"=" * 80)
    
    if success:
        print(f"🎊 SUCCESS! Mistral 7B can run on this MacBook Air")
        print(f"✅ Working configuration: {quantization}")
        print(f"✅ Device: {resources['device']}")
        print(f"✅ Available RAM: {resources['available_ram_gb']:.1f} GB")
        print(f"\n🚀 READY TO SWITCH TO MISTRAL 7B!")
        print(f"💡 This will enable testing with real LoRA adapters from HuggingFace")
    else:
        print(f"❌ FAILED: Cannot run Mistral 7B on this system")
        print(f"💡 Consider using a smaller model or cloud computing")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
