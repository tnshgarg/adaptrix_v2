"""
Simple test for Mistral model compatibility with focus on working models.
"""

import sys
import os
import psutil
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))


def test_smaller_mistral_models():
    """Test smaller Mistral-based models that should work on MacBook Air."""
    print("🧪 Testing Smaller Mistral-Based Models")
    print("=" * 60)
    
    # Start with smaller models that are more likely to work
    model_candidates = [
        {
            "name": "microsoft/DialoGPT-medium",
            "size": "1.5B",
            "description": "Medium conversational model"
        },
        {
            "name": "microsoft/DialoGPT-large", 
            "size": "3B",
            "description": "Large conversational model"
        },
        {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "size": "1.1B", 
            "description": "Tiny Llama chat model"
        }
    ]
    
    working_models = []
    
    for model_info in model_candidates:
        print(f"\n🔍 Testing {model_info['name']} ({model_info['size']})")
        print(f"   Description: {model_info['description']}")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Test tokenizer first
            print("   Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_info['name'])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("   ✅ Tokenizer loaded")
            
            # Test model loading
            print("   Loading model...")
            
            # Check available memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 4:
                print(f"   ⚠️  Low memory ({available_gb:.1f} GB), trying CPU only")
                device = "cpu"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            
            # Load model with appropriate settings
            model = AutoModelForCausalLM.from_pretrained(
                model_info['name'],
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map="auto" if device != "mps" else None,
                trust_remote_code=True
            )
            
            if device == "mps":
                model = model.to(device)
            
            print(f"   ✅ Model loaded on {device}")
            
            # Test generation
            print("   Testing generation...")
            test_prompt = "Hello, how are you?"
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"   ✅ Generation successful: '{response.strip()[:50]}...'")
            
            # Check memory usage
            memory_after = psutil.virtual_memory()
            memory_used = (memory_after.used - memory.used) / (1024**3)
            print(f"   📊 Memory used: {memory_used:.1f} GB")
            
            working_models.append({
                **model_info,
                "device": device,
                "memory_used_gb": memory_used,
                "response": response.strip()
            })
            
            # Clean up
            del model
            del tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:100]}...")
    
    return working_models


def test_real_lora_adapters():
    """Test downloading and examining real LoRA adapters."""
    print(f"\n🔧 Testing Real LoRA Adapter Access")
    print("=" * 60)
    
    # LoRA adapters we want to test
    lora_adapters = [
        "yspkm/Mistral-7B-Instruct-v0.3-lora-math",
        "TheBloke/Mistral-7B-codealpaca-lora-GPTQ",
        "microsoft/DialoGPT-medium-lora",  # If it exists
    ]
    
    accessible_adapters = []
    
    for adapter_name in lora_adapters:
        print(f"\n🔍 Checking {adapter_name}")
        
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            
            # Try to list files in the repo
            files = list_repo_files(adapter_name)
            print(f"   ✅ Accessible - {len(files)} files found")
            
            # Look for LoRA-specific files
            lora_files = [f for f in files if any(keyword in f.lower() for keyword in ['adapter', 'lora', 'peft'])]
            if lora_files:
                print(f"   🎯 LoRA files found: {lora_files[:3]}...")
                accessible_adapters.append({
                    "name": adapter_name,
                    "files": files,
                    "lora_files": lora_files
                })
            else:
                print(f"   ⚠️  No obvious LoRA files found")
                
        except Exception as e:
            print(f"   ❌ Cannot access: {str(e)[:100]}...")
    
    return accessible_adapters


def recommend_best_approach():
    """Recommend the best approach based on test results."""
    print(f"\n💡 Testing System Resources")
    print("=" * 60)
    
    memory = psutil.virtual_memory()
    total_gb = memory.total / (1024**3)
    available_gb = memory.available / (1024**3)
    
    print(f"💾 Total RAM: {total_gb:.1f} GB")
    print(f"💾 Available RAM: {available_gb:.1f} GB")
    
    # Device capabilities
    if torch.backends.mps.is_available():
        print("🚀 MPS (Metal) acceleration available")
        device_rec = "mps"
    elif torch.cuda.is_available():
        print("🚀 CUDA acceleration available") 
        device_rec = "cuda"
    else:
        print("💻 CPU only")
        device_rec = "cpu"
    
    print(f"\n📋 Recommendations:")
    
    if available_gb >= 8:
        print("✅ GOOD: 8+ GB available - can try larger models")
        print("   Recommended: Try Mistral 7B with 4-bit quantization")
        print("   Alternative: Use 3B models without quantization")
    elif available_gb >= 4:
        print("⚠️  MODERATE: 4-8 GB available - use smaller models")
        print("   Recommended: Use 1-3B models")
        print("   Alternative: Try Mistral 7B with heavy quantization")
    else:
        print("❌ LIMITED: <4 GB available - very small models only")
        print("   Recommended: Close other applications first")
        print("   Alternative: Use cloud computing")
    
    return {
        "total_ram_gb": total_gb,
        "available_ram_gb": available_gb,
        "device": device_rec,
        "can_run_7b": available_gb >= 6,
        "can_run_3b": available_gb >= 3,
        "can_run_1b": available_gb >= 1.5
    }


def main():
    """Run comprehensive compatibility test."""
    print("🎯 MISTRAL MODEL COMPATIBILITY TEST")
    print("=" * 80)
    print("Finding the best Mistral-based model for your MacBook Air")
    print("=" * 80)
    
    # Test system capabilities
    system_info = recommend_best_approach()
    
    # Test smaller models first
    working_models = test_smaller_mistral_models()
    
    # Test LoRA adapter access
    accessible_adapters = test_real_lora_adapters()
    
    # Final recommendations
    print(f"\n" + "=" * 80)
    print(f"🎊 COMPATIBILITY TEST RESULTS")
    print(f"=" * 80)
    
    print(f"💻 System: {system_info['total_ram_gb']:.1f} GB RAM, {system_info['device']} acceleration")
    print(f"📊 Working Models: {len(working_models)}")
    print(f"🔧 Accessible LoRA Adapters: {len(accessible_adapters)}")
    
    if working_models:
        print(f"\n✅ WORKING MODELS:")
        for model in working_models:
            print(f"   • {model['name']} ({model['size']}) - {model['memory_used_gb']:.1f} GB")
    
    if accessible_adapters:
        print(f"\n🔧 ACCESSIBLE LORA ADAPTERS:")
        for adapter in accessible_adapters:
            print(f"   • {adapter['name']} - {len(adapter['lora_files'])} LoRA files")
    
    # Strategy recommendation
    print(f"\n🎯 RECOMMENDED STRATEGY:")
    
    if working_models and accessible_adapters:
        print("🎊 EXCELLENT: Both models and adapters accessible!")
        print("   1. Use one of the working models as base")
        print("   2. Test with accessible LoRA adapters")
        print("   3. Demonstrate real adapter switching")
        
        # Pick best model
        best_model = min(working_models, key=lambda x: x['memory_used_gb'])
        print(f"\n🏆 RECOMMENDED BASE MODEL: {best_model['name']}")
        print(f"   Memory usage: {best_model['memory_used_gb']:.1f} GB")
        print(f"   Device: {best_model['device']}")
        
    elif working_models:
        print("✅ GOOD: Models work, need to find compatible adapters")
        print("   1. Use working model as base")
        print("   2. Create synthetic adapters for demonstration")
        print("   3. Look for more LoRA adapters")
        
    else:
        print("⚠️  CHALLENGING: Need to optimize further")
        print("   1. Try with more aggressive quantization")
        print("   2. Consider cloud computing")
        print("   3. Use current DeepSeek setup as fallback")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
