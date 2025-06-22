"""
Test the training framework components without full training.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))


def test_imports():
    """Test that all training components can be imported."""
    print("🧪 Testing Training Framework Imports")
    print("=" * 50)
    
    try:
        print("Testing config import...")
        from src.training.config import TrainingConfig, MATH_CONFIG
        print("✅ Config import successful")
        
        print("Testing data handler import...")
        from src.training.data_handler import DatasetHandler, GSM8KHandler
        print("✅ Data handler import successful")
        
        print("Testing evaluator import...")
        from src.training.evaluator import AdapterEvaluator
        print("✅ Evaluator import successful")
        
        print("Testing trainer import...")
        from src.training.trainer import LoRATrainer
        print("✅ Trainer import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration creation and serialization."""
    print("\n🔧 Testing Configuration System")
    print("=" * 50)
    
    try:
        from src.training.config import TrainingConfig, MATH_CONFIG, get_config_for_domain
        
        # Test default config
        config = TrainingConfig()
        print(f"✅ Default config created: {config.adapter_name}")
        
        # Test math config
        math_config = MATH_CONFIG
        print(f"✅ Math config loaded: {math_config.adapter_name}")
        
        # Test domain config
        domain_config = get_config_for_domain('math')
        print(f"✅ Domain config loaded: {domain_config.adapter_name}")
        
        # Test serialization
        config_dict = config.to_dict()
        print(f"✅ Config serialization: {len(config_dict)} keys")
        
        # Test deserialization
        new_config = TrainingConfig.from_dict(config_dict)
        print(f"✅ Config deserialization: {new_config.adapter_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_handler():
    """Test dataset handler without actually loading data."""
    print("\n📊 Testing Dataset Handler")
    print("=" * 50)
    
    try:
        from src.training.config import MATH_CONFIG
        from src.training.data_handler import get_dataset_handler, GSM8KHandler
        
        # Test handler creation
        handler = get_dataset_handler(MATH_CONFIG)
        print(f"✅ Handler created: {type(handler).__name__}")
        
        # Test GSM8K specific handler
        gsm8k_handler = GSM8KHandler(MATH_CONFIG)
        print(f"✅ GSM8K handler created: {gsm8k_handler.config.dataset_name}")
        
        # Test answer extraction
        test_solution = "Let me solve this step by step. First, I calculate 2 + 2 = 4. Then I multiply by 3 to get 12. #### 12"
        answer = gsm8k_handler._extract_answer_from_solution(test_solution)
        print(f"✅ Answer extraction: '{answer}'")
        
        # Test prompt formatting
        test_instruction = "What is 2 + 2?"
        formatted = gsm8k_handler.format_for_inference(test_instruction)
        print(f"✅ Prompt formatting: {len(formatted)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset handler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test basic model and tokenizer loading."""
    print("\n🤖 Testing Model Loading")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        model_name = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
        
        print(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer loaded: {len(tokenizer)} tokens")
        
        print(f"Loading model for {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None,  # Don't use device_map for CPU
            trust_remote_code=True
        )
        print(f"✅ Model loaded: {model.config.model_type}")
        
        # Test tokenization
        test_text = "Hello, this is a test."
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenization test: {tokens['input_ids'].shape}")
        
        # Test generation (very short)
        with torch.no_grad():
            outputs = model.generate(
                tokens['input_ids'],
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Generation test: '{generated[:50]}...'")
        
        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lora_setup():
    """Test LoRA configuration without full training."""
    print("\n🔧 Testing LoRA Setup")
    print("=" * 50)
    
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoModelForCausalLM
        import torch
        
        # Load a small model for testing
        model_name = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
        
        print("Loading model for LoRA test...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True
        )
        
        # Create LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "self_attn.q_proj",
                "self_attn.v_proj",
                "self_attn.k_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj"
            ],
            bias="none"
        )
        print("✅ LoRA config created")
        
        # Apply LoRA
        peft_model = get_peft_model(model, lora_config)
        print("✅ LoRA applied to model")
        
        # Check trainable parameters
        peft_model.print_trainable_parameters()
        
        # Cleanup
        del peft_model
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print(f"❌ LoRA setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all framework tests."""
    print("🎯 TRAINING FRAMEWORK COMPONENT TESTS")
    print("=" * 80)
    print("Testing all components before full training")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Dataset Handler", test_dataset_handler),
        ("Model Loading", test_model_loading),
        ("LoRA Setup", test_lora_setup)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"🎊 TEST RESULTS SUMMARY")
    print(f"=" * 80)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print(f"\n🎊 ALL TESTS PASSED! Training framework is ready!")
        print(f"✅ Ready to train custom LoRA adapters")
        print(f"✅ All components working correctly")
        print(f"🚀 Proceed with actual training")
    elif passed >= total * 0.8:
        print(f"\n✅ MOSTLY WORKING! Minor issues to fix")
        print(f"⚠️  Some components need attention")
        print(f"🔧 Fix failing tests before training")
    else:
        print(f"\n❌ SIGNIFICANT ISSUES DETECTED")
        print(f"🔧 Fix major components before proceeding")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
