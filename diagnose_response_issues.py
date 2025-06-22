"""
Diagnose response generation issues and test with proper parameters.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import torch
from transformers import AutoModel, AutoTokenizer
from src.core.engine import AdaptrixEngine


def test_raw_model_generation():
    """Test the raw model without our engine to see baseline quality."""
    print("🔍 Testing Raw Model Generation")
    print("=" * 60)
    
    try:
        model_name = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
        
        print(f"📋 Loading {model_name} directly...")
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        test_queries = [
            "Hello, how are you today?",
            "What is 2 + 2? Please explain step by step.",
            "Tell me about quantum physics in simple terms.",
            "Write a short story about a robot."
        ]
        
        print("\n💬 Raw Model Responses:")
        for i, query in enumerate(test_queries, 1):
            try:
                # Proper tokenization
                inputs = tokenizer(query, return_tensors="pt", padding=True)
                
                # Better generation parameters
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=100,  # Generate new tokens, not total length
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                # Decode only the new tokens
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                print(f"   {i}. '{query}'")
                print(f"      → '{response.strip()}'")
                print()
                
            except Exception as e:
                print(f"   {i}. '{query}' → ERROR: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Raw model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_engine_generation_params():
    """Test our engine with better generation parameters."""
    print("\n🔧 Testing Engine with Better Parameters")
    print("=" * 60)
    
    try:
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        test_queries = [
            "Hello, how are you today?",
            "What is 2 + 2? Please explain step by step.",
            "Tell me about quantum physics in simple terms."
        ]
        
        print("\n💬 Engine Responses with Better Parameters:")
        for i, query in enumerate(test_queries, 1):
            try:
                # Test with better parameters
                response = engine.generate(
                    query,
                    max_length=150,  # Longer responses
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1
                )
                print(f"   {i}. '{query}'")
                print(f"      → '{response.strip()}'")
                print()
                
            except Exception as e:
                print(f"   {i}. '{query}' → ERROR: {e}")
        
        engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_tokenizer_issues():
    """Check for tokenizer configuration issues."""
    print("\n🔍 Checking Tokenizer Configuration")
    print("=" * 60)
    
    try:
        model_name = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"📊 Tokenizer Info:")
        print(f"   Vocab size: {tokenizer.vocab_size}")
        print(f"   Model max length: {tokenizer.model_max_length}")
        print(f"   Pad token: {tokenizer.pad_token}")
        print(f"   EOS token: {tokenizer.eos_token}")
        print(f"   BOS token: {tokenizer.bos_token}")
        print(f"   UNK token: {tokenizer.unk_token}")
        
        # Test tokenization
        test_text = "Hello, how are you?"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        print(f"\n🧪 Tokenization Test:")
        print(f"   Input: '{test_text}'")
        print(f"   Tokens: {tokens}")
        print(f"   Decoded: '{decoded}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer check failed: {e}")
        return False


def main():
    """Run comprehensive response generation diagnosis."""
    print("🚀 Response Generation Diagnosis")
    print("=" * 80)
    print("Diagnosing poor response quality and generation issues")
    print("=" * 80)
    
    # Test 1: Raw model
    raw_working = test_raw_model_generation()
    
    # Test 2: Tokenizer check
    tokenizer_ok = check_tokenizer_issues()
    
    # Test 3: Engine with better params
    if raw_working:
        engine_working = test_engine_generation_params()
    else:
        engine_working = False
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"🎉 Diagnosis Results")
    print(f"=" * 80)
    print(f"🤖 Raw model: {'✅ WORKING' if raw_working else '❌ FAILED'}")
    print(f"🔤 Tokenizer: {'✅ OK' if tokenizer_ok else '❌ ISSUES'}")
    print(f"🔧 Engine: {'✅ WORKING' if engine_working else '❌ FAILED'}")
    
    if raw_working and engine_working:
        print(f"\n✅ GENERATION CAN BE FIXED!")
        print(f"🔧 Need to improve generation parameters in engine")
    elif raw_working:
        print(f"\n⚠️  ENGINE ISSUES DETECTED")
        print(f"🔧 Raw model works, but engine has problems")
    else:
        print(f"\n❌ FUNDAMENTAL MODEL ISSUES")
        print(f"🔧 Need to check model loading and configuration")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
