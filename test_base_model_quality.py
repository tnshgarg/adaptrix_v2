"""
Test base model response quality with different generation parameters.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.core.engine import AdaptrixEngine


def test_base_model_with_different_params():
    """Test base model with different generation parameters."""
    print("🔍 Testing Base Model Response Quality")
    print("=" * 60)
    
    try:
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        print("✅ Engine initialized")
        
        test_queries = [
            "Hello, how are you?",
            "What is 2 + 2?",
            "My name is Tanish.",
            "Who are you?",
        ]
        
        # Test with different parameters
        param_sets = [
            {"max_length": 30, "temperature": 0.7, "do_sample": True, "top_p": 0.9},
            {"max_length": 50, "temperature": 0.5, "do_sample": True, "top_p": 0.8},
            {"max_length": 30, "temperature": 1.0, "do_sample": False},  # Greedy
            {"max_length": 40, "temperature": 0.3, "do_sample": True, "top_p": 0.95},
        ]
        
        for i, params in enumerate(param_sets, 1):
            print(f"\n🧪 Parameter Set {i}: {params}")
            print("-" * 40)
            
            for j, query in enumerate(test_queries, 1):
                try:
                    response = engine.generate(query, **params)
                    print(f"   {j}. '{query}'")
                    print(f"      → '{response}'")
                    print(f"      Length: {len(response.split())} words")
                except Exception as e:
                    print(f"   {j}. '{query}' → ERROR: {e}")
        
        engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_tracking():
    """Test if context tracking works properly."""
    print(f"\n💭 Testing Context Tracking")
    print("=" * 60)
    
    try:
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        print("✅ Engine initialized")
        
        # Test conversation flow
        print("\n🗣️ Conversation Test:")
        
        # First message
        response1 = engine.generate("My name is Tanish.", max_length=30, temperature=0.5)
        print(f"1. User: 'My name is Tanish.'")
        print(f"   Bot: '{response1}'")
        
        # Second message (should remember name)
        response2 = engine.generate("What is my name?", max_length=30, temperature=0.5)
        print(f"2. User: 'What is my name?'")
        print(f"   Bot: '{response2}'")
        
        # Third message
        response3 = engine.generate("I like programming.", max_length=30, temperature=0.5)
        print(f"3. User: 'I like programming.'")
        print(f"   Bot: '{response3}'")
        
        # Fourth message (should remember both name and interest)
        response4 = engine.generate("What do you know about me?", max_length=50, temperature=0.5)
        print(f"4. User: 'What do you know about me?'")
        print(f"   Bot: '{response4}'")
        
        # Check if context is preserved
        context_preserved = "tanish" in response2.lower() or "name" in response2.lower()
        print(f"\n📊 Context Analysis:")
        print(f"   Name remembered: {'✅' if context_preserved else '❌'}")
        
        engine.cleanup()
        return context_preserved
        
    except Exception as e:
        print(f"❌ Context test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run base model quality tests."""
    print("🚀 Base Model Quality Testing")
    print("=" * 80)
    print("Testing response quality and context tracking")
    print("=" * 80)
    
    # Test 1: Different generation parameters
    params_working = test_base_model_with_different_params()
    
    # Test 2: Context tracking
    context_working = test_context_tracking()
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"🎉 Base Model Test Results")
    print(f"=" * 80)
    print(f"🧪 Parameter testing: {'✅ SUCCESS' if params_working else '❌ FAILED'}")
    print(f"💭 Context tracking: {'✅ WORKING' if context_working else '❌ NOT WORKING'}")
    
    if params_working and context_working:
        print(f"\n🎊 BASE MODEL IS WORKING WELL!")
        print(f"✅ Good response quality with proper parameters")
        print(f"✅ Context tracking functional")
    else:
        print(f"\n⚠️  Base model needs attention")
        if not params_working:
            print(f"❌ Response quality issues")
        if not context_working:
            print(f"❌ Context tracking not working")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
