"""
Test to verify that adapters are actually affecting model responses.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.core.engine import AdaptrixEngine


def test_adapter_effectiveness():
    """Test that adapters actually change model behavior."""
    print("🧪 Testing Adapter Effectiveness")
    print("=" * 60)
    
    try:
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        test_queries = [
            "Hello, how are you?",
            "What is 2 + 2?",
            "Tell me about yourself"
        ]
        
        print("💬 Baseline Responses (No Adapter):")
        baseline_responses = {}
        for i, query in enumerate(test_queries, 1):
            response = engine.query(query, max_length=20)
            baseline_responses[query] = response
            print(f"   {i}. '{query}' → '{response}'")
        
        print("\n🔧 Testing General Adapter:")
        engine.load_adapter("deepseek_general")
        general_responses = {}
        for i, query in enumerate(test_queries, 1):
            response = engine.query(query, max_length=20)
            general_responses[query] = response
            print(f"   {i}. '{query}' → '{response}'")
        
        engine.unload_adapter("deepseek_general")
        
        print("\n🧮 Testing Math Adapter:")
        engine.load_adapter("deepseek_math")
        math_responses = {}
        for i, query in enumerate(test_queries, 1):
            response = engine.query(query, max_length=20)
            math_responses[query] = response
            print(f"   {i}. '{query}' → '{response}'")
        
        engine.unload_adapter("deepseek_math")
        
        # Analyze differences
        print("\n📊 Response Analysis:")
        total_queries = len(test_queries)
        general_different = 0
        math_different = 0
        
        for query in test_queries:
            baseline = baseline_responses[query]
            general = general_responses[query]
            math = math_responses[query]
            
            print(f"\n   Query: '{query}'")
            print(f"   Baseline:  '{baseline}'")
            print(f"   General:   '{general}'")
            print(f"   Math:      '{math}'")
            
            if baseline != general:
                general_different += 1
                print(f"   ✅ General adapter changed response")
            else:
                print(f"   ⚠️  General adapter had no effect")
            
            if baseline != math:
                math_different += 1
                print(f"   ✅ Math adapter changed response")
            else:
                print(f"   ⚠️  Math adapter had no effect")
        
        print(f"\n📈 Effectiveness Summary:")
        print(f"   General adapter effectiveness: {general_different}/{total_queries} queries affected")
        print(f"   Math adapter effectiveness: {math_different}/{total_queries} queries affected")
        
        # Consider test successful if at least 50% of queries are affected
        general_effective = general_different >= total_queries * 0.5
        math_effective = math_different >= total_queries * 0.5
        
        engine.cleanup()
        
        return general_effective and math_effective
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_tracking():
    """Test if the system can track context across multiple queries."""
    print("\n🔄 Testing Context Tracking")
    print("=" * 60)
    
    try:
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        # Test context tracking
        print("💬 Context Tracking Test:")
        
        # First query - establish context
        query1 = "My name is Alice and I'm 25 years old."
        response1 = engine.query(query1, max_length=15)
        print(f"   1. '{query1}' → '{response1}'")
        
        # Second query - test if context is remembered
        query2 = "What is my name?"
        response2 = engine.query(query2, max_length=15)
        print(f"   2. '{query2}' → '{response2}'")
        
        # Third query - test age context
        query3 = "How old am I?"
        response3 = engine.query(query3, max_length=15)
        print(f"   3. '{query3}' → '{response3}'")
        
        # Simple heuristic: check if name appears in response
        context_working = "alice" in response2.lower() or "25" in response3
        
        print(f"\n📊 Context Analysis:")
        print(f"   Name query response: '{response2}'")
        print(f"   Age query response: '{response3}'")
        print(f"   Context tracking: {'✅ WORKING' if context_working else '❌ NOT WORKING'}")
        
        engine.cleanup()
        return context_working
        
    except Exception as e:
        print(f"❌ Context test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive effectiveness tests."""
    print("🚀 Adapter Effectiveness Testing")
    print("=" * 80)
    print("Testing if adapters actually change model behavior and context tracking")
    print("=" * 80)
    
    # Test 1: Adapter effectiveness
    adapters_effective = test_adapter_effectiveness()
    
    # Test 2: Context tracking
    context_working = test_context_tracking()
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"🎉 Effectiveness Test Results")
    print(f"=" * 80)
    print(f"🔧 Adapter effectiveness: {'✅ WORKING' if adapters_effective else '❌ NOT WORKING'}")
    print(f"🔄 Context tracking: {'✅ WORKING' if context_working else '❌ NOT WORKING'}")
    
    overall_success = adapters_effective and context_working
    
    if overall_success:
        print(f"\n🎊 SYSTEM FULLY FUNCTIONAL!")
        print(f"✅ Adapters are changing model behavior")
        print(f"✅ Context tracking is working")
        print(f"🚀 Ready for production use!")
    elif adapters_effective:
        print(f"\n✅ ADAPTERS WORKING!")
        print(f"✅ Adapters are changing model behavior")
        print(f"⚠️  Context tracking needs improvement")
        print(f"🔧 System is functional but context could be better")
    else:
        print(f"\n⚠️  ISSUES DETECTED")
        print(f"❌ Adapters may not be working properly")
        print(f"🔧 Need to debug adapter injection")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
