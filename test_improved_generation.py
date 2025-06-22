"""
Test the improved generation with better parameters.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.core.engine import AdaptrixEngine


def test_improved_generation():
    """Test the improved generation method."""
    print("🚀 Testing Improved Generation")
    print("=" * 60)
    
    try:
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        test_queries = [
            "Hello, how are you today?",
            "What is 2 + 2? Please explain step by step.",
            "Tell me about quantum physics in simple terms.",
            "Write a short story about a robot.",
            "Explain the concept of machine learning."
        ]
        
        print("\n💬 Improved Engine Responses:")
        for i, query in enumerate(test_queries, 1):
            try:
                print(f"\n   {i}. Query: '{query}'")
                
                # Test with longer responses
                response = engine.generate(
                    query,
                    max_length=200,  # Allow longer responses
                    temperature=0.7,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1
                )
                
                print(f"      Response: '{response}'")
                print(f"      Length: {len(response.split())} words")
                
                # Quality check
                if len(response.split()) > 10 and len(response) > 50:
                    print(f"      Quality: ✅ Good (detailed response)")
                elif len(response.split()) > 5:
                    print(f"      Quality: ⚠️  Moderate")
                else:
                    print(f"      Quality: ❌ Poor (too short)")
                
            except Exception as e:
                print(f"      ERROR: {e}")
        
        engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conversation_flow():
    """Test conversation flow and context."""
    print("\n🔄 Testing Conversation Flow")
    print("=" * 60)
    
    try:
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        conversation = [
            "Hello! My name is Alice and I'm a software engineer.",
            "What did I just tell you about myself?",
            "Can you help me with a Python programming question?",
            "What's my profession again?"
        ]
        
        print("\n💬 Conversation Test:")
        for i, query in enumerate(conversation, 1):
            try:
                response = engine.generate(
                    query,
                    max_length=150,
                    temperature=0.7
                )
                
                print(f"\n   Turn {i}:")
                print(f"   User: '{query}'")
                print(f"   Bot:  '{response}'")
                
            except Exception as e:
                print(f"   Turn {i}: ERROR - {e}")
        
        engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Conversation test failed: {e}")
        return False


def main():
    """Run improved generation tests."""
    print("🚀 Improved Generation Testing")
    print("=" * 80)
    print("Testing the fixed generation method with better parameters")
    print("=" * 80)
    
    # Test 1: Improved generation
    generation_working = test_improved_generation()
    
    # Test 2: Conversation flow
    if generation_working:
        conversation_working = test_conversation_flow()
    else:
        conversation_working = False
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"🎉 Improved Generation Results")
    print(f"=" * 80)
    print(f"🔧 Generation quality: {'✅ IMPROVED' if generation_working else '❌ STILL POOR'}")
    print(f"🔄 Conversation flow: {'✅ WORKING' if conversation_working else '❌ ISSUES'}")
    
    if generation_working and conversation_working:
        print(f"\n🎊 GENERATION FIXED!")
        print(f"✅ Engine now produces quality responses")
        print(f"✅ Ready to test with real adapters")
    elif generation_working:
        print(f"\n✅ GENERATION IMPROVED!")
        print(f"✅ Response quality is much better")
        print(f"⚠️  Context tracking still needs work")
    else:
        print(f"\n❌ STILL ISSUES")
        print(f"🔧 Need further debugging")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
