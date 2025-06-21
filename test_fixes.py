"""
Quick test script to verify the dimension mismatch fixes.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.core.engine import AdaptrixEngine


def test_dimension_fixes():
    """Test the dimension mismatch fixes."""
    print("🔧 Testing Dimension Mismatch Fixes")
    print("=" * 50)
    
    try:
        engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
        engine.initialize()
        
        print("✅ Engine initialized")
        
        # Load an existing adapter
        adapters = engine.list_adapters()
        if adapters:
            adapter_name = adapters[0]
            print(f"🎯 Testing with adapter: {adapter_name}")
            
            success = engine.load_adapter(adapter_name)
            if success:
                print("✅ Adapter loaded successfully")
                
                # Test generation
                test_queries = [
                    "Hello there!",
                    "How are you today?",
                    "Tell me a story about a robot."
                ]
                
                print("\n💬 Testing generation:")
                for i, query in enumerate(test_queries, 1):
                    try:
                        response = engine.query(query, max_length=20)
                        print(f"   {i}. '{query}'")
                        print(f"      -> '{response}'")
                        
                        if response and response.strip():
                            print(f"      ✅ Generation working!")
                        else:
                            print(f"      ⚠️  Empty response")
                    except Exception as e:
                        print(f"      ❌ Generation failed: {e}")
                
                # Get context statistics
                context_stats = engine.layer_injector.context_injector.get_context_statistics()
                print(f"\n📊 Context Statistics:")
                print(f"   Layers with context: {context_stats['layers_with_context']}")
                print(f"   Total injections: {context_stats['total_injections']}")
                print(f"   Avg processing time: {context_stats['average_processing_time']:.4f}s")
                
                # Get system status
                status = engine.get_system_status()
                print(f"\n📊 System Status:")
                print(f"   Active adapters: {status['loaded_adapters']}")
                print(f"   Memory usage: {status.get('memory_usage', 'unknown')}")
                
                engine.unload_adapter(adapter_name)
                print(f"✅ Adapter unloaded successfully")
            else:
                print("❌ Failed to load adapter")
        else:
            print("❌ No adapters available")
        
        engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the test."""
    print("🚀 Quick Fix Verification Test")
    print("=" * 50)
    
    success = test_dimension_fixes()
    
    print(f"\n" + "=" * 50)
    if success:
        print("🎉 Fixes appear to be working!")
    else:
        print("⚠️  Issues still remain")
    print("=" * 50)


if __name__ == "__main__":
    main()
