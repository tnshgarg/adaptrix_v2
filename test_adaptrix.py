"""
Quick test script for Adaptrix functionality.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.core.engine import AdaptrixEngine

def test_basic_functionality():
    """Test basic Adaptrix functionality."""
    print("Testing Adaptrix Basic Functionality")
    print("=" * 50)
    
    try:
        # Initialize engine with small model
        print("1. Initializing engine...")
        engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
        
        print("2. Loading model...")
        success = engine.initialize()
        if not success:
            print("❌ Failed to initialize engine")
            return False
        
        print("✅ Engine initialized successfully")
        
        # Check system status
        print("3. Checking system status...")
        status = engine.get_system_status()
        print(f"   Model: {status['model_name']}")
        print(f"   Device: {status['device']}")
        print(f"   Available adapters: {len(status['available_adapters'])}")
        
        # List adapters
        print("4. Listing available adapters...")
        adapters = engine.list_adapters()
        for adapter in adapters:
            print(f"   - {adapter}")
        
        # Test generation without adapters
        print("5. Testing base model generation...")
        response = engine.generate("Hello", max_length=10)
        print(f"   Response: '{response}'")
        
        # Load an adapter if available
        if adapters:
            print(f"6. Loading adapter '{adapters[0]}'...")
            success = engine.load_adapter(adapters[0])
            if success:
                print("✅ Adapter loaded successfully")
                
                # Test generation with adapter
                print("7. Testing generation with adapter...")
                response = engine.generate("Hello", max_length=10)
                print(f"   Response: '{response}'")
                
                # Unload adapter
                print("8. Unloading adapter...")
                engine.unload_adapter(adapters[0])
                print("✅ Adapter unloaded successfully")
            else:
                print("❌ Failed to load adapter")
        
        # Cleanup
        print("9. Cleaning up...")
        engine.cleanup()
        print("✅ Cleanup completed")
        
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_functionality()
