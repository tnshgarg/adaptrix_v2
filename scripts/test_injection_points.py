#!/usr/bin/env python3
"""
🔧 INJECTION POINTS TEST

Quick test to verify injection points are being registered correctly.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def test_injection_points():
    """Test injection point registration."""
    
    print("🔧" * 60)
    print("🔧 INJECTION POINTS TEST")
    print("🔧" * 60)
    
    try:
        from src.core.engine import AdaptrixEngine
        
        # Initialize engine
        print("🚀 Initializing engine...")
        engine = AdaptrixEngine("microsoft/phi-2", "cpu")
        
        if not engine.initialize():
            print("❌ Failed to initialize engine")
            return False
        
        print("✅ Engine initialized!")
        
        # Check injection points
        injection_points = engine.layer_injector.get_injection_points()
        print(f"\n📊 Injection points registered: {len(injection_points)}")
        
        if len(injection_points) == 0:
            print("❌ No injection points registered!")
            return False
        
        # Show first few injection points
        print("📍 First 10 injection points:")
        for i, (layer_idx, module_name) in enumerate(injection_points[:10]):
            print(f"   {i+1}. Layer {layer_idx}, Module {module_name}")
        
        if len(injection_points) > 10:
            print(f"   ... and {len(injection_points) - 10} more")
        
        # Check target modules
        target_modules = engine.layer_injector.target_modules
        print(f"\n🎯 Target modules: {target_modules}")
        
        # Check if we have the expected Phi-2 modules
        expected_phi2_modules = ['self_attn.q_proj', 'self_attn.v_proj', 'self_attn.k_proj', 'self_attn.dense', 'mlp.fc1', 'mlp.fc2']
        
        print(f"\n✅ Expected Phi-2 modules found:")
        for module in expected_phi2_modules:
            if module in target_modules:
                print(f"   ✅ {module}")
            else:
                print(f"   ❌ {module} (missing)")
        
        # Test adapter loading with a simple adapter
        print(f"\n🧪 Testing adapter loading...")
        available_adapters = engine.list_adapters()
        
        if available_adapters:
            test_adapter = available_adapters[0]
            print(f"📦 Testing with adapter: {test_adapter}")
            
            if engine.load_adapter(test_adapter):
                print(f"✅ {test_adapter} loaded successfully!")
                
                # Check active adapters
                active_adapters = engine.layer_injector.get_active_adapters()
                print(f"🔧 Active adapters: {list(active_adapters.keys())}")
                
                # Unload
                engine.unload_adapter(test_adapter)
                print(f"✅ {test_adapter} unloaded successfully!")
            else:
                print(f"❌ Failed to load {test_adapter}")
        else:
            print("⚠️ No adapters available for testing")
        
        # Cleanup
        engine.cleanup()
        
        print("\n" + "✅" * 60)
        print("✅ INJECTION POINTS TEST COMPLETE!")
        print("✅" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    success = test_injection_points()
    
    if success:
        print("\n🎯 INJECTION POINTS WORKING CORRECTLY!")
    else:
        print("\n❌ Injection points test failed")


if __name__ == "__main__":
    main()
