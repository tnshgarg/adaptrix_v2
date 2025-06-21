"""
Test script to verify the PEFT conversion fixes work.
"""

import sys
import os
import tempfile
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.adapters.peft_converter import PEFTConverter


def test_conversion_fix():
    """Test the fixed PEFT conversion."""
    print("🔧 Testing Fixed PEFT Conversion")
    print("=" * 50)
    
    # Test with a simple real adapter
    adapter_id = "tloen/alpaca-lora-7b"
    output_dir = tempfile.mkdtemp()
    
    try:
        print(f"📥 Testing conversion of {adapter_id}...")
        
        converter = PEFTConverter(target_layers=[3, 6, 9])
        
        success = converter.convert_from_hub(
            adapter_id=adapter_id,
            output_dir=output_dir,
            base_model_name="microsoft/DialoGPT-small"
        )
        
        if success:
            print("✅ Conversion reported success!")
            
            # Check what was created
            print(f"\n📁 Output directory contents:")
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    print(f"   {item}: {size:,} bytes")
            
            # Check metadata
            metadata_path = os.path.join(output_dir, "metadata.json")
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                print(f"\n📋 Metadata:")
                print(f"   Target layers: {metadata.get('target_layers', [])}")
                print(f"   Target modules: {metadata.get('target_modules', [])}")
                print(f"   Rank: {metadata.get('rank', 'unknown')}")
                print(f"   Alpha: {metadata.get('alpha', 'unknown')}")
                
                # Check if we have layer files
                target_layers = metadata.get('target_layers', [])
                if target_layers:
                    print(f"\n🏋️  Layer files:")
                    for layer_idx in target_layers:
                        layer_file = os.path.join(output_dir, f"layer_{layer_idx}.pt")
                        if os.path.exists(layer_file):
                            import torch
                            layer_weights = torch.load(layer_file, map_location='cpu')
                            print(f"   layer_{layer_idx}.pt: {len(layer_weights)} modules")
                            for module_name, module_data in layer_weights.items():
                                lora_A_shape = module_data['lora_A'].shape
                                lora_B_shape = module_data['lora_B'].shape
                                print(f"      {module_name}: A{lora_A_shape}, B{lora_B_shape}")
                        else:
                            print(f"   layer_{layer_idx}.pt: MISSING")
                    
                    if target_layers:
                        print(f"✅ Conversion successful with {len(target_layers)} layers!")
                        return True
                    else:
                        print(f"❌ No target layers found")
                        return False
                else:
                    print(f"❌ No target layers in metadata")
                    return False
            else:
                print(f"❌ No metadata file found")
                return False
        else:
            print(f"❌ Conversion failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def test_existing_adapter():
    """Test that existing adapters still work."""
    print(f"\n🧪 Testing Existing Adapter Compatibility")
    print("=" * 50)
    
    try:
        from src.core.engine import AdaptrixEngine
        
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
                response = engine.query("Hello there!", max_length=15)
                print(f"💬 Response: '{response}'")
                
                if response and response.strip():
                    print("✅ Generation working!")
                    
                    # Get context statistics
                    context_stats = engine.layer_injector.context_injector.get_context_statistics()
                    print(f"📊 Context injections: {context_stats['total_injections']}")
                    
                    engine.unload_adapter(adapter_name)
                    engine.cleanup()
                    return True
                else:
                    print("⚠️  Empty response")
                    engine.cleanup()
                    return False
            else:
                print("❌ Failed to load adapter")
                engine.cleanup()
                return False
        else:
            print("❌ No adapters available")
            engine.cleanup()
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the tests."""
    print("🚀 Testing PEFT Conversion Fixes")
    print("=" * 60)
    
    # Test 1: Existing adapter compatibility
    existing_works = test_existing_adapter()
    
    # Test 2: Real adapter conversion
    conversion_works = test_conversion_fix()
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"🎉 Test Results Summary")
    print(f"=" * 60)
    print(f"✅ Existing Adapter Test: {'PASSED' if existing_works else 'FAILED'}")
    print(f"✅ Real Adapter Conversion: {'PASSED' if conversion_works else 'FAILED'}")
    
    if existing_works and conversion_works:
        print(f"\n🎊 ALL TESTS PASSED!")
        print(f"🚀 Ready to proceed with full demo!")
    elif existing_works:
        print(f"\n✅ Existing adapters working!")
        print(f"⚠️  Real adapter conversion needs more work")
    elif conversion_works:
        print(f"\n✅ Real adapter conversion working!")
        print(f"⚠️  Existing adapter issues detected")
    else:
        print(f"\n⚠️  Both tests failed - need more debugging")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
