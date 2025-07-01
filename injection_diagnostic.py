#!/usr/bin/env python3
"""
🔧 ADAPTRIX INJECTION DIAGNOSTIC & FIX 🔧

This script diagnoses and fixes the injection point registration issues.
"""

import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core.engine import AdaptrixEngine

def diagnose_injection_points():
    """Diagnose injection point registration issues."""
    print("🔧 DIAGNOSING ADAPTRIX INJECTION SYSTEM")
    print("=" * 60)
    
    model_name = "Qwen/Qwen3-1.7B"
    device = "cpu"
    
    # Initialize Adaptrix Engine
    print("Initializing Adaptrix Engine...")
    adaptrix_engine = AdaptrixEngine(model_name, device)
    success = adaptrix_engine.initialize()
    
    if not success:
        print("❌ Failed to initialize Adaptrix engine")
        return
    
    print("✅ Adaptrix Engine initialized")
    
    # Target layers and modules
    target_layers = [9, 14, 19]
    target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']
    
    # Check what modules actually exist
    print(f"\n🔍 Checking module existence for layers {target_layers}...")
    
    for layer_idx in target_layers:
        print(f"\n📋 Layer {layer_idx}:")
        
        for module_name in target_modules:
            # Try to get the module
            module = adaptrix_engine.layer_injector._get_module_by_path(layer_idx, module_name)
            
            if module is not None:
                print(f"   ✅ {module_name}: {type(module)} - {module.weight.shape if hasattr(module, 'weight') else 'No weight'}")
                
                # Try to register injection point
                adaptrix_engine.layer_injector.register_injection_point(layer_idx, module_name)
                
                # Check if it was registered
                if (layer_idx, module_name) in adaptrix_engine.layer_injector.injection_points:
                    print(f"      ✅ Injection point registered")
                else:
                    print(f"      ❌ Failed to register injection point")
            else:
                print(f"   ❌ {module_name}: Not found")
    
    # Show registered injection points
    print(f"\n📊 Registered injection points: {len(adaptrix_engine.layer_injector.injection_points)}")
    for (layer_idx, module_name), module in adaptrix_engine.layer_injector.injection_points.items():
        print(f"   {layer_idx}.{module_name}")
    
    return adaptrix_engine

def test_injection_with_dummy_weights():
    """Test injection with dummy weights."""
    print("\n🧪 TESTING INJECTION WITH DUMMY WEIGHTS")
    print("=" * 60)
    
    adaptrix_engine = diagnose_injection_points()
    
    if not adaptrix_engine:
        return False
    
    # Create dummy adapter data
    dummy_adapter_data = {}
    target_layers = [9, 14, 19]
    
    for layer_idx in target_layers:
        layer_weights = {}
        
        # Only create weights for modules that were successfully registered
        for (reg_layer, reg_module), module in adaptrix_engine.layer_injector.injection_points.items():
            if reg_layer == layer_idx:
                # Get module dimensions
                if hasattr(module, 'weight'):
                    in_features = module.weight.shape[1]
                    out_features = module.weight.shape[0]
                    
                    layer_weights[reg_module] = {
                        'lora_A': torch.randn(8, in_features) * 0.01,
                        'lora_B': torch.randn(out_features, 8) * 0.01,
                        'scaling': 0.5,
                        'rank': 8,
                        'alpha': 16
                    }
                    
                    print(f"   ✅ Created dummy weights for {layer_idx}.{reg_module}")
        
        if layer_weights:
            dummy_adapter_data[layer_idx] = layer_weights
    
    # Test injection
    print(f"\n🔬 Testing injection for {len(dummy_adapter_data)} layers...")
    
    success_count = 0
    for layer_idx, layer_weights in dummy_adapter_data.items():
        for module_name, adapter_data in layer_weights.items():
            success = adaptrix_engine.layer_injector.inject_adapter(
                "test_adapter", layer_idx, module_name, adapter_data
            )
            if success:
                success_count += 1
                print(f"   ✅ Injected {layer_idx}.{module_name}")
            else:
                print(f"   ❌ Failed to inject {layer_idx}.{module_name}")
    
    print(f"\n📊 Injection success rate: {success_count}/{len([item for sublist in dummy_adapter_data.values() for item in sublist])}")
    
    # Test generation
    if success_count > 0:
        print("\n🚀 Testing generation with injected weights...")
        try:
            response = adaptrix_engine.generate("Write a simple Python function:", max_length=50)
            print(f"   ✅ Generation successful: {response[:100]}...")
            return True
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            return False
    else:
        print("   ❌ No successful injections, skipping generation test")
        return False

if __name__ == "__main__":
    success = test_injection_with_dummy_weights()
    
    if success:
        print("\n🎉 DIAGNOSTIC COMPLETE: INJECTION SYSTEM WORKING")
        print("The issue may be with weight loading, not injection mechanism")
    else:
        print("\n❌ DIAGNOSTIC COMPLETE: INJECTION SYSTEM NEEDS FIXES")
        print("Need to fix the injection point registration and/or injection process") 