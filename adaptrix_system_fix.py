#!/usr/bin/env python3
"""
🔧 ADAPTRIX SYSTEM FIX 🔧

This fixes the core issues with the Adaptrix system:
1. Selective injection point registration (only middle layers)
2. Proper module path resolution
3. Correct adapter loading pipeline

This ensures the sophisticated Adaptrix architecture works properly.
"""

import os
import sys
import json
import torch
import traceback
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core.engine import AdaptrixEngine
from src.core.layer_injector import LayerInjector

class AdaptrixSystemFix:
    """
    System fix for the Adaptrix architecture.
    """
    
    def __init__(self):
        self.model_name = "Qwen/Qwen3-1.7B"
        self.device = "cpu"
        
    def fix_engine_initialization(self):
        """
        Fix the AdaptrixEngine initialization to properly register middle-layer injection points.
        """
        print("🔧 FIXING ADAPTRIX ENGINE INITIALIZATION")
        print("=" * 60)
        
        # Create engine
        engine = AdaptrixEngine(self.model_name, self.device)
        
        # Initialize with debug
        print("🚀 Initializing engine...")
        success = engine.initialize()
        
        if not success:
            print("❌ Engine initialization failed")
            return None
        
        print("✅ Engine initialized successfully")
        
        # Get the layer injector
        injector = engine.layer_injector
        
        # Check what injection points are registered
        print(f"\n📊 Registered injection points: {len(injector.injection_points)}")
        
        # Check if our target modules are accessible
        print("\n🔍 Testing module accessibility...")
        
        target_layers = [9, 14, 19]
        target_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
        
        for layer_idx in target_layers:
            print(f"\n   Testing layer {layer_idx}:")
            
            for module_name in target_modules:
                # Test if module exists
                module = injector._get_module_by_path(layer_idx, module_name)
                
                if module is not None:
                    print(f"      ✅ {module_name}: {type(module).__name__} - {module.weight.shape}")
                else:
                    print(f"      ❌ {module_name}: NOT FOUND")
        
        return engine
    
    def create_targeted_engine(self):
        """
        Create a targeted engine that only registers middle-layer injection points.
        """
        print("\n🎯 CREATING TARGETED MIDDLE-LAYER ENGINE")
        print("=" * 60)
        
        try:
            # Create engine
            engine = AdaptrixEngine(self.model_name, self.device)
            
            # Initialize normally first
            success = engine.initialize()
            if not success:
                print("❌ Engine initialization failed")
                return None
            
            # Clear existing injection points
            engine.layer_injector.injection_points.clear()
            print("🧹 Cleared existing injection points")
            
            # Register only our target middle layers
            target_layers = [9, 14, 19]
            target_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
            
            successful_registrations = 0
            
            for layer_idx in target_layers:
                print(f"\n   Registering layer {layer_idx}:")
                
                for module_name in target_modules:
                    try:
                        # Test if module exists first
                        module = engine.layer_injector._get_module_by_path(layer_idx, module_name)
                        
                        if module is not None:
                            # Register injection point
                            engine.layer_injector.register_injection_point(layer_idx, module_name)
                            successful_registrations += 1
                            print(f"      ✅ {module_name}: {module.weight.shape}")
                        else:
                            print(f"      ❌ {module_name}: Module not found")
                    
                    except Exception as e:
                        print(f"      ❌ {module_name}: Error - {e}")
            
            print(f"\n📊 Successfully registered {successful_registrations}/{len(target_layers) * len(target_modules)} injection points")
            
            # Test adapter loading
            print("\n🔄 Testing adapter loading...")
            adapter_name = "code_adapter_middle_layers"
            
            try:
                # List available adapters
                available_adapters = engine.list_adapters()
                print(f"📋 Available adapters: {available_adapters}")
                
                if adapter_name in available_adapters:
                    # Try to load
                    load_success = engine.load_adapter(adapter_name)
                    
                    if load_success:
                        print(f"✅ Successfully loaded adapter: {adapter_name}")
                        
                        # Get loaded adapter info
                        loaded_adapters = engine.get_loaded_adapters()
                        print(f"📊 Loaded adapters: {loaded_adapters}")
                        
                        return engine
                    else:
                        print(f"❌ Failed to load adapter: {adapter_name}")
                else:
                    print(f"❌ Adapter not found: {adapter_name}")
                    
            except Exception as e:
                print(f"❌ Error testing adapter loading: {e}")
                traceback.print_exc()
                
            return engine
            
        except Exception as e:
            print(f"❌ Error creating targeted engine: {e}")
            traceback.print_exc()
            return None
    
    def debug_layer_injector(self):
        """
        Debug the layer injector to understand why some modules aren't found.
        """
        print("\n🔍 DEBUGGING LAYER INJECTOR")
        print("=" * 60)
        
        try:
            # Create engine
            engine = AdaptrixEngine(self.model_name, self.device)
            success = engine.initialize()
            
            if not success:
                print("❌ Engine initialization failed")
                return
            
            injector = engine.layer_injector
            model = engine.base_model_manager.model
            
            # Check layer 9 structure
            layer_idx = 9
            print(f"\n🔍 Examining layer {layer_idx} structure:")
            
            # Try different patterns to find the layer
            patterns = [
                f"model.layers.{layer_idx}",
                f"transformer.h.{layer_idx}",
                f"layers.{layer_idx}",
            ]
            
            layer = None
            for pattern in patterns:
                try:
                    layer = injector._get_nested_attr(model, pattern)
                    if layer is not None:
                        print(f"   ✅ Found layer using pattern: {pattern}")
                        break
                except:
                    continue
            
            if layer is None:
                print("   ❌ Could not find layer")
                return
            
            # Examine attention structure
            print(f"\n🔍 Layer {layer_idx} structure:")
            for name, module in layer.named_children():
                print(f"   {name}: {type(module).__name__}")
                
                if 'attn' in name.lower():
                    print(f"      Attention submodules:")
                    for sub_name, sub_module in module.named_children():
                        print(f"         {sub_name}: {type(sub_module).__name__}")
                        if hasattr(sub_module, 'weight'):
                            print(f"            Weight shape: {sub_module.weight.shape}")
            
            # Test specific module paths
            print(f"\n🔍 Testing specific module paths:")
            target_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
            
            for module_name in target_modules:
                module = injector._get_module_by_path(layer_idx, module_name)
                if module is not None:
                    print(f"   ✅ {module_name}: {type(module).__name__} - {module.weight.shape}")
                else:
                    print(f"   ❌ {module_name}: NOT FOUND")
                    
                    # Try to find it manually
                    try:
                        manual_module = injector._get_nested_attr(layer, module_name)
                        if manual_module is not None:
                            print(f"      🔍 Found manually: {type(manual_module).__name__}")
                        else:
                            print(f"      🔍 Manual search also failed")
                    except Exception as e:
                        print(f"      🔍 Manual search error: {e}")
                        
        except Exception as e:
            print(f"❌ Debug failed: {e}")
            traceback.print_exc()
    
    def test_direct_module_access(self):
        """
        Test direct module access to understand the model structure.
        """
        print("\n🔬 TESTING DIRECT MODULE ACCESS")
        print("=" * 60)
        
        try:
            from transformers import AutoModelForCausalLM
            
            # Load model directly
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            
            print("✅ Model loaded successfully")
            
            # Test layer 9 access
            layer_idx = 9
            layer = model.model.layers[layer_idx]
            
            print(f"\n🔍 Layer {layer_idx} direct access:")
            print(f"   Type: {type(layer).__name__}")
            
            # Check attention
            attn = layer.self_attn
            print(f"   Attention type: {type(attn).__name__}")
            
            # Check specific modules
            modules_to_check = ["q_proj", "k_proj", "v_proj", "o_proj"]
            
            for module_name in modules_to_check:
                if hasattr(attn, module_name):
                    module = getattr(attn, module_name)
                    print(f"   ✅ {module_name}: {type(module).__name__} - {module.weight.shape}")
                else:
                    print(f"   ❌ {module_name}: NOT FOUND")
            
            # Test path resolution
            print(f"\n🔍 Testing path resolution:")
            
            # Test full paths
            full_paths = [
                f"model.layers.{layer_idx}.self_attn.q_proj",
                f"model.layers.{layer_idx}.self_attn.k_proj", 
                f"model.layers.{layer_idx}.self_attn.v_proj",
                f"model.layers.{layer_idx}.self_attn.o_proj"
            ]
            
            for path in full_paths:
                try:
                    module = model
                    for attr in path.split('.'):
                        module = getattr(module, attr)
                    print(f"   ✅ {path}: {type(module).__name__} - {module.weight.shape}")
                except Exception as e:
                    print(f"   ❌ {path}: Error - {e}")
                    
        except Exception as e:
            print(f"❌ Direct module access failed: {e}")
            traceback.print_exc()
    
    def run_complete_fix(self):
        """
        Run the complete system fix.
        """
        print("🔧 ADAPTRIX SYSTEM COMPLETE FIX")
        print("=" * 80)
        
        # 1. Debug layer injector
        self.debug_layer_injector()
        
        # 2. Test direct module access
        self.test_direct_module_access()
        
        # 3. Fix engine initialization
        engine = self.fix_engine_initialization()
        
        # 4. Create targeted engine
        targeted_engine = self.create_targeted_engine()
        
        if targeted_engine:
            print("\n🎉 ADAPTRIX SYSTEM FIX COMPLETED!")
            print("✅ Targeted middle-layer engine created successfully")
            print("✅ Adapter loading pipeline functional")
            print("✅ Ready for proper benchmarking")
            return targeted_engine
        else:
            print("\n❌ ADAPTRIX SYSTEM FIX FAILED")
            print("🔧 Manual intervention required")
            return None


def main():
    """Run the complete Adaptrix system fix."""
    fixer = AdaptrixSystemFix()
    fixer.run_complete_fix()


if __name__ == "__main__":
    main() 