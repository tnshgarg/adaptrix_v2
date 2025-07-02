#!/usr/bin/env python3
"""
🔄 PROPER ADAPTRIX CONVERTER 🔄

Converts your trained PEFT adapter to proper Adaptrix format using the correct
middle-layer injection strategy (layers 9, 14, 19).

This respects your sophisticated Adaptrix system architecture.
"""

import os
import sys
import json
import torch
from pathlib import Path
from safetensors import safe_open
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.adapters.adapter_manager import AdapterManager

class ProperAdaptrixConverter:
    """
    Proper converter that respects the Adaptrix system architecture.
    """
    
    def __init__(self):
        self.adapter_manager = AdapterManager()
        self.peft_adapter_path = "adapters/code_adapter"
        self.output_adapter_name = "code_adapter_middle_layers"
        
        # Proper middle-layer strategy
        self.middle_layers = [9, 14, 19]  # Strategic middle positions
        self.target_modules = [
            "self_attn.q_proj",
            "self_attn.k_proj", 
            "self_attn.v_proj",
            "self_attn.o_proj"
        ]
    
    def convert_peft_to_adaptrix(self) -> bool:
        """
        Convert your trained PEFT adapter to proper Adaptrix format.
        
        Returns:
            True if conversion successful
        """
        print("🔄 CONVERTING PEFT ADAPTER TO PROPER ADAPTRIX FORMAT")
        print("=" * 70)
        
        try:
            # Load your trained PEFT weights
            peft_weights_file = os.path.join(self.peft_adapter_path, "adapter_model.safetensors")
            
            if not os.path.exists(peft_weights_file):
                print(f"❌ PEFT weights file not found: {peft_weights_file}")
                return False
            
            print(f"📁 Loading PEFT weights from: {peft_weights_file}")
            
            # Load PEFT configuration
            config_file = os.path.join(self.peft_adapter_path, "adapter_config.json")
            with open(config_file, 'r') as f:
                peft_config = json.load(f)
            
            print(f"📋 PEFT Config: rank={peft_config['r']}, alpha={peft_config['lora_alpha']}")
            
            # Extract weights for middle layers
            adaptrix_weights = {}
            
            with safe_open(peft_weights_file, framework="pt", device="cpu") as f:
                print(f"\n🎯 Converting to middle layers: {self.middle_layers}")
                
                for target_layer in self.middle_layers:
                    print(f"\n   Processing layer {target_layer}:")
                    layer_weights = {}
                    
                    for module_name in self.target_modules:
                        # Strategy: Use weights from source layers that map well to target layers
                        source_layers = self._get_source_layers_for_target(target_layer)
                        
                        collected_a_weights = []
                        collected_b_weights = []
                        
                        for source_layer in source_layers:
                            lora_a_key = f"base_model.model.model.layers.{source_layer}.{module_name}.lora_A.weight"
                            lora_b_key = f"base_model.model.model.layers.{source_layer}.{module_name}.lora_B.weight"
                            
                            try:
                                lora_a = f.get_tensor(lora_a_key)
                                lora_b = f.get_tensor(lora_b_key)
                                
                                collected_a_weights.append(lora_a)
                                collected_b_weights.append(lora_b)
                                
                            except KeyError:
                                continue  # Skip if this layer/module doesn't exist
                        
                        if collected_a_weights and collected_b_weights:
                            # Create optimized weights for middle layer
                            optimized_a, optimized_b = self._optimize_for_middle_layer(
                                collected_a_weights, collected_b_weights, target_layer
                            )
                            
                            layer_weights[module_name] = {
                                'lora_A': optimized_a,
                                'lora_B': optimized_b,
                                'rank': peft_config['r'],
                                'alpha': peft_config['lora_alpha'],
                                'scaling': peft_config['lora_alpha'] / peft_config['r']  # Proper scaling
                            }
                            
                            print(f"      ✅ {module_name} (A: {optimized_a.shape}, B: {optimized_b.shape})")
                        else:
                            print(f"      ⚠️  No weights found for {module_name}")
                    
                    if layer_weights:
                        adaptrix_weights[target_layer] = layer_weights
                        print(f"      📊 Layer {target_layer}: {len(layer_weights)} modules")
            
            # Create proper metadata
            metadata = {
                "name": self.output_adapter_name,
                "version": "1.0.0",
                "description": "Your manually trained LoRA converted to proper middle-layer injection",
                "source": "converted_from_manual_training",
                "base_model": "Qwen/Qwen3-1.7B",
                "target_layers": self.middle_layers,
                "target_modules": self.target_modules,
                "rank": peft_config['r'],
                "alpha": peft_config['lora_alpha'],
                "training_steps": 4600,
                "training_data": "manual_coding_dataset_4600_samples",
                "conversion_strategy": "optimized_middle_layer_mapping",
                "converted_date": datetime.now().isoformat()
            }
            
            # Save using proper AdapterManager
            print(f"\n💾 Saving as Adaptrix adapter: {self.output_adapter_name}")
            success = self.adapter_manager.save_adapter(
                self.output_adapter_name,
                adaptrix_weights,
                metadata
            )
            
            if success:
                print(f"✅ Successfully converted to proper Adaptrix format!")
                print(f"📂 Saved to: adapters/{self.output_adapter_name}/")
                print(f"🎯 Target layers: {self.middle_layers}")
                print(f"🔧 Target modules: {len(self.target_modules)}")
                print(f"📊 Total layers: {len(adaptrix_weights)}")
                return True
            else:
                print(f"❌ Failed to save Adaptrix adapter")
                return False
                
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_source_layers_for_target(self, target_layer: int) -> list:
        """
        Get optimal source layers for a target middle layer.
        
        Args:
            target_layer: Target layer index (9, 14, or 19)
            
        Returns:
            List of source layer indices
        """
        if target_layer == 9:
            # Early-middle: Use early layers with some mid-layer knowledge
            return [0, 1, 2, 3, 4, 8, 9, 10]
        elif target_layer == 14:
            # True-middle: Use balanced early and late layers
            return [6, 7, 8, 13, 14, 15, 16]
        elif target_layer == 19:
            # Late-middle: Use late layers with some final-layer knowledge  
            return [16, 17, 18, 19, 20, 21, 22]
        else:
            # Fallback: Use nearby layers
            return list(range(max(0, target_layer-3), min(28, target_layer+4)))
    
    def _optimize_for_middle_layer(self, a_weights: list, b_weights: list, target_layer: int) -> tuple:
        """
        Optimize weights specifically for middle-layer injection.
        
        Args:
            a_weights: List of LoRA A matrices from source layers
            b_weights: List of LoRA B matrices from source layers
            target_layer: Target layer index
            
        Returns:
            Tuple of (optimized_A, optimized_B)
        """
        if not a_weights or not b_weights:
            raise ValueError("No weights provided for optimization")
        
        # Convert to tensors
        a_tensors = torch.stack(a_weights)
        b_tensors = torch.stack(b_weights)
        
        # Optimization strategy based on layer position
        if target_layer == 9:
            # Early-middle: Emphasize pattern learning, moderate scaling
            weights = torch.softmax(torch.tensor([0.3, 0.25, 0.2, 0.15, 0.1] + [0.0] * (len(a_weights)-5)), dim=0)
            scaling = 0.8
        elif target_layer == 14:
            # True-middle: Balanced approach, full strength
            weights = torch.ones(len(a_weights)) / len(a_weights)  # Equal weighting
            scaling = 1.0
        elif target_layer == 19:
            # Late-middle: Emphasize reasoning, stronger effect
            weights = torch.softmax(torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3] + [0.0] * (len(a_weights)-5)), dim=0)
            scaling = 1.2
        else:
            # Default: Simple average
            weights = torch.ones(len(a_weights)) / len(a_weights)
            scaling = 0.9
        
        # Apply weighted averaging
        weights = weights[:len(a_weights)]  # Ensure same length
        optimized_a = torch.sum(a_tensors * weights.view(-1, 1, 1), dim=0) * scaling
        optimized_b = torch.sum(b_tensors * weights.view(-1, 1, 1), dim=0) * scaling
        
        return optimized_a, optimized_b
    
    def verify_conversion(self) -> bool:
        """
        Verify the converted adapter can be loaded by AdapterManager.
        
        Returns:
            True if verification successful
        """
        print(f"\n🔍 VERIFYING CONVERTED ADAPTER")
        print("-" * 50)
        
        try:
            # Try to load the converted adapter
            adapter_data = self.adapter_manager.load_adapter(self.output_adapter_name)
            
            if adapter_data is None:
                print("❌ Failed to load converted adapter")
                return False
            
            metadata = adapter_data['metadata']
            weights = adapter_data['weights']
            
            print(f"✅ Adapter loaded successfully!")
            print(f"📋 Name: {metadata['name']}")
            print(f"🎯 Target layers: {metadata['target_layers']}")
            print(f"🔧 Target modules: {metadata['target_modules']}")
            print(f"📊 Loaded layers: {list(weights.keys())}")
            
            # Verify each layer has all required modules
            for layer_idx in metadata['target_layers']:
                if layer_idx in weights:
                    layer_weights = weights[layer_idx]
                    print(f"   Layer {layer_idx}: {len(layer_weights)} modules")
                    
                    for module_name in metadata['target_modules']:
                        if module_name in layer_weights:
                            module_data = layer_weights[module_name]
                            lora_a = module_data['lora_A']
                            lora_b = module_data['lora_B']
                            print(f"      ✅ {module_name}: A{lora_a.shape} B{lora_b.shape}")
                        else:
                            print(f"      ❌ Missing {module_name}")
                            return False
                else:
                    print(f"❌ Missing layer {layer_idx}")
                    return False
            
            print("✅ Verification successful - adapter ready for use!")
            return True
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False


def main():
    """Convert PEFT adapter to proper Adaptrix format."""
    converter = ProperAdaptrixConverter()
    
    print("🚀 STARTING PROPER ADAPTRIX CONVERSION")
    print("=" * 70)
    
    # Convert
    if not converter.convert_peft_to_adaptrix():
        print("❌ Conversion failed!")
        return False
    
    # Verify
    if not converter.verify_conversion():
        print("❌ Verification failed!")
        return False
    
    print("\n🎉 CONVERSION COMPLETED SUCCESSFULLY!")
    print("📂 Your trained adapter is now in proper Adaptrix format")
    print("🎯 Ready for middle-layer injection testing")
    
    return True


if __name__ == "__main__":
    main() 