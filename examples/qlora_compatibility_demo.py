"""
Demonstration of QLoRA compatibility and context preservation features.
"""

import sys
import os
import torch
import tempfile
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.engine import AdaptrixEngine
from src.adapters.peft_converter import PEFTConverter
from src.adapters.adapter_manager import AdapterManager
from src.models.architecture_registry import architecture_registry


def demonstrate_architecture_support():
    """Demonstrate support for different model architectures."""
    print("🏗️  Architecture Support Demonstration")
    print("=" * 60)
    
    # Test different model architectures
    models_to_test = [
        ("microsoft/DialoGPT-small", "GPT-2 Style"),
        ("gpt2", "GPT-2 Original"),
    ]
    
    for model_name, description in models_to_test:
        print(f"\n📋 Testing {description}: {model_name}")
        
        try:
            # Get architecture info
            arch_info = architecture_registry.get_architecture_info(model_name)
            
            print(f"   ✅ Architecture: {arch_info['architecture_type']}")
            print(f"   📊 Layers: {arch_info['layer_count']}")
            print(f"   🔢 Hidden Size: {arch_info['hidden_size']}")
            print(f"   🎯 Target Modules: {len(arch_info['target_modules'])} modules")
            print(f"   ⚡ Recommended Injection Layers: {arch_info['recommended_middle_layers']}")
            
            # Test with Adaptrix engine
            engine = AdaptrixEngine(model_name, "cpu")
            success = engine.initialize()
            
            if success:
                print(f"   ✅ Engine initialization successful")
                
                # Test basic generation
                response = engine.generate("Hello", max_length=5)
                print(f"   💬 Sample generation: '{response}'")
                
                engine.cleanup()
            else:
                print(f"   ❌ Engine initialization failed")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n✅ Architecture support demonstration completed!")


def demonstrate_context_preservation():
    """Demonstrate context preservation across multiple layers."""
    print("\n🧠 Context Preservation Demonstration")
    print("=" * 60)
    
    try:
        # Initialize engine with context preservation
        engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
        engine.initialize()
        
        print(f"🔧 Context preservation enabled: {engine.layer_injector.enable_context_preservation}")
        
        # Load an adapter for testing
        adapters = engine.list_adapters()
        if not adapters:
            print("❌ No adapters available for testing")
            return
        
        adapter_name = adapters[0]
        print(f"🎯 Testing with adapter: {adapter_name}")
        
        # Load adapter with multiple layers
        arch_info = engine.base_model_manager.architecture_info
        target_layers = arch_info['recommended_middle_layers'][:2]  # Use first 2 layers
        
        success = engine.load_adapter(adapter_name, layer_indices=target_layers)
        if not success:
            print("❌ Failed to load adapter")
            return
        
        print(f"✅ Adapter loaded into layers: {target_layers}")
        
        # Test context preservation with multiple queries
        test_queries = [
            "What is artificial intelligence?",
            "Explain machine learning",
            "How do neural networks work?"
        ]
        
        print("\n📝 Testing context preservation across queries:")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n   Query {i}: {query}")
            
            # Set query anchor for context preservation
            engine.layer_injector.context_injector.set_query_anchor(
                torch.randn(1, 10, 768)  # Mock query embedding
            )
            
            response = engine.query(query, max_length=20)
            print(f"   Response: {response}")
            
            # Get context statistics
            context_stats = engine.layer_injector.context_injector.get_context_statistics()
            print(f"   Context layers: {context_stats['layers_with_context']}")
            print(f"   Total injections: {context_stats['total_injections']}")
        
        # Final context statistics
        final_stats = engine.layer_injector.context_injector.get_context_statistics()
        print(f"\n📊 Final Context Statistics:")
        print(f"   Layers with context: {final_stats['layers_with_context']}")
        print(f"   Total injections: {final_stats['total_injections']}")
        print(f"   Average processing time: {final_stats['average_processing_time']:.4f}s")
        
        engine.cleanup()
        print("\n✅ Context preservation demonstration completed!")
        
    except Exception as e:
        print(f"❌ Context preservation demo failed: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_qlora_conversion():
    """Demonstrate QLoRA/PEFT adapter conversion."""
    print("\n🔄 QLoRA Conversion Demonstration")
    print("=" * 60)
    
    # Create a realistic mock PEFT adapter
    temp_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    
    try:
        print("🏗️  Creating realistic mock PEFT adapter...")
        
        # Create adapter config (realistic PEFT format)
        adapter_config = {
            "alpha": 32,
            "auto_mapping": None,
            "base_model_name_or_path": "microsoft/DialoGPT-small",
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": 16,
            "revision": None,
            "target_modules": [
                "attn.c_attn",
                "mlp.c_fc"
            ],
            "task_type": "CAUSAL_LM"
        }
        
        # Save config
        import json
        config_path = os.path.join(temp_dir, "adapter_config.json")
        with open(config_path, 'w') as f:
            json.dump(adapter_config, f, indent=2)
        
        # Create realistic LoRA weights
        mock_weights = {}
        
        # Create weights for multiple layers (simulating a real PEFT adapter)
        layers_with_weights = [2, 5, 8, 11]  # Some layers have weights
        
        for layer_idx in layers_with_weights:
            for module in ["attn.c_attn", "mlp.c_fc"]:
                if module == "attn.c_attn":
                    in_dim, out_dim = 768, 2304  # DialoGPT dimensions
                else:
                    in_dim, out_dim = 768, 3072
                
                rank = 16
                
                # Create realistic LoRA weights
                lora_A_key = f"base_model.model.transformer.h.{layer_idx}.{module}.lora_A.weight"
                lora_B_key = f"base_model.model.transformer.h.{layer_idx}.{module}.lora_B.weight"
                
                # A matrix: small random values
                mock_weights[lora_A_key] = torch.randn(rank, in_dim) * 0.02
                # B matrix: zeros (standard LoRA initialization)
                mock_weights[lora_B_key] = torch.zeros(out_dim, rank)
        
        # Save weights
        weights_path = os.path.join(temp_dir, "adapter_model.bin")
        torch.save(mock_weights, weights_path)
        
        print(f"✅ Mock PEFT adapter created")
        print(f"   📁 Path: {temp_dir}")
        print(f"   🎯 Target modules: {adapter_config['target_modules']}")
        print(f"   📊 Layers with weights: {layers_with_weights}")
        print(f"   🔢 Total weight tensors: {len(mock_weights)}")
        
        # Convert using PEFTConverter
        print("\n🔄 Converting PEFT adapter to Adaptrix format...")
        
        converter = PEFTConverter(target_layers=[3, 6, 9])
        
        success = converter.convert_from_local(
            adapter_path=temp_dir,
            output_dir=output_dir,
            base_model_name="microsoft/DialoGPT-small"
        )
        
        if success:
            print("✅ PEFT conversion successful!")
            
            # Examine converted adapter
            adapter_manager = AdapterManager(adapter_dir=os.path.dirname(output_dir))
            converted_name = os.path.basename(output_dir)
            
            # Load converted adapter
            converted_adapter = adapter_manager.load_adapter(converted_name)
            
            if converted_adapter:
                metadata = converted_adapter['metadata']
                weights = converted_adapter['weights']
                
                print(f"\n📋 Converted Adapter Details:")
                print(f"   📛 Name: {metadata['name']}")
                print(f"   🎯 Target layers: {metadata['target_layers']}")
                print(f"   🔧 Target modules: {metadata['target_modules']}")
                print(f"   📊 Rank: {metadata['rank']}, Alpha: {metadata['alpha']}")
                print(f"   💾 Weight layers: {list(weights.keys())}")
                
                # Test with Adaptrix engine
                print(f"\n🧪 Testing converted adapter with Adaptrix...")
                
                engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
                engine.initialize()
                
                # Copy adapter to adapters directory for testing
                import shutil
                target_adapter_dir = os.path.join("adapters", "converted_test_adapter")
                if os.path.exists(target_adapter_dir):
                    shutil.rmtree(target_adapter_dir)
                shutil.copytree(output_dir, target_adapter_dir)
                
                load_success = engine.load_adapter("converted_test_adapter")
                
                if load_success:
                    print("✅ Converted adapter loaded successfully!")
                    
                    # Test generation
                    test_query = "Hello, how are you?"
                    response = engine.query(test_query, max_length=15)
                    print(f"   💬 Test query: {test_query}")
                    print(f"   🤖 Response: {response}")
                    
                    # Show active adapters
                    active = engine.get_loaded_adapters()
                    print(f"   🎯 Active adapters: {active}")
                    
                else:
                    print("❌ Failed to load converted adapter in engine")
                
                engine.cleanup()
                
                # Cleanup test adapter
                if os.path.exists(target_adapter_dir):
                    shutil.rmtree(target_adapter_dir)
                
            else:
                print("❌ Failed to load converted adapter")
        else:
            print("❌ PEFT conversion failed")
        
        print("\n✅ QLoRA conversion demonstration completed!")
        
    except Exception as e:
        print(f"❌ QLoRA conversion demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def main():
    """Run all QLoRA compatibility demonstrations."""
    print("🚀 Adaptrix QLoRA Compatibility & Context Preservation Demo")
    print("=" * 70)
    print("This demo showcases the enhanced Adaptrix features:")
    print("• 🏗️  Multi-architecture support")
    print("• 🧠 Context preservation across layers")
    print("• 🔄 QLoRA/PEFT adapter conversion")
    print("• ⚡ Dynamic middle-layer injection")
    print("=" * 70)
    
    # Run demonstrations
    demonstrate_architecture_support()
    demonstrate_context_preservation()
    demonstrate_qlora_conversion()
    
    print("\n" + "=" * 70)
    print("🎉 All QLoRA compatibility demonstrations completed!")
    print("🚀 Adaptrix is now ready for real-world QLoRA adapters!")
    print("=" * 70)


if __name__ == "__main__":
    main()
