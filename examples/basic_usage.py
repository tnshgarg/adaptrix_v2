"""
Basic usage example for Adaptrix system.
"""

import sys
import os
import torch


# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.engine import AdaptrixEngine
from src.adapters.adapter_manager import AdapterManager


def create_sample_adapter():
    """Create a sample adapter for demonstration."""
    print("Creating sample math reasoning adapter...")
    
    adapter_manager = AdapterManager()
    
    # Create sample metadata
    metadata = {
        'name': 'math_reasoning_demo',
        'version': '1.0.0',
        'description': 'Demo math reasoning adapter',
        'target_layers': [3, 6, 9],  # Updated for DialoGPT (12 layers total)
        'rank': 16,
        'alpha': 32,
        'target_modules': ['attn.c_attn', 'mlp.c_fc'],  # Updated for DialoGPT
        'performance_metrics': {
            'accuracy': 0.85,
            'latency_ms': 120
        }
    }
    
    # Create sample LoRA weights (normally these would be trained)
    weights = {}

    for layer_idx in [3, 6, 9]:  # Updated for DialoGPT
        layer_weights = {}

        for module_name in ['attn.c_attn', 'mlp.c_fc']:  # Updated for DialoGPT
            # Create random LoRA weights for demonstration
            # In practice, these would be trained weights
            if module_name == 'attn.c_attn':
                # DialoGPT c_attn combines Q, K, V (768 -> 2304)
                in_features, out_features = 768, 2304
            else:  # mlp.c_fc
                in_features, out_features = 768, 3072
            
            rank = 16
            
            layer_weights[module_name] = {
                'lora_A': torch.randn(rank, in_features) * 0.01,  # Small random weights
                'lora_B': torch.zeros(out_features, rank),  # Initialize B to zero
                'rank': rank,
                'alpha': 32
            }
        
        weights[layer_idx] = layer_weights
    
    # Save the adapter
    success = adapter_manager.save_adapter('math_reasoning_demo', weights, metadata)
    
    if success:
        print("✓ Sample adapter created successfully!")
        return True
    else:
        print("✗ Failed to create sample adapter")
        return False


def demonstrate_basic_usage():
    """Demonstrate basic Adaptrix usage."""
    print("=" * 60)
    print("Adaptrix Basic Usage Demonstration")
    print("=" * 60)
    
    try:
        # Create sample adapter first
        if not create_sample_adapter():
            return
        
        print("\n1. Initializing Adaptrix Engine...")
        engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
        
        print("2. Loading base model...")
        if not engine.initialize():
            print("✗ Failed to initialize engine")
            return
        
        print("✓ Engine initialized successfully!")
        
        # Show system status
        print("\n3. System Status:")
        status = engine.get_system_status()
        print(f"   Model: {status['model_name']}")
        print(f"   Device: {status['device']}")
        print(f"   Available adapters: {len(status['available_adapters'])}")
        
        # List available adapters
        print("\n4. Available Adapters:")
        adapters = engine.list_adapters()
        for adapter in adapters:
            info = engine.get_adapter_info(adapter)
            if info:
                print(f"   - {adapter}: {info['metadata'].get('description', 'No description')}")
        
        # Load an adapter
        print("\n5. Loading math reasoning adapter...")
        if engine.load_adapter('math_reasoning_demo'):
            print("✓ Adapter loaded successfully!")
        else:
            print("✗ Failed to load adapter")
            return
        
        # Show loaded adapters
        loaded = engine.get_loaded_adapters()
        print(f"   Currently loaded: {loaded}")
        
        # Test generation without specific prompt (base model behavior)
        print("\n6. Testing base model generation...")
        base_response = engine.generate("Hello, how are you?", max_length=50)
        print(f"   Base response: {base_response}")
        
        # Test with math-related prompt
        print("\n7. Testing with math-related prompt...")
        math_prompt = "What is 15 + 27?"
        math_response = engine.query(math_prompt, max_length=50)
        print(f"   Math prompt: {math_prompt}")
        print(f"   Response: {math_response}")
        
        # Show memory usage
        print("\n8. Memory Usage:")
        memory_info = engine.dynamic_loader.get_memory_usage()
        print(f"   LoRA memory: {memory_info['injector_memory']['memory_mb']:.2f} MB")
        print(f"   Cache memory: {memory_info['cache_memory_mb']:.2f} MB")
        print(f"   Loaded adapters: {memory_info['loaded_adapters']}")
        
        # Unload adapter
        print("\n9. Unloading adapter...")
        if engine.unload_adapter('math_reasoning_demo'):
            print("✓ Adapter unloaded successfully!")
        
        # Cleanup
        print("\n10. Cleaning up...")
        engine.cleanup()
        print("✓ Cleanup completed!")
        
        print("\n" + "=" * 60)
        print("Demonstration completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_adapter_switching():
    """Demonstrate hot-swapping adapters."""
    print("\n" + "=" * 60)
    print("Adapter Hot-Swapping Demonstration")
    print("=" * 60)
    
    try:
        # Create a second sample adapter
        print("Creating second sample adapter...")
        adapter_manager = AdapterManager()
        
        metadata = {
            'name': 'creative_writing_demo',
            'version': '1.0.0',
            'description': 'Demo creative writing adapter',
            'target_layers': [3, 6, 9],  # Updated for DialoGPT
            'rank': 16,
            'alpha': 32,
            'target_modules': ['attn.c_attn', 'mlp.c_fc']  # Updated for DialoGPT
        }
        
        # Create different weights for creative writing
        weights = {}
        for layer_idx in [3, 6, 9]:  # Updated for DialoGPT
            layer_weights = {}
            for module_name in ['attn.c_attn', 'mlp.c_fc']:  # Updated for DialoGPT
                if module_name == 'attn.c_attn':
                    in_features, out_features = 768, 2304  # DialoGPT c_attn
                else:
                    in_features, out_features = 768, 3072
                
                rank = 16
                layer_weights[module_name] = {
                    'lora_A': torch.randn(rank, in_features) * 0.02,  # Slightly different weights
                    'lora_B': torch.zeros(out_features, rank),
                    'rank': rank,
                    'alpha': 32
                }
            weights[layer_idx] = layer_weights
        
        adapter_manager.save_adapter('creative_writing_demo', weights, metadata)
        
        # Initialize engine
        engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
        engine.initialize()
        
        # Load first adapter
        print("\nLoading math reasoning adapter...")
        engine.load_adapter('math_reasoning_demo')
        
        # Test with math prompt
        response1 = engine.query("Calculate 25 * 4", max_length=30)
        print(f"Math adapter response: {response1}")
        
        # Switch to creative writing adapter
        print("\nSwitching to creative writing adapter...")
        engine.switch_adapter('math_reasoning_demo', 'creative_writing_demo')
        
        # Test with creative prompt
        response2 = engine.query("Write a short story about", max_length=50)
        print(f"Creative adapter response: {response2}")
        
        # Switch back
        print("\nSwitching back to math adapter...")
        engine.switch_adapter('creative_writing_demo', 'math_reasoning_demo')
        
        response3 = engine.query("What is 100 divided by 5?", max_length=30)
        print(f"Math adapter response: {response3}")
        
        engine.cleanup()
        print("\n✓ Hot-swapping demonstration completed!")
        
    except Exception as e:
        print(f"\n✗ Error during hot-swapping demo: {e}")


if __name__ == "__main__":
    print("Adaptrix System Demonstration")
    print("This example shows basic usage of the Adaptrix system.")
    print("Note: This will download the DialoGPT-small model (~500MB) if not cached.")
    
    response = input("\nProceed with demonstration? (y/n): ")
    if response.lower() in ['y', 'yes']:
        demonstrate_basic_usage()
        
        response2 = input("\nRun adapter switching demo? (y/n): ")
        if response2.lower() in ['y', 'yes']:
            demonstrate_adapter_switching()
    else:
        print("Demonstration cancelled.")
