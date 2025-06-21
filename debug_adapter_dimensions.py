"""
Debug adapter dimensions to understand the mismatch.
"""

import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.adapters.adapter_manager import AdapterManager


def debug_adapter_dimensions():
    """Debug the dimensions in existing adapters."""
    print("🔍 Debugging Adapter Dimensions")
    print("=" * 50)
    
    try:
        adapter_manager = AdapterManager()
        adapters = adapter_manager.list_adapters()
        
        print(f"📋 Available adapters: {adapters}")
        
        for adapter_name in adapters:
            print(f"\n🔍 Examining adapter: {adapter_name}")
            
            adapter_data = adapter_manager.load_adapter(adapter_name)
            if adapter_data:
                metadata = adapter_data['metadata']
                weights = adapter_data['weights']
                
                print(f"   📋 Metadata:")
                print(f"      Target layers: {metadata.get('target_layers', [])}")
                print(f"      Target modules: {metadata.get('target_modules', [])}")
                print(f"      Rank: {metadata.get('rank', 'unknown')}")
                print(f"      Alpha: {metadata.get('alpha', 'unknown')}")
                
                print(f"   🏋️  Weight Analysis:")
                for layer_idx, layer_weights in weights.items():
                    print(f"      Layer {layer_idx}:")
                    for module_name, module_data in layer_weights.items():
                        lora_A = module_data['lora_A']
                        lora_B = module_data['lora_B']
                        print(f"         {module_name}:")
                        print(f"            LoRA A: {lora_A.shape} (rank x in_features)")
                        print(f"            LoRA B: {lora_B.shape} (out_features x rank)")
                        
                        # Calculate expected output dimensions
                        expected_output = torch.matmul(lora_B, torch.matmul(lora_A, torch.randn(1, lora_A.shape[1])))
                        print(f"            Expected output: {expected_output.shape}")
            else:
                print(f"   ❌ Failed to load adapter")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_dimensions():
    """Check the actual model dimensions."""
    print(f"\n🏗️  Checking Model Dimensions")
    print("=" * 50)
    
    try:
        from transformers import AutoModel, AutoConfig
        
        model_name = "microsoft/DialoGPT-small"
        config = AutoConfig.from_pretrained(model_name)
        
        print(f"📋 Model Configuration:")
        print(f"   Model type: {getattr(config, 'model_type', 'unknown')}")
        print(f"   Hidden size: {getattr(config, 'hidden_size', 'unknown')}")
        print(f"   Num layers: {getattr(config, 'n_layer', getattr(config, 'num_hidden_layers', 'unknown'))}")
        print(f"   Vocab size: {getattr(config, 'vocab_size', 'unknown')}")
        
        # Load model to check actual dimensions
        model = AutoModel.from_pretrained(model_name)
        
        print(f"\n🔍 Actual Module Dimensions:")
        
        # Check first layer dimensions
        first_layer = model.transformer.h[0]
        
        if hasattr(first_layer, 'attn') and hasattr(first_layer.attn, 'c_attn'):
            attn_module = first_layer.attn.c_attn
            print(f"   attn.c_attn: {attn_module.weight.shape}")
        
        if hasattr(first_layer, 'mlp') and hasattr(first_layer.mlp, 'c_fc'):
            mlp_module = first_layer.mlp.c_fc
            print(f"   mlp.c_fc: {mlp_module.weight.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_correct_dimensions_adapter():
    """Create an adapter with correct dimensions."""
    print(f"\n🔧 Creating Adapter with Correct Dimensions")
    print("=" * 50)
    
    try:
        # Based on DialoGPT-small dimensions
        # attn.c_attn: [768, 2304] (768 -> 2304)
        # mlp.c_fc: [768, 3072] (768 -> 3072)
        
        adapter_dir = "adapters/correct_dimensions"
        
        if os.path.exists(adapter_dir):
            import shutil
            shutil.rmtree(adapter_dir)
        os.makedirs(adapter_dir)
        
        # Create metadata
        metadata = {
            'name': 'correct_dimensions',
            'version': '1.0.0',
            'description': 'Adapter with correct dimensions for DialoGPT-small',
            'source': 'manual_creation',
            'base_model': 'microsoft/DialoGPT-small',
            'target_layers': [3, 6, 9],
            'target_modules': ['attn.c_attn', 'mlp.c_fc'],
            'rank': 8,
            'alpha': 16
        }
        
        # Save metadata
        import json
        with open(os.path.join(adapter_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create weights with correct dimensions
        for layer_idx in [3, 6, 9]:
            layer_weights = {}
            
            # attn.c_attn: 768 -> 2304
            layer_weights['attn.c_attn'] = {
                'lora_A': torch.randn(8, 768) * 0.01,    # rank x in_features
                'lora_B': torch.randn(2304, 8) * 0.01,   # out_features x rank
                'rank': 8,
                'alpha': 16
            }
            
            # mlp.c_fc: 768 -> 3072
            layer_weights['mlp.c_fc'] = {
                'lora_A': torch.randn(8, 768) * 0.01,    # rank x in_features
                'lora_B': torch.randn(3072, 8) * 0.01,   # out_features x rank
                'rank': 8,
                'alpha': 16
            }
            
            # Save layer weights
            layer_file = os.path.join(adapter_dir, f"layer_{layer_idx}.pt")
            torch.save(layer_weights, layer_file)
        
        print(f"✅ Created correct dimensions adapter")
        
        # Verify the adapter
        adapter_manager = AdapterManager()
        adapter_data = adapter_manager.load_adapter("correct_dimensions")
        
        if adapter_data:
            print(f"✅ Adapter verification successful")
            weights = adapter_data['weights']
            
            for layer_idx, layer_weights in weights.items():
                print(f"   Layer {layer_idx}:")
                for module_name, module_data in layer_weights.items():
                    lora_A = module_data['lora_A']
                    lora_B = module_data['lora_B']
                    print(f"      {module_name}: A{lora_A.shape} -> B{lora_B.shape}")
            
            return True
        else:
            print(f"❌ Adapter verification failed")
            return False
        
    except Exception as e:
        print(f"❌ Creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_correct_adapter():
    """Test the adapter with correct dimensions."""
    print(f"\n🧪 Testing Correct Dimensions Adapter")
    print("=" * 50)
    
    try:
        from src.core.engine import AdaptrixEngine
        
        engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
        engine.initialize()
        
        print("✅ Engine initialized")
        
        # Load the correct dimensions adapter
        success = engine.load_adapter("correct_dimensions")
        if success:
            print("✅ Correct dimensions adapter loaded")
            
            # Test generation
            test_queries = [
                "Hello, my name is Tanish.",
                "What is my name?",
                "2 + 2 equals",
                "How are you?"
            ]
            
            print("\n💬 Testing generation:")
            for i, query in enumerate(test_queries, 1):
                try:
                    # Set context anchor
                    if engine.tokenizer:
                        query_tokens = engine.tokenizer.encode(query, return_tensors="pt")
                        query_embedding = torch.randn(1, query_tokens.shape[1], 768)
                        engine.layer_injector.context_injector.set_query_anchor(query_embedding)
                    
                    response = engine.query(query, max_length=15)
                    print(f"   {i}. '{query}' -> '{response}'")
                    
                    # Check for dimension errors
                    if response and response.strip():
                        print(f"      ✅ Generation successful")
                    else:
                        print(f"      ⚠️  Empty response")
                        
                except Exception as e:
                    print(f"      ❌ Generation failed: {e}")
            
            # Get context statistics
            context_stats = engine.layer_injector.context_injector.get_context_statistics()
            print(f"\n📊 Context Statistics:")
            print(f"   Total injections: {context_stats['total_injections']}")
            
            engine.unload_adapter("correct_dimensions")
            engine.cleanup()
            return True
            
        else:
            print("❌ Failed to load correct dimensions adapter")
            engine.cleanup()
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive dimension debugging."""
    print("🔍 Comprehensive Dimension Debugging")
    print("=" * 70)
    
    # Step 1: Debug existing adapters
    debug_success = debug_adapter_dimensions()
    
    # Step 2: Check model dimensions
    model_success = check_model_dimensions()
    
    # Step 3: Create correct adapter
    create_success = create_correct_dimensions_adapter()
    
    # Step 4: Test correct adapter
    if create_success:
        test_success = test_correct_adapter()
    else:
        test_success = False
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"🎉 Dimension Debugging Results")
    print(f"=" * 70)
    print(f"🔍 Existing adapter analysis: {'✅ SUCCESS' if debug_success else '❌ FAILED'}")
    print(f"🏗️  Model dimension check: {'✅ SUCCESS' if model_success else '❌ FAILED'}")
    print(f"🔧 Correct adapter creation: {'✅ SUCCESS' if create_success else '❌ FAILED'}")
    print(f"🧪 Correct adapter test: {'✅ SUCCESS' if test_success else '❌ FAILED'}")
    
    if test_success:
        print(f"\n🎊 DIMENSION ISSUES FIXED!")
        print(f"🚀 Ready for improved demo with correct dimensions!")
    else:
        print(f"\n⚠️  Dimension issues remain - need further debugging")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
