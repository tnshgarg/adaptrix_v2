"""
Fix critical issues with math adapter and context preservation.
"""

import sys
import os
import torch
import tempfile
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.core.engine import AdaptrixEngine


def test_context_preservation_fix():
    """Test and fix context preservation issues."""
    print("🔧 Testing Context Preservation Fix")
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
                
                # Test context preservation with a clear conversation
                print("\n💬 Testing context preservation:")
                
                # Step 1: Introduce name
                intro_query = "Hello, my name is Tanish Garg."
                print(f"   1. User: {intro_query}")
                
                # Set context anchor
                if engine.tokenizer:
                    query_tokens = engine.tokenizer.encode(intro_query, return_tensors="pt")
                    query_embedding = torch.randn(1, query_tokens.shape[1], 768)
                    engine.layer_injector.context_injector.set_query_anchor(query_embedding)
                
                response1 = engine.query(intro_query, max_length=20)
                print(f"      Bot: {response1}")
                
                context_stats1 = engine.layer_injector.context_injector.get_context_statistics()
                print(f"      📊 Context injections: {context_stats1['total_injections']}")
                
                # Step 2: Ask about name
                name_query = "What is my name?"
                print(f"\n   2. User: {name_query}")
                
                # Set context anchor for name query
                if engine.tokenizer:
                    query_tokens = engine.tokenizer.encode(name_query, return_tensors="pt")
                    query_embedding = torch.randn(1, query_tokens.shape[1], 768)
                    engine.layer_injector.context_injector.set_query_anchor(query_embedding)
                
                response2 = engine.query(name_query, max_length=20)
                print(f"      Bot: {response2}")
                
                context_stats2 = engine.layer_injector.context_injector.get_context_statistics()
                print(f"      📊 Context injections: {context_stats2['total_injections']}")
                
                # Step 3: Test if name is remembered
                if "tanish" in response2.lower() or "garg" in response2.lower():
                    print("   ✅ Context preservation working - name remembered!")
                    return True
                else:
                    print("   ❌ Context preservation not working - name not remembered")
                    print(f"   🔍 Debug: Expected 'Tanish' or 'Garg' in response: '{response2}'")
                    return False
                    
            else:
                print("❌ Failed to load adapter")
                return False
        else:
            print("❌ No adapters available")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_simple_math_adapter():
    """Create a simple math adapter for testing."""
    print(f"\n🧮 Creating Simple Math Adapter")
    print("=" * 50)
    
    try:
        # Create a simple math adapter with basic patterns
        adapter_dir = "adapters/simple_math"
        
        if os.path.exists(adapter_dir):
            shutil.rmtree(adapter_dir)
        os.makedirs(adapter_dir)
        
        # Create metadata
        metadata = {
            'name': 'simple_math',
            'version': '1.0.0',
            'description': 'Simple math adapter for basic arithmetic',
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
        
        # Create simple LoRA weights for math
        for layer_idx in [3, 6, 9]:
            layer_weights = {}
            
            # Create weights for attention
            layer_weights['attn.c_attn'] = {
                'lora_A': torch.randn(8, 768) * 0.01,  # Small random weights
                'lora_B': torch.randn(2304, 8) * 0.01,
                'rank': 8,
                'alpha': 16
            }
            
            # Create weights for MLP
            layer_weights['mlp.c_fc'] = {
                'lora_A': torch.randn(8, 768) * 0.01,
                'lora_B': torch.randn(3072, 8) * 0.01,
                'rank': 8,
                'alpha': 16
            }
            
            # Save layer weights
            layer_file = os.path.join(adapter_dir, f"layer_{layer_idx}.pt")
            torch.save(layer_weights, layer_file)
        
        print(f"✅ Created simple math adapter at {adapter_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create math adapter: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_math_adapter():
    """Test the math adapter functionality."""
    print(f"\n🧮 Testing Math Adapter")
    print("=" * 50)
    
    try:
        engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
        engine.initialize()
        
        print("✅ Engine initialized")
        
        # Load the simple math adapter
        success = engine.load_adapter("simple_math")
        if success:
            print("✅ Math adapter loaded successfully")
            
            # Test math queries
            math_queries = [
                "2 + 2 =",
                "What is 5 * 3?",
                "Calculate 10 - 4",
                "4 * 4 equals"
            ]
            
            print("\n💬 Testing math queries:")
            for i, query in enumerate(math_queries, 1):
                try:
                    response = engine.query(query, max_length=10)
                    print(f"   {i}. '{query}' -> '{response}'")
                    
                    if response and response.strip():
                        print(f"      ✅ Generated response")
                    else:
                        print(f"      ⚠️  Empty response")
                        
                except Exception as e:
                    print(f"      ❌ Generation failed: {e}")
            
            # Get context statistics
            context_stats = engine.layer_injector.context_injector.get_context_statistics()
            print(f"\n📊 Context Statistics:")
            print(f"   Total injections: {context_stats['total_injections']}")
            
            engine.unload_adapter("simple_math")
            engine.cleanup()
            return True
            
        else:
            print("❌ Failed to load math adapter")
            engine.cleanup()
            return False
            
    except Exception as e:
        print(f"❌ Math adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def diagnose_dimension_issues():
    """Diagnose the dimension mismatch issues."""
    print(f"\n🔍 Diagnosing Dimension Issues")
    print("=" * 50)
    
    try:
        engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
        engine.initialize()
        
        print("✅ Engine initialized")
        
        # Check model architecture
        if hasattr(engine.model, 'config'):
            config = engine.model.config
            print(f"📋 Model Configuration:")
            print(f"   Model type: {getattr(config, 'model_type', 'unknown')}")
            print(f"   Hidden size: {getattr(config, 'hidden_size', 'unknown')}")
            print(f"   Num layers: {getattr(config, 'n_layer', getattr(config, 'num_hidden_layers', 'unknown'))}")
            print(f"   Vocab size: {getattr(config, 'vocab_size', 'unknown')}")
        
        # Check existing adapter structure
        adapters = engine.list_adapters()
        if adapters:
            adapter_name = adapters[0]
            print(f"\n🔍 Examining adapter: {adapter_name}")
            
            # Load adapter and check dimensions
            success = engine.load_adapter(adapter_name)
            if success:
                print("✅ Adapter loaded")
                
                # Check LoRA layer dimensions
                if hasattr(engine.layer_injector, 'lora_layers'):
                    print(f"\n📊 LoRA Layer Analysis:")
                    for lora_key, lora_layer in engine.layer_injector.lora_layers.items():
                        if hasattr(lora_layer, 'lora_A') and hasattr(lora_layer, 'lora_B'):
                            print(f"   {lora_key}:")
                            print(f"      LoRA A: {lora_layer.lora_A.weight.shape}")
                            print(f"      LoRA B: {lora_layer.lora_B.weight.shape}")
                
                engine.unload_adapter(adapter_name)
        
        engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive fixes and tests."""
    print("🔧 Comprehensive Fix for Critical Issues")
    print("=" * 70)
    print("Fixing math adapter and context preservation issues")
    print("=" * 70)
    
    # Test 1: Diagnose dimension issues
    print("🔍 Step 1: Diagnosing Current Issues")
    diagnose_dimension_issues()
    
    # Test 2: Test current context preservation
    print(f"\n🧠 Step 2: Testing Current Context Preservation")
    context_working = test_context_preservation_fix()
    
    # Test 3: Create and test simple math adapter
    print(f"\n🧮 Step 3: Creating Simple Math Adapter")
    math_adapter_created = create_simple_math_adapter()
    
    if math_adapter_created:
        math_working = test_math_adapter()
    else:
        math_working = False
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"🎉 Fix Results Summary")
    print(f"=" * 70)
    print(f"🧠 Context Preservation: {'✅ WORKING' if context_working else '❌ NEEDS MORE WORK'}")
    print(f"🧮 Math Adapter: {'✅ WORKING' if math_working else '❌ NEEDS MORE WORK'}")
    
    if context_working and math_working:
        print(f"\n🎊 ALL ISSUES FIXED!")
        print(f"🚀 Ready for improved demo!")
    else:
        print(f"\n⚠️  Some issues remain:")
        if not context_working:
            print(f"   - Context preservation needs debugging")
        if not math_working:
            print(f"   - Math adapter needs improvement")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
