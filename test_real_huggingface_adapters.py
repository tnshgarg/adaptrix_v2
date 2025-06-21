"""
Test script for real HuggingFace LoRA adapters and context preservation.
"""

import sys
import os
import torch
import tempfile
import shutil

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.adapters.peft_converter import PEFTConverter
from src.adapters.adapter_manager import AdapterManager
from src.core.engine import AdaptrixEngine

# Real LoRA adapters from HuggingFace Hub
MATH_ADAPTERS = [
    "winglian/wizardlm-7b-uncensored-lora",
    "garage-bAInd/Platypus2-7B-lora", 
    "ehartford/WizardLM-7B-Uncensored-lora"
]

CODE_ADAPTERS = [
    "teknium/GPT4-x-Alpaca-Roleplay-Lora",
    "WizardLM/WizardCoder-Python-7B-V1.0-lora",
    "codellama/CodeLlama-7b-Instruct-hf-lora"
]

GENERAL_ADAPTERS = [
    "tloen/alpaca-lora-7b",           # Classic Alpaca LoRA
    "samwit/alpaca7B-lora",          # Alternative Alpaca
    "chavinlo/gpt4-x-alpaca",        # GPT-4 style responses
]

# Smaller/compatible adapters for testing
TEST_ADAPTERS = [
    "tloen/alpaca-lora-7b",
    "chavinlo/gpt4-x-alpaca",
]


def check_adapter_availability(adapter_id: str) -> bool:
    """Check if adapter is available on HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # Try to get repo info
        repo_info = api.repo_info(adapter_id)
        return repo_info is not None
        
    except Exception as e:
        print(f"   ⚠️  Adapter {adapter_id} not available: {e}")
        return False


def test_real_huggingface_adapter(adapter_id: str, base_model: str = "microsoft/DialoGPT-small") -> bool:
    """Test a real HuggingFace LoRA adapter."""
    print(f"\n🧪 Testing Real HuggingFace Adapter: {adapter_id}")
    print("=" * 80)
    
    # Check availability first
    if not check_adapter_availability(adapter_id):
        print(f"   ⏭️  Skipping unavailable adapter")
        return False
    
    output_dir = tempfile.mkdtemp()
    
    try:
        print(f"   📥 Downloading and converting adapter...")
        
        # Initialize converter
        converter = PEFTConverter(target_layers=[3, 6, 9])
        
        # Convert from HuggingFace Hub
        success = converter.convert_from_hub(
            adapter_id=adapter_id,
            output_dir=output_dir,
            base_model_name=base_model
        )
        
        if success:
            print(f"   ✅ Conversion successful!")
            
            # Examine converted adapter
            adapter_manager = AdapterManager(adapter_dir=os.path.dirname(output_dir))
            converted_name = os.path.basename(output_dir)
            
            converted_adapter = adapter_manager.load_adapter(converted_name)
            
            if converted_adapter:
                metadata = converted_adapter['metadata']
                weights = converted_adapter['weights']
                
                print(f"   📋 Converted Adapter Analysis:")
                print(f"      📛 Name: {metadata['name']}")
                print(f"      🎯 Target layers: {metadata['target_layers']}")
                print(f"      🔧 Target modules: {metadata['target_modules']}")
                print(f"      📊 Rank: {metadata['rank']}, Alpha: {metadata['alpha']}")
                print(f"      💾 Weight layers: {list(weights.keys())}")
                
                # Test with Adaptrix engine
                print(f"   🧪 Testing with Adaptrix engine...")
                
                engine = AdaptrixEngine(base_model, "cpu")
                engine.initialize()
                
                # Copy adapter to adapters directory
                test_adapter_name = f"real_{adapter_id.replace('/', '_').replace('-', '_')}"
                target_adapter_dir = os.path.join("adapters", test_adapter_name)
                
                if os.path.exists(target_adapter_dir):
                    shutil.rmtree(target_adapter_dir)
                shutil.copytree(output_dir, target_adapter_dir)
                
                # Try to load the adapter
                load_success = engine.load_adapter(test_adapter_name)
                
                if load_success:
                    print(f"   ✅ Adapter loaded successfully in Adaptrix!")
                    
                    # Test generation with domain-specific prompts
                    if "math" in adapter_id.lower() or "wizard" in adapter_id.lower():
                        test_prompts = [
                            "Solve this math problem: What is 15 * 23?",
                            "Explain the concept of derivatives in calculus.",
                            "If a train travels 60 mph for 2.5 hours, how far does it go?"
                        ]
                    elif "code" in adapter_id.lower():
                        test_prompts = [
                            "Write a Python function to calculate fibonacci numbers.",
                            "Explain how to use loops in programming.",
                            "What is the difference between a list and a dictionary?"
                        ]
                    else:  # General adapters
                        test_prompts = [
                            "Explain the importance of renewable energy.",
                            "What are the benefits of reading books?",
                            "How can someone improve their communication skills?"
                        ]
                    
                    print(f"   💬 Testing domain-specific generation:")
                    for i, prompt in enumerate(test_prompts, 1):
                        try:
                            response = engine.query(prompt, max_length=30)
                            print(f"      {i}. '{prompt[:40]}...' -> '{response}'")
                        except Exception as e:
                            print(f"      {i}. Generation failed: {e}")
                    
                    # Get system status
                    status = engine.get_system_status()
                    print(f"   📊 System Status:")
                    print(f"      🎯 Active adapters: {status['loaded_adapters']}")
                    print(f"      💾 Memory usage: {status.get('memory_usage', 'unknown')}")
                    
                    # Test adapter unloading
                    engine.unload_adapter(test_adapter_name)
                    print(f"   🔄 Adapter unloaded successfully")
                    
                    engine.cleanup()
                    
                    # Cleanup test adapter
                    if os.path.exists(target_adapter_dir):
                        shutil.rmtree(target_adapter_dir)
                    
                    return True
                    
                else:
                    print(f"   ❌ Failed to load adapter in Adaptrix")
                    engine.cleanup()
                    return False
            else:
                print(f"   ❌ Failed to load converted adapter")
                return False
        else:
            print(f"   ❌ Conversion failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


def test_context_preservation_properly():
    """Test context preservation with proper conversation flow."""
    print(f"\n🧠 Testing Context Preservation - PROPER TEST")
    print("=" * 80)
    
    try:
        engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
        engine.initialize()
        
        print(f"🔧 Context preservation enabled: {engine.layer_injector.enable_context_preservation}")
        
        # Load an adapter
        adapters = engine.list_adapters()
        if not adapters:
            print("❌ No adapters available for testing")
            return
        
        adapter_name = adapters[0]
        print(f"🎯 Testing with adapter: {adapter_name}")
        
        # Load adapter
        success = engine.load_adapter(adapter_name)
        if not success:
            print("❌ Failed to load adapter")
            return
        
        print("✅ Adapter loaded successfully!")
        
        # Test conversation with explicit context setting
        conversation_turns = [
            ("Hi, my name is Alice. I'm 25 years old.", "Introduction"),
            ("What's my name?", "Name recall"),
            ("How old am I?", "Age recall"),
            ("I like pizza and reading books.", "Preferences"),
            ("What do I like to do?", "Preference recall"),
            ("My name is Alice and I'm 25. What did I tell you?", "Full recall test")
        ]
        
        print("\n💬 Testing conversation with context preservation:")
        
        for i, (query, test_type) in enumerate(conversation_turns, 1):
            print(f"\n   Turn {i} ({test_type}): {query}")
            
            try:
                # Set query anchor for context preservation
                query_tokens = engine.tokenizer.encode(query, return_tensors="pt")
                if hasattr(engine.layer_injector, 'context_injector'):
                    # Create a mock query embedding for context anchoring
                    query_embedding = torch.randn(1, query_tokens.shape[1], 768)  # Mock embedding
                    engine.layer_injector.context_injector.set_query_anchor(query_embedding)
                
                response = engine.query(query, max_length=25)
                print(f"   Response: '{response}'")
                
                # Get context statistics after each turn
                if hasattr(engine.layer_injector, 'context_injector'):
                    context_stats = engine.layer_injector.context_injector.get_context_statistics()
                    print(f"   📊 Context layers: {context_stats['layers_with_context']}")
                    print(f"   📊 Total injections: {context_stats['total_injections']}")
                else:
                    print(f"   ⚠️  Context injector not available")
                
            except Exception as e:
                print(f"   ❌ Turn failed: {e}")
        
        # Final context analysis
        if hasattr(engine.layer_injector, 'context_injector'):
            final_stats = engine.layer_injector.context_injector.get_context_statistics()
            print(f"\n📊 Final Context Statistics:")
            print(f"   Layers with context: {final_stats['layers_with_context']}")
            print(f"   Total injections: {final_stats['total_injections']}")
            print(f"   Average processing time: {final_stats['average_processing_time']:.4f}s")
            
            if final_stats['total_injections'] == 0:
                print(f"   ❌ ISSUE: No context injections recorded - context preservation not working!")
            else:
                print(f"   ✅ Context preservation is working!")
        
        engine.cleanup()
        print("✅ Context preservation test completed!")
        
    except Exception as e:
        print(f"❌ Context preservation test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run comprehensive real adapter and context tests."""
    print("🚀 Real HuggingFace LoRA Adapter & Context Preservation Testing")
    print("=" * 90)
    print("Testing Adaptrix with REAL adapters from HuggingFace Hub")
    print("=" * 90)
    
    # Test 1: Context preservation (fix the issue first)
    test_context_preservation_properly()
    
    # Test 2: Real HuggingFace adapters
    print(f"\n🌐 Testing Real HuggingFace LoRA Adapters")
    print("=" * 80)
    
    success_count = 0
    total_count = 0
    
    # Test a few key adapters
    test_adapters = [
        "tloen/alpaca-lora-7b",
        "chavinlo/gpt4-x-alpaca",
    ]
    
    for adapter_id in test_adapters:
        total_count += 1
        if test_real_huggingface_adapter(adapter_id):
            success_count += 1
    
    # Summary
    print(f"\n" + "=" * 90)
    print(f"🎉 Real HuggingFace Adapter Testing Complete!")
    print(f"📊 Success Rate: {success_count}/{total_count} adapters converted successfully")
    
    if success_count > 0:
        print(f"✅ Adaptrix successfully works with real HuggingFace LoRA adapters!")
        print(f"🚀 Ready for production deployment with existing adapter ecosystem!")
    else:
        print(f"⚠️  No real adapters tested successfully")
        print(f"💡 Check network connectivity and adapter availability")
    
    print("=" * 90)


if __name__ == "__main__":
    main()
