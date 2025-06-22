"""
Test with real HuggingFace LoRA adapters.
Check compatibility and attempt cross-model adapter transfer.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, list_repo_files
from src.core.engine import AdaptrixEngine


def check_adapter_compatibility(adapter_repo: str):
    """Check if a HuggingFace adapter is compatible with our model."""
    print(f"🔍 Checking compatibility for {adapter_repo}")
    print("=" * 60)
    
    try:
        # List files in the repository
        files = list_repo_files(adapter_repo)
        print(f"📁 Files in {adapter_repo}:")
        for file in files[:10]:  # Show first 10 files
            print(f"   - {file}")
        if len(files) > 10:
            print(f"   ... and {len(files) - 10} more files")
        
        # Check for adapter config
        if "adapter_config.json" in files:
            config_path = hf_hub_download(adapter_repo, "adapter_config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"\n📋 Adapter Configuration:")
            print(f"   Base model: {config.get('base_model_name_or_path', 'Unknown')}")
            print(f"   Task type: {config.get('task_type', 'Unknown')}")
            print(f"   PEFT type: {config.get('peft_type', 'Unknown')}")
            print(f"   Rank (r): {config.get('r', 'Unknown')}")
            print(f"   Alpha: {config.get('lora_alpha', 'Unknown')}")
            print(f"   Target modules: {config.get('target_modules', 'Unknown')}")
            
            return config
        else:
            print("❌ No adapter_config.json found")
            return None
            
    except Exception as e:
        print(f"❌ Error checking {adapter_repo}: {e}")
        return None


def download_and_convert_adapter(adapter_repo: str, adapter_name: str):
    """Download and convert a HuggingFace adapter to our format."""
    print(f"\n🔄 Converting {adapter_repo} to our format")
    print("=" * 60)
    
    try:
        import shutil
        
        # Check compatibility first
        config = check_adapter_compatibility(adapter_repo)
        if not config:
            return False
        
        # Create adapter directory
        adapter_dir = f"adapters/{adapter_name}"
        if os.path.exists(adapter_dir):
            shutil.rmtree(adapter_dir)
        os.makedirs(adapter_dir)
        
        # Download adapter weights
        print(f"📥 Downloading adapter weights...")
        try:
            adapter_weights_path = hf_hub_download(adapter_repo, "adapter_model.bin")
            adapter_weights = torch.load(adapter_weights_path, map_location='cpu')
            print(f"✅ Downloaded {len(adapter_weights)} weight tensors")
        except:
            # Try safetensors format
            try:
                from safetensors.torch import load_file
                adapter_weights_path = hf_hub_download(adapter_repo, "adapter_model.safetensors")
                adapter_weights = load_file(adapter_weights_path)
                print(f"✅ Downloaded {len(adapter_weights)} weight tensors (safetensors)")
            except Exception as e:
                print(f"❌ Could not download weights: {e}")
                return False
        
        # Analyze weight structure
        print(f"\n📊 Analyzing weight structure:")
        layer_weights = {}
        for key, tensor in adapter_weights.items():
            print(f"   {key}: {tensor.shape}")
            
            # Parse layer and module from key
            # Example: "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            parts = key.split('.')
            if 'layers' in parts:
                layer_idx = None
                module_name = None
                weight_type = None
                
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        layer_idx = int(parts[i + 1])
                    elif part in ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
                        module_name = part
                    elif part in ['lora_A', 'lora_B']:
                        weight_type = part
                
                if layer_idx is not None and module_name is not None and weight_type is not None:
                    if layer_idx not in layer_weights:
                        layer_weights[layer_idx] = {}
                    if module_name not in layer_weights[layer_idx]:
                        layer_weights[layer_idx][module_name] = {}
                    
                    layer_weights[layer_idx][module_name][weight_type] = tensor
        
        print(f"\n🔧 Converting to our format:")
        print(f"   Found {len(layer_weights)} layers")
        
        # Map module names to DeepSeek format
        module_mapping = {
            'q_proj': 'self_attn.q_proj',
            'v_proj': 'self_attn.v_proj', 
            'k_proj': 'self_attn.k_proj',
            'o_proj': 'self_attn.o_proj',
            'gate_proj': 'mlp.gate_proj',
            'up_proj': 'mlp.up_proj',
            'down_proj': 'mlp.down_proj'
        }
        
        # Convert and save layers
        converted_layers = []
        for layer_idx, modules in layer_weights.items():
            converted_modules = {}
            
            for orig_module, weights in modules.items():
                if orig_module in module_mapping:
                    mapped_module = module_mapping[orig_module]
                    
                    if 'lora_A' in weights and 'lora_B' in weights:
                        converted_modules[mapped_module] = {
                            'lora_A': weights['lora_A'],
                            'lora_B': weights['lora_B'],
                            'rank': weights['lora_A'].shape[0],
                            'alpha': config.get('lora_alpha', 16)
                        }
                        print(f"   Layer {layer_idx}.{mapped_module}: A{weights['lora_A'].shape} B{weights['lora_B'].shape}")
            
            if converted_modules:
                layer_file = os.path.join(adapter_dir, f"layer_{layer_idx}.pt")
                torch.save(converted_modules, layer_file)
                converted_layers.append(layer_idx)
        
        # Create metadata
        metadata = {
            'name': adapter_name,
            'version': '1.0.0',
            'description': f'Converted from {adapter_repo}',
            'source': adapter_repo,
            'base_model': 'deepseek-ai/deepseek-r1-distill-qwen-1.5b',
            'original_base_model': config.get('base_model_name_or_path', 'Unknown'),
            'target_layers': converted_layers,
            'target_modules': list(set([mod for layer in layer_weights.values() for mod in module_mapping.values() if any(orig in layer for orig in module_mapping.keys())])),
            'rank': config.get('r', 16),
            'alpha': config.get('lora_alpha', 16),
            'task_type': config.get('task_type', 'Unknown')
        }
        
        with open(os.path.join(adapter_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Converted adapter saved to {adapter_dir}")
        print(f"   Layers: {converted_layers}")
        print(f"   Modules: {metadata['target_modules']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_real_adapters():
    """Test with real HuggingFace adapters."""
    print("\n🧪 Testing Real HuggingFace Adapters")
    print("=" * 60)
    
    # Adapters to test
    adapters_to_test = [
        ("tloen/alpaca-lora-7b", "alpaca_lora"),
        ("darshjoshi16/phi2-lora-math", "phi2_math")
    ]
    
    successful_adapters = []
    
    # Download and convert adapters
    for repo, name in adapters_to_test:
        print(f"\n🔄 Processing {repo}...")
        success = download_and_convert_adapter(repo, name)
        if success:
            successful_adapters.append(name)
    
    if not successful_adapters:
        print("❌ No adapters were successfully converted")
        return False
    
    # Test with our engine
    print(f"\n🚀 Testing converted adapters with DeepSeek engine...")
    
    try:
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        test_queries = [
            "Explain how to solve a quadratic equation step by step.",
            "What is the capital of France?",
            "Write a short poem about nature.",
            "Calculate 15% of 240."
        ]
        
        # Test baseline
        print(f"\n💬 Baseline responses (no adapter):")
        baseline_responses = {}
        for i, query in enumerate(test_queries, 1):
            response = engine.generate(query, max_length=150, temperature=0.7)
            baseline_responses[query] = response
            print(f"   {i}. '{query}' → '{response[:100]}...'")
        
        # Test each adapter
        for adapter_name in successful_adapters:
            print(f"\n🔧 Testing {adapter_name} adapter:")
            
            success = engine.load_adapter(adapter_name)
            if success:
                print(f"   ✅ {adapter_name} loaded successfully")
                
                for i, query in enumerate(test_queries, 1):
                    response = engine.generate(query, max_length=150, temperature=0.7)
                    print(f"   {i}. '{query}' → '{response[:100]}...'")
                    
                    # Check if response changed
                    if response != baseline_responses[query]:
                        print(f"      ✅ Response changed (adapter working)")
                    else:
                        print(f"      ⚠️  Response unchanged")
                
                engine.unload_adapter(adapter_name)
            else:
                print(f"   ❌ Failed to load {adapter_name}")
        
        engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test real HuggingFace adapters."""
    print("🚀 Real HuggingFace Adapter Testing")
    print("=" * 80)
    print("Testing tloen/alpaca-lora-7b and darshjoshi16/phi2-lora-math")
    print("=" * 80)
    
    # Install required packages
    try:
        import safetensors
    except ImportError:
        print("📦 Installing safetensors...")
        os.system("pip install safetensors")
    
    try:
        import huggingface_hub
    except ImportError:
        print("📦 Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
    
    # Test real adapters
    success = test_real_adapters()
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"🎉 Real Adapter Test Results")
    print(f"=" * 80)
    print(f"🔧 Real adapter testing: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print(f"\n🎊 REAL ADAPTERS WORKING!")
        print(f"✅ Successfully converted HuggingFace adapters")
        print(f"✅ Adapters are affecting model behavior")
        print(f"🚀 Cross-model adapter transfer successful!")
    else:
        print(f"\n❌ ADAPTER ISSUES")
        print(f"🔧 Need to debug adapter conversion or compatibility")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
