"""
Debug script to examine real adapter structures and fix conversion issues.
"""

import sys
import os
import torch
import tempfile
import shutil
from huggingface_hub import hf_hub_download, snapshot_download
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))


def examine_real_adapter(adapter_id: str):
    """Download and examine a real adapter structure."""
    print(f"\n🔍 Examining {adapter_id}")
    print("=" * 60)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Download the adapter
        print("📥 Downloading adapter...")
        adapter_path = snapshot_download(
            repo_id=adapter_id,
            local_dir=temp_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Downloaded to {adapter_path}")
        
        # List all files
        print("\n📁 Files in adapter:")
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, temp_dir)
                size = os.path.getsize(file_path)
                print(f"   {rel_path}: {size:,} bytes")
        
        # Examine config
        config_path = os.path.join(temp_dir, "adapter_config.json")
        if os.path.exists(config_path):
            print("\n📋 Adapter Config:")
            with open(config_path, 'r') as f:
                config = json.load(f)
                for key, value in config.items():
                    print(f"   {key}: {value}")
        
        # Examine weights
        weight_files = []
        for file in os.listdir(temp_dir):
            if file.endswith(('.bin', '.safetensors')):
                weight_files.append(file)
        
        if weight_files:
            print(f"\n🏋️  Weight Files: {weight_files}")
            
            for weight_file in weight_files:
                weight_path = os.path.join(temp_dir, weight_file)
                print(f"\n📊 Examining {weight_file}:")
                
                try:
                    if weight_file.endswith('.safetensors'):
                        from safetensors import safe_open
                        with safe_open(weight_path, framework="pt", device="cpu") as f:
                            keys = list(f.keys())
                    else:
                        weights = torch.load(weight_path, map_location='cpu')
                        keys = list(weights.keys())
                    
                    print(f"   Total keys: {len(keys)}")
                    print(f"   Sample keys:")
                    for i, key in enumerate(keys[:10]):
                        if weight_file.endswith('.safetensors'):
                            with safe_open(weight_path, framework="pt", device="cpu") as f:
                                tensor = f.get_tensor(key)
                                shape = tensor.shape
                        else:
                            shape = weights[key].shape
                        print(f"     {i+1}. {key}: {shape}")
                    
                    if len(keys) > 10:
                        print(f"     ... and {len(keys) - 10} more")
                    
                    # Analyze layer patterns
                    layer_patterns = {}
                    module_patterns = set()
                    
                    for key in keys:
                        # Extract layer numbers
                        import re
                        layer_match = re.search(r'\.(\d+)\.', key)
                        if layer_match:
                            layer_num = int(layer_match.group(1))
                            if layer_num not in layer_patterns:
                                layer_patterns[layer_num] = []
                            layer_patterns[layer_num].append(key)
                        
                        # Extract module patterns
                        if 'lora_A' in key or 'lora_B' in key:
                            # Extract module name
                            parts = key.split('.')
                            for i, part in enumerate(parts):
                                if part in ['lora_A', 'lora_B']:
                                    if i > 0:
                                        module_name = '.'.join(parts[max(0, i-3):i])
                                        module_patterns.add(module_name)
                    
                    print(f"\n   📊 Analysis:")
                    print(f"     Layers found: {sorted(layer_patterns.keys())}")
                    print(f"     Module patterns: {sorted(module_patterns)}")
                    
                except Exception as e:
                    print(f"   ❌ Error examining weights: {e}")
        
        return temp_dir
        
    except Exception as e:
        print(f"❌ Error examining adapter: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return None


def create_fixed_converter():
    """Create a fixed version of the PEFT converter."""
    print("\n🔧 Creating Fixed PEFT Converter")
    print("=" * 60)
    
    # Based on our analysis, create a more robust converter
    converter_code = '''
def _create_module_mapping_fixed(self, peft_target_modules: List[str], architecture: str) -> Dict[str, str]:
    """Create mapping from PEFT modules to Adaptrix modules with better detection."""
    mapping = {}
    
    # Analyze the actual PEFT target modules to create mapping
    for peft_module in peft_target_modules:
        # Map common patterns
        if 'q_proj' in peft_module or 'query' in peft_module:
            mapping['attn.c_attn'] = peft_module
        elif 'k_proj' in peft_module or 'key' in peft_module:
            if 'attn.c_attn' not in mapping:  # Only if we don't have q_proj
                mapping['attn.c_attn'] = peft_module
        elif 'v_proj' in peft_module or 'value' in peft_module:
            if 'attn.c_attn' not in mapping:  # Only if we don't have q_proj/k_proj
                mapping['attn.c_attn'] = peft_module
        elif 'o_proj' in peft_module or 'dense' in peft_module and 'attention' in peft_module:
            mapping['attn.c_proj'] = peft_module
        elif 'gate_proj' in peft_module or 'up_proj' in peft_module or ('dense' in peft_module and 'intermediate' in peft_module):
            mapping['mlp.c_fc'] = peft_module
        elif 'down_proj' in peft_module or ('dense' in peft_module and 'output' in peft_module):
            mapping['mlp.c_proj'] = peft_module
    
    return mapping

def _convert_weights_fixed(self, peft_weights: Dict[str, torch.Tensor], 
                          peft_config: str, base_model_name: str) -> Dict[str, Any]:
    """Convert PEFT weights to Adaptrix format with better layer detection."""
    
    # Extract configuration
    rank = 16  # Default
    alpha = 16  # Default
    
    # Try to extract from config string
    import re
    rank_match = re.search(r'r=(\d+)', peft_config)
    if rank_match:
        rank = int(rank_match.group(1))
    
    alpha_match = re.search(r'lora_alpha=(\d+)', peft_config)
    if alpha_match:
        alpha = int(alpha_match.group(1))
    
    # Extract target modules from config
    target_modules_match = re.search(r"target_modules=\{([^}]+)\}", peft_config)
    if target_modules_match:
        target_modules_str = target_modules_match.group(1)
        peft_target_modules = [m.strip().strip("'\"") for m in target_modules_str.split(',')]
    else:
        # Fallback: detect from weight keys
        peft_target_modules = set()
        for key in peft_weights.keys():
            if 'lora_A' in key or 'lora_B' in key:
                parts = key.split('.')
                for i, part in enumerate(parts):
                    if part in ['lora_A', 'lora_B'] and i > 0:
                        module_name = parts[i-1]
                        peft_target_modules.add(module_name)
        peft_target_modules = list(peft_target_modules)
    
    print(f"   Detected PEFT target modules: {peft_target_modules}")
    
    # Get model info
    model_info = self._get_model_architecture_info(base_model_name)
    
    # Create module mapping
    module_mapping = self._create_module_mapping_fixed(peft_target_modules, model_info['architecture'])
    print(f"   Module mapping: {module_mapping}")
    
    # Find all layers in the PEFT weights
    layer_numbers = set()
    for key in peft_weights.keys():
        layer_match = re.search(r'\.(\d+)\.', key)
        if layer_match:
            layer_numbers.add(int(layer_match.group(1)))
    
    print(f"   Found layers in PEFT weights: {sorted(layer_numbers)}")
    
    # Convert weights
    converted_weights = {}
    
    for target_layer in self.target_layers:
        layer_weights = {}
        
        for adaptrix_module, peft_module in module_mapping.items():
            # Try to find weights for this layer and module
            lora_A = None
            lora_B = None
            
            # Search patterns for this module and layer
            patterns = [
                f".{target_layer}.{peft_module}.lora_A.weight",
                f".{target_layer}.{peft_module}.lora_B.weight",
                f"layers.{target_layer}.{peft_module}.lora_A.weight",
                f"layers.{target_layer}.{peft_module}.lora_B.weight",
                f"h.{target_layer}.{peft_module}.lora_A.weight",
                f"h.{target_layer}.{peft_module}.lora_B.weight",
            ]
            
            for key in peft_weights.keys():
                if str(target_layer) in key and peft_module in key:
                    if 'lora_A' in key:
                        lora_A = peft_weights[key]
                    elif 'lora_B' in key:
                        lora_B = peft_weights[key]
            
            # If we found both weights, add them
            if lora_A is not None and lora_B is not None:
                layer_weights[adaptrix_module] = {
                    'lora_A': lora_A,
                    'lora_B': lora_B,
                    'rank': rank,
                    'alpha': alpha
                }
                print(f"   ✅ Found weights for layer {target_layer}, module {adaptrix_module}")
            else:
                # Try to redistribute from available layers
                redistributed = self._redistribute_weights_fixed(
                    peft_weights, peft_module, target_layer, rank, alpha
                )
                if redistributed:
                    layer_weights[adaptrix_module] = redistributed
                    print(f"   🔄 Redistributed weights for layer {target_layer}, module {adaptrix_module}")
        
        if layer_weights:
            converted_weights[target_layer] = layer_weights
    
    # Create metadata
    metadata = {
        'name': f"converted_peft_adapter",
        'version': '1.0.0',
        'description': f"Converted from PEFT adapter for {base_model_name}",
        'source': 'peft_conversion',
        'base_model': base_model_name,
        'target_layers': list(converted_weights.keys()),
        'target_modules': list(module_mapping.keys()),
        'rank': rank,
        'alpha': alpha,
        'original_peft_config': peft_config
    }
    
    print(f"   ✅ Converted {len(converted_weights)} layers with weights")
    
    return {
        'metadata': metadata,
        'weights': converted_weights
    }

def _redistribute_weights_fixed(self, peft_weights: Dict[str, torch.Tensor], 
                               module_name: str, target_layer: int, 
                               rank: int, alpha: float) -> Optional[Dict[str, Any]]:
    """Redistribute weights from available layers."""
    available_weights = []
    
    for key in peft_weights.keys():
        if module_name in key and 'lora_A' in key:
            layer_match = re.search(r'\.(\d+)\.', key)
            if layer_match:
                layer_num = int(layer_match.group(1))
                b_key = key.replace('lora_A', 'lora_B')
                if b_key in peft_weights:
                    available_weights.append({
                        'layer': layer_num,
                        'lora_A': peft_weights[key],
                        'lora_B': peft_weights[b_key]
                    })
    
    if available_weights:
        # Use the first available weights (could be improved with better selection)
        weights = available_weights[0]
        return {
            'lora_A': weights['lora_A'].clone(),
            'lora_B': weights['lora_B'].clone(),
            'rank': rank,
            'alpha': alpha
        }
    
    return None
'''
    
    print("✅ Fixed converter methods created")
    return converter_code


def main():
    """Main debug function."""
    print("🔍 Real Adapter Structure Analysis")
    print("=" * 80)
    
    adapters_to_examine = [
        "tloen/alpaca-lora-7b",
        "darshjoshi16/phi2-lora-math"
    ]
    
    temp_dirs = []
    
    try:
        for adapter_id in adapters_to_examine:
            temp_dir = examine_real_adapter(adapter_id)
            if temp_dir:
                temp_dirs.append(temp_dir)
        
        # Create fixed converter
        create_fixed_converter()
        
    finally:
        # Cleanup
        for temp_dir in temp_dirs:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        print(f"\n🧹 Cleaned up {len(temp_dirs)} temporary directories")


if __name__ == "__main__":
    main()
