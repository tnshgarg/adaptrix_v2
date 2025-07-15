#!/usr/bin/env python3
"""
🔄 CONVERT CODE ADAPTER TO ADAPTRIX FORMAT

Converts the existing code adapter from standard LoRA format to Adaptrix 
middle-layer injection format for proper testing.
"""

import sys
import os
import json
import shutil
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def convert_code_adapter_to_adaptrix():
    """Convert the code adapter to Adaptrix format."""
    
    print("🔄" * 80)
    print("🔄 CONVERTING CODE ADAPTER TO ADAPTRIX FORMAT 🔄")
    print("🔄" * 80)
    
    try:
        from src.conversion.dynamic_lora_converter import DynamicLoRAConverter
        
        # Check if code adapter exists
        code_adapter_path = Path("adapters/code")
        if not code_adapter_path.exists():
            print("❌ Code adapter not found at adapters/code")
            return False
        
        print(f"📁 Found code adapter at: {code_adapter_path}")
        
        # Check adapter configuration
        config_path = code_adapter_path / "adapter_config.json"
        if not config_path.exists():
            print("❌ adapter_config.json not found")
            return False
        
        with open(config_path, 'r') as f:
            adapter_config = json.load(f)
        
        print(f"📋 Adapter Configuration:")
        print(f"   Base Model: {adapter_config.get('base_model_name_or_path', 'Unknown')}")
        print(f"   PEFT Type: {adapter_config.get('peft_type', 'Unknown')}")
        print(f"   Rank: {adapter_config.get('r', 'Unknown')}")
        print(f"   Alpha: {adapter_config.get('lora_alpha', 'Unknown')}")
        print(f"   Target Modules: {adapter_config.get('target_modules', [])}")
        
        # Check if adapter is already in Adaptrix format
        adaptrix_metadata_path = code_adapter_path / "adaptrix_metadata.json"
        if adaptrix_metadata_path.exists():
            print("✅ Adapter already has Adaptrix metadata")
            
            # Check for layer structure
            layer_files = list(code_adapter_path.glob("layer_*.json"))
            if layer_files:
                print(f"✅ Found {len(layer_files)} layer files - already in Adaptrix format")
                return True
            else:
                print("⚠️ Has metadata but missing layer files - needs conversion")
        
        # Initialize converter
        print("\n🔧 Initializing Dynamic LoRA Converter...")
        converter = DynamicLoRAConverter()
        
        # Determine base model from config
        base_model = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen3-1.7B')
        
        # Create a temporary HuggingFace-style directory for conversion
        temp_hf_dir = Path("temp_code_adapter_hf")
        if temp_hf_dir.exists():
            shutil.rmtree(temp_hf_dir)
        
        temp_hf_dir.mkdir()
        
        try:
            # Copy adapter files to temp directory
            for file in code_adapter_path.iterdir():
                if file.is_file():
                    shutil.copy2(file, temp_hf_dir / file.name)
            
            print(f"📁 Created temporary HF directory: {temp_hf_dir}")
            
            # Convert using dynamic converter
            print("\n🔄 Converting to Adaptrix format...")
            success = converter.convert_adapter(
                hf_repo=str(temp_hf_dir),  # Use local path
                adapter_name="code",
                description="Code generation specialist adapter for programming tasks",
                capabilities=["python", "javascript", "code_generation", "debugging", "algorithms"],
                domain="programming",
                training_data="Code generation and programming datasets"
            )
            
            if success:
                print("✅ Conversion successful!")
                
                # Verify conversion
                verify_conversion()
                
                return True
            else:
                print("❌ Conversion failed")
                return False
                
        finally:
            # Cleanup temp directory
            if temp_hf_dir.exists():
                shutil.rmtree(temp_hf_dir)
                print("🗑️ Cleaned up temporary files")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_conversion():
    """Verify that the conversion was successful."""
    
    print("\n🔍 VERIFYING CONVERSION...")
    print("-" * 50)
    
    code_adapter_path = Path("adapters/code")
    
    # Check for Adaptrix metadata
    metadata_path = code_adapter_path / "adaptrix_metadata.json"
    if metadata_path.exists():
        print("✅ Adaptrix metadata found")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"   Name: {metadata.get('name', 'Unknown')}")
        print(f"   Domain: {metadata.get('domain', 'Unknown')}")
        print(f"   Capabilities: {metadata.get('capabilities', [])}")
        print(f"   Target Modules: {metadata.get('target_modules', [])}")
    else:
        print("❌ Adaptrix metadata missing")
        return False
    
    # Check for layer files
    layer_files = list(code_adapter_path.glob("layer_*.json"))
    if layer_files:
        print(f"✅ Found {len(layer_files)} layer files")
        
        # Check a sample layer file
        sample_layer = layer_files[0]
        with open(sample_layer, 'r') as f:
            layer_data = json.load(f)
        
        print(f"   Sample layer: {sample_layer.name}")
        print(f"   Modules in layer: {list(layer_data.keys())}")
    else:
        print("❌ No layer files found")
        return False
    
    # Check for weights
    weights_files = list(code_adapter_path.glob("*.safetensors")) + list(code_adapter_path.glob("*.bin"))
    if weights_files:
        print(f"✅ Found {len(weights_files)} weight files")
    else:
        print("❌ No weight files found")
        return False
    
    print("✅ Conversion verification successful!")
    return True


def test_converted_adapter():
    """Test the converted adapter with the Adaptrix engine."""
    
    print("\n🧪 TESTING CONVERTED ADAPTER...")
    print("-" * 50)
    
    try:
        from src.core.engine import AdaptrixEngine
        
        # Initialize engine
        print("🚀 Initializing Adaptrix engine...")
        engine = AdaptrixEngine("Qwen/Qwen3-1.7B", "cpu")
        
        if not engine.initialize():
            print("❌ Failed to initialize engine")
            return False
        
        print("✅ Engine initialized!")
        
        # List available adapters
        available_adapters = engine.list_adapters()
        print(f"📦 Available adapters: {available_adapters}")
        
        if "code" not in available_adapters:
            print("❌ Code adapter not found in available adapters")
            return False
        
        # Load the code adapter
        print("\n🔌 Loading code adapter...")
        if engine.load_adapter("code"):
            print("✅ Code adapter loaded successfully!")
            
            # Test generation
            print("\n🧪 Testing generation...")
            test_prompt = "Write a Python function to calculate the factorial of a number"
            
            response = engine.generate(
                test_prompt,
                max_length=300,
                temperature=0.3
            )
            
            print(f"📝 Test prompt: {test_prompt}")
            print(f"🤖 Response: {response[:200]}...")
            
            # Unload adapter
            engine.unload_adapter("code")
            print("✅ Adapter unloaded successfully!")
            
            # Cleanup
            engine.cleanup()
            
            return True
        else:
            print("❌ Failed to load code adapter")
            return False
            
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main conversion function."""
    
    print("🔄 Starting Code Adapter Conversion to Adaptrix Format...")
    
    # Convert adapter
    conversion_success = convert_code_adapter_to_adaptrix()
    
    if conversion_success:
        print("\n🎊 CONVERSION SUCCESSFUL!")
        
        # Test the converted adapter
        test_success = test_converted_adapter()
        
        if test_success:
            print("\n🎊 ADAPTER TESTING SUCCESSFUL!")
            print("✅ Code adapter is now ready for Adaptrix middle-layer injection testing")
        else:
            print("\n⚠️ Conversion successful but testing failed")
            print("🔧 Adapter may need manual adjustments")
    else:
        print("\n❌ CONVERSION FAILED")
        print("🔧 Manual conversion may be required")
    
    return conversion_success


if __name__ == "__main__":
    main()
