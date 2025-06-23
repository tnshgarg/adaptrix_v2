#!/usr/bin/env python3
"""
Debug script to test adapter loading and injection.
"""

import sys
import os
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.core.engine import AdaptrixEngine
from src.adapters.adapter_manager import AdapterManager
from src.composition.adapter_composer import CompositionStrategy

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_adapter_manager():
    """Test the adapter manager directly."""
    print("🔍 Testing AdapterManager directly...")
    
    adapter_manager = AdapterManager()
    
    # List available adapters
    adapters = adapter_manager.list_adapters()
    print(f"Available adapters: {adapters}")
    
    if not adapters:
        print("❌ No adapters found!")
        return False
    
    # Test loading first adapter
    adapter_name = adapters[0]
    print(f"\n🔍 Testing loading adapter: {adapter_name}")
    
    adapter_data = adapter_manager.load_adapter(adapter_name)
    if adapter_data:
        print("✅ Adapter loaded successfully!")
        print(f"Metadata: {adapter_data['metadata']['name']}")
        print(f"Target layers: {adapter_data['metadata']['target_layers']}")
        print(f"Target modules: {adapter_data['metadata']['target_modules']}")
        print(f"Weights loaded for layers: {list(adapter_data['weights'].keys())}")
        return True
    else:
        print("❌ Failed to load adapter!")
        return False


def test_engine_initialization():
    """Test engine initialization."""
    print("\n🔍 Testing AdaptrixEngine initialization...")
    
    engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
    
    if engine.initialize():
        print("✅ Engine initialized successfully!")
        
        # Test listing adapters through engine
        adapters = engine.list_adapters()
        print(f"Adapters via engine: {adapters}")
        
        return engine
    else:
        print("❌ Engine initialization failed!")
        return None


def test_adapter_loading_via_engine(engine):
    """Test adapter loading through the engine."""
    print("\n🔍 Testing adapter loading via engine...")
    
    adapters = engine.list_adapters()
    if not adapters:
        print("❌ No adapters available!")
        return False
    
    adapter_name = adapters[0]
    print(f"Testing loading: {adapter_name}")
    
    # Test loading
    success = engine.load_adapter(adapter_name)
    print(f"Load result: {success}")
    
    if success:
        print("✅ Adapter loaded via engine!")
        
        # Check if it's actually loaded
        status = engine.get_system_status()
        print(f"Loaded adapters: {status.get('loaded_adapters', [])}")
        
        # Test generation
        print("\n🔍 Testing generation with loaded adapter...")
        response = engine.generate("What is 5 + 3?", max_length=50)
        print(f"Response: {response}")
        
        return True
    else:
        print("❌ Failed to load adapter via engine!")
        return False


def test_composition(engine):
    """Test adapter composition."""
    print("\n🔍 Testing adapter composition...")
    
    adapters = engine.list_adapters()
    if len(adapters) < 1:
        print("❌ Need at least 1 adapter for composition test!")
        return False
    
    # Test with single adapter first
    test_adapters = adapters[:1]
    print(f"Testing composition with: {test_adapters}")
    
    result = engine.compose_adapters(test_adapters, CompositionStrategy.PARALLEL)
    print(f"Composition result: {result}")
    
    if result.get('success'):
        print("✅ Composition successful!")
        
        # Test generation with composition
        print("\n🔍 Testing generation with composition...")
        response = engine.generate_with_composition(
            "Calculate 23 * 92",
            test_adapters,
            CompositionStrategy.PARALLEL,
            max_length=100
        )
        print(f"Composed response: {response}")
        
        return True
    else:
        print(f"❌ Composition failed: {result.get('error')}")
        return False


def main():
    """Main diagnostic function."""
    print("🔍" * 50)
    print("🔍 ADAPTRIX ADAPTER LOADING DIAGNOSTIC 🔍")
    print("🔍" * 50)
    
    # Test 1: Adapter Manager
    if not test_adapter_manager():
        print("\n❌ Adapter Manager test failed - stopping here")
        return
    
    # Test 2: Engine Initialization
    engine = test_engine_initialization()
    if not engine:
        print("\n❌ Engine initialization failed - stopping here")
        return
    
    # Test 3: Adapter Loading via Engine
    if not test_adapter_loading_via_engine(engine):
        print("\n❌ Adapter loading via engine failed - stopping here")
        return
    
    # Test 4: Composition
    if not test_composition(engine):
        print("\n❌ Composition test failed")
        return
    
    print("\n✅" * 20)
    print("✅ ALL TESTS PASSED! ✅")
    print("✅" * 20)
    
    # Cleanup
    engine.cleanup()


if __name__ == "__main__":
    main()
