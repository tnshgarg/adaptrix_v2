#!/usr/bin/env python3
"""
Complete Integration Test for Adaptrix System.

This script validates that all components work together:
- Qwen-3 1.7B base model
- Universal adapter management
- MoE task classification
- RAG document retrieval
- Layer injection system
- Adapter composition
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_core_components():
    """Test core component imports and basic functionality."""
    
    print("🔧" * 80)
    print("🔧 TESTING CORE COMPONENT INTEGRATION")
    print("🔧" * 80)
    
    try:
        # Test 1: Core imports
        print("\n📦 Test 1: Core Component Imports")
        print("-" * 50)
        
        from src.core.base_model_interface import ModelFactory, ModelDetector, ModelFamily
        from src.core.modular_engine import ModularAdaptrixEngine
        from src.core.universal_adapter_manager import UniversalAdapterManager
        from src.core.layer_injector import LayerInjector
        from src.composition.adapter_composer import AdapterComposer
        
        print("✅ All core components imported successfully")
        
        # Test 2: Model family detection
        print("\n🔍 Test 2: Model Family Detection")
        print("-" * 50)
        
        test_models = [
            ("Qwen/Qwen3-1.7B", ModelFamily.QWEN),
            ("microsoft/phi-2", ModelFamily.PHI),
            ("meta-llama/Llama-2-7b-hf", ModelFamily.LLAMA)
        ]
        
        for model_id, expected_family in test_models:
            detected = ModelDetector.detect_family(model_id)
            status = "✅" if detected == expected_family else "❌"
            print(f"   {status} {model_id} -> {detected.value}")
        
        # Test 3: Model factory
        print("\n🏭 Test 3: Model Factory")
        print("-" * 50)
        
        qwen_model = ModelFactory.create_model("Qwen/Qwen3-1.7B", "cpu")
        print(f"✅ Created Qwen model: {type(qwen_model).__name__}")
        
        # Test 4: Engine creation
        print("\n🚀 Test 4: Engine Creation")
        print("-" * 50)
        
        engine = ModularAdaptrixEngine(
            model_id="Qwen/Qwen3-1.7B",
            device="cpu",
            adapters_dir="adapters"
        )
        print("✅ Modular engine created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Core component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_moe_rag_integration():
    """Test MoE and RAG integration."""
    
    print("\n🧠" * 80)
    print("🧠 TESTING MOE-RAG INTEGRATION")
    print("🧠" * 80)
    
    try:
        # Test 1: MoE imports
        print("\n📦 Test 1: MoE Component Imports")
        print("-" * 50)
        
        from src.moe.moe_engine import MoEAdaptrixEngine
        from src.moe.classifier import TaskClassifier
        from src.rag.vector_store import FAISSVectorStore
        from src.rag.retriever import DocumentRetriever
        
        print("✅ All MoE-RAG components imported successfully")
        
        # Test 2: Check if trained models exist
        print("\n🔍 Test 2: Trained Model Availability")
        print("-" * 50)
        
        classifier_path = Path("models/classifier")
        vector_store_path = Path("models/rag_vector_store")
        
        classifier_exists = classifier_path.exists()
        vector_store_exists = vector_store_path.exists()
        
        print(f"   {'✅' if classifier_exists else '❌'} Task classifier: {classifier_path}")
        print(f"   {'✅' if vector_store_exists else '❌'} Vector store: {vector_store_path}")
        
        if not (classifier_exists and vector_store_exists):
            print("⚠️ Some trained models missing - will test without them")
        
        # Test 3: MoE engine creation
        print("\n🚀 Test 3: MoE Engine Creation")
        print("-" * 50)
        
        moe_engine = MoEAdaptrixEngine(
            model_id="Qwen/Qwen3-1.7B",
            device="cpu",
            adapters_dir="adapters",
            classifier_path="models/classifier" if classifier_exists else None,
            enable_auto_selection=classifier_exists,
            rag_vector_store_path="models/rag_vector_store" if vector_store_exists else None,
            enable_rag=vector_store_exists
        )
        print("✅ MoE engine created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ MoE-RAG integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adapter_system():
    """Test adapter management and composition system."""
    
    print("\n🔌" * 80)
    print("🔌 TESTING ADAPTER SYSTEM")
    print("🔌" * 80)
    
    try:
        # Test 1: Check adapter directory
        print("\n📁 Test 1: Adapter Directory Structure")
        print("-" * 50)
        
        adapters_dir = Path("adapters")
        if not adapters_dir.exists():
            adapters_dir.mkdir(exist_ok=True)
            print("✅ Created adapters directory")
        else:
            print("✅ Adapters directory exists")
        
        # List any existing adapters
        adapter_files = list(adapters_dir.glob("*"))
        print(f"   📊 Found {len(adapter_files)} items in adapters directory")
        
        # Test 2: Adapter manager creation
        print("\n🔧 Test 2: Adapter Manager Creation")
        print("-" * 50)
        
        from src.core.base_model_interface import ModelFactory
        from src.core.universal_adapter_manager import UniversalAdapterManager
        
        # Create a dummy model for testing
        dummy_model = ModelFactory.create_model("Qwen/Qwen3-1.7B", "cpu")
        adapter_manager = UniversalAdapterManager(dummy_model)
        
        print("✅ Adapter manager created successfully")
        
        # Test 3: Layer injector
        print("\n💉 Test 3: Layer Injector")
        print("-" * 50)
        
        from src.core.layer_injector import LayerInjector
        
        # Note: We can't fully test without initializing the model
        print("✅ Layer injector imported successfully")
        
        # Test 4: Adapter composer
        print("\n🎼 Test 4: Adapter Composer")
        print("-" * 50)
        
        from src.composition.adapter_composer import AdapterComposer, CompositionStrategy
        
        print("✅ Adapter composer imported successfully")
        print(f"   📊 Available strategies: {[s.value for s in CompositionStrategy]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Adapter system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_system_initialization():
    """Test full system initialization with all components."""
    
    print("\n🌟" * 80)
    print("🌟 TESTING FULL SYSTEM INITIALIZATION")
    print("🌟" * 80)
    
    try:
        # Test 1: Basic engine initialization
        print("\n🚀 Test 1: Basic Engine Initialization")
        print("-" * 50)
        
        from src.core.modular_engine import ModularAdaptrixEngine
        
        engine = ModularAdaptrixEngine(
            model_id="Qwen/Qwen3-1.7B",
            device="cpu",
            adapters_dir="adapters"
        )
        
        start_time = time.time()
        success = engine.initialize()
        init_time = time.time() - start_time
        
        if success:
            print(f"✅ Basic engine initialized in {init_time:.2f}s")
            
            # Get system status
            status = engine.get_system_status()
            print(f"   📊 Model: {status.get('model_info', {}).get('model_id', 'Unknown')}")
            print(f"   📊 Parameters: {status.get('model_info', {}).get('total_parameters', 'Unknown'):,}")
            print(f"   📊 Adapters: {len(status.get('adapters', []))}")
            
            engine.cleanup()
        else:
            print("❌ Basic engine initialization failed")
            return False
        
        # Test 2: MoE engine initialization (if components available)
        print("\n🧠 Test 2: MoE Engine Initialization")
        print("-" * 50)
        
        classifier_exists = Path("models/classifier").exists()
        vector_store_exists = Path("models/rag_vector_store").exists()
        
        if classifier_exists or vector_store_exists:
            from src.moe.moe_engine import MoEAdaptrixEngine
            
            moe_engine = MoEAdaptrixEngine(
                model_id="Qwen/Qwen3-1.7B",
                device="cpu",
                adapters_dir="adapters",
                classifier_path="models/classifier" if classifier_exists else None,
                enable_auto_selection=classifier_exists,
                rag_vector_store_path="models/rag_vector_store" if vector_store_exists else None,
                enable_rag=vector_store_exists
            )
            
            start_time = time.time()
            success = moe_engine.initialize()
            init_time = time.time() - start_time
            
            if success:
                print(f"✅ MoE engine initialized in {init_time:.2f}s")
                
                # Get MoE status
                moe_status = moe_engine.get_moe_status()
                moe_info = moe_status.get('moe', {})
                
                print(f"   🧠 Classifier: {moe_info.get('classifier_initialized', False)}")
                print(f"   🔍 RAG: {moe_info.get('rag_initialized', False)}")
                print(f"   📊 Vector Store Docs: {moe_info.get('vector_store_stats', {}).get('num_documents', 0)}")
                
                moe_engine.cleanup()
            else:
                print("❌ MoE engine initialization failed")
                return False
        else:
            print("⚠️ Skipping MoE engine test - trained models not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Full system initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive integration tests."""
    
    print("🚀" * 100)
    print("🚀 ADAPTRIX COMPLETE INTEGRATION TEST SUITE")
    print("🚀" * 100)
    
    tests = [
        ("Core Components", test_core_components),
        ("MoE-RAG Integration", test_moe_rag_integration),
        ("Adapter System", test_adapter_system),
        ("Full System Initialization", test_full_system_initialization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊" * 50)
    print("📊 INTEGRATION TEST SUMMARY")
    print("📊" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("🎉 SYSTEM IS READY FOR NEXT PHASES!")
        return True
    else:
        print("\n❌ SOME INTEGRATION TESTS FAILED.")
        print("❌ PLEASE FIX ISSUES BEFORE PROCEEDING.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
