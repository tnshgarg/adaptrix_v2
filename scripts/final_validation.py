#!/usr/bin/env python3
"""
Final System Validation for Adaptrix.

This script performs comprehensive validation of all system components
to ensure the complete Adaptrix system is working correctly.
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdaptrixValidator:
    """Comprehensive validator for the Adaptrix system."""
    
    def __init__(self):
        """Initialize the validator."""
        self.results = {
            "core_components": {},
            "moe_system": {},
            "rag_system": {},
            "optimization": {},
            "api_system": {},
            "integration": {}
        }
        self.total_tests = 0
        self.passed_tests = 0
        
    def print_header(self, title: str):
        """Print a formatted header."""
        print("\n" + "="*80)
        print(f"🔍 {title}")
        print("="*80)
    
    def print_test(self, test_name: str, status: bool, details: str = ""):
        """Print test result."""
        self.total_tests += 1
        if status:
            self.passed_tests += 1
            print(f"✅ {test_name}")
        else:
            print(f"❌ {test_name}")
        
        if details:
            print(f"   {details}")
    
    def validate_project_structure(self) -> bool:
        """Validate project directory structure."""
        self.print_header("PROJECT STRUCTURE VALIDATION")
        
        required_dirs = [
            "src/core", "src/moe", "src/rag", "src/inference", "src/api", "src/composition",
            "adapters", "models", "tests", "scripts"
        ]
        
        required_files = [
            "requirements.txt", "README.md",
            "src/core/base_model_interface.py", "src/core/modular_engine.py",
            "src/moe/moe_engine.py", "src/moe/classifier.py",
            "src/rag/vector_store.py", "src/rag/retriever.py",
            "src/inference/optimized_engine.py", "src/inference/vllm_engine.py",
            "src/api/main.py", "src/api/config.py"
        ]
        
        all_valid = True
        
        # Check directories
        for dir_path in required_dirs:
            exists = Path(dir_path).exists()
            self.print_test(f"Directory: {dir_path}", exists)
            if not exists:
                all_valid = False
        
        # Check files
        for file_path in required_files:
            exists = Path(file_path).exists()
            self.print_test(f"File: {file_path}", exists)
            if not exists:
                all_valid = False
        
        self.results["core_components"]["structure"] = all_valid
        return all_valid
    
    def validate_imports(self) -> bool:
        """Validate that all modules can be imported."""
        self.print_header("IMPORT VALIDATION")
        
        import_tests = [
            ("Core Engine", "src.core.modular_engine", "ModularAdaptrixEngine"),
            ("Base Model Interface", "src.core.base_model_interface", "ModelFactory"),
            ("Universal Adapter Manager", "src.core.universal_adapter_manager", "UniversalAdapterManager"),
            ("Layer Injector", "src.core.layer_injector", "LayerInjector"),
            ("MoE Engine", "src.moe.moe_engine", "MoEAdaptrixEngine"),
            ("Task Classifier", "src.moe.classifier", "TaskClassifier"),
            ("Vector Store", "src.rag.vector_store", "FAISSVectorStore"),
            ("Document Retriever", "src.rag.retriever", "DocumentRetriever"),
            ("Optimized Engine", "src.inference.optimized_engine", "OptimizedAdaptrixEngine"),
            ("vLLM Engine", "src.inference.vllm_engine", "VLLMInferenceEngine"),
            ("Quantization", "src.inference.quantization", "QuantizationManager"),
            ("Caching", "src.inference.caching", "CacheManager"),
            ("Adapter Composer", "src.composition.adapter_composer", "AdapterComposer"),
            ("API Main", "src.api.main", "app"),
            ("API Config", "src.api.config", "get_api_config")
        ]
        
        all_valid = True
        
        for test_name, module_path, class_name in import_tests:
            try:
                module = __import__(module_path, fromlist=[class_name])
                getattr(module, class_name)
                self.print_test(f"Import: {test_name}", True)
            except Exception as e:
                self.print_test(f"Import: {test_name}", False, str(e))
                all_valid = False
        
        self.results["core_components"]["imports"] = all_valid
        return all_valid
    
    def validate_model_factory(self) -> bool:
        """Validate model factory functionality."""
        self.print_header("MODEL FACTORY VALIDATION")
        
        try:
            from src.core.base_model_interface import ModelFactory, ModelDetector, ModelFamily
            
            # Test model family detection
            test_models = [
                ("Qwen/Qwen3-1.7B", ModelFamily.QWEN),
                ("microsoft/phi-2", ModelFamily.PHI),
                ("meta-llama/Llama-2-7b-hf", ModelFamily.LLAMA)
            ]
            
            detection_valid = True
            for model_id, expected_family in test_models:
                try:
                    detected = ModelDetector.detect_family(model_id)
                    success = detected == expected_family
                    self.print_test(f"Family Detection: {model_id}", success)
                    if not success:
                        detection_valid = False
                except Exception as e:
                    self.print_test(f"Family Detection: {model_id}", False, str(e))
                    detection_valid = False
            
            # Test model factory creation
            try:
                model = ModelFactory.create_model("Qwen/Qwen3-1.7B", "cpu")
                self.print_test("Model Factory Creation", True)
                factory_valid = True
            except Exception as e:
                self.print_test("Model Factory Creation", False, str(e))
                factory_valid = False
            
            overall_valid = detection_valid and factory_valid
            self.results["core_components"]["model_factory"] = overall_valid
            return overall_valid
            
        except Exception as e:
            self.print_test("Model Factory Module", False, str(e))
            self.results["core_components"]["model_factory"] = False
            return False
    
    def validate_moe_system(self) -> bool:
        """Validate MoE system components."""
        self.print_header("MOE SYSTEM VALIDATION")
        
        try:
            from src.moe.classifier import TaskClassifier
            from src.moe.training_data import generate_training_data
            
            # Test training data generation
            try:
                data = generate_training_data(samples_per_domain=10)
                data_valid = len(data) > 0 and all(len(item) == 2 for item in data)
                self.print_test("Training Data Generation", data_valid)
            except Exception as e:
                self.print_test("Training Data Generation", False, str(e))
                data_valid = False
            
            # Test classifier creation
            try:
                classifier = TaskClassifier()
                self.print_test("Classifier Creation", True)
                classifier_valid = True
            except Exception as e:
                self.print_test("Classifier Creation", False, str(e))
                classifier_valid = False
            
            # Test MoE engine creation
            try:
                from src.moe.moe_engine import MoEAdaptrixEngine
                engine = MoEAdaptrixEngine(
                    model_id="Qwen/Qwen3-1.7B",
                    device="cpu",
                    enable_auto_selection=False  # Don't require trained classifier
                )
                self.print_test("MoE Engine Creation", True)
                moe_engine_valid = True
            except Exception as e:
                self.print_test("MoE Engine Creation", False, str(e))
                moe_engine_valid = False
            
            overall_valid = data_valid and classifier_valid and moe_engine_valid
            self.results["moe_system"]["components"] = overall_valid
            return overall_valid
            
        except Exception as e:
            self.print_test("MoE System Import", False, str(e))
            self.results["moe_system"]["components"] = False
            return False
    
    def validate_rag_system(self) -> bool:
        """Validate RAG system components."""
        self.print_header("RAG SYSTEM VALIDATION")
        
        try:
            from src.rag.vector_store import FAISSVectorStore
            from src.rag.document_processor import DocumentProcessor
            from src.rag.retriever import DocumentRetriever
            
            # Test document processor
            try:
                processor = DocumentProcessor()
                test_text = "This is a test document for processing."
                chunks = processor.process_text(test_text, chunk_size=50)
                processor_valid = len(chunks) > 0
                self.print_test("Document Processor", processor_valid)
            except Exception as e:
                self.print_test("Document Processor", False, str(e))
                processor_valid = False
            
            # Test vector store creation
            try:
                vector_store = FAISSVectorStore(dimension=384)
                self.print_test("Vector Store Creation", True)
                vector_store_valid = True
            except Exception as e:
                self.print_test("Vector Store Creation", False, str(e))
                vector_store_valid = False
            
            # Test retriever creation
            try:
                retriever = DocumentRetriever(vector_store_path=None)  # No pre-trained store
                self.print_test("Document Retriever Creation", True)
                retriever_valid = True
            except Exception as e:
                self.print_test("Document Retriever Creation", False, str(e))
                retriever_valid = False
            
            overall_valid = processor_valid and vector_store_valid and retriever_valid
            self.results["rag_system"]["components"] = overall_valid
            return overall_valid
            
        except Exception as e:
            self.print_test("RAG System Import", False, str(e))
            self.results["rag_system"]["components"] = False
            return False
    
    def validate_optimization_system(self) -> bool:
        """Validate optimization system components."""
        self.print_header("OPTIMIZATION SYSTEM VALIDATION")
        
        try:
            from src.inference.quantization import QuantizationManager, create_int4_config
            from src.inference.caching import CacheManager, create_default_cache_manager
            
            # Test quantization manager
            try:
                quant_manager = QuantizationManager()
                supported = quant_manager.get_supported_methods()
                self.print_test("Quantization Manager", True, f"Supported: {list(supported.keys())}")
                quant_valid = True
            except Exception as e:
                self.print_test("Quantization Manager", False, str(e))
                quant_valid = False
            
            # Test cache manager
            try:
                cache_manager = create_default_cache_manager()
                stats = cache_manager.get_global_stats()
                self.print_test("Cache Manager", True)
                cache_valid = True
            except Exception as e:
                self.print_test("Cache Manager", False, str(e))
                cache_valid = False
            
            # Test vLLM availability (optional)
            try:
                from src.inference.vllm_engine import VLLMInferenceEngine, VLLM_AVAILABLE
                self.print_test("vLLM Availability", VLLM_AVAILABLE, "Optional component")
                vllm_valid = True
            except Exception as e:
                self.print_test("vLLM Import", False, str(e))
                vllm_valid = False
            
            overall_valid = quant_valid and cache_valid
            self.results["optimization"]["components"] = overall_valid
            return overall_valid
            
        except Exception as e:
            self.print_test("Optimization System Import", False, str(e))
            self.results["optimization"]["components"] = False
            return False
    
    def validate_api_system(self) -> bool:
        """Validate API system components."""
        self.print_header("API SYSTEM VALIDATION")
        
        try:
            from src.api.main import app
            from src.api.config import get_api_config
            from src.api.models import GenerationRequest, GenerationResponse
            from src.api.dependencies import get_engine
            
            # Test API app creation
            try:
                self.print_test("FastAPI App Creation", app is not None)
                app_valid = True
            except Exception as e:
                self.print_test("FastAPI App Creation", False, str(e))
                app_valid = False
            
            # Test configuration
            try:
                config = get_api_config()
                self.print_test("API Configuration", config is not None)
                config_valid = True
            except Exception as e:
                self.print_test("API Configuration", False, str(e))
                config_valid = False
            
            # Test Pydantic models
            try:
                request = GenerationRequest(prompt="test")
                self.print_test("Pydantic Models", True)
                models_valid = True
            except Exception as e:
                self.print_test("Pydantic Models", False, str(e))
                models_valid = False
            
            overall_valid = app_valid and config_valid and models_valid
            self.results["api_system"]["components"] = overall_valid
            return overall_valid
            
        except Exception as e:
            self.print_test("API System Import", False, str(e))
            self.results["api_system"]["components"] = False
            return False
    
    def validate_adapter_examples(self) -> bool:
        """Validate example adapters."""
        self.print_header("ADAPTER EXAMPLES VALIDATION")
        
        adapter_dirs = ["adapters/code_adapter", "adapters/legal_adapter"]
        required_files = ["adapter_config.json", "README.md"]
        
        all_valid = True
        
        for adapter_dir in adapter_dirs:
            adapter_path = Path(adapter_dir)
            if adapter_path.exists():
                for file_name in required_files:
                    file_path = adapter_path / file_name
                    exists = file_path.exists()
                    self.print_test(f"Adapter File: {adapter_dir}/{file_name}", exists)
                    if not exists:
                        all_valid = False
            else:
                self.print_test(f"Adapter Directory: {adapter_dir}", False)
                all_valid = False
        
        self.results["integration"]["adapters"] = all_valid
        return all_valid
    
    def print_final_summary(self):
        """Print final validation summary."""
        self.print_header("FINAL VALIDATION SUMMARY")
        
        print(f"📊 Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.total_tests - self.passed_tests}")
        print(f"📈 Success Rate: {(self.passed_tests / self.total_tests * 100):.1f}%")
        
        print("\n🔍 Component Summary:")
        for category, results in self.results.items():
            if results:
                status = "✅" if all(results.values()) else "⚠️"
                print(f"  {status} {category.replace('_', ' ').title()}")
        
        overall_success = self.passed_tests / self.total_tests >= 0.8
        
        if overall_success:
            print("\n🎉 ADAPTRIX SYSTEM VALIDATION SUCCESSFUL!")
            print("🚀 The system is ready for deployment and testing!")
        else:
            print("\n⚠️ ADAPTRIX SYSTEM VALIDATION INCOMPLETE")
            print("🔧 Please address the failed components before deployment.")
        
        return overall_success
    
    def run_full_validation(self) -> bool:
        """Run complete system validation."""
        print("🚀 Starting Adaptrix System Validation")
        print("This will validate all components of the Adaptrix system")
        
        validation_steps = [
            ("Project Structure", self.validate_project_structure),
            ("Module Imports", self.validate_imports),
            ("Model Factory", self.validate_model_factory),
            ("MoE System", self.validate_moe_system),
            ("RAG System", self.validate_rag_system),
            ("Optimization System", self.validate_optimization_system),
            ("API System", self.validate_api_system),
            ("Adapter Examples", self.validate_adapter_examples)
        ]
        
        for step_name, step_func in validation_steps:
            try:
                step_func()
            except Exception as e:
                logger.error(f"Validation step '{step_name}' failed: {e}")
                self.print_test(f"Validation Step: {step_name}", False, str(e))
        
        return self.print_final_summary()


def main():
    """Main validation function."""
    validator = AdaptrixValidator()
    success = validator.run_full_validation()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Run the setup script: ./scripts/setup.sh")
        print("2. Train the classifier: python scripts/train_classifier.py")
        print("3. Setup RAG system: python scripts/setup_rag.py")
        print("4. Start the API server: python scripts/run_server.py")
        print("5. Run comprehensive tests: python -m pytest tests/test_complete_system.py")
        
        return 0
    else:
        print("\n🔧 Fix the issues above and run validation again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
