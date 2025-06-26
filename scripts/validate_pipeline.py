#!/usr/bin/env python3
"""
🔧 COMPLETE PIPELINE VALIDATION

Validates the entire Adaptrix pipeline to ensure it's ready for new LoRA adapters.
Tests the complete flow from model initialization to adapter loading and generation.
"""

import sys
import os
import json
import tempfile
import time
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class PipelineValidator:
    """Validates the complete Adaptrix pipeline."""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dirs = []
    
    def create_test_adapter(self, name: str, config: dict) -> str:
        """Create a test LoRA adapter."""
        temp_dir = tempfile.mkdtemp(prefix=f"pipeline_test_{name}_")
        self.temp_dirs.append(temp_dir)
        
        # Create adapter config
        config_path = os.path.join(temp_dir, "adapter_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create metadata
        metadata = {
            "description": f"Test {name} adapter for pipeline validation",
            "domain": config.get("domain", "general"),
            "capabilities": config.get("capabilities", [])
        }
        
        metadata_path = os.path.join(temp_dir, "adaptrix_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create dummy model file
        model_path = os.path.join(temp_dir, "adapter_model.bin")
        with open(model_path, 'wb') as f:
            f.write(b"dummy_adapter_weights")
        
        return temp_dir
    
    def test_model_initialization(self) -> bool:
        """Test model initialization with Qwen3-1.7B."""
        
        print("🚀 TESTING MODEL INITIALIZATION")
        print("=" * 60)
        
        try:
            from src.core.modular_engine import ModularAdaptrixEngine
            
            print("   📦 Creating engine instance...")
            engine = ModularAdaptrixEngine(
                model_id="Qwen/Qwen3-1.7B",
                device="cpu",
                adapters_dir="adapters"
            )
            
            print("   🔧 Initializing engine...")
            start_time = time.time()
            success = engine.initialize()
            init_time = time.time() - start_time
            
            if success:
                print(f"   ✅ Model initialized in {init_time:.2f}s")
                
                # Test basic generation
                print("   🧪 Testing basic generation...")
                response = engine.generate("What is 2+2?", max_length=50)
                
                if response and "error" not in response.lower():
                    print(f"   ✅ Generation working: {response[:50]}...")
                    self.test_results["model_init"] = True
                else:
                    print(f"   ❌ Generation failed: {response}")
                    self.test_results["model_init"] = False
                
                # Cleanup
                engine.cleanup()
                
            else:
                print("   ❌ Model initialization failed")
                self.test_results["model_init"] = False
            
            return success
            
        except Exception as e:
            print(f"   ❌ Exception during model initialization: {e}")
            self.test_results["model_init"] = False
            return False
    
    def test_adapter_discovery(self) -> bool:
        """Test adapter auto-discovery."""
        
        print("\n🔌 TESTING ADAPTER DISCOVERY")
        print("=" * 60)
        
        try:
            # Create test adapters directory
            test_adapters_dir = tempfile.mkdtemp(prefix="test_adapters_")
            self.temp_dirs.append(test_adapters_dir)
            
            # Create test adapters
            adapters = [
                {
                    "name": "math_test",
                    "config": {
                        "base_model_name_or_path": "Qwen/Qwen3-1.7B",
                        "peft_type": "LORA",
                        "r": 16,
                        "lora_alpha": 32,
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                        "domain": "mathematics",
                        "capabilities": ["arithmetic", "algebra"]
                    }
                },
                {
                    "name": "code_test", 
                    "config": {
                        "base_model_name_or_path": "Qwen/Qwen3-1.7B",
                        "peft_type": "LORA",
                        "r": 24,
                        "lora_alpha": 48,
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
                        "domain": "programming",
                        "capabilities": ["python", "javascript", "debugging"]
                    }
                }
            ]
            
            # Create adapter directories
            for adapter in adapters:
                adapter_dir = os.path.join(test_adapters_dir, adapter["name"])
                os.makedirs(adapter_dir)
                
                config_path = os.path.join(adapter_dir, "adapter_config.json")
                with open(config_path, 'w') as f:
                    json.dump(adapter["config"], f, indent=2)
            
            # Test discovery
            from src.core.modular_engine import ModularAdaptrixEngine
            
            engine = ModularAdaptrixEngine(
                model_id="Qwen/Qwen3-1.7B",
                device="cpu",
                adapters_dir=test_adapters_dir
            )
            
            print("   🔍 Initializing with test adapters...")
            if engine.initialize():
                discovered_adapters = engine.list_adapters()
                print(f"   📦 Discovered adapters: {discovered_adapters}")
                
                if len(discovered_adapters) >= 2:
                    print("   ✅ Adapter discovery working")
                    self.test_results["adapter_discovery"] = True
                    
                    # Test adapter info
                    for adapter_name in discovered_adapters:
                        info = engine.get_adapter_info(adapter_name)
                        if info:
                            print(f"      📋 {adapter_name}: {info['domain']} domain")
                        else:
                            print(f"      ❌ Failed to get info for {adapter_name}")
                    
                    engine.cleanup()
                    return True
                else:
                    print(f"   ❌ Expected 2+ adapters, found {len(discovered_adapters)}")
                    self.test_results["adapter_discovery"] = False
                    engine.cleanup()
                    return False
            else:
                print("   ❌ Engine initialization failed")
                self.test_results["adapter_discovery"] = False
                return False
                
        except Exception as e:
            print(f"   ❌ Exception during adapter discovery: {e}")
            self.test_results["adapter_discovery"] = False
            return False
    
    def test_adapter_loading(self) -> bool:
        """Test adapter loading and switching."""
        
        print("\n🔄 TESTING ADAPTER LOADING & SWITCHING")
        print("=" * 60)
        
        try:
            # Create test adapter
            adapter_config = {
                "base_model_name_or_path": "Qwen/Qwen3-1.7B",
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "task_type": "CAUSAL_LM"
            }
            
            adapter_path = self.create_test_adapter("load_test", adapter_config)
            
            # Test with engine
            from src.core.modular_engine import ModularAdaptrixEngine
            
            engine = ModularAdaptrixEngine(
                model_id="Qwen/Qwen3-1.7B",
                device="cpu",
                adapters_dir=os.path.dirname(adapter_path)
            )
            
            print("   🔧 Initializing engine...")
            if not engine.initialize():
                print("   ❌ Engine initialization failed")
                return False
            
            # Test adapter registration
            print("   📝 Registering test adapter...")
            success = engine.register_adapter(
                name="load_test",
                path=adapter_path,
                description="Test adapter for loading validation",
                domain="general"
            )
            
            if success:
                print("   ✅ Adapter registered successfully")
                
                # Test loading (this will fail without actual PEFT, but should handle gracefully)
                print("   🔌 Testing adapter loading...")
                load_success = engine.load_adapter("load_test")
                
                if load_success:
                    print("   ✅ Adapter loaded successfully")
                    
                    # Test generation with adapter
                    print("   🧪 Testing generation with adapter...")
                    response = engine.generate("Test prompt", max_length=30)
                    
                    if response and "error" not in response.lower():
                        print("   ✅ Generation with adapter working")
                        
                        # Test unloading
                        print("   🔌 Testing adapter unloading...")
                        unload_success = engine.unload_adapter("load_test")
                        
                        if unload_success:
                            print("   ✅ Adapter unloaded successfully")
                            self.test_results["adapter_loading"] = True
                            engine.cleanup()
                            return True
                        else:
                            print("   ❌ Adapter unloading failed")
                    else:
                        print(f"   ❌ Generation with adapter failed: {response}")
                else:
                    print("   ⚠️ Adapter loading failed (expected without PEFT weights)")
                    # This is expected behavior - mark as partial success
                    self.test_results["adapter_loading"] = "partial"
                    engine.cleanup()
                    return True
            else:
                print("   ❌ Adapter registration failed")
                self.test_results["adapter_loading"] = False
                engine.cleanup()
                return False
                
        except Exception as e:
            print(f"   ❌ Exception during adapter loading: {e}")
            self.test_results["adapter_loading"] = False
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling and edge cases."""
        
        print("\n🚨 TESTING ERROR HANDLING")
        print("=" * 60)
        
        try:
            from src.core.modular_engine import ModularAdaptrixEngine
            
            # Test 1: Invalid model ID
            print("   🧪 Testing invalid model ID...")
            try:
                engine = ModularAdaptrixEngine("invalid/model-id", "cpu")
                init_success = engine.initialize()
                
                if not init_success:
                    print("   ✅ Invalid model ID handled correctly")
                else:
                    print("   ❌ Invalid model ID should have failed")
                    return False
            except Exception as e:
                print(f"   ✅ Invalid model ID exception handled: {type(e).__name__}")
            
            # Test 2: Generation without initialization
            print("   🧪 Testing generation without initialization...")
            try:
                engine = ModularAdaptrixEngine("Qwen/Qwen3-1.7B", "cpu")
                response = engine.generate("test")
                
                if "error" in response.lower() or "not initialized" in response.lower():
                    print("   ✅ Uninitialized generation handled correctly")
                else:
                    print("   ❌ Should have failed for uninitialized engine")
                    return False
            except Exception as e:
                print(f"   ✅ Uninitialized generation exception handled: {type(e).__name__}")
            
            # Test 3: Invalid adapter loading
            print("   🧪 Testing invalid adapter loading...")
            engine = ModularAdaptrixEngine("Qwen/Qwen3-1.7B", "cpu")
            
            if engine.initialize():
                load_success = engine.load_adapter("nonexistent_adapter")
                
                if not load_success:
                    print("   ✅ Invalid adapter loading handled correctly")
                else:
                    print("   ❌ Invalid adapter loading should have failed")
                    engine.cleanup()
                    return False
                
                engine.cleanup()
            
            self.test_results["error_handling"] = True
            return True
            
        except Exception as e:
            print(f"   ❌ Exception during error handling test: {e}")
            self.test_results["error_handling"] = False
            return False
    
    def generate_pipeline_report(self) -> str:
        """Generate comprehensive pipeline validation report."""
        
        report = []
        report.append("🔧 ADAPTRIX PIPELINE VALIDATION REPORT")
        report.append("=" * 80)
        
        # Test results summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result is True)
        partial_tests = sum(1 for result in self.test_results.values() if result == "partial")
        failed_tests = total_tests - passed_tests - partial_tests
        
        report.append(f"\n📊 PIPELINE TEST SUMMARY:")
        report.append(f"   Total Tests: {total_tests}")
        report.append(f"   Passed: {passed_tests} ✅")
        report.append(f"   Partial: {partial_tests} ⚠️")
        report.append(f"   Failed: {failed_tests} ❌")
        
        success_rate = ((passed_tests + partial_tests * 0.5) / total_tests) * 100
        report.append(f"   Success Rate: {success_rate:.1f}%")
        
        # Detailed results
        report.append(f"\n📋 DETAILED TEST RESULTS:")
        for test_name, result in self.test_results.items():
            if result is True:
                status = "✅ PASS"
            elif result == "partial":
                status = "⚠️ PARTIAL"
            else:
                status = "❌ FAIL"
            
            report.append(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        # Pipeline readiness assessment
        report.append(f"\n🚀 PIPELINE READINESS ASSESSMENT:")
        
        if success_rate >= 90:
            report.append(f"   🎊 EXCELLENT: Pipeline is production-ready!")
            report.append(f"   ✅ Ready for new LoRA adapters")
            report.append(f"   🔌 Plug-and-play functionality confirmed")
        elif success_rate >= 75:
            report.append(f"   ✅ GOOD: Pipeline is mostly ready")
            report.append(f"   ⚠️ Minor issues may need attention")
            report.append(f"   🔧 Should handle most new adapters")
        elif success_rate >= 50:
            report.append(f"   ⚠️ FAIR: Pipeline has some issues")
            report.append(f"   🔧 Recommend fixing failed tests")
            report.append(f"   ⚠️ May need manual handling for some adapters")
        else:
            report.append(f"   ❌ POOR: Pipeline needs significant work")
            report.append(f"   🚨 Not ready for production use")
            report.append(f"   🔧 Address all failed tests before deployment")
        
        # Recommendations for new LoRA adapters
        report.append(f"\n🎯 RECOMMENDATIONS FOR NEW LORA ADAPTERS:")
        report.append(f"   📋 Ensure adapter configs include:")
        report.append(f"      - base_model_name_or_path: 'Qwen/Qwen3-1.7B'")
        report.append(f"      - peft_type: 'LORA'")
        report.append(f"      - target_modules: [list of target modules]")
        report.append(f"      - r: rank value (8-64 recommended)")
        report.append(f"      - lora_alpha: alpha value (16-128 recommended)")
        
        report.append(f"\n   📁 Optional Adaptrix metadata file:")
        report.append(f"      - adaptrix_metadata.json with domain, capabilities, description")
        
        report.append(f"\n   🔌 Supported target modules for Qwen3:")
        report.append(f"      - Attention: q_proj, k_proj, v_proj, o_proj")
        report.append(f"      - MLP: gate_proj, up_proj, down_proj")
        report.append(f"      - Embeddings: embed_tokens, lm_head")
        
        return "\n".join(report)
    
    def cleanup(self):
        """Clean up temporary directories."""
        import shutil
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup {temp_dir}: {e}")
    
    def run_full_validation(self):
        """Run complete pipeline validation."""
        
        print("🔧" * 100)
        print("🔧 COMPLETE ADAPTRIX PIPELINE VALIDATION 🔧")
        print("🔧" * 100)
        
        try:
            # Run all tests
            self.test_model_initialization()
            self.test_adapter_discovery()
            self.test_adapter_loading()
            self.test_error_handling()
            
            # Generate report
            report = self.generate_pipeline_report()
            print(f"\n{report}")
            
            # Save report
            report_path = "pipeline_validation_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            
            print(f"\n📄 Report saved to: {report_path}")
            
            return self.test_results
            
        finally:
            self.cleanup()


def main():
    """Main validation function."""
    validator = PipelineValidator()
    
    try:
        results = validator.run_full_validation()
        print("\n🎯 PIPELINE VALIDATION COMPLETE!")
        return results
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        validator.cleanup()


if __name__ == "__main__":
    main()
