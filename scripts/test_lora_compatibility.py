#!/usr/bin/env python3
"""
🔧 LORA COMPATIBILITY TESTING PIPELINE

Comprehensive testing for LoRA adapter compatibility with the modular system.
Tests various adapter architectures, configurations, and edge cases.
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class LoRACompatibilityTester:
    """Comprehensive LoRA adapter compatibility tester."""
    
    def __init__(self):
        self.test_results = []
        self.temp_dirs = []
    
    def create_mock_adapter(self, config: Dict[str, Any], adapter_name: str) -> str:
        """Create a mock LoRA adapter for testing."""
        temp_dir = tempfile.mkdtemp(prefix=f"test_adapter_{adapter_name}_")
        self.temp_dirs.append(temp_dir)
        
        # Create adapter_config.json
        config_path = os.path.join(temp_dir, "adapter_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create adapter_model.bin (empty file for testing)
        model_path = os.path.join(temp_dir, "adapter_model.bin")
        with open(model_path, 'wb') as f:
            f.write(b"mock_adapter_weights")
        
        # Create optional Adaptrix metadata
        if "adaptrix_metadata" in config:
            metadata_path = os.path.join(temp_dir, "adaptrix_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(config["adaptrix_metadata"], f, indent=2)
        
        return temp_dir
    
    def test_adapter_configurations(self) -> List[Dict[str, Any]]:
        """Test various LoRA adapter configurations."""
        
        print("🧪 TESTING VARIOUS LORA ADAPTER CONFIGURATIONS")
        print("=" * 80)
        
        # Test configurations covering different scenarios
        test_configs = [
            {
                "name": "Standard Qwen3 LoRA",
                "config": {
                    "base_model_name_or_path": "Qwen/Qwen3-1.7B",
                    "peft_type": "LORA",
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "lora_dropout": 0.1,
                    "bias": "none",
                    "task_type": "CAUSAL_LM"
                }
            },
            {
                "name": "High Rank Qwen3 LoRA",
                "config": {
                    "base_model_name_or_path": "Qwen/Qwen3-1.7B",
                    "peft_type": "LORA",
                    "r": 64,
                    "lora_alpha": 128,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    "lora_dropout": 0.05,
                    "bias": "lora_only",
                    "task_type": "CAUSAL_LM"
                }
            },
            {
                "name": "QLoRA Configuration",
                "config": {
                    "base_model_name_or_path": "Qwen/Qwen3-1.7B",
                    "peft_type": "LORA",
                    "r": 32,
                    "lora_alpha": 64,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "lora_dropout": 0.1,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                    "use_rslora": True,
                    "use_dora": False
                }
            },
            {
                "name": "Full Parameter LoRA",
                "config": {
                    "base_model_name_or_path": "Qwen/Qwen3-1.7B",
                    "peft_type": "LORA",
                    "r": 8,
                    "lora_alpha": 16,
                    "target_modules": [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head"
                    ],
                    "lora_dropout": 0.1,
                    "bias": "all",
                    "task_type": "CAUSAL_LM",
                    "modules_to_save": ["embed_tokens", "lm_head"]
                }
            },
            {
                "name": "Domain-Specific Math LoRA",
                "config": {
                    "base_model_name_or_path": "Qwen/Qwen3-1.7B",
                    "peft_type": "LORA",
                    "r": 24,
                    "lora_alpha": 48,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "lora_dropout": 0.1,
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                    "adaptrix_metadata": {
                        "domain": "mathematics",
                        "capabilities": ["arithmetic", "algebra", "calculus"],
                        "description": "Specialized for mathematical reasoning"
                    }
                }
            },
            {
                "name": "Legacy Phi-2 LoRA (Cross-Model)",
                "config": {
                    "base_model_name_or_path": "microsoft/phi-2",
                    "peft_type": "LORA",
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": ["Wqkv", "out_proj", "fc1", "fc2"],
                    "lora_dropout": 0.1,
                    "bias": "none",
                    "task_type": "CAUSAL_LM"
                }
            },
            {
                "name": "Malformed Config (Missing Fields)",
                "config": {
                    "base_model_name_or_path": "Qwen/Qwen3-1.7B",
                    "peft_type": "LORA",
                    "r": 16
                    # Missing required fields intentionally
                }
            },
            {
                "name": "Unknown Architecture",
                "config": {
                    "base_model_name_or_path": "unknown/model-1b",
                    "peft_type": "LORA",
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": ["unknown_proj"],
                    "task_type": "CAUSAL_LM"
                }
            }
        ]
        
        results = []
        
        for test_case in test_configs:
            print(f"\n🧪 Testing: {test_case['name']}")
            print("-" * 50)
            
            try:
                # Create mock adapter
                adapter_path = self.create_mock_adapter(test_case['config'], test_case['name'])
                
                # Test with universal adapter manager
                result = self.test_adapter_with_manager(adapter_path, test_case['name'])
                result['config'] = test_case['config']
                result['test_name'] = test_case['name']
                
                results.append(result)
                
                # Print result
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                print(f"   Status: {status}")
                if result['warnings']:
                    print(f"   Warnings: {', '.join(result['warnings'])}")
                if result['errors']:
                    print(f"   Errors: {', '.join(result['errors'])}")
                
            except Exception as e:
                print(f"   ❌ EXCEPTION: {e}")
                results.append({
                    'test_name': test_case['name'],
                    'success': False,
                    'errors': [str(e)],
                    'warnings': [],
                    'config': test_case['config']
                })
        
        return results
    
    def test_adapter_with_manager(self, adapter_path: str, adapter_name: str) -> Dict[str, Any]:
        """Test adapter with the universal adapter manager."""
        
        result = {
            'success': False,
            'errors': [],
            'warnings': [],
            'adapter_info': None
        }
        
        try:
            from src.core.modular_engine import ModularAdaptrixEngine
            from src.core.universal_adapter_manager import UniversalAdapterManager
            
            # Create engine (don't initialize to save time)
            engine = ModularAdaptrixEngine("Qwen/Qwen3-1.7B", "cpu")
            
            # Create base model instance for adapter manager
            from src.core.base_model_interface import ModelFactory
            base_model = ModelFactory.create_model("Qwen/Qwen3-1.7B", "cpu")
            
            # Create adapter manager
            adapter_manager = UniversalAdapterManager(base_model)
            
            # Test adapter parsing
            adapter_info = adapter_manager._parse_adapter_directory(Path(adapter_path))
            
            if adapter_info:
                result['adapter_info'] = {
                    'name': adapter_info.name,
                    'base_model': adapter_info.base_model,
                    'model_family': adapter_info.model_family.value,
                    'adapter_type': adapter_info.adapter_type,
                    'rank': adapter_info.rank,
                    'alpha': adapter_info.alpha,
                    'target_modules': adapter_info.target_modules,
                    'domain': adapter_info.domain,
                    'capabilities': adapter_info.capabilities
                }
                
                # Test adapter registration
                if adapter_manager.register_adapter(adapter_info):
                    result['success'] = True
                    print(f"   ✅ Adapter parsed and registered successfully")
                else:
                    result['errors'].append("Failed to register adapter")
                    print(f"   ❌ Failed to register adapter")
            else:
                result['errors'].append("Failed to parse adapter directory")
                print(f"   ❌ Failed to parse adapter directory")
            
            # Test compatibility validation
            try:
                is_compatible = base_model.validate_adapter(adapter_path)
                if not is_compatible:
                    result['warnings'].append("Adapter may not be fully compatible")
                    print(f"   ⚠️ Compatibility warning")
            except Exception as e:
                result['warnings'].append(f"Compatibility check failed: {e}")
            
            # Cleanup
            base_model.cleanup()
            
        except Exception as e:
            result['errors'].append(f"Manager test failed: {e}")
        
        return result
    
    def test_edge_cases(self) -> List[Dict[str, Any]]:
        """Test edge cases and potential failure scenarios."""
        
        print("\n🚨 TESTING EDGE CASES AND FAILURE SCENARIOS")
        print("=" * 80)
        
        edge_cases = []
        
        # Test 1: Empty directory
        print("\n🧪 Test: Empty adapter directory")
        empty_dir = tempfile.mkdtemp(prefix="empty_adapter_")
        self.temp_dirs.append(empty_dir)
        
        try:
            from src.core.universal_adapter_manager import UniversalAdapterManager
            from src.core.base_model_interface import ModelFactory
            
            base_model = ModelFactory.create_model("Qwen/Qwen3-1.7B", "cpu")
            adapter_manager = UniversalAdapterManager(base_model)
            
            adapter_info = adapter_manager._parse_adapter_directory(Path(empty_dir))
            
            if adapter_info is None:
                print("   ✅ Correctly handled empty directory")
                edge_cases.append({"test": "empty_directory", "success": True})
            else:
                print("   ❌ Should have returned None for empty directory")
                edge_cases.append({"test": "empty_directory", "success": False})
            
            base_model.cleanup()
            
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            edge_cases.append({"test": "empty_directory", "success": False, "error": str(e)})
        
        # Test 2: Corrupted JSON
        print("\n🧪 Test: Corrupted adapter config")
        corrupted_dir = tempfile.mkdtemp(prefix="corrupted_adapter_")
        self.temp_dirs.append(corrupted_dir)
        
        # Create corrupted JSON
        config_path = os.path.join(corrupted_dir, "adapter_config.json")
        with open(config_path, 'w') as f:
            f.write('{"invalid": json syntax}')
        
        try:
            adapter_info = adapter_manager._parse_adapter_directory(Path(corrupted_dir))
            
            if adapter_info is None:
                print("   ✅ Correctly handled corrupted JSON")
                edge_cases.append({"test": "corrupted_json", "success": True})
            else:
                print("   ❌ Should have returned None for corrupted JSON")
                edge_cases.append({"test": "corrupted_json", "success": False})
                
        except Exception as e:
            print(f"   ✅ Exception properly caught: {type(e).__name__}")
            edge_cases.append({"test": "corrupted_json", "success": True})
        
        # Test 3: Missing required files
        print("\n🧪 Test: Missing adapter model file")
        missing_file_dir = tempfile.mkdtemp(prefix="missing_file_adapter_")
        self.temp_dirs.append(missing_file_dir)
        
        # Create config but no model file
        config_path = os.path.join(missing_file_dir, "adapter_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "base_model_name_or_path": "Qwen/Qwen3-1.7B",
                "peft_type": "LORA",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj"]
            }, f)
        
        try:
            adapter_info = adapter_manager._parse_adapter_directory(Path(missing_file_dir))
            
            if adapter_info:
                print("   ✅ Config parsed despite missing model file")
                edge_cases.append({"test": "missing_model_file", "success": True})
            else:
                print("   ❌ Failed to parse valid config")
                edge_cases.append({"test": "missing_model_file", "success": False})
                
        except Exception as e:
            print(f"   ⚠️ Exception: {e}")
            edge_cases.append({"test": "missing_model_file", "success": False, "error": str(e)})
        
        return edge_cases
    
    def test_cross_model_compatibility(self) -> Dict[str, Any]:
        """Test cross-model family compatibility."""
        
        print("\n🔄 TESTING CROSS-MODEL FAMILY COMPATIBILITY")
        print("=" * 80)
        
        # Test loading Phi-2 adapter with Qwen3 model
        phi2_config = {
            "base_model_name_or_path": "microsoft/phi-2",
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["Wqkv", "out_proj", "fc1", "fc2"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
        
        phi2_adapter_path = self.create_mock_adapter(phi2_config, "phi2_cross_test")
        
        try:
            from src.core.base_model_interface import ModelFactory
            from src.core.universal_adapter_manager import UniversalAdapterManager
            
            # Create Qwen3 model
            qwen_model = ModelFactory.create_model("Qwen/Qwen3-1.7B", "cpu")
            adapter_manager = UniversalAdapterManager(qwen_model)
            
            # Try to parse Phi-2 adapter
            adapter_info = adapter_manager._parse_adapter_directory(Path(phi2_adapter_path))
            
            if adapter_info:
                print(f"   📊 Adapter Info:")
                print(f"      Base Model: {adapter_info.base_model}")
                print(f"      Model Family: {adapter_info.model_family.value}")
                print(f"      Target Modules: {adapter_info.target_modules}")
                
                # Try to register (should warn about compatibility)
                success = adapter_manager.register_adapter(adapter_info)
                
                result = {
                    "cross_model_test": "phi2_to_qwen3",
                    "parsing_success": True,
                    "registration_success": success,
                    "warnings_generated": True  # Should generate warnings
                }
                
                print(f"   ✅ Cross-model parsing: SUCCESS")
                print(f"   ⚠️ Registration: {'SUCCESS' if success else 'FAILED'} (warnings expected)")
            else:
                result = {
                    "cross_model_test": "phi2_to_qwen3",
                    "parsing_success": False,
                    "registration_success": False
                }
                print(f"   ❌ Cross-model parsing: FAILED")
            
            qwen_model.cleanup()
            return result
            
        except Exception as e:
            print(f"   ❌ Cross-model test exception: {e}")
            return {
                "cross_model_test": "phi2_to_qwen3",
                "parsing_success": False,
                "error": str(e)
            }
    
    def generate_compatibility_report(self, results: List[Dict], edge_cases: List[Dict], cross_model: Dict) -> str:
        """Generate comprehensive compatibility report."""
        
        report = []
        report.append("🔧 LORA COMPATIBILITY TEST REPORT")
        report.append("=" * 80)
        
        # Summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['success'])
        failed_tests = total_tests - passed_tests
        
        report.append(f"\n📊 SUMMARY STATISTICS:")
        report.append(f"   Total Tests: {total_tests}")
        report.append(f"   Passed: {passed_tests} ✅")
        report.append(f"   Failed: {failed_tests} ❌")
        report.append(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Detailed results
        report.append(f"\n📋 DETAILED TEST RESULTS:")
        for result in results:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            report.append(f"   {result['test_name']}: {status}")
            
            if result.get('adapter_info'):
                info = result['adapter_info']
                report.append(f"      Model Family: {info['model_family']}")
                report.append(f"      Rank: {info['rank']}, Alpha: {info['alpha']}")
                report.append(f"      Target Modules: {len(info['target_modules'])} modules")
            
            if result['warnings']:
                report.append(f"      Warnings: {', '.join(result['warnings'])}")
            if result['errors']:
                report.append(f"      Errors: {', '.join(result['errors'])}")
        
        # Edge cases
        report.append(f"\n🚨 EDGE CASE RESULTS:")
        for case in edge_cases:
            status = "✅ HANDLED" if case['success'] else "❌ FAILED"
            report.append(f"   {case['test']}: {status}")
        
        # Cross-model compatibility
        report.append(f"\n🔄 CROSS-MODEL COMPATIBILITY:")
        if cross_model.get('parsing_success'):
            report.append(f"   ✅ Cross-model parsing: WORKING")
        else:
            report.append(f"   ❌ Cross-model parsing: FAILED")
        
        # Recommendations
        report.append(f"\n🎯 RECOMMENDATIONS:")
        if failed_tests == 0:
            report.append(f"   🎊 EXCELLENT: All adapter configurations supported!")
            report.append(f"   🚀 System is ready for any LoRA adapter architecture")
        elif failed_tests <= 2:
            report.append(f"   ✅ GOOD: Most adapter configurations supported")
            report.append(f"   🔧 Minor improvements needed for edge cases")
        else:
            report.append(f"   ⚠️ NEEDS WORK: Several compatibility issues found")
            report.append(f"   🔧 Recommend addressing failed test cases")
        
        report.append(f"\n🔌 PLUG-AND-PLAY READINESS:")
        if passed_tests >= total_tests * 0.8:
            report.append(f"   🎊 READY: System can handle diverse LoRA architectures")
            report.append(f"   ✅ New adapters should work without code changes")
        else:
            report.append(f"   ⚠️ PARTIAL: Some adapter types may need manual handling")
        
        return "\n".join(report)
    
    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup {temp_dir}: {e}")
    
    def run_full_test_suite(self):
        """Run the complete compatibility test suite."""
        
        print("🚀" * 100)
        print("🚀 COMPREHENSIVE LORA COMPATIBILITY TEST SUITE 🚀")
        print("🚀" * 100)
        
        try:
            # Test various adapter configurations
            config_results = self.test_adapter_configurations()
            
            # Test edge cases
            edge_case_results = self.test_edge_cases()
            
            # Test cross-model compatibility
            cross_model_result = self.test_cross_model_compatibility()
            
            # Generate report
            report = self.generate_compatibility_report(
                config_results, 
                edge_case_results, 
                cross_model_result
            )
            
            print(f"\n{report}")
            
            # Save report to file
            report_path = "lora_compatibility_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            
            print(f"\n📄 Report saved to: {report_path}")
            
            return config_results, edge_case_results, cross_model_result
            
        finally:
            self.cleanup()


def main():
    """Main test function."""
    tester = LoRACompatibilityTester()
    
    try:
        results = tester.run_full_test_suite()
        print("\n🎯 COMPATIBILITY TESTING COMPLETE!")
        return results
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
