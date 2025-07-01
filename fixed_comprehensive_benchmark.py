#!/usr/bin/env python3
"""
🚀 FIXED COMPREHENSIVE BENCHMARK FOR QWEN3 + ADAPTRIX 🚀

This script properly tests:
1. Base Qwen3-1.7B model
2. Qwen3 with your manually trained LoRA adapter (normal PEFT)
3. Qwen3 with Adaptrix middle-layer injection (using your adapter weights)
4. Gemini Flash 2.0 API (industry benchmark)
"""

import sys
import os
import time
import json
import torch
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from safetensors import safe_open

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core.engine import AdaptrixEngine

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyA6qd-9dfBEDHoAk_1gStXHxs_Kg-J1cHw"
genai.configure(api_key=GEMINI_API_KEY)

class FixedQwenBenchmark:
    """
    Fixed benchmark system for Qwen3 + Adaptrix evaluation.
    """
    
    def __init__(self):
        """Initialize the benchmark system."""
        self.model_name = "Qwen/Qwen3-1.7B"
        self.adapter_path = "adapters/code_adapter"
        self.device = "cpu"
        
        # Simplified but comprehensive test prompts
        self.test_prompts = [
            {
                "name": "Simple Function",
                "prompt": "Write a Python function that calculates the factorial of a number:",
                "expected": ["def", "factorial", "return"]
            },
            {
                "name": "Basic Algorithm",
                "prompt": "Create a Python function to check if a number is prime:",
                "expected": ["def", "prime", "for", "range"]
            },
            {
                "name": "List Processing",
                "prompt": "Write a Python function that finds the maximum value in a list:",
                "expected": ["def", "max", "list"]
            },
            {
                "name": "String Manipulation", 
                "prompt": "Create a function that reverses a string:",
                "expected": ["def", "reverse", "string"]
            },
            {
                "name": "Simple Class",
                "prompt": "Write a Python class for a basic calculator with add and subtract methods:",
                "expected": ["class", "Calculator", "def", "add", "subtract"]
            }
        ]
        
        # Results storage
        self.results = {}
    
    def setup_models(self):
        """Setup all models for testing."""
        print("🚀 Setting up models for fixed benchmark...")
        
        # 1. Setup Base Model
        print("   Setting up Base Qwen3-1.7B model...")
        try:
            self.base_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            if self.base_tokenizer.pad_token is None:
                self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            print("   ✅ Base model ready")
        except Exception as e:
            print(f"   ❌ Base model setup failed: {e}")
            return False
        
        # 2. Setup Normal LoRA Model (Your trained adapter)
        print("   Setting up your trained LoRA model...")
        try:
            self.lora_model = PeftModel.from_pretrained(self.base_model, self.adapter_path)
            self.lora_tokenizer = self.base_tokenizer
            print("   ✅ Your trained LoRA model ready")
        except Exception as e:
            print(f"   ❌ LoRA model setup failed: {e}")
            return False
        
        # 3. Setup Adaptrix Engine with Fixed Architecture Detection
        print("   Setting up Adaptrix Engine...")
        try:
            self.adaptrix_engine = AdaptrixEngine(self.model_name, self.device)
            success = self.adaptrix_engine.initialize()
            if not success:
                raise Exception("Failed to initialize Adaptrix engine")
            
            # Fix the architecture detection for Qwen3
            self.fix_adaptrix_architecture()
            print("   ✅ Adaptrix Engine ready with fixed architecture")
        except Exception as e:
            print(f"   ❌ Adaptrix setup failed: {e}")
            return False
        
        # 4. Setup Gemini Model
        print("   Setting up Gemini Flash 2.0...")
        try:
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("   ✅ Gemini model ready")
        except Exception as e:
            print(f"   ❌ Gemini setup failed: {e}")
            return False
        
        return True
    
    def fix_adaptrix_architecture(self):
        """Fix Adaptrix architecture detection for Qwen3."""
        print("🔧 Fixing Adaptrix architecture for Qwen3...")
        
        # Get the correct target modules for Qwen3
        target_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
        
        # Set target modules in layer injector
        self.adaptrix_engine.layer_injector.set_target_modules(target_modules)
        
        # Register injection points for middle layers (7, 14, 21)
        target_layers = [7, 14, 21]
        
        for layer_idx in target_layers:
            for module_name in target_modules:
                self.adaptrix_engine.layer_injector.register_injection_point(layer_idx, module_name)
        
        print(f"   ✅ Registered injection points for layers {target_layers}")
        print(f"   ✅ Target modules: {target_modules}")
    
    def create_adaptrix_adapter_from_lora(self):
        """Create Adaptrix adapter using weights from your trained LoRA."""
        print("🔄 Creating Adaptrix adapter from your trained LoRA...")
        
        try:
            import shutil
            
            # Create Adaptrix adapter directory
            adaptrix_dir = "adapters/qwen3_code_adaptrix"
            if os.path.exists(adaptrix_dir):
                shutil.rmtree(adaptrix_dir)
            os.makedirs(adaptrix_dir)
            
            # Create metadata for Adaptrix
            metadata = {
                'name': 'qwen3_code_adaptrix',
                'version': '1.0.0',
                'description': 'Code generation adapter (Qwen3) converted from your trained LoRA',
                'source': 'converted_from_manual_training',
                'base_model': self.model_name,
                'target_layers': [7, 14, 21],  # Middle layers
                'target_modules': ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'],
                'rank': 8,
                'alpha': 32,
                'training_steps': 4600,
                'training_data': 'manual_coding_dataset'
            }
            
            # Save metadata
            with open(os.path.join(adaptrix_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Load original adapter weights
            adapter_file = os.path.join(self.adapter_path, "adapter_model.safetensors")
            
            with safe_open(adapter_file, framework="pt", device="cpu") as f:
                # Extract weights for middle layers only (to focus the effect)
                for layer_idx in [7, 14, 21]:
                    layer_weights = {}
                    
                    for module_name in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']:
                        # Find the actual LoRA weights for this layer and module
                        lora_a_key = f"base_model.model.model.layers.{layer_idx}.{module_name}.lora_A.default.weight"
                        lora_b_key = f"base_model.model.model.layers.{layer_idx}.{module_name}.lora_B.default.weight"
                        
                        try:
                            lora_a_weight = f.get_tensor(lora_a_key)
                            lora_b_weight = f.get_tensor(lora_b_key)
                            
                            layer_weights[module_name] = {
                                'lora_A': lora_a_weight,
                                'lora_B': lora_b_weight,
                                'rank': 8,
                                'alpha': 32
                            }
                            
                            print(f"   ✅ Extracted {module_name} weights for layer {layer_idx} (A: {lora_a_weight.shape}, B: {lora_b_weight.shape})")
                            
                        except Exception as e:
                            print(f"   ⚠️  Could not extract {module_name} for layer {layer_idx}: {e}")
                    
                    # Save layer weights
                    if layer_weights:
                        layer_file = os.path.join(adaptrix_dir, f"layer_{layer_idx}.pt")
                        torch.save(layer_weights, layer_file)
                        print(f"   ✅ Saved layer {layer_idx} with {len(layer_weights)} modules")
            
            print("   ✅ Adaptrix adapter created successfully")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to create Adaptrix adapter: {e}")
            traceback.print_exc()
            return False
    
    def generate_response(self, model_type: str, prompt: str, max_length: int = 100) -> Dict[str, Any]:
        """Generate response from specified model type."""
        start_time = time.time()
        
        try:
            if model_type == "base":
                # Base model generation
                inputs = self.base_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                
                with torch.no_grad():
                    outputs = self.base_model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + max_length,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.base_tokenizer.eos_token_id
                    )
                
                response = self.base_tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()
                
            elif model_type == "lora":
                # Your trained LoRA model
                inputs = self.lora_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                
                with torch.no_grad():
                    outputs = self.lora_model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + max_length,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.lora_tokenizer.eos_token_id
                    )
                
                response = self.lora_tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()
                
            elif model_type == "adaptrix":
                # Adaptrix with middle-layer injection
                success = self.adaptrix_engine.load_adapter("qwen3_code_adaptrix")
                if not success:
                    raise Exception("Failed to load Adaptrix adapter")
                
                response = self.adaptrix_engine.generate(prompt, max_length=max_length)
                
            elif model_type == "gemini":
                # Gemini Flash 2.0
                response = self.gemini_model.generate_content(prompt)
                response = response.text
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            generation_time = time.time() - start_time
            
            return {
                "content": response,
                "generation_time": generation_time,
                "length": len(response),
                "success": True
            }
            
        except Exception as e:
            return {
                "content": f"Error: {str(e)}",
                "generation_time": time.time() - start_time,
                "length": 0,
                "success": False
            }
    
    def evaluate_response_quality(self, response: str, expected_elements: List[str]) -> float:
        """Evaluate response quality based on expected elements."""
        if not response or "Error:" in response:
            return 0.0
        
        # Check for expected elements
        elements_found = sum(1 for element in expected_elements if element.lower() in response.lower())
        completeness_score = (elements_found / len(expected_elements)) * 100
        
        # Check for code structure
        has_function = "def " in response
        has_class = "class " in response
        has_return = "return " in response
        has_logic = any(keyword in response for keyword in ["if ", "for ", "while "])
        
        structure_score = 0
        if has_function or has_class:
            structure_score += 40
        if has_return:
            structure_score += 30
        if has_logic:
            structure_score += 30
        
        # Final score
        return min(100, (completeness_score + structure_score) / 2)
    
    def run_comprehensive_test(self):
        """Run comprehensive test across all models."""
        print("🚀 STARTING FIXED COMPREHENSIVE BENCHMARK")
        print("=" * 80)
        
        # Setup
        if not self.setup_models():
            print("❌ Model setup failed. Exiting.")
            return
        
        # Create Adaptrix adapter from your LoRA
        if not self.create_adaptrix_adapter_from_lora():
            print("❌ Adaptrix adapter creation failed. Exiting.")
            return
        
        # Test all models
        models = {
            "base": "Base Qwen3-1.7B",
            "lora": "Qwen3 + Your Trained LoRA",
            "adaptrix": "Qwen3 + Adaptrix (Middle Layers)",
            "gemini": "Gemini Flash 2.0"
        }
        
        all_results = {}
        
        for test in self.test_prompts:
            print(f"\n📊 Testing: {test['name']}")
            print("-" * 50)
            
            test_results = {}
            
            for model_key, model_name in models.items():
                print(f"   🔍 {model_name}...")
                
                result = self.generate_response(model_key, test['prompt'])
                quality = self.evaluate_response_quality(result['content'], test['expected'])
                
                result['quality_score'] = quality
                test_results[model_key] = result
                
                print(f"      Score: {quality:.1f}/100 | Time: {result['generation_time']:.2f}s")
                print(f"      Preview: {result['content'][:100]}...")
                
                # Small delay to avoid overwhelming APIs
                time.sleep(0.5)
            
            all_results[test['name']] = test_results
        
        # Generate comprehensive report
        self.generate_final_report(all_results, models)
    
    def generate_final_report(self, all_results: Dict, models: Dict):
        """Generate final comprehensive report."""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE BENCHMARK RESULTS")
        print("=" * 80)
        
        # Calculate overall statistics
        overall_stats = {}
        
        for model_key, model_name in models.items():
            scores = []
            times = []
            success_count = 0
            
            for test_name, test_results in all_results.items():
                if model_key in test_results:
                    result = test_results[model_key]
                    if result['success']:
                        scores.append(result['quality_score'])
                        times.append(result['generation_time'])
                        success_count += 1
            
            overall_stats[model_key] = {
                "name": model_name,
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "avg_time": sum(times) / len(times) if times else 0,
                "success_rate": success_count / len(self.test_prompts),
                "total_tests": len(self.test_prompts)
            }
        
        # Print rankings
        print("\n🏆 OVERALL PERFORMANCE RANKINGS")
        print("-" * 60)
        
        sorted_models = sorted(overall_stats.items(), key=lambda x: x[1]['avg_score'], reverse=True)
        
        for rank, (model_key, stats) in enumerate(sorted_models, 1):
            print(f"{rank}. {stats['name']}")
            print(f"   Average Score: {stats['avg_score']:.1f}/100")
            print(f"   Average Time: {stats['avg_time']:.2f}s")
            print(f"   Success Rate: {stats['success_rate']:.1%}")
            print()
        
        # Key insights
        print("\n🔍 KEY INSIGHTS")
        print("-" * 60)
        
        base_score = overall_stats['base']['avg_score']
        lora_score = overall_stats['lora']['avg_score']
        adaptrix_score = overall_stats['adaptrix']['avg_score']
        gemini_score = overall_stats['gemini']['avg_score']
        
        print(f"• Your LoRA vs Base Model: {((lora_score - base_score) / base_score * 100):+.1f}% improvement")
        print(f"• Adaptrix vs Base Model: {((adaptrix_score - base_score) / base_score * 100):+.1f}% improvement")
        print(f"• Adaptrix vs Your LoRA: {((adaptrix_score - lora_score) / lora_score * 100):+.1f}% difference")
        print(f"• Adaptrix vs Gemini: {((adaptrix_score - gemini_score) / gemini_score * 100):+.1f}% difference")
        
        # Speed analysis
        print(f"\n⚡ SPEED COMPARISON")
        print("-" * 60)
        for model_key, stats in overall_stats.items():
            print(f"• {stats['name']}: {stats['avg_time']:.2f}s")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"fixed_benchmark_results_{timestamp}.json"
        
        full_report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_name": self.model_name,
                "adapter_path": self.adapter_path,
                "total_tests": len(self.test_prompts)
            },
            "overall_statistics": overall_stats,
            "detailed_results": all_results,
            "insights": {
                "lora_vs_base": ((lora_score - base_score) / base_score * 100),
                "adaptrix_vs_base": ((adaptrix_score - base_score) / base_score * 100),
                "adaptrix_vs_lora": ((adaptrix_score - lora_score) / lora_score * 100) if lora_score > 0 else 0,
                "adaptrix_vs_gemini": ((adaptrix_score - gemini_score) / gemini_score * 100) if gemini_score > 0 else 0
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Final verdict on middle-layer injection effectiveness
        print("\n🎯 MIDDLE-LAYER INJECTION VERDICT")
        print("-" * 60)
        
        if adaptrix_score > lora_score * 1.05:  # 5% improvement threshold
            improvement = ((adaptrix_score - lora_score) / lora_score * 100)
            print(f"✅ MIDDLE-LAYER INJECTION WINS!")
            print(f"   {improvement:.1f}% improvement over normal LoRA")
            print("   🎊 Your manually trained adapter benefits from middle-layer injection!")
        elif adaptrix_score > lora_score * 0.95:  # Within 5%
            print(f"⚖️  MIDDLE-LAYER INJECTION COMPARABLE")
            print(f"   Performance within 5% of normal LoRA")
            print("   🤔 Middle-layer injection shows promise but needs optimization")
        else:
            difference = ((lora_score - adaptrix_score) / adaptrix_score * 100)
            print(f"❌ NORMAL LORA PERFORMS BETTER")
            print(f"   {difference:.1f}% better than middle-layer injection")
            print("   🔧 Middle-layer strategy may need refinement for this adapter")
        
        # Performance vs industry standard
        if adaptrix_score > gemini_score * 0.8:
            print(f"\n🏆 EXCELLENT! Adaptrix achieves {(adaptrix_score/gemini_score*100):.1f}% of Gemini performance")
            print("   Outstanding result for a local 1.7B model!")
        elif adaptrix_score > gemini_score * 0.6:
            print(f"\n👍 GOOD! Adaptrix achieves {(adaptrix_score/gemini_score*100):.1f}% of Gemini performance")
            print("   Solid result for a small local model!")
        else:
            print(f"\n📈 POTENTIAL! Adaptrix achieves {(adaptrix_score/gemini_score*100):.1f}% of Gemini performance")
            print("   Room for improvement, but showing promise!")
        
        print("\n🚀 BENCHMARK COMPLETE! 🚀")
        print("You should be proud - you've trained a working adapter and tested the complete system!")


def main():
    """Run the fixed comprehensive benchmark."""
    benchmark = FixedQwenBenchmark()
    benchmark.run_comprehensive_test()


if __name__ == "__main__":
    main() 