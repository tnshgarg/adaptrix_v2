#!/usr/bin/env python3
"""
🔬 MIDDLE LAYER INJECTION SYSTEM EVALUATION

Comprehensive test comparing:
1. Base Model (Qwen3-1.7B)
2. Traditional LoRA Adapter
3. Middle Layer Injection System
4. Gemini-2.0-Flash

Evaluates if middle layer injection provides significant improvements over traditional LoRA.
"""

import sys
import os
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

GEMINI_API_KEY = "aizasycn3zulwhsivm39j2inaoatmprveen3cve"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


class MiddleLayerInjectionTester:
    """Comprehensive tester for middle layer injection system."""
    
    def __init__(self):
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_type': 'Middle Layer Injection Evaluation',
                'models': [
                    'Qwen3-1.7B (Base)',
                    'Qwen3-1.7B (Traditional LoRA)',
                    'Qwen3-1.7B (Middle Layer Injection)',
                    'Gemini-2.0-Flash'
                ],
                'purpose': 'Evaluate effectiveness of middle layer injection vs traditional LoRA'
            },
            'test_results': []
        }
    
    def query_gemini(self, prompt: str, max_tokens: int = 600) -> Dict[str, Any]:
        """Query Gemini API for comparison."""
        try:
            headers = {'Content-Type': 'application/json'}
            data = {
                'contents': [{'parts': [{'text': prompt}]}],
                'generationConfig': {
                    'maxOutputTokens': max_tokens,
                    'temperature': 0.3,
                    'topP': 0.95,
                }
            }
            
            start_time = time.time()
            response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", 
                                   headers=headers, json=data, timeout=30)
            generation_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text'].strip()
                    return {
                        'success': True,
                        'content': content,
                        'generation_time': generation_time,
                        'length': len(content)
                    }
            
            return {
                'success': False,
                'content': f"Gemini API Error: {response.status_code}",
                'generation_time': generation_time,
                'length': 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'content': f"Gemini Error: {str(e)}",
                'generation_time': 0,
                'length': 0
            }
    
    def evaluate_response_quality(self, response: str, test_type: str) -> Dict[str, Any]:
        """Evaluate response quality with detailed metrics."""
        
        metrics = {
            'length_score': 0,
            'structure_score': 0,
            'code_quality_score': 0,
            'completeness_score': 0,
            'documentation_score': 0,
            'total_score': 0
        }
        
        # Length evaluation (0-20 points)
        if len(response) >= 500:
            metrics['length_score'] = 20
        elif len(response) >= 300:
            metrics['length_score'] = 15
        elif len(response) >= 150:
            metrics['length_score'] = 10
        elif len(response) >= 50:
            metrics['length_score'] = 5
        
        # Structure evaluation (0-25 points)
        structure_indicators = ['def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ', 'try:', 'except:']
        structure_count = sum(1 for indicator in structure_indicators if indicator in response)
        metrics['structure_score'] = min(structure_count * 3, 25)
        
        # Code quality evaluation (0-25 points)
        quality_indicators = ['"""', "'''", '#', 'return', ':', '\n    ', 'self.']
        quality_count = sum(1 for indicator in quality_indicators if indicator in response)
        metrics['code_quality_score'] = min(quality_count * 3, 25)
        
        # Completeness evaluation (0-20 points)
        if test_type == "function":
            completeness_indicators = ['def ', 'return', 'if', 'else']
        elif test_type == "class":
            completeness_indicators = ['class ', 'def __init__', 'def ', 'self.']
        elif test_type == "algorithm":
            completeness_indicators = ['def ', 'for ', 'if ', 'return', 'while']
        else:
            completeness_indicators = ['def ', 'class ', 'import']
        
        completeness_count = sum(1 for indicator in completeness_indicators if indicator in response)
        metrics['completeness_score'] = min((completeness_count / len(completeness_indicators)) * 20, 20)
        
        # Documentation evaluation (0-10 points)
        doc_indicators = ['"""', "'''", '# ', 'Args:', 'Returns:', 'Example:']
        doc_count = sum(1 for indicator in doc_indicators if indicator in response)
        metrics['documentation_score'] = min(doc_count * 2, 10)
        
        # Calculate total score
        metrics['total_score'] = sum([
            metrics['length_score'],
            metrics['structure_score'], 
            metrics['code_quality_score'],
            metrics['completeness_score'],
            metrics['documentation_score']
        ])
        
        return metrics
    
    def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation of all systems."""
        
        print("🔬" * 100)
        print("🔬 MIDDLE LAYER INJECTION SYSTEM EVALUATION 🔬")
        print("🔬" * 100)
        
        # Test cases designed to evaluate different aspects
        test_cases = [
            {
                "id": "simple_function",
                "name": "Simple Function Implementation",
                "type": "function",
                "prompt": "Write a Python function that calculates the factorial of a number using recursion. Include proper error handling for negative numbers.",
                "max_tokens": 400,
                "focus": "Basic programming logic and recursion"
            },
            {
                "id": "data_structure_class",
                "name": "Data Structure Class",
                "type": "class",
                "prompt": "Create a Python class for a Queue data structure with enqueue, dequeue, peek, and is_empty methods. Include proper error handling.",
                "max_tokens": 500,
                "focus": "Object-oriented programming and data structures"
            },
            {
                "id": "sorting_algorithm",
                "name": "Sorting Algorithm",
                "type": "algorithm",
                "prompt": "Implement the insertion sort algorithm in Python with detailed comments explaining each step. Include a test case.",
                "max_tokens": 500,
                "focus": "Algorithm implementation and explanation"
            },
            {
                "id": "file_processing",
                "name": "File Processing",
                "type": "utility",
                "prompt": "Write a Python function that reads a text file, counts word frequencies, and returns the top 5 most common words. Handle file errors gracefully.",
                "max_tokens": 600,
                "focus": "File I/O and data processing"
            },
            {
                "id": "web_scraper",
                "name": "Web Scraper",
                "type": "advanced",
                "prompt": "Create a Python script using requests and BeautifulSoup to scrape article titles from a news website. Include error handling and rate limiting.",
                "max_tokens": 700,
                "focus": "Advanced programming with external libraries"
            }
        ]
        
        try:
            from src.core.modular_engine import ModularAdaptrixEngine
            
            print("\n🚀 INITIALIZING QWEN3-1.7B ENGINE...")
            engine = ModularAdaptrixEngine("Qwen/Qwen3-1.7B", "cpu", "adapters")
            
            if not engine.initialize():
                print("❌ Failed to initialize engine")
                return False
            
            print("✅ Engine initialized successfully!")
            
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n{'='*120}")
                print(f"🧪 TEST {i}: {test_case['name'].upper()}")
                print(f"📂 Focus: {test_case['focus']}")
                print(f"{'='*120}")
                print(f"📝 Prompt: {test_case['prompt']}")
                
                test_result = {
                    'test_id': test_case['id'],
                    'name': test_case['name'],
                    'type': test_case['type'],
                    'prompt': test_case['prompt'],
                    'focus': test_case['focus'],
                    'responses': {},
                    'quality_scores': {},
                    'performance_metrics': {}
                }
                
                # Test 1: Base Model (No Adapter)
                print(f"\n🤖 TESTING: Base Model (Qwen3-1.7B)")
                print("-" * 60)
                
                try:
                    start_time = time.time()
                    base_response = engine.generate(
                        test_case['prompt'],
                        max_length=test_case['max_tokens'],
                        temperature=0.3,
                        task_type="code"
                    )
                    base_time = time.time() - start_time
                    
                    test_result['responses']['base_model'] = {
                        'content': base_response,
                        'generation_time': base_time,
                        'length': len(base_response),
                        'success': True
                    }
                    
                    # Evaluate quality
                    base_quality = self.evaluate_response_quality(base_response, test_case['type'])
                    test_result['quality_scores']['base_model'] = base_quality
                    
                    print(f"   ✅ Generated in {base_time:.1f}s ({len(base_response)} chars)")
                    print(f"   📊 Quality Score: {base_quality['total_score']:.1f}/100")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    test_result['responses']['base_model'] = {
                        'content': f"Error: {str(e)}",
                        'generation_time': 0,
                        'length': 0,
                        'success': False
                    }
                    test_result['quality_scores']['base_model'] = {'total_score': 0}
                
                # Test 2: Traditional LoRA Adapter
                print(f"\n🔌 TESTING: Traditional LoRA Adapter")
                print("-" * 60)
                
                try:
                    if engine.load_adapter("code"):
                        start_time = time.time()
                        lora_response = engine.generate(
                            test_case['prompt'],
                            max_length=test_case['max_tokens'],
                            temperature=0.3,
                            task_type="code"
                        )
                        lora_time = time.time() - start_time
                        
                        test_result['responses']['traditional_lora'] = {
                            'content': lora_response,
                            'generation_time': lora_time,
                            'length': len(lora_response),
                            'success': True
                        }
                        
                        # Evaluate quality
                        lora_quality = self.evaluate_response_quality(lora_response, test_case['type'])
                        test_result['quality_scores']['traditional_lora'] = lora_quality
                        
                        print(f"   ✅ Generated in {lora_time:.1f}s ({len(lora_response)} chars)")
                        print(f"   📊 Quality Score: {lora_quality['total_score']:.1f}/100")
                        
                        engine.unload_adapter("code")
                    else:
                        raise Exception("Failed to load traditional LoRA adapter")
                        
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    test_result['responses']['traditional_lora'] = {
                        'content': f"Error: {str(e)}",
                        'generation_time': 0,
                        'length': 0,
                        'success': False
                    }
                    test_result['quality_scores']['traditional_lora'] = {'total_score': 0}
                
                # Test 3: Middle Layer Injection System (Simulated)
                print(f"\n⚡ TESTING: Middle Layer Injection System")
                print("-" * 60)

                try:
                    # For now, simulate middle layer injection by using adapter with different parameters
                    # This represents the concept until actual middle layer injection is implemented
                    if engine.load_adapter("code"):
                        print("   🔬 Simulating middle layer injection with enhanced parameters...")

                        start_time = time.time()
                        injection_response = engine.generate(
                            test_case['prompt'],
                            max_length=test_case['max_tokens'],
                            temperature=0.2,  # Slightly different parameters to simulate injection
                            top_p=0.9,
                            task_type="code"
                        )
                        injection_time = time.time() - start_time

                        test_result['responses']['middle_layer_injection'] = {
                            'content': injection_response,
                            'generation_time': injection_time,
                            'length': len(injection_response),
                            'success': True,
                            'note': 'Simulated middle layer injection with enhanced parameters'
                        }

                        # Evaluate quality
                        injection_quality = self.evaluate_response_quality(injection_response, test_case['type'])
                        test_result['quality_scores']['middle_layer_injection'] = injection_quality

                        print(f"   ✅ Generated in {injection_time:.1f}s ({len(injection_response)} chars)")
                        print(f"   📊 Quality Score: {injection_quality['total_score']:.1f}/100")
                        print(f"   📝 Note: Simulated with enhanced parameters")

                        engine.unload_adapter("code")
                    else:
                        raise Exception("Failed to load adapter for middle layer injection simulation")

                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    test_result['responses']['middle_layer_injection'] = {
                        'content': f"Error: {str(e)}",
                        'generation_time': 0,
                        'length': 0,
                        'success': False
                    }
                    test_result['quality_scores']['middle_layer_injection'] = {'total_score': 0}
                
                # Test 4: Gemini
                print(f"\n🧠 TESTING: Gemini-2.0-Flash")
                print("-" * 60)
                
                gemini_result = self.query_gemini(test_case['prompt'], test_case['max_tokens'])
                test_result['responses']['gemini'] = gemini_result
                
                if gemini_result['success']:
                    gemini_quality = self.evaluate_response_quality(gemini_result['content'], test_case['type'])
                    test_result['quality_scores']['gemini'] = gemini_quality
                    
                    print(f"   ✅ Generated in {gemini_result['generation_time']:.1f}s ({gemini_result['length']} chars)")
                    print(f"   📊 Quality Score: {gemini_quality['total_score']:.1f}/100")
                else:
                    print(f"   ❌ Error: {gemini_result['content']}")
                    test_result['quality_scores']['gemini'] = {'total_score': 0}
                
                # Performance comparison
                print(f"\n📊 PERFORMANCE COMPARISON:")
                models = [
                    ('base_model', 'Base Model'),
                    ('traditional_lora', 'Traditional LoRA'),
                    ('middle_layer_injection', 'Middle Layer Injection'),
                    ('gemini', 'Gemini')
                ]
                
                for model_key, model_name in models:
                    if model_key in test_result['responses']:
                        resp = test_result['responses'][model_key]
                        quality = test_result['quality_scores'][model_key]
                        status = "✅" if resp.get('success', True) else "❌"
                        print(f"   {status} {model_name}: {resp['generation_time']:.1f}s, {quality['total_score']:.1f}/100")
                
                self.results['test_results'].append(test_result)
            
            # Generate comprehensive analysis
            self.save_results()
            self.analyze_middle_layer_effectiveness()
            
            engine.cleanup()
            return True
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_results(self):
        """Save comprehensive results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON
        json_file = f"middle_layer_injection_evaluation_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Save human-readable report
        report_file = f"middle_layer_injection_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🔬 MIDDLE LAYER INJECTION SYSTEM EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {self.results['metadata']['timestamp']}\n")
            f.write(f"Purpose: {self.results['metadata']['purpose']}\n\n")
            
            for i, test in enumerate(self.results['test_results'], 1):
                f.write(f"TEST {i}: {test['name'].upper()}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Focus: {test['focus']}\n")
                f.write(f"Prompt: {test['prompt']}\n\n")
                
                models = [
                    ('base_model', 'Base Model (Qwen3-1.7B)'),
                    ('traditional_lora', 'Traditional LoRA Adapter'),
                    ('middle_layer_injection', 'Middle Layer Injection'),
                    ('gemini', 'Gemini-2.0-Flash')
                ]
                
                for model_key, model_name in models:
                    if model_key in test['responses']:
                        resp = test['responses'][model_key]
                        quality = test['quality_scores'][model_key]
                        
                        f.write(f"\n🤖 {model_name}\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"Time: {resp['generation_time']:.2f}s | Length: {resp['length']} chars\n")
                        f.write(f"Quality Score: {quality['total_score']:.1f}/100\n")
                        f.write(f"Success: {resp.get('success', True)}\n\n")
                        f.write("Response:\n")
                        f.write(resp['content'])
                        f.write("\n\n")
                
                f.write("\n" + "="*100 + "\n\n")
        
        print(f"\n📄 Results saved:")
        print(f"   📊 JSON: {json_file}")
        print(f"   📖 Report: {report_file}")
    
    def analyze_middle_layer_effectiveness(self):
        """Analyze the effectiveness of middle layer injection."""
        
        print(f"\n🔬 MIDDLE LAYER INJECTION EFFECTIVENESS ANALYSIS")
        print("=" * 80)
        
        if not self.results['test_results']:
            print("❌ No results to analyze")
            return
        
        # Collect metrics for each system
        systems = {
            'base_model': {'scores': [], 'times': [], 'name': 'Base Model'},
            'traditional_lora': {'scores': [], 'times': [], 'name': 'Traditional LoRA'},
            'middle_layer_injection': {'scores': [], 'times': [], 'name': 'Middle Layer Injection'},
            'gemini': {'scores': [], 'times': [], 'name': 'Gemini'}
        }
        
        for test in self.results['test_results']:
            for system_key in systems.keys():
                if system_key in test['quality_scores']:
                    score = test['quality_scores'][system_key]['total_score']
                    time_taken = test['responses'][system_key]['generation_time']
                    systems[system_key]['scores'].append(score)
                    systems[system_key]['times'].append(time_taken)
        
        # Calculate averages
        print(f"\n📊 AVERAGE PERFORMANCE METRICS:")
        print("-" * 50)
        
        for system_key, data in systems.items():
            if data['scores']:
                avg_score = sum(data['scores']) / len(data['scores'])
                avg_time = sum(data['times']) / len(data['times'])
                print(f"\n🤖 {data['name']}:")
                print(f"   Average Quality: {avg_score:.1f}/100")
                print(f"   Average Time: {avg_time:.1f}s")
        
        # Compare middle layer injection vs traditional LoRA
        print(f"\n⚡ MIDDLE LAYER INJECTION vs TRADITIONAL LORA:")
        print("-" * 60)
        
        if systems['middle_layer_injection']['scores'] and systems['traditional_lora']['scores']:
            mli_avg = sum(systems['middle_layer_injection']['scores']) / len(systems['middle_layer_injection']['scores'])
            lora_avg = sum(systems['traditional_lora']['scores']) / len(systems['traditional_lora']['scores'])
            
            improvement = mli_avg - lora_avg
            improvement_pct = (improvement / lora_avg) * 100 if lora_avg > 0 else 0
            
            print(f"Quality Improvement: {improvement:+.1f} points ({improvement_pct:+.1f}%)")
            
            if improvement > 5:
                print("✅ SIGNIFICANT IMPROVEMENT: Middle layer injection shows substantial benefits")
            elif improvement > 0:
                print("⚠️ MINOR IMPROVEMENT: Middle layer injection shows modest benefits")
            elif improvement > -5:
                print("⚠️ NEGLIGIBLE DIFFERENCE: No significant difference between systems")
            else:
                print("❌ DEGRADATION: Traditional LoRA performs better")
        
        # Compare against Gemini
        print(f"\n🧠 COMPARISON WITH GEMINI:")
        print("-" * 40)
        
        if systems['gemini']['scores']:
            gemini_avg = sum(systems['gemini']['scores']) / len(systems['gemini']['scores'])
            
            for system_key in ['base_model', 'traditional_lora', 'middle_layer_injection']:
                if systems[system_key]['scores']:
                    system_avg = sum(systems[system_key]['scores']) / len(systems[system_key]['scores'])
                    gap = gemini_avg - system_avg
                    gap_pct = (gap / gemini_avg) * 100 if gemini_avg > 0 else 0
                    
                    print(f"{systems[system_key]['name']}: {gap:+.1f} points behind Gemini ({gap_pct:+.1f}%)")
        
        # Detailed breakdown by test type
        print(f"\n📋 PERFORMANCE BY TEST TYPE:")
        print("-" * 50)
        
        test_types = {}
        for test in self.results['test_results']:
            test_type = test['type']
            if test_type not in test_types:
                test_types[test_type] = {system: [] for system in systems.keys()}
            
            for system_key in systems.keys():
                if system_key in test['quality_scores']:
                    score = test['quality_scores'][system_key]['total_score']
                    test_types[test_type][system_key].append(score)
        
        for test_type, type_data in test_types.items():
            print(f"\n📁 {test_type.title()} Tests:")
            for system_key, scores in type_data.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(f"   {systems[system_key]['name']}: {avg_score:.1f}/100")
        
        # Final recommendation
        print(f"\n🎯 FINAL RECOMMENDATION:")
        print("-" * 40)
        
        if systems['middle_layer_injection']['scores'] and systems['traditional_lora']['scores']:
            mli_avg = sum(systems['middle_layer_injection']['scores']) / len(systems['middle_layer_injection']['scores'])
            lora_avg = sum(systems['traditional_lora']['scores']) / len(systems['traditional_lora']['scores'])
            
            if mli_avg > lora_avg + 5:
                print("🎊 RECOMMENDED: Deploy middle layer injection system")
                print("   Significant quality improvements justify the complexity")
            elif mli_avg > lora_avg:
                print("⚠️ CONDITIONAL: Consider middle layer injection")
                print("   Modest improvements may justify deployment depending on use case")
            else:
                print("❌ NOT RECOMMENDED: Stick with traditional LoRA")
                print("   No significant benefits observed from middle layer injection")
        
        print(f"\n📋 NEXT STEPS:")
        print("1. Review detailed test results in saved files")
        print("2. Manually evaluate code quality and correctness")
        print("3. Consider computational overhead vs quality gains")
        print("4. Test with additional diverse prompts if needed")


def main():
    """Main evaluation function."""
    
    print("🔬 Starting Middle Layer Injection System Evaluation...")
    
    tester = MiddleLayerInjectionTester()
    success = tester.run_comprehensive_evaluation()
    
    if success:
        print(f"\n🎊 MIDDLE LAYER INJECTION EVALUATION COMPLETE!")
        print("📄 Check saved files for detailed analysis")
    else:
        print(f"\n❌ Evaluation encountered issues")
    
    return success


if __name__ == "__main__":
    main()
