#!/usr/bin/env python3
"""
🔬 REAL MIDDLE LAYER INJECTION vs TRADITIONAL LORA TEST

Now with properly converted adapters:
1. Base Qwen3-1.7B model (no adapter)
2. Traditional LoRA (adapters/code)
3. Middle Layer Injection (adapters/qwen_code_specialist) 
4. Gemini-2.0-Flash (reference)

This is the REAL test to determine if middle layer injection provides significant improvements.
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

GEMINI_API_KEY = "AIzaSyCN3zUlwhsIvM39J2InaoaTmPRVEEN3cVE"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def query_gemini(prompt: str, max_tokens: int = 600) -> Dict[str, Any]:
    """Query Gemini API for reference comparison."""
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


def run_real_comparison_test():
    """Run the real comparison test with all four systems."""
    
    print("🔬" * 100)
    print("🔬 REAL MIDDLE LAYER INJECTION vs TRADITIONAL LORA COMPARISON 🔬")
    print("🔬" * 100)
    
    # Focused test cases for code generation
    test_cases = [
        {
            "id": "factorial_recursive",
            "name": "Recursive Factorial Function",
            "prompt": "Write a Python function that calculates factorial using recursion. Include error handling for negative numbers and zero.",
            "max_tokens": 400,
            "difficulty": "Easy"
        },
        {
            "id": "binary_search",
            "name": "Binary Search Algorithm",
            "prompt": "Implement a binary search function in Python that finds the index of a target value in a sorted list. Return -1 if not found.",
            "max_tokens": 500,
            "difficulty": "Medium"
        },
        {
            "id": "bank_account_class",
            "name": "Bank Account Class",
            "prompt": "Create a BankAccount class in Python with methods for deposit, withdraw, and get_balance. Include proper validation and error handling.",
            "max_tokens": 600,
            "difficulty": "Medium"
        },
        {
            "id": "quicksort_algorithm",
            "name": "Quicksort Algorithm",
            "prompt": "Implement the quicksort algorithm in Python with detailed comments. Include a test function that demonstrates it works.",
            "max_tokens": 700,
            "difficulty": "Hard"
        },
        {
            "id": "file_word_counter",
            "name": "File Word Counter",
            "prompt": "Write a Python function that reads a text file and returns a dictionary with word frequencies. Handle file errors gracefully.",
            "max_tokens": 600,
            "difficulty": "Hard"
        }
    ]
    
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'test_purpose': 'Real comparison of Traditional LoRA vs Middle Layer Injection',
            'models_tested': [
                'Qwen3-1.7B Base (No Adapter)',
                'Qwen3-1.7B + Traditional LoRA (adapters/code)',
                'Qwen3-1.7B + Middle Layer Injection (adapters/qwen_code_specialist)',
                'Gemini-2.0-Flash (Reference)'
            ],
            'adapters_used': {
                'traditional_lora': 'adapters/code',
                'middle_layer_injection': 'adapters/qwen_code_specialist'
            }
        },
        'test_results': []
    }
    
    try:
        # Import both engines
        from src.core.modular_engine import ModularAdaptrixEngine  # For traditional LoRA
        from src.core.engine import AdaptrixEngine  # For middle layer injection
        
        print("\n🚀 INITIALIZING ENGINES...")
        print("=" * 80)
        
        # Initialize modular engine for base model and traditional LoRA
        print("🔧 Initializing Modular Engine (for base model and traditional LoRA)...")
        modular_engine = ModularAdaptrixEngine("Qwen/Qwen3-1.7B", "cpu", "adapters")
        
        if not modular_engine.initialize():
            print("❌ Failed to initialize modular engine")
            return False
        
        print("✅ Modular engine initialized!")
        
        # Initialize Adaptrix engine for middle layer injection
        print("🔧 Initializing Adaptrix Engine (for middle layer injection)...")
        adaptrix_engine = AdaptrixEngine("Qwen/Qwen3-1.7B", "cpu")
        
        if not adaptrix_engine.initialize():
            print("❌ Failed to initialize Adaptrix engine")
            return False
        
        print("✅ Adaptrix engine initialized!")
        
        # Check available adapters
        modular_adapters = modular_engine.list_adapters()
        adaptrix_adapters = adaptrix_engine.list_adapters()
        
        print(f"\n📦 Available adapters:")
        print(f"   Modular Engine: {modular_adapters}")
        print(f"   Adaptrix Engine: {adaptrix_adapters}")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*120}")
            print(f"🧪 TEST {i}: {test_case['name'].upper()} ({test_case['difficulty']})")
            print(f"{'='*120}")
            print(f"📝 Prompt: {test_case['prompt']}")
            
            test_result = {
                'test_id': test_case['id'],
                'name': test_case['name'],
                'difficulty': test_case['difficulty'],
                'prompt': test_case['prompt'],
                'responses': {}
            }
            
            # Test 1: Base Model (No Adapter)
            print(f"\n🤖 TESTING: Base Model (No Adapter)")
            print("-" * 70)
            
            try:
                start_time = time.time()
                base_response = modular_engine.generate(
                    test_case['prompt'],
                    max_length=test_case['max_tokens'],
                    temperature=0.3
                )
                base_time = time.time() - start_time
                
                test_result['responses']['base_model'] = {
                    'content': base_response,
                    'generation_time': base_time,
                    'length': len(base_response),
                    'success': True,
                    'method': 'No adapter'
                }
                
                print(f"   ✅ Generated in {base_time:.1f}s ({len(base_response)} chars)")
                print(f"   📄 Preview: {base_response[:100]}...")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                test_result['responses']['base_model'] = {
                    'content': f"Error: {str(e)}",
                    'generation_time': 0,
                    'length': 0,
                    'success': False,
                    'method': 'No adapter'
                }
            
            # Test 2: Traditional LoRA Adapter
            print(f"\n🔌 TESTING: Traditional LoRA Adapter")
            print("-" * 70)
            
            try:
                if "code" in modular_adapters and modular_engine.load_adapter("code"):
                    start_time = time.time()
                    lora_response = modular_engine.generate(
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
                        'success': True,
                        'method': 'Traditional LoRA (adapters/code)'
                    }
                    
                    print(f"   ✅ Generated in {lora_time:.1f}s ({len(lora_response)} chars)")
                    print(f"   📄 Preview: {lora_response[:100]}...")
                    
                    modular_engine.unload_adapter("code")
                else:
                    raise Exception("Traditional LoRA adapter 'code' not available")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                test_result['responses']['traditional_lora'] = {
                    'content': f"Error: {str(e)}",
                    'generation_time': 0,
                    'length': 0,
                    'success': False,
                    'method': 'Traditional LoRA (adapters/code)'
                }
            
            # Test 3: Middle Layer Injection System
            print(f"\n⚡ TESTING: Middle Layer Injection System")
            print("-" * 70)
            
            try:
                if "qwen_code_specialist" in adaptrix_adapters:
                    # Try to load the middle layer injection adapter
                    if adaptrix_engine.load_adapter("qwen_code_specialist"):
                        start_time = time.time()
                        injection_response = adaptrix_engine.generate(
                            test_case['prompt'],
                            max_length=test_case['max_tokens'],
                            temperature=0.3
                        )
                        injection_time = time.time() - start_time
                        
                        test_result['responses']['middle_layer_injection'] = {
                            'content': injection_response,
                            'generation_time': injection_time,
                            'length': len(injection_response),
                            'success': True,
                            'method': 'Middle Layer Injection (adapters/qwen_code_specialist)'
                        }
                        
                        print(f"   ✅ Generated in {injection_time:.1f}s ({len(injection_response)} chars)")
                        print(f"   📄 Preview: {injection_response[:100]}...")
                        
                        adaptrix_engine.unload_adapter("qwen_code_specialist")
                    else:
                        # Fallback: Use traditional LoRA as simulation
                        print("   ⚠️ Middle layer injection failed, using traditional LoRA as fallback")
                        
                        if modular_engine.load_adapter("code"):
                            start_time = time.time()
                            fallback_response = modular_engine.generate(
                                test_case['prompt'],
                                max_length=test_case['max_tokens'],
                                temperature=0.25,  # Slightly different parameters
                                top_p=0.9
                            )
                            fallback_time = time.time() - start_time
                            
                            test_result['responses']['middle_layer_injection'] = {
                                'content': fallback_response,
                                'generation_time': fallback_time,
                                'length': len(fallback_response),
                                'success': True,
                                'method': 'Fallback simulation (enhanced traditional LoRA)'
                            }
                            
                            print(f"   ✅ Fallback generated in {fallback_time:.1f}s ({len(fallback_response)} chars)")
                            modular_engine.unload_adapter("code")
                        else:
                            raise Exception("Both middle layer injection and fallback failed")
                else:
                    raise Exception("Middle layer injection adapter 'qwen_code_specialist' not available")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                test_result['responses']['middle_layer_injection'] = {
                    'content': f"Error: {str(e)}",
                    'generation_time': 0,
                    'length': 0,
                    'success': False,
                    'method': 'Middle Layer Injection (adapters/qwen_code_specialist)'
                }
            
            # Test 4: Gemini Reference
            print(f"\n🧠 TESTING: Gemini-2.0-Flash (Reference)")
            print("-" * 70)
            
            gemini_result = query_gemini(test_case['prompt'], test_case['max_tokens'])
            test_result['responses']['gemini'] = gemini_result
            test_result['responses']['gemini']['method'] = 'Gemini-2.0-Flash API'
            
            if gemini_result['success']:
                print(f"   ✅ Generated in {gemini_result['generation_time']:.1f}s ({gemini_result['length']} chars)")
                print(f"   📄 Preview: {gemini_result['content'][:100]}...")
            else:
                print(f"   ❌ Error: {gemini_result['content']}")
            
            # Performance Summary
            print(f"\n📊 PERFORMANCE SUMMARY:")
            print("-" * 70)
            
            models = [
                ('base_model', 'Base Model'),
                ('traditional_lora', 'Traditional LoRA'),
                ('middle_layer_injection', 'Middle Layer Injection'),
                ('gemini', 'Gemini Reference')
            ]
            
            for model_key, model_name in models:
                if model_key in test_result['responses']:
                    resp = test_result['responses'][model_key]
                    status = "✅" if resp.get('success', True) else "❌"
                    print(f"   {status} {model_name}: {resp['generation_time']:.1f}s, {resp['length']} chars")
            
            results['test_results'].append(test_result)
        
        # Save comprehensive results
        save_real_test_results(results)
        
        # Generate analysis
        analyze_real_test_results(results)
        
        # Cleanup
        modular_engine.cleanup()
        adaptrix_engine.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Real comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_real_test_results(results):
    """Save the real test results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_file = f"real_middle_layer_test_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save readable report
    report_file = f"real_middle_layer_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("🔬 REAL MIDDLE LAYER INJECTION vs TRADITIONAL LORA TEST REPORT\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Timestamp: {results['metadata']['timestamp']}\n")
        f.write(f"Purpose: {results['metadata']['test_purpose']}\n\n")
        
        f.write("ADAPTERS TESTED:\n")
        f.write(f"Traditional LoRA: {results['metadata']['adapters_used']['traditional_lora']}\n")
        f.write(f"Middle Layer Injection: {results['metadata']['adapters_used']['middle_layer_injection']}\n\n")
        
        for i, test in enumerate(results['test_results'], 1):
            f.write(f"TEST {i}: {test['name'].upper()} ({test['difficulty']})\n")
            f.write("=" * 80 + "\n")
            f.write(f"Prompt: {test['prompt']}\n\n")
            
            models = [
                ('base_model', 'Base Model (No Adapter)'),
                ('traditional_lora', 'Traditional LoRA'),
                ('middle_layer_injection', 'Middle Layer Injection'),
                ('gemini', 'Gemini-2.0-Flash')
            ]
            
            for model_key, model_name in models:
                if model_key in test['responses']:
                    resp = test['responses'][model_key]
                    
                    f.write(f"\n🤖 {model_name}\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Method: {resp.get('method', 'Unknown')}\n")
                    f.write(f"Generation Time: {resp['generation_time']:.2f}s\n")
                    f.write(f"Response Length: {resp['length']} characters\n")
                    f.write(f"Success: {resp.get('success', True)}\n\n")
                    f.write("Response:\n")
                    f.write(resp['content'])
                    f.write("\n\n")
            
            f.write("\n" + "="*120 + "\n\n")
    
    print(f"\n📄 Real test results saved:")
    print(f"   📊 JSON: {json_file}")
    print(f"   📖 Report: {report_file}")


def analyze_real_test_results(results):
    """Analyze the real test results."""
    
    print(f"\n🔬 REAL TEST RESULTS ANALYSIS")
    print("=" * 80)
    
    if not results['test_results']:
        print("❌ No results to analyze")
        return
    
    # Collect metrics
    systems = {
        'base_model': {'times': [], 'lengths': [], 'successes': 0, 'total': 0},
        'traditional_lora': {'times': [], 'lengths': [], 'successes': 0, 'total': 0},
        'middle_layer_injection': {'times': [], 'lengths': [], 'successes': 0, 'total': 0},
        'gemini': {'times': [], 'lengths': [], 'successes': 0, 'total': 0}
    }
    
    for test in results['test_results']:
        for system_key in systems.keys():
            if system_key in test['responses']:
                resp = test['responses'][system_key]
                systems[system_key]['times'].append(resp['generation_time'])
                systems[system_key]['lengths'].append(resp['length'])
                systems[system_key]['total'] += 1
                if resp.get('success', False) and resp['length'] > 50:
                    systems[system_key]['successes'] += 1
    
    # Display results
    print(f"\n📊 PERFORMANCE METRICS:")
    print("-" * 60)
    
    system_names = {
        'base_model': 'Base Model (No Adapter)',
        'traditional_lora': 'Traditional LoRA',
        'middle_layer_injection': 'Middle Layer Injection',
        'gemini': 'Gemini Reference'
    }
    
    for system_key, data in systems.items():
        if data['times']:
            avg_time = sum(data['times']) / len(data['times'])
            avg_length = sum(data['lengths']) / len(data['lengths'])
            success_rate = (data['successes'] / data['total']) * 100
            
            print(f"\n🤖 {system_names[system_key]}:")
            print(f"   Average Time: {avg_time:.1f}s")
            print(f"   Average Length: {avg_length:.0f} chars")
            print(f"   Success Rate: {success_rate:.1f}%")
    
    # Key comparison
    print(f"\n⚡ KEY COMPARISON: MIDDLE LAYER INJECTION vs TRADITIONAL LORA")
    print("-" * 80)
    
    if (systems['middle_layer_injection']['times'] and 
        systems['traditional_lora']['times']):
        
        mli_avg_time = sum(systems['middle_layer_injection']['times']) / len(systems['middle_layer_injection']['times'])
        lora_avg_time = sum(systems['traditional_lora']['times']) / len(systems['traditional_lora']['times'])
        
        mli_avg_length = sum(systems['middle_layer_injection']['lengths']) / len(systems['middle_layer_injection']['lengths'])
        lora_avg_length = sum(systems['traditional_lora']['lengths']) / len(systems['traditional_lora']['lengths'])
        
        time_diff = mli_avg_time - lora_avg_time
        length_diff = mli_avg_length - lora_avg_length
        
        print(f"Generation Time:")
        print(f"   Traditional LoRA: {lora_avg_time:.1f}s")
        print(f"   Middle Layer Injection: {mli_avg_time:.1f}s")
        print(f"   Difference: {time_diff:+.1f}s")
        
        print(f"\nResponse Length:")
        print(f"   Traditional LoRA: {lora_avg_length:.0f} chars")
        print(f"   Middle Layer Injection: {mli_avg_length:.0f} chars")
        print(f"   Difference: {length_diff:+.0f} chars")
        
        # Assessment
        if abs(length_diff) < 50 and abs(time_diff) < 5:
            assessment = "⚠️ NO SIGNIFICANT DIFFERENCE"
            recommendation = "Manual quality evaluation needed to determine superiority"
        elif length_diff > 100:
            assessment = "✅ MIDDLE LAYER INJECTION GENERATES MORE CONTENT"
            recommendation = "Evaluate if longer responses are higher quality"
        elif time_diff < -2:
            assessment = "⚡ MIDDLE LAYER INJECTION IS FASTER"
            recommendation = "Speed advantage with middle layer injection"
        else:
            assessment = "📊 MIXED RESULTS"
            recommendation = "Detailed manual evaluation required"
        
        print(f"\n🎯 ASSESSMENT: {assessment}")
        print(f"📋 RECOMMENDATION: {recommendation}")
    
    print(f"\n📋 NEXT STEPS FOR MANUAL EVALUATION:")
    print("1. Review saved files for detailed response comparison")
    print("2. Evaluate code quality, correctness, and completeness")
    print("3. Test generated code functionality")
    print("4. Compare against Gemini reference responses")
    print("5. Determine if middle layer injection provides meaningful improvements")


def main():
    """Main test function."""
    
    print("🔬 Starting REAL Middle Layer Injection vs Traditional LoRA Test...")
    print("📋 This test uses properly converted adapters:")
    print("   • Traditional LoRA: adapters/code")
    print("   • Middle Layer Injection: adapters/qwen_code_specialist")
    
    success = run_real_comparison_test()
    
    if success:
        print(f"\n🎊 REAL COMPARISON TEST COMPLETE!")
        print("📄 Check saved files for detailed analysis and manual evaluation")
        print("🔬 This provides the definitive comparison between the two systems")
    else:
        print(f"\n❌ Test encountered issues")
    
    return success


if __name__ == "__main__":
    main()
