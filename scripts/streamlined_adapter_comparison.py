#!/usr/bin/env python3
"""
🎯 STREAMLINED ADAPTER COMPARISON TEST

Focused comparison of:
1. Qwen3-1.7B Base Model
2. Qwen3-1.7B + Code Adapter
3. Gemini-2.0-Flash

Generates comprehensive report with manual evaluation guidelines.
"""

import sys
import os
import json
import time
import requests
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

GEMINI_API_KEY = "AIzaSyCN3zUlwhsIvM39J2InaoaTmPRVEEN3cVE"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def query_gemini(prompt: str, max_tokens: int = 600) -> dict:
    """Query Gemini API."""
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


def run_streamlined_comparison():
    """Run streamlined comparison test."""
    
    print("🎯" * 80)
    print("🎯 STREAMLINED ADAPTER COMPARISON TEST 🎯")
    print("🎯" * 80)
    
    # Focused test cases
    test_cases = [
        {
            "id": "factorial_function",
            "name": "Factorial Function",
            "prompt": "Write a Python function that calculates factorial using recursion. Include error handling for negative numbers.",
            "max_tokens": 400
        },
        {
            "id": "binary_search",
            "name": "Binary Search Algorithm",
            "prompt": "Implement binary search in Python. Return the index of target value or -1 if not found.",
            "max_tokens": 400
        },
        {
            "id": "class_design",
            "name": "Class Design",
            "prompt": "Create a BankAccount class with deposit, withdraw, and get_balance methods. Include validation.",
            "max_tokens": 500
        }
    ]
    
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'models': ['Qwen3-1.7B Base', 'Qwen3-1.7B + Code Adapter', 'Gemini-2.0-Flash'],
            'total_tests': len(test_cases)
        },
        'test_results': []
    }
    
    try:
        from src.core.modular_engine import ModularAdaptrixEngine
        
        print("\n🚀 Initializing Qwen3-1.7B...")
        engine = ModularAdaptrixEngine("Qwen/Qwen3-1.7B", "cpu", "adapters")
        
        if not engine.initialize():
            print("❌ Failed to initialize engine")
            return False
        
        print("✅ Engine initialized!")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*80}")
            print(f"🧪 TEST {i}: {test_case['name']}")
            print(f"{'='*80}")
            print(f"📝 Prompt: {test_case['prompt']}")
            
            test_result = {
                'test_id': test_case['id'],
                'name': test_case['name'],
                'prompt': test_case['prompt'],
                'responses': {}
            }
            
            # Test 1: Base Model
            print(f"\n🤖 Testing Base Model...")
            try:
                start_time = time.time()
                base_response = engine.generate(
                    test_case['prompt'],
                    max_length=test_case['max_tokens'],
                    temperature=0.3
                )
                base_time = time.time() - start_time
                
                test_result['responses']['base_model'] = {
                    'content': base_response,
                    'generation_time': base_time,
                    'length': len(base_response),
                    'success': True
                }
                
                print(f"   ✅ Generated in {base_time:.1f}s ({len(base_response)} chars)")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                test_result['responses']['base_model'] = {
                    'content': f"Error: {str(e)}",
                    'generation_time': 0,
                    'length': 0,
                    'success': False
                }
            
            # Test 2: Code Adapter
            print(f"\n🔌 Testing Code Adapter...")
            try:
                if engine.load_adapter("code"):
                    start_time = time.time()
                    adapter_response = engine.generate(
                        test_case['prompt'],
                        max_length=test_case['max_tokens'],
                        temperature=0.3
                    )
                    adapter_time = time.time() - start_time
                    
                    test_result['responses']['code_adapter'] = {
                        'content': adapter_response,
                        'generation_time': adapter_time,
                        'length': len(adapter_response),
                        'success': True
                    }
                    
                    print(f"   ✅ Generated in {adapter_time:.1f}s ({len(adapter_response)} chars)")
                    engine.unload_adapter("code")
                    
                else:
                    raise Exception("Failed to load code adapter")
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                test_result['responses']['code_adapter'] = {
                    'content': f"Error: {str(e)}",
                    'generation_time': 0,
                    'length': 0,
                    'success': False
                }
            
            # Test 3: Gemini
            print(f"\n🧠 Testing Gemini...")
            gemini_result = query_gemini(test_case['prompt'], test_case['max_tokens'])
            test_result['responses']['gemini'] = gemini_result
            
            if gemini_result['success']:
                print(f"   ✅ Generated in {gemini_result['generation_time']:.1f}s ({gemini_result['length']} chars)")
            else:
                print(f"   ❌ Error: {gemini_result['content']}")
            
            # Quick comparison
            print(f"\n📊 Quick Comparison:")
            for model_key, model_name in [('base_model', 'Base Model'), 
                                         ('code_adapter', 'Code Adapter'), 
                                         ('gemini', 'Gemini')]:
                if model_key in test_result['responses']:
                    resp = test_result['responses'][model_key]
                    status = "✅" if resp.get('success', True) else "❌"
                    print(f"   {status} {model_name}: {resp['generation_time']:.1f}s, {resp['length']} chars")
            
            results['test_results'].append(test_result)
        
        # Save results
        save_results(results)
        
        # Generate analysis
        analyze_results(results)
        
        engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_results(results):
    """Save results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_file = f"adapter_comparison_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save readable format
    readable_file = f"adapter_comparison_{timestamp}.txt"
    with open(readable_file, 'w', encoding='utf-8') as f:
        f.write("🎯 ADAPTER COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {results['metadata']['timestamp']}\n")
        f.write(f"Models: {', '.join(results['metadata']['models'])}\n")
        f.write(f"Total Tests: {results['metadata']['total_tests']}\n\n")
        
        for i, test in enumerate(results['test_results'], 1):
            f.write(f"TEST {i}: {test['name'].upper()}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Prompt: {test['prompt']}\n\n")
            
            for model_key, model_name in [('base_model', 'Qwen3-1.7B Base'), 
                                         ('code_adapter', 'Qwen3-1.7B + Code Adapter'), 
                                         ('gemini', 'Gemini-2.0-Flash')]:
                if model_key in test['responses']:
                    resp = test['responses'][model_key]
                    f.write(f"\n🤖 {model_name}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Time: {resp['generation_time']:.2f}s | Length: {resp['length']} chars | Success: {resp.get('success', True)}\n\n")
                    f.write("Response:\n")
                    f.write(resp['content'])
                    f.write("\n\n")
            
            f.write("\n" + "="*100 + "\n\n")
    
    print(f"\n📄 Results saved:")
    print(f"   📊 JSON: {json_file}")
    print(f"   📖 Readable: {readable_file}")


def analyze_results(results):
    """Analyze and summarize results."""
    
    print(f"\n📊 COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    if not results['test_results']:
        print("❌ No results to analyze")
        return
    
    # Performance metrics
    models = [
        ('base_model', 'Base Model'),
        ('code_adapter', 'Code Adapter'), 
        ('gemini', 'Gemini')
    ]
    
    print(f"\n📈 PERFORMANCE SUMMARY:")
    print("-" * 40)
    
    for model_key, model_name in models:
        times = []
        lengths = []
        successes = 0
        total = 0
        
        for test in results['test_results']:
            if model_key in test['responses']:
                resp = test['responses'][model_key]
                times.append(resp['generation_time'])
                lengths.append(resp['length'])
                total += 1
                if resp.get('success', True) and resp['length'] > 50:
                    successes += 1
        
        if total > 0:
            avg_time = sum(times) / len(times)
            avg_length = sum(lengths) / len(lengths)
            success_rate = (successes / total) * 100
            
            print(f"\n🤖 {model_name}:")
            print(f"   Avg Time: {avg_time:.1f}s")
            print(f"   Avg Length: {avg_length:.0f} chars")
            print(f"   Success Rate: {success_rate:.1f}%")
    
    print(f"\n🎯 MANUAL EVALUATION GUIDELINES:")
    print("-" * 40)
    print("For each response, evaluate on a scale of 1-10:")
    print("1. ✅ Code Correctness - Does the code work as intended?")
    print("2. 🏗️ Code Completeness - Are all requirements addressed?")
    print("3. 📚 Code Quality - Is it well-structured and readable?")
    print("4. 🔧 Best Practices - Does it follow Python conventions?")
    print("5. 📖 Documentation - Clear comments and explanations?")
    print("6. 🚨 Error Handling - Proper exception handling?")
    print("7. 🎯 Efficiency - Is the solution appropriately optimized?")
    
    print(f"\n📋 EVALUATION TEMPLATE:")
    print("-" * 40)
    print("Test: [Test Name]")
    print("Model: [Model Name]")
    print("Correctness: [1-10]")
    print("Completeness: [1-10]")
    print("Quality: [1-10]")
    print("Best Practices: [1-10]")
    print("Documentation: [1-10]")
    print("Error Handling: [1-10]")
    print("Efficiency: [1-10]")
    print("Overall Score: [1-10]")
    print("Notes: [Additional comments]")
    
    print(f"\n📊 COMPARISON FRAMEWORK:")
    print("-" * 40)
    print("Compare models across:")
    print("• Response quality and accuracy")
    print("• Code structure and organization")
    print("• Adherence to requirements")
    print("• Professional coding standards")
    print("• Explanation clarity")
    print("• Error handling robustness")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Review saved files for detailed comparison")
    print("2. Test generated code functionality")
    print("3. Score each response using evaluation guidelines")
    print("4. Calculate average scores per model")
    print("5. Determine which model performs best overall")


def main():
    """Main function."""
    print("🎯 Starting streamlined adapter comparison...")
    
    success = run_streamlined_comparison()
    
    if success:
        print(f"\n🎊 COMPARISON TEST COMPLETE!")
        print("📄 Check saved files for detailed manual evaluation")
    else:
        print(f"\n❌ Test encountered issues")


if __name__ == "__main__":
    main()
