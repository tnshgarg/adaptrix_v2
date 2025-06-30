#!/usr/bin/env python3
"""
🥊 CODE ADAPTER VS GEMINI COMPREHENSIVE COMPARISON

Thorough testing of Qwen3 (baseline), Qwen3 (code adapter), and Gemini
across diverse coding challenges with manual quality assessment.
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


class CodeComparisonTester:
    """Comprehensive code generation comparison tester."""
    
    def __init__(self):
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'models': ['Qwen3-1.7B (Baseline)', 'Qwen3-1.7B (Code Adapter)', 'Gemini-2.0-Flash'],
                'total_tests': 0
            },
            'test_results': []
        }
    
    def query_gemini(self, prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """Query Gemini API for code generation."""
        try:
            headers = {'Content-Type': 'application/json'}
            data = {
                'contents': [{'parts': [{'text': prompt}]}],
                'generationConfig': {
                    'maxOutputTokens': max_tokens,
                    'temperature': 0.3,  # Lower temperature for code
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
    
    def test_comprehensive_coding_challenges(self):
        """Run comprehensive coding challenges across all models."""
        
        print("🥊" * 100)
        print("🥊 COMPREHENSIVE CODE GENERATION COMPARISON 🥊")
        print("🥊" * 100)
        
        # Diverse coding challenges covering different aspects
        test_cases = [
            {
                "id": "basic_function",
                "category": "Basic Programming",
                "difficulty": "Easy",
                "prompt": "Write a Python function that takes a list of numbers and returns the sum of all even numbers in the list. Include error handling for invalid inputs.",
                "evaluation_criteria": [
                    "Correct function definition",
                    "Proper list iteration", 
                    "Even number detection logic",
                    "Error handling implementation",
                    "Code readability and comments"
                ]
            },
            {
                "id": "data_structure",
                "category": "Data Structures",
                "difficulty": "Medium",
                "prompt": "Implement a Stack class in Python with push, pop, peek, and is_empty methods. Include proper error handling for empty stack operations and demonstrate usage with examples.",
                "evaluation_criteria": [
                    "Complete class implementation",
                    "All required methods present",
                    "Proper error handling",
                    "Usage examples provided",
                    "Code organization and documentation"
                ]
            },
            {
                "id": "algorithm_sorting",
                "category": "Algorithms",
                "difficulty": "Medium",
                "prompt": "Implement the merge sort algorithm in Python. Include detailed comments explaining the divide-and-conquer approach and provide a test case with performance analysis.",
                "evaluation_criteria": [
                    "Correct merge sort implementation",
                    "Proper divide-and-conquer logic",
                    "Merge function implementation",
                    "Detailed comments and explanation",
                    "Test cases and performance discussion"
                ]
            },
            {
                "id": "web_development",
                "category": "Web Development",
                "difficulty": "Medium",
                "prompt": "Create a Flask web application with a REST API that manages a simple todo list. Include endpoints for GET (list todos), POST (add todo), PUT (update todo), and DELETE (remove todo). Include proper JSON handling and basic validation.",
                "evaluation_criteria": [
                    "Complete Flask application structure",
                    "All CRUD endpoints implemented",
                    "Proper HTTP methods usage",
                    "JSON request/response handling",
                    "Input validation and error responses"
                ]
            },
            {
                "id": "database_operations",
                "category": "Database",
                "difficulty": "Medium",
                "prompt": "Write Python code using SQLAlchemy to define a User model with relationships to a Post model (one-to-many). Include methods to create, read, update, and delete users and posts. Show how to query for all posts by a specific user.",
                "evaluation_criteria": [
                    "Proper SQLAlchemy model definitions",
                    "Correct relationship setup",
                    "CRUD operations implementation",
                    "Query examples provided",
                    "Database best practices"
                ]
            },
            {
                "id": "async_programming",
                "category": "Async Programming",
                "difficulty": "Hard",
                "prompt": "Create an asynchronous Python program that fetches data from multiple URLs concurrently using asyncio and aiohttp. Include proper error handling, timeout management, and demonstrate how to process the results efficiently.",
                "evaluation_criteria": [
                    "Correct async/await usage",
                    "Concurrent request handling",
                    "Proper error handling",
                    "Timeout management",
                    "Result processing efficiency"
                ]
            },
            {
                "id": "design_patterns",
                "category": "Design Patterns",
                "difficulty": "Hard",
                "prompt": "Implement the Observer design pattern in Python for a news publisher system. Create a NewsPublisher class that can notify multiple subscribers when news is published. Include different types of subscribers (EmailSubscriber, SMSSubscriber) and demonstrate the pattern in action.",
                "evaluation_criteria": [
                    "Correct Observer pattern implementation",
                    "Abstract base classes/interfaces",
                    "Multiple subscriber types",
                    "Proper notification mechanism",
                    "Demonstration of pattern usage"
                ]
            },
            {
                "id": "testing_debugging",
                "category": "Testing & Debugging",
                "difficulty": "Medium",
                "prompt": "Write a comprehensive test suite using pytest for a Calculator class that performs basic arithmetic operations. Include unit tests, edge cases, exception testing, and parametrized tests. Also include a simple debugging example showing how to identify and fix a common bug.",
                "evaluation_criteria": [
                    "Complete pytest test suite",
                    "Coverage of edge cases",
                    "Exception testing",
                    "Parametrized test usage",
                    "Debugging example and explanation"
                ]
            },
            {
                "id": "performance_optimization",
                "category": "Performance",
                "difficulty": "Hard",
                "prompt": "Write Python code that demonstrates performance optimization techniques. Include examples of: 1) Using list comprehensions vs loops, 2) Memory-efficient generators, 3) Caching with functools.lru_cache, 4) Profiling code performance. Provide before/after comparisons with timing measurements.",
                "evaluation_criteria": [
                    "Multiple optimization techniques shown",
                    "Before/after code comparisons",
                    "Performance measurement examples",
                    "Proper use of optimization tools",
                    "Clear explanations of improvements"
                ]
            },
            {
                "id": "real_world_problem",
                "category": "Real-World Application",
                "difficulty": "Hard",
                "prompt": "Create a Python script that processes a large CSV file containing sales data. The script should: 1) Read the CSV efficiently, 2) Calculate monthly sales totals by product category, 3) Identify top-performing products, 4) Generate a summary report, 5) Handle memory efficiently for large files. Include proper error handling and logging.",
                "evaluation_criteria": [
                    "Efficient CSV processing",
                    "Correct data aggregation",
                    "Memory-efficient handling",
                    "Comprehensive error handling",
                    "Professional logging and reporting"
                ]
            }
        ]
        
        try:
            from src.core.modular_engine import ModularAdaptrixEngine
            
            print("\n🚀 INITIALIZING QWEN3-1.7B ENGINE...")
            engine = ModularAdaptrixEngine(
                model_id="Qwen/Qwen3-1.7B",
                device="cpu",
                adapters_dir="adapters"
            )
            
            if not engine.initialize():
                print("❌ Failed to initialize Qwen3 engine")
                return False
            
            print("✅ Qwen3-1.7B engine initialized!")
            
            # Test each challenge
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n{'='*120}")
                print(f"🧪 TEST {i}: {test_case['id'].upper()} ({test_case['difficulty']})")
                print(f"📂 Category: {test_case['category']}")
                print(f"{'='*120}")
                print(f"📝 Challenge: {test_case['prompt']}")
                
                test_result = {
                    'test_id': test_case['id'],
                    'category': test_case['category'],
                    'difficulty': test_case['difficulty'],
                    'prompt': test_case['prompt'],
                    'evaluation_criteria': test_case['evaluation_criteria'],
                    'responses': {}
                }
                
                # Test 1: Qwen3 Baseline (No Adapter)
                print(f"\n🤖 TESTING: Qwen3-1.7B (Baseline)")
                print("-" * 60)
                
                start_time = time.time()
                baseline_response = engine.generate(
                    test_case['prompt'],
                    task_type="code",
                    max_length=800,
                    temperature=0.3
                )
                baseline_time = time.time() - start_time
                
                test_result['responses']['qwen3_baseline'] = {
                    'content': baseline_response,
                    'generation_time': baseline_time,
                    'length': len(baseline_response),
                    'model': 'Qwen3-1.7B (Baseline)'
                }
                
                print(f"⏱️ Generated in {baseline_time:.2f}s")
                print(f"📏 Length: {len(baseline_response)} characters")
                print(f"📄 Preview: {baseline_response[:200]}...")
                
                # Test 2: Qwen3 with Code Adapter
                print(f"\n🔌 TESTING: Qwen3-1.7B (Code Adapter)")
                print("-" * 60)
                
                if engine.load_adapter("code"):
                    start_time = time.time()
                    adapter_response = engine.generate(
                        test_case['prompt'],
                        task_type="code",
                        max_length=800,
                        temperature=0.3
                    )
                    adapter_time = time.time() - start_time
                    
                    test_result['responses']['qwen3_adapter'] = {
                        'content': adapter_response,
                        'generation_time': adapter_time,
                        'length': len(adapter_response),
                        'model': 'Qwen3-1.7B (Code Adapter)'
                    }
                    
                    print(f"⏱️ Generated in {adapter_time:.2f}s")
                    print(f"📏 Length: {len(adapter_response)} characters")
                    print(f"📄 Preview: {adapter_response[:200]}...")
                    
                    engine.unload_adapter("code")
                else:
                    print("❌ Failed to load code adapter")
                    test_result['responses']['qwen3_adapter'] = {
                        'content': "Error: Failed to load code adapter",
                        'generation_time': 0,
                        'length': 0,
                        'model': 'Qwen3-1.7B (Code Adapter)'
                    }
                
                # Test 3: Gemini
                print(f"\n🧠 TESTING: Gemini-2.0-Flash")
                print("-" * 60)
                
                gemini_result = self.query_gemini(test_case['prompt'], max_tokens=800)
                
                test_result['responses']['gemini'] = {
                    'content': gemini_result['content'],
                    'generation_time': gemini_result['generation_time'],
                    'length': gemini_result['length'],
                    'model': 'Gemini-2.0-Flash',
                    'success': gemini_result['success']
                }
                
                print(f"⏱️ Generated in {gemini_result['generation_time']:.2f}s")
                print(f"📏 Length: {gemini_result['length']} characters")
                print(f"📄 Preview: {gemini_result['content'][:200]}...")
                
                # Add to results
                self.results['test_results'].append(test_result)
                
                # Brief comparison
                print(f"\n📊 QUICK COMPARISON:")
                print(f"   Qwen3 Baseline: {baseline_time:.1f}s, {len(baseline_response)} chars")
                print(f"   Qwen3 + Adapter: {adapter_time:.1f}s, {len(adapter_response)} chars")
                print(f"   Gemini: {gemini_result['generation_time']:.1f}s, {gemini_result['length']} chars")
            
            # Update metadata
            self.results['metadata']['total_tests'] = len(test_cases)
            
            # Save all responses to file
            self.save_results()
            
            # Generate analysis
            self.analyze_results()
            
            # Cleanup
            engine.cleanup()
            
            return True
            
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_results(self):
        """Save all test results to files for manual review."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results as JSON
        json_filename = f"code_comparison_results_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Save human-readable format
        readable_filename = f"code_comparison_readable_{timestamp}.txt"
        with open(readable_filename, 'w', encoding='utf-8') as f:
            f.write("🥊 CODE GENERATION COMPARISON RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            for i, test in enumerate(self.results['test_results'], 1):
                f.write(f"TEST {i}: {test['test_id'].upper()}\n")
                f.write(f"Category: {test['category']} | Difficulty: {test['difficulty']}\n")
                f.write(f"Prompt: {test['prompt']}\n")
                f.write("-" * 80 + "\n\n")
                
                for model_key, response in test['responses'].items():
                    f.write(f"🤖 {response['model']}\n")
                    f.write(f"Time: {response['generation_time']:.2f}s | Length: {response['length']} chars\n")
                    f.write("Response:\n")
                    f.write(response['content'])
                    f.write("\n\n" + "="*40 + "\n\n")
                
                f.write("\n" + "="*120 + "\n\n")
        
        print(f"\n📄 Results saved:")
        print(f"   JSON: {json_filename}")
        print(f"   Readable: {readable_filename}")
    
    def analyze_results(self):
        """Perform comprehensive analysis of results."""
        
        print(f"\n📊 COMPREHENSIVE ANALYSIS")
        print("=" * 80)
        
        if not self.results['test_results']:
            print("❌ No results to analyze")
            return
        
        # Performance metrics
        total_tests = len(self.results['test_results'])
        
        # Calculate average metrics for each model
        models = ['qwen3_baseline', 'qwen3_adapter', 'gemini']
        model_names = ['Qwen3 Baseline', 'Qwen3 + Code Adapter', 'Gemini']
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print("-" * 50)
        
        for model_key, model_name in zip(models, model_names):
            times = []
            lengths = []
            successes = 0
            
            for test in self.results['test_results']:
                if model_key in test['responses']:
                    response = test['responses'][model_key]
                    times.append(response['generation_time'])
                    lengths.append(response['length'])
                    if response.get('success', True) and len(response['content']) > 50:
                        successes += 1
            
            if times:
                avg_time = sum(times) / len(times)
                avg_length = sum(lengths) / len(lengths)
                success_rate = (successes / len(times)) * 100
                
                print(f"\n🤖 {model_name}:")
                print(f"   Average Time: {avg_time:.2f}s")
                print(f"   Average Length: {avg_length:.0f} characters")
                print(f"   Success Rate: {success_rate:.1f}%")
        
        # Category analysis
        print(f"\n📂 PERFORMANCE BY CATEGORY:")
        print("-" * 50)
        
        categories = {}
        for test in self.results['test_results']:
            category = test['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(test)
        
        for category, tests in categories.items():
            print(f"\n📁 {category} ({len(tests)} tests):")
            for model_key, model_name in zip(models, model_names):
                avg_length = 0
                count = 0
                for test in tests:
                    if model_key in test['responses']:
                        avg_length += test['responses'][model_key]['length']
                        count += 1
                if count > 0:
                    avg_length /= count
                    print(f"   {model_name}: {avg_length:.0f} chars avg")
        
        # Manual evaluation guidelines
        print(f"\n📋 MANUAL EVALUATION GUIDELINES:")
        print("-" * 50)
        print("For each test, evaluate responses on:")
        print("1. ✅ Correctness - Does the code work as intended?")
        print("2. 🏗️ Completeness - Are all requirements addressed?")
        print("3. 📚 Code Quality - Is it well-structured and readable?")
        print("4. 🔧 Best Practices - Does it follow coding standards?")
        print("5. 📖 Documentation - Are comments and explanations clear?")
        print("6. 🚨 Error Handling - Are edge cases properly handled?")
        print("7. 🎯 Efficiency - Is the solution optimized appropriately?")
        
        print(f"\n🎯 NEXT STEPS:")
        print("1. Review saved files for detailed manual analysis")
        print("2. Test generated code for functionality")
        print("3. Compare code quality and best practices")
        print("4. Evaluate documentation and explanations")


def main():
    """Main testing function."""
    
    tester = CodeComparisonTester()
    
    print("🎯 Starting comprehensive code generation comparison...")
    
    success = tester.test_comprehensive_coding_challenges()
    
    if success:
        print(f"\n🎊 COMPREHENSIVE TESTING COMPLETE!")
        print("📄 Check saved files for detailed manual review")
    else:
        print(f"\n❌ Testing encountered issues")
    
    return success


if __name__ == "__main__":
    main()
