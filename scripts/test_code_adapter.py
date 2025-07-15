#!/usr/bin/env python3
"""
🧪 CODE ADAPTER COMPREHENSIVE TESTING

Tests the new "code" adapter in the adapters folder with Qwen3-1.7B model.
Performs detailed evaluation of code generation capabilities.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def inspect_code_adapter():
    """Inspect the code adapter configuration."""
    
    print("🔍 INSPECTING CODE ADAPTER")
    print("=" * 60)
    
    adapter_path = Path("adapters/code")
    
    if not adapter_path.exists():
        print(f"❌ Code adapter not found at {adapter_path}")
        return None
    
    # Check adapter files
    config_file = adapter_path / "adapter_config.json"
    metadata_file = adapter_path / "adaptrix_metadata.json"
    model_file = adapter_path / "adapter_model.safetensors"
    
    print(f"📁 Adapter directory: {adapter_path}")
    print(f"   Config file: {'✅' if config_file.exists() else '❌'} {config_file}")
    print(f"   Metadata file: {'✅' if metadata_file.exists() else '❌'} {metadata_file}")
    print(f"   Model file: {'✅' if model_file.exists() else '❌'} {model_file}")
    
    adapter_info = {}
    
    # Load configuration
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            print(f"\n📋 ADAPTER CONFIGURATION:")
            print(f"   Base Model: {config.get('base_model_name_or_path', 'Unknown')}")
            print(f"   PEFT Type: {config.get('peft_type', 'Unknown')}")
            print(f"   Rank (r): {config.get('r', 'Unknown')}")
            print(f"   Alpha: {config.get('lora_alpha', 'Unknown')}")
            print(f"   Target Modules: {config.get('target_modules', [])}")
            print(f"   Dropout: {config.get('lora_dropout', 'Unknown')}")
            print(f"   Task Type: {config.get('task_type', 'Unknown')}")
            
            adapter_info['config'] = config
            
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
    
    # Load metadata
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print(f"\n📊 ADAPTER METADATA:")
            print(f"   Description: {metadata.get('description', 'No description')}")
            print(f"   Domain: {metadata.get('domain', 'Unknown')}")
            print(f"   Capabilities: {metadata.get('capabilities', [])}")
            print(f"   Training Dataset: {metadata.get('training_dataset', 'Unknown')}")
            
            adapter_info['metadata'] = metadata
            
        except Exception as e:
            print(f"❌ Failed to load metadata: {e}")
    
    return adapter_info


def test_code_generation_tasks():
    """Test various code generation tasks with the code adapter."""
    
    print("\n🧪 CODE GENERATION TESTING")
    print("=" * 60)
    
    # Comprehensive test cases for code generation
    test_cases = [
        {
            "name": "Simple Function",
            "prompt": "Write a Python function to calculate the factorial of a number",
            "expected_elements": ["def", "factorial", "return", "if", "else"],
            "difficulty": "Easy"
        },
        {
            "name": "Data Structure",
            "prompt": "Create a Python class for a binary search tree with insert and search methods",
            "expected_elements": ["class", "def __init__", "def insert", "def search", "self"],
            "difficulty": "Medium"
        },
        {
            "name": "Algorithm Implementation",
            "prompt": "Implement quicksort algorithm in Python with comments explaining each step",
            "expected_elements": ["def quicksort", "partition", "pivot", "recursion", "#"],
            "difficulty": "Medium"
        },
        {
            "name": "Web API",
            "prompt": "Create a Flask REST API endpoint that accepts JSON data and returns a response",
            "expected_elements": ["from flask", "@app.route", "request.json", "jsonify"],
            "difficulty": "Medium"
        },
        {
            "name": "Database Query",
            "prompt": "Write a SQL query to find the top 5 customers by total order value",
            "expected_elements": ["SELECT", "JOIN", "GROUP BY", "ORDER BY", "LIMIT"],
            "difficulty": "Easy"
        },
        {
            "name": "Error Handling",
            "prompt": "Write a Python function that reads a file and handles potential errors gracefully",
            "expected_elements": ["try:", "except:", "with open", "return"],
            "difficulty": "Medium"
        },
        {
            "name": "Complex Algorithm",
            "prompt": "Implement a function to find the longest common subsequence between two strings",
            "expected_elements": ["def", "dynamic programming", "matrix", "for", "if"],
            "difficulty": "Hard"
        },
        {
            "name": "Object-Oriented Design",
            "prompt": "Design a Python class hierarchy for different types of vehicles with inheritance",
            "expected_elements": ["class Vehicle", "class Car", "def __init__", "super()", "inheritance"],
            "difficulty": "Medium"
        }
    ]
    
    try:
        from src.core.modular_engine import ModularAdaptrixEngine
        
        print("🚀 Initializing Adaptrix with Qwen3-1.7B...")
        engine = ModularAdaptrixEngine(
            model_id="Qwen/Qwen3-1.7B",
            device="cpu",
            adapters_dir="adapters"
        )
        
        if not engine.initialize():
            print("❌ Failed to initialize engine")
            return False
        
        print("✅ Engine initialized successfully!")
        
        # Check if code adapter is available
        available_adapters = engine.list_adapters()
        print(f"📦 Available adapters: {available_adapters}")
        
        if "code" not in available_adapters:
            print("❌ Code adapter not found in available adapters")
            return False
        
        # Load the code adapter
        print("\n🔌 Loading code adapter...")
        if not engine.load_adapter("code"):
            print("❌ Failed to load code adapter")
            return False
        
        print("✅ Code adapter loaded successfully!")
        
        # Test each case
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}: {test_case['name']} ({test_case['difficulty']})")
            print("-" * 50)
            print(f"📝 Prompt: {test_case['prompt']}")
            
            start_time = time.time()
            
            # Generate with code-specific parameters
            response = engine.generate(
                test_case['prompt'],
                task_type="code",
                max_length=512,
                temperature=0.3,  # Lower temperature for more deterministic code
                top_p=0.95
            )
            
            generation_time = time.time() - start_time
            
            print(f"⏱️ Generated in {generation_time:.2f}s")
            print(f"📏 Response length: {len(response)} characters")
            
            # Evaluate response quality
            quality_score = evaluate_code_response(response, test_case['expected_elements'])
            
            print(f"📊 Quality Score: {quality_score:.1f}%")
            
            # Show response
            print(f"\n💻 GENERATED CODE:")
            print("```python")
            print(response[:800] + ("..." if len(response) > 800 else ""))
            print("```")
            
            # Store results
            results.append({
                'name': test_case['name'],
                'difficulty': test_case['difficulty'],
                'prompt': test_case['prompt'],
                'response': response,
                'quality_score': quality_score,
                'generation_time': generation_time,
                'response_length': len(response)
            })
        
        # Generate comprehensive report
        generate_code_adapter_report(results)
        
        # Cleanup
        engine.unload_adapter("code")
        engine.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def evaluate_code_response(response: str, expected_elements: list) -> float:
    """Evaluate the quality of a code generation response."""
    
    score = 0
    max_score = 100
    
    # Check for expected elements (40 points)
    elements_found = 0
    for element in expected_elements:
        if element.lower() in response.lower():
            elements_found += 1
    
    element_score = (elements_found / len(expected_elements)) * 40 if expected_elements else 0
    score += element_score
    
    # Check for code structure (20 points)
    structure_indicators = ['def ', 'class ', 'import ', 'from ', 'return', 'if ', 'for ', 'while ']
    structure_found = sum(1 for indicator in structure_indicators if indicator in response)
    structure_score = min(structure_found * 3, 20)
    score += structure_score
    
    # Check for proper formatting (15 points)
    formatting_score = 0
    if '```' in response or 'def ' in response:
        formatting_score += 5
    if ':' in response and '\n' in response:
        formatting_score += 5
    if any(comment in response for comment in ['#', '"""', "'''"]):
        formatting_score += 5
    score += formatting_score
    
    # Check response length (10 points)
    if len(response) >= 100:
        length_score = min(len(response) / 50, 10)
    else:
        length_score = len(response) / 10
    score += length_score
    
    # Check for completeness (15 points)
    completeness_score = 0
    if response.count('def ') >= 1:
        completeness_score += 5
    if 'return' in response:
        completeness_score += 5
    if len(response.split('\n')) >= 5:
        completeness_score += 5
    score += completeness_score
    
    return min(score, max_score)


def test_code_adapter_vs_baseline():
    """Compare code adapter performance vs baseline model."""
    
    print("\n📊 CODE ADAPTER VS BASELINE COMPARISON")
    print("=" * 60)
    
    comparison_prompts = [
        "Write a Python function to reverse a string",
        "Create a simple calculator class in Python",
        "Implement bubble sort algorithm"
    ]
    
    try:
        from src.core.modular_engine import ModularAdaptrixEngine
        
        engine = ModularAdaptrixEngine("Qwen/Qwen3-1.7B", "cpu", "adapters")
        
        if not engine.initialize():
            print("❌ Failed to initialize engine")
            return
        
        for prompt in comparison_prompts:
            print(f"\n🧪 Testing: {prompt}")
            print("-" * 40)
            
            # Test without adapter (baseline)
            print("📊 Baseline (No Adapter):")
            baseline_response = engine.generate(prompt, task_type="code", max_length=300)
            baseline_score = evaluate_code_response(baseline_response, ["def", "return"])
            print(f"   Quality: {baseline_score:.1f}%")
            print(f"   Length: {len(baseline_response)} chars")
            
            # Test with code adapter
            if engine.load_adapter("code"):
                print("\n🔌 With Code Adapter:")
                adapter_response = engine.generate(prompt, task_type="code", max_length=300)
                adapter_score = evaluate_code_response(adapter_response, ["def", "return"])
                print(f"   Quality: {adapter_score:.1f}%")
                print(f"   Length: {len(adapter_response)} chars")
                
                improvement = adapter_score - baseline_score
                print(f"   📈 Improvement: {improvement:+.1f}%")
                
                engine.unload_adapter("code")
            else:
                print("❌ Failed to load code adapter")
        
        engine.cleanup()
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")


def generate_code_adapter_report(results: list):
    """Generate comprehensive report for code adapter testing."""
    
    print(f"\n📊 CODE ADAPTER COMPREHENSIVE REPORT")
    print("=" * 80)
    
    if not results:
        print("❌ No results to report")
        return
    
    # Calculate statistics
    total_tests = len(results)
    avg_quality = sum(r['quality_score'] for r in results) / total_tests
    avg_time = sum(r['generation_time'] for r in results) / total_tests
    avg_length = sum(r['response_length'] for r in results) / total_tests
    
    # Quality distribution
    excellent = sum(1 for r in results if r['quality_score'] >= 80)
    good = sum(1 for r in results if 60 <= r['quality_score'] < 80)
    fair = sum(1 for r in results if 40 <= r['quality_score'] < 60)
    poor = sum(1 for r in results if r['quality_score'] < 40)
    
    print(f"\n📈 OVERALL PERFORMANCE:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Average Quality: {avg_quality:.1f}%")
    print(f"   Average Generation Time: {avg_time:.2f}s")
    print(f"   Average Response Length: {avg_length:.0f} characters")
    
    print(f"\n📊 QUALITY DISTRIBUTION:")
    print(f"   Excellent (80%+): {excellent} tests ({excellent/total_tests*100:.1f}%)")
    print(f"   Good (60-79%): {good} tests ({good/total_tests*100:.1f}%)")
    print(f"   Fair (40-59%): {fair} tests ({fair/total_tests*100:.1f}%)")
    print(f"   Poor (<40%): {poor} tests ({poor/total_tests*100:.1f}%)")
    
    print(f"\n📋 DETAILED RESULTS:")
    for result in results:
        status = "🟢" if result['quality_score'] >= 80 else "🟡" if result['quality_score'] >= 60 else "🔴"
        print(f"   {status} {result['name']}: {result['quality_score']:.1f}% ({result['difficulty']})")
    
    # Performance by difficulty
    difficulties = {}
    for result in results:
        diff = result['difficulty']
        if diff not in difficulties:
            difficulties[diff] = []
        difficulties[diff].append(result['quality_score'])
    
    print(f"\n📊 PERFORMANCE BY DIFFICULTY:")
    for difficulty, scores in difficulties.items():
        avg_score = sum(scores) / len(scores)
        print(f"   {difficulty}: {avg_score:.1f}% (n={len(scores)})")
    
    # Overall assessment
    print(f"\n🎯 CODE ADAPTER ASSESSMENT:")
    if avg_quality >= 80:
        print(f"   🎊 EXCELLENT: Code adapter performs exceptionally well!")
        print(f"   ✅ Ready for production code generation tasks")
    elif avg_quality >= 70:
        print(f"   ✅ GOOD: Code adapter shows strong performance")
        print(f"   🔧 Minor optimizations could improve results")
    elif avg_quality >= 60:
        print(f"   ⚠️ FAIR: Code adapter has decent performance")
        print(f"   🔧 Recommend additional training or parameter tuning")
    else:
        print(f"   ❌ POOR: Code adapter needs significant improvement")
        print(f"   🔧 Consider retraining with better data or parameters")
    
    # Save detailed report
    report_path = "code_adapter_test_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            'summary': {
                'total_tests': total_tests,
                'avg_quality': avg_quality,
                'avg_time': avg_time,
                'avg_length': avg_length,
                'quality_distribution': {
                    'excellent': excellent,
                    'good': good,
                    'fair': fair,
                    'poor': poor
                }
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_path}")


def main():
    """Main testing function."""
    
    print("🧪" * 80)
    print("🧪 CODE ADAPTER COMPREHENSIVE TESTING 🧪")
    print("🧪" * 80)
    
    # Inspect adapter
    adapter_info = inspect_code_adapter()
    
    if adapter_info:
        # Test code generation
        success = test_code_generation_tasks()
        
        # Compare with baseline
        test_code_adapter_vs_baseline()
        
        if success:
            print(f"\n🎊 CODE ADAPTER TESTING COMPLETE!")
        else:
            print(f"\n❌ Testing encountered issues")
    else:
        print(f"\n❌ Could not inspect code adapter")


if __name__ == "__main__":
    main()
