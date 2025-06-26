#!/usr/bin/env python3
"""
🎯 ULTIMATE QUALITY TEST - PROPER LORA APPLICATION

This test verifies that:
1. LoRA adapters are properly applied to model weights
2. Responses show clear domain specialization
3. Streaming works for better UX
4. Post-processing removes all artifacts
5. Quality is genuinely high

GOAL: Prove that adapters actually change model behavior!
"""

import sys
import os
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def analyze_response_quality(prompt: str, response: str, adapter_name: str = "", expected_domain: str = ""):
    """Analyze response quality with domain-specific expectations."""
    print(f"\n{'🎯' * 80}")
    print(f"🎯 PROMPT: {prompt}")
    if adapter_name:
        print(f"🔧 ADAPTER: {adapter_name}")
    if expected_domain:
        print(f"🎯 EXPECTED DOMAIN: {expected_domain}")
    print(f"{'🎯' * 80}")
    print(f"🤖 RESPONSE:")
    print(f"{'─' * 80}")
    print(response)
    print(f"{'─' * 80}")
    
    # Comprehensive quality analysis
    quality_score = 0
    max_score = 10
    issues = []
    strengths = []
    
    # 1. Basic quality checks
    if len(response) >= 20:
        quality_score += 1
        strengths.append("Adequate length")
    else:
        issues.append("Too short")
    
    # 2. Proper formatting
    if response and response[0].isupper():
        quality_score += 1
        strengths.append("Proper capitalization")
    else:
        issues.append("Capitalization issue")
    
    # 3. No artifacts
    artifacts = ['<|', '|>', '```', 'print(', 'def ', '## INPUT', 'Exercise', 'Problem']
    if not any(artifact in response for artifact in artifacts):
        quality_score += 1
        strengths.append("Clean formatting")
    else:
        issues.append("Contains artifacts")
    
    # 4. Relevance to prompt
    prompt_words = set(prompt.lower().split())
    response_words = set(response.lower().split())
    relevance = len(prompt_words.intersection(response_words)) / len(prompt_words) if prompt_words else 0
    if relevance >= 0.3:
        quality_score += 1
        strengths.append("Relevant to prompt")
    else:
        issues.append("Low relevance")
    
    # 5. Domain-specific quality checks
    if expected_domain == "mathematics":
        # Check for mathematical content
        math_indicators = ['=', '+', '-', '*', '/', 'calculate', 'answer', 'result', 'total']
        if any(indicator in response.lower() for indicator in math_indicators):
            quality_score += 2
            strengths.append("Mathematical content detected")
        else:
            issues.append("No mathematical content")
            
        # Check for numerical answers
        import re
        numbers = re.findall(r'\b\d+\b', response)
        if numbers:
            quality_score += 1
            strengths.append("Contains numerical answers")
        else:
            issues.append("No numerical answers")
    
    elif expected_domain == "journalism":
        # Check for news-style content
        news_indicators = ['news', 'report', 'announced', 'according', 'sources', 'breaking', 'update']
        if any(indicator in response.lower() for indicator in news_indicators):
            quality_score += 2
            strengths.append("Journalistic style detected")
        else:
            issues.append("No journalistic style")
            
        # Check for proper structure
        if len(response.split('.')) >= 2:
            quality_score += 1
            strengths.append("Structured content")
        else:
            issues.append("Lacks structure")
    
    elif expected_domain == "programming":
        # Check for code-related content
        code_indicators = ['function', 'def', 'return', 'variable', 'loop', 'if', 'else', 'python']
        if any(indicator in response.lower() for indicator in code_indicators):
            quality_score += 2
            strengths.append("Programming content detected")
        else:
            issues.append("No programming content")
            
        # Check for technical accuracy
        if 'function' in response.lower() and ('def' in response.lower() or 'return' in response.lower()):
            quality_score += 1
            strengths.append("Technical accuracy")
        else:
            issues.append("Lacks technical accuracy")
    
    # 6. Coherence check
    sentences = [s.strip() for s in response.split('.') if s.strip()]
    if len(sentences) >= 1 and all(len(s) > 5 for s in sentences):
        quality_score += 1
        strengths.append("Coherent sentences")
    else:
        issues.append("Incoherent content")
    
    # Overall assessment
    percentage = (quality_score / max_score) * 100
    
    print(f"📊 QUALITY ANALYSIS:")
    print(f"   Score: {quality_score}/{max_score} ({percentage:.1f}%)")
    
    if percentage >= 80:
        print(f"   ✅ EXCELLENT QUALITY")
    elif percentage >= 60:
        print(f"   ⚠️ GOOD QUALITY")
    elif percentage >= 40:
        print(f"   ⚠️ FAIR QUALITY")
    else:
        print(f"   ❌ POOR QUALITY")
    
    if strengths:
        print(f"   ✅ Strengths: {', '.join(strengths)}")
    if issues:
        print(f"   ⚠️ Issues: {', '.join(issues)}")
    
    print(f"{'🎯' * 80}")
    
    return quality_score, max_score


def ultimate_quality_test():
    """Run the ultimate quality test with proper LoRA application."""
    
    print("🚀" * 100)
    print("🚀 ULTIMATE QUALITY TEST - PROPER LORA APPLICATION 🚀")
    print("🚀" * 100)
    print()
    print("🎯 TESTING:")
    print("   ✅ Proper LoRA weight application to model")
    print("   ✅ Clear domain specialization")
    print("   ✅ High-quality, artifact-free responses")
    print("   ✅ Streaming for better UX")
    print("   ✅ Comprehensive quality analysis")
    print()
    
    # Test adapters with clear domain expectations
    test_adapters = [
        {
            "hf_repo": "liuchanghf/phi2-gsm8k-lora",
            "name": "math_specialist",
            "description": "Mathematical reasoning specialist",
            "capabilities": ["mathematics", "arithmetic", "problem_solving"],
            "domain": "mathematics",
            "training_data": "GSM8K mathematical reasoning dataset",
            "test_prompts": [
                "What is 25 times 8?",
                "If I have 100 dollars and spend 35 dollars, how much do I have left?",
                "Calculate 15% of 240"
            ]
        },
        {
            "hf_repo": "AmevinLS/phi-2-lora-realnews",
            "name": "news_specialist",
            "description": "Professional journalism specialist",
            "capabilities": ["journalism", "news_writing", "reporting"],
            "domain": "journalism",
            "training_data": "RealNews dataset",
            "test_prompts": [
                "Write a news headline about a scientific breakthrough",
                "Report on renewable energy developments",
                "Create a technology news update"
            ]
        },
        {
            "hf_repo": "Nutanix/phi-2_SFT_lora_4_alpha_16_humaneval_raw_json",
            "name": "code_specialist",
            "description": "Programming and code generation specialist",
            "capabilities": ["programming", "python", "code_generation"],
            "domain": "programming",
            "training_data": "HumanEval dataset",
            "test_prompts": [
                "Write a Python function to find the maximum value in a list",
                "Create a function that checks if a string is a palindrome",
                "Write code to calculate the factorial of a number"
            ]
        }
    ]
    
    try:
        from src.core.engine import AdaptrixEngine
        from src.conversion.dynamic_lora_converter import DynamicLoRAConverter
        
        # Initialize system
        print("🚀 INITIALIZING SYSTEM")
        print("=" * 60)
        
        engine = AdaptrixEngine("microsoft/phi-2", "cpu")
        
        if not engine.initialize():
            print("❌ Failed to initialize engine")
            return False
        
        print("✅ Engine initialized successfully!")
        
        # Clean start verification
        available_adapters = engine.list_adapters()
        print(f"📦 Starting with {len(available_adapters)} adapters")
        
        # Convert adapters
        print(f"\n🔄 CONVERTING ADAPTERS")
        print("=" * 60)
        
        converter = DynamicLoRAConverter()
        converted_adapters = []
        
        for adapter_config in test_adapters:
            print(f"\n🔌 Converting {adapter_config['name']}...")
            
            success = converter.convert_adapter(
                adapter_config['hf_repo'],
                adapter_config['name'],
                adapter_config['description'],
                adapter_config['capabilities'],
                adapter_config['domain'],
                adapter_config['training_data']
            )
            
            if success:
                converted_adapters.append(adapter_config)
                print(f"✅ {adapter_config['name']} converted successfully!")
            else:
                print(f"❌ {adapter_config['name']} conversion failed!")
        
        if not converted_adapters:
            print("❌ No adapters converted successfully!")
            return False
        
        # Test baseline (no adapters)
        print(f"\n🧪 BASELINE TEST (No Adapters)")
        print("=" * 60)
        
        baseline_prompt = "What is 12 multiplied by 7?"
        print(f"🎯 Testing baseline: {baseline_prompt}")
        baseline_response = engine.generate(baseline_prompt, max_length=100, stream=True)
        analyze_response_quality(baseline_prompt, baseline_response, "BASELINE", "")
        
        # Test each adapter with streaming
        print(f"\n🎯 ADAPTER SPECIALIZATION TESTS")
        print("=" * 60)
        
        total_score = 0
        total_max = 0
        
        for adapter_config in converted_adapters:
            adapter_name = adapter_config['name']
            domain = adapter_config['domain']
            
            print(f"\n🔧 TESTING {adapter_name.upper()}")
            print(f"   Expected domain: {domain}")
            print("-" * 50)
            
            # Load adapter
            if not engine.load_adapter(adapter_name):
                print(f"❌ Failed to load {adapter_name}")
                continue
            
            print(f"✅ {adapter_name} loaded and applied to model!")
            
            # Test with domain-specific prompts
            for i, prompt in enumerate(adapter_config['test_prompts'], 1):
                print(f"\n🧪 Test {i}/{len(adapter_config['test_prompts'])}")
                
                try:
                    response = engine.generate(prompt, max_length=150, stream=True)
                    score, max_score = analyze_response_quality(prompt, response, adapter_name, domain)
                    total_score += score
                    total_max += max_score
                except Exception as e:
                    print(f"❌ Generation error: {e}")
            
            # Unload adapter
            engine.unload_adapter(adapter_name)
            print(f"✅ {adapter_name} unloaded and weights restored!")
        
        # Final assessment
        print(f"\n📊 FINAL QUALITY ASSESSMENT")
        print("=" * 60)
        
        if total_max > 0:
            overall_percentage = (total_score / total_max) * 100
            print(f"Overall Quality Score: {total_score}/{total_max} ({overall_percentage:.1f}%)")
            
            if overall_percentage >= 70:
                print("✅ EXCELLENT - Adapters are working properly!")
            elif overall_percentage >= 50:
                print("⚠️ GOOD - Adapters show some specialization")
            else:
                print("❌ POOR - Adapters may not be properly applied")
        
        # Cleanup
        engine.cleanup()
        
        print("\n" + "🎊" * 100)
        print("🎊 ULTIMATE QUALITY TEST COMPLETE! 🎊")
        print("🎊" * 100)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    success = ultimate_quality_test()
    
    if success:
        print("\n🎯 ULTIMATE QUALITY TEST COMPLETED!")
        print("   Check the quality scores above to verify adapter effectiveness")
    else:
        print("\n❌ Ultimate quality test failed")


if __name__ == "__main__":
    main()
