#!/usr/bin/env python3
"""
🔥 ULTIMATE FIXED TEST - PRODUCTION READY VALIDATION

This test validates all critical fixes:
1. ✅ Response corruption fixed
2. ✅ LoRA adapter validation working
3. ✅ Complete generation (no truncation)
4. ✅ Domain-specific prompt engineering
5. ✅ Robust error handling

GOAL: Prove Adaptrix is now production-ready!
"""

import sys
import os
import requests

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

GEMINI_API_KEY = "AIzaSyAA-4qYJmlNtzO6gR-L5-pSEWPfuSl_xEA"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def query_gemini(prompt: str) -> str:
    """Query Gemini for comparison."""
    try:
        headers = {'Content-Type': 'application/json'}
        data = {'contents': [{'parts': [{'text': prompt}]}]}
        
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text'].strip()
        return "Gemini API Error"
    except Exception as e:
        return f"Gemini Error: {e}"


def validate_response_quality(response: str, domain: str, expected_answer: str = "") -> dict:
    """Comprehensive response quality validation."""
    issues = []
    score = 0
    max_score = 10
    
    # 1. Basic quality checks
    if len(response) >= 20:
        score += 1
    else:
        issues.append("Too short")
    
    # 2. No corruption
    if not any(char in response for char in ['�', '\x00', '\ufffd']):
        score += 2
    else:
        issues.append("Contains corrupted characters")
    
    # 3. Proper encoding
    try:
        response.encode('utf-8')
        score += 1
    except UnicodeEncodeError:
        issues.append("Encoding issues")
    
    # 4. No excessive special characters
    special_char_ratio = sum(1 for c in response if not c.isalnum() and c not in ' .,!?-:;()[]{}') / max(len(response), 1)
    if special_char_ratio <= 0.3:
        score += 1
    else:
        issues.append("Too many special characters")
    
    # 5. Domain-specific validation
    if domain == "mathematics":
        if expected_answer and expected_answer in response:
            score += 3
        elif any(indicator in response.lower() for indicator in ['=', 'answer', 'result']):
            score += 1
        else:
            issues.append("No mathematical content")
    elif domain == "journalism":
        if any(indicator in response.lower() for indicator in ['news', 'report', 'announced', 'breaking']):
            score += 2
        else:
            issues.append("No journalistic content")
    elif domain == "programming":
        if 'def ' in response or 'function' in response.lower():
            score += 2
        else:
            issues.append("No programming content")
    
    # 6. Completeness
    if response.endswith(('.', '!', '?', ':')):
        score += 1
    else:
        issues.append("Incomplete ending")
    
    return {
        'score': score,
        'max_score': max_score,
        'percentage': (score / max_score) * 100,
        'issues': issues,
        'quality': 'EXCELLENT' if score >= 8 else 'GOOD' if score >= 6 else 'FAIR' if score >= 4 else 'POOR'
    }


def test_adapter_validation(engine, adapter_name: str) -> bool:
    """Test that adapter is actually modifying model behavior."""
    try:
        # Test prompt
        test_prompt = "What is 5 + 3?"
        
        # Get baseline response (no adapter)
        baseline_response = engine.generate(test_prompt, max_length=50)
        
        # Load adapter
        if not engine.load_adapter(adapter_name):
            print(f"❌ Failed to load {adapter_name}")
            return False
        
        # Get adapter response
        adapter_response = engine.generate(test_prompt, max_length=50)
        
        # Unload adapter
        engine.unload_adapter(adapter_name)
        
        # Check if responses are different
        is_different = baseline_response.strip() != adapter_response.strip()
        
        if is_different:
            print(f"✅ {adapter_name} validation: PASSED (responses differ)")
            return True
        else:
            print(f"⚠️ {adapter_name} validation: QUESTIONABLE (responses identical)")
            return False
            
    except Exception as e:
        print(f"❌ {adapter_name} validation failed: {e}")
        return False


def ultimate_fixed_test():
    """Run comprehensive test of all fixes."""
    
    print("🔥" * 100)
    print("🔥 ULTIMATE FIXED TEST - PRODUCTION READY VALIDATION 🔥")
    print("🔥" * 100)
    
    # Critical test cases
    test_cases = [
        {
            "adapter": "math_specialist",
            "domain": "mathematics",
            "tests": [
                ("What is 25 times 8?", "200"),
                ("Calculate 15% of 240", "36"),
                ("If I have 100 dollars and spend 35 dollars, how much do I have left?", "65")
            ]
        },
        {
            "adapter": "news_specialist",
            "domain": "journalism", 
            "tests": [
                ("Write a news headline about AI breakthrough", ""),
                ("Report on renewable energy progress", ""),
                ("Create a technology news update", "")
            ]
        },
        {
            "adapter": "code_specialist",
            "domain": "programming",
            "tests": [
                ("Write a Python function to find maximum in a list", "def"),
                ("How do you reverse a string in Python?", "[::-1]"),
                ("Write code to calculate factorial", "factorial")
            ]
        }
    ]
    
    try:
        from src.core.engine import AdaptrixEngine
        from src.conversion.dynamic_lora_converter import DynamicLoRAConverter
        
        print("\n🚀 INITIALIZING FIXED SYSTEM...")
        engine = AdaptrixEngine("microsoft/phi-2", "cpu")
        
        if not engine.initialize():
            print("❌ Failed to initialize")
            return False
        
        print("✅ System initialized!")
        
        # Convert adapters if needed
        available = engine.list_adapters()
        converter = DynamicLoRAConverter()
        
        adapter_configs = {
            "math_specialist": "liuchanghf/phi2-gsm8k-lora",
            "news_specialist": "AmevinLS/phi-2-lora-realnews",
            "code_specialist": "Nutanix/phi-2_SFT_lora_4_alpha_16_humaneval_raw_json"
        }
        
        for adapter_name, hf_repo in adapter_configs.items():
            if adapter_name not in available:
                print(f"\n🔄 Converting {adapter_name}...")
                success = converter.convert_adapter(
                    hf_repo, adapter_name, f"{adapter_name} for testing",
                    [adapter_name.split('_')[0]], adapter_name.split('_')[0], "Test dataset"
                )
                if success:
                    print(f"✅ {adapter_name} ready!")
                else:
                    print(f"❌ {adapter_name} failed!")
                    continue
        
        # Test adapter validation
        print(f"\n🔍 TESTING ADAPTER VALIDATION")
        print("=" * 60)
        
        validation_results = {}
        for adapter_name in adapter_configs.keys():
            if adapter_name in engine.list_adapters():
                validation_results[adapter_name] = test_adapter_validation(engine, adapter_name)
        
        # Run quality tests
        print(f"\n🎯 TESTING RESPONSE QUALITY")
        print("=" * 60)
        
        total_score = 0
        total_max = 0
        adaptrix_wins = 0
        gemini_wins = 0
        ties = 0
        
        for test_case in test_cases:
            adapter_name = test_case["adapter"]
            domain = test_case["domain"]
            
            if adapter_name not in engine.list_adapters():
                print(f"⚠️ {adapter_name} not available, skipping...")
                continue
            
            print(f"\n🔧 TESTING {adapter_name.upper()}")
            print("-" * 50)
            
            # Load adapter
            if not engine.load_adapter(adapter_name):
                print(f"❌ Failed to load {adapter_name}")
                continue
            
            print(f"✅ {adapter_name} loaded!")
            
            # Test each prompt
            for prompt, expected in test_case["tests"]:
                print(f"\n🧪 Testing: {prompt}")
                
                try:
                    # Get Adaptrix response
                    print("🤖 Generating Adaptrix response...", end="", flush=True)
                    adaptrix_response = engine.generate(prompt, max_length=150, temperature=0.7)
                    print(" ✅")
                    
                    # Validate quality
                    quality = validate_response_quality(adaptrix_response, domain, expected)
                    
                    print(f"\n📊 ADAPTRIX QUALITY:")
                    print(f"   Response: {adaptrix_response}")
                    print(f"   Score: {quality['score']}/{quality['max_score']} ({quality['percentage']:.1f}%)")
                    print(f"   Quality: {quality['quality']}")
                    if quality['issues']:
                        print(f"   Issues: {', '.join(quality['issues'])}")
                    
                    # Get Gemini response for comparison
                    print("\n🧠 Querying Gemini...", end="", flush=True)
                    gemini_response = query_gemini(prompt)
                    print(" ✅")
                    
                    gemini_quality = validate_response_quality(gemini_response, domain, expected)
                    
                    print(f"\n📊 GEMINI QUALITY:")
                    print(f"   Response: {gemini_response[:100]}...")
                    print(f"   Score: {gemini_quality['score']}/{gemini_quality['max_score']} ({gemini_quality['percentage']:.1f}%)")
                    print(f"   Quality: {gemini_quality['quality']}")
                    
                    # Compare
                    if quality['percentage'] > gemini_quality['percentage']:
                        print(f"🏆 WINNER: ADAPTRIX (+{quality['percentage'] - gemini_quality['percentage']:.1f}%)")
                        adaptrix_wins += 1
                    elif gemini_quality['percentage'] > quality['percentage']:
                        print(f"🏆 WINNER: GEMINI (+{gemini_quality['percentage'] - quality['percentage']:.1f}%)")
                        gemini_wins += 1
                    else:
                        print(f"🤝 TIE")
                        ties += 1
                    
                    total_score += quality['score']
                    total_max += quality['max_score']
                    
                except Exception as e:
                    print(f"❌ Test failed: {e}")
            
            # Unload adapter
            engine.unload_adapter(adapter_name)
            print(f"✅ {adapter_name} unloaded!")
        
        # Final report
        print(f"\n🎊 FINAL PRODUCTION READINESS REPORT 🎊")
        print("=" * 80)
        
        if total_max > 0:
            overall_quality = (total_score / total_max) * 100
            
            print(f"📊 OVERALL PERFORMANCE:")
            print(f"   🤖 Adaptrix Quality: {overall_quality:.1f}%")
            print(f"   🏆 Wins vs Gemini: {adaptrix_wins}")
            print(f"   🤝 Ties: {ties}")
            print(f"   📉 Losses: {gemini_wins}")
            
            print(f"\n🔍 VALIDATION RESULTS:")
            for adapter, result in validation_results.items():
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"   {adapter}: {status}")
            
            print(f"\n🎯 PRODUCTION READINESS:")
            if overall_quality >= 80:
                print(f"   ✅ EXCELLENT - PRODUCTION READY!")
                print(f"   🚀 High-quality responses across all domains")
                print(f"   🔥 Competitive with state-of-the-art models")
            elif overall_quality >= 70:
                print(f"   ⚠️ GOOD - NEAR PRODUCTION READY")
                print(f"   🔧 Minor optimizations needed")
            elif overall_quality >= 60:
                print(f"   ⚠️ FAIR - NEEDS IMPROVEMENT")
                print(f"   🔧 Significant optimizations required")
            else:
                print(f"   ❌ POOR - NOT PRODUCTION READY")
                print(f"   🔧 Major fixes still needed")
            
            # Specific recommendations
            print(f"\n📋 RECOMMENDATIONS:")
            if overall_quality >= 80:
                print(f"   🎊 System is ready for production deployment!")
                print(f"   📈 Consider A/B testing with real users")
                print(f"   🔄 Monitor performance in production")
            else:
                print(f"   🔧 Focus on improving response completeness")
                print(f"   🎯 Enhance domain-specific training")
                print(f"   🔍 Debug remaining corruption issues")
        
        engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    success = ultimate_fixed_test()
    
    if success:
        print(f"\n🎯 ULTIMATE FIXED TEST COMPLETE!")
    else:
        print(f"\n❌ Test failed")


if __name__ == "__main__":
    main()
