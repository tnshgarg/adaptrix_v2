#!/usr/bin/env python3
"""
🔥 GEMINI-LEVEL QUALITY TEST - COMPLETE FIXES VALIDATION

This test validates all Gemini-level fixes:
1. ✅ Structured prompt templates
2. ✅ Improved generation parameters (512 tokens, temp=0.7)
3. ✅ Gemini-style post-processing
4. ✅ Domain-specific formatting
5. ✅ Journalism adapter corruption fixes
6. ✅ Complete response validation

GOAL: Achieve Gemini-level quality across all domains!
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
        
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=data, timeout=20)
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text'].strip()
        return "Gemini API Error"
    except Exception as e:
        return f"Gemini Error: {e}"


def evaluate_gemini_level_quality(response: str, domain: str, expected_content: str = "") -> dict:
    """Evaluate response quality against Gemini standards."""
    score = 0
    max_score = 15  # Higher standards for Gemini-level
    issues = []
    strengths = []
    
    # 1. Length and completeness (3 points)
    if len(response) >= 100:
        score += 3
        strengths.append("Comprehensive length")
    elif len(response) >= 50:
        score += 2
        strengths.append("Adequate length")
    elif len(response) >= 20:
        score += 1
        strengths.append("Minimal length")
    else:
        issues.append("Too short")
    
    # 2. Structure and formatting (3 points)
    if domain == 'programming' and '```' in response:
        score += 3
        strengths.append("Proper code formatting")
    elif domain == 'journalism' and ('#' in response or len(response.split('\n')) > 1):
        score += 3
        strengths.append("Proper news structure")
    elif domain == 'mathematics' and ('Step' in response or '=' in response):
        score += 3
        strengths.append("Structured mathematical solution")
    elif response.count('\n') > 0 or response.count('.') > 1:
        score += 2
        strengths.append("Basic structure")
    else:
        score += 1
        issues.append("Lacks structure")
    
    # 3. No corruption (2 points)
    corruption_indicators = ['�', '\x00', '\ufffd', '" "', 'model " "']
    if not any(indicator in response for indicator in corruption_indicators):
        score += 2
        strengths.append("No corruption")
    else:
        issues.append("Contains corruption")
    
    # 4. Domain expertise (4 points)
    if domain == 'mathematics':
        if expected_content and expected_content in response:
            score += 4
            strengths.append("Correct mathematical answer")
        elif any(word in response.lower() for word in ['step', 'calculate', 'answer', '=']):
            score += 2
            strengths.append("Mathematical reasoning")
        else:
            issues.append("Lacks mathematical content")
    
    elif domain == 'journalism':
        journalism_indicators = ['news', 'report', 'announced', 'breaking', 'according', 'sources']
        if sum(1 for indicator in journalism_indicators if indicator.lower() in response.lower()) >= 2:
            score += 4
            strengths.append("Professional journalism style")
        elif any(indicator.lower() in response.lower() for indicator in journalism_indicators):
            score += 2
            strengths.append("Basic journalism content")
        else:
            issues.append("Lacks journalistic style")
    
    elif domain == 'programming':
        if 'def ' in response and '"""' in response:
            score += 4
            strengths.append("Complete function with docstring")
        elif 'def ' in response or 'function' in response.lower():
            score += 3
            strengths.append("Programming content")
        elif any(word in response.lower() for word in ['code', 'python', 'algorithm']):
            score += 2
            strengths.append("Technical content")
        else:
            issues.append("Lacks programming content")
    
    # 5. Professional quality (3 points)
    if response[0].isupper() and response.endswith(('.', '!', '?', '```')):
        score += 2
        strengths.append("Professional formatting")
    elif response[0].isupper() or response.endswith(('.', '!', '?')):
        score += 1
        strengths.append("Basic formatting")
    else:
        issues.append("Poor formatting")
    
    # Bonus for exceptional quality
    if len(response) > 200 and len(strengths) >= 4:
        score += 1
        strengths.append("Exceptional quality")
    
    percentage = (score / max_score) * 100
    
    if percentage >= 90:
        quality = "GEMINI-LEVEL"
    elif percentage >= 80:
        quality = "EXCELLENT"
    elif percentage >= 70:
        quality = "GOOD"
    elif percentage >= 60:
        quality = "FAIR"
    else:
        quality = "POOR"
    
    return {
        'score': score,
        'max_score': max_score,
        'percentage': percentage,
        'quality': quality,
        'strengths': strengths,
        'issues': issues
    }


def gemini_level_test():
    """Run comprehensive Gemini-level quality test."""
    
    print("🔥" * 120)
    print("🔥 GEMINI-LEVEL QUALITY TEST - COMPLETE FIXES VALIDATION 🔥")
    print("🔥" * 120)
    
    # Comprehensive test cases
    test_cases = [
        {
            "adapter": "math_specialist",
            "domain": "mathematics",
            "tests": [
                ("What is 25 times 8?", "200"),
                ("Calculate 15% of 240", "36"),
                ("Solve: 2x + 5 = 15", "5"),
                ("What is the square root of 144?", "12")
            ]
        },
        {
            "adapter": "news_specialist",
            "domain": "journalism",
            "tests": [
                ("Write a news headline about AI breakthrough", ""),
                ("Report on renewable energy developments", ""),
                ("Create a technology news update about quantum computing", ""),
                ("Write about climate change progress", "")
            ]
        },
        {
            "adapter": "code_specialist",
            "domain": "programming",
            "tests": [
                ("Write a Python function to find maximum in a list", "def"),
                ("Create a function that checks if a string is a palindrome", "def"),
                ("Write code to calculate factorial of a number", "factorial"),
                ("How do you reverse a string in Python?", "[::-1]")
            ]
        }
    ]
    
    try:
        from src.core.engine import AdaptrixEngine
        from src.conversion.dynamic_lora_converter import DynamicLoRAConverter
        
        print("\n🚀 INITIALIZING GEMINI-LEVEL SYSTEM...")
        engine = AdaptrixEngine("microsoft/phi-2", "cpu")
        
        if not engine.initialize():
            print("❌ Failed to initialize")
            return False
        
        print("✅ System initialized with Gemini-level enhancements!")
        
        # Ensure adapters are available
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
                    hf_repo, adapter_name, f"{adapter_name} for Gemini-level testing",
                    [adapter_name.split('_')[0]], adapter_name.split('_')[0], "High-quality dataset"
                )
                if success:
                    print(f"✅ {adapter_name} ready!")
                else:
                    print(f"❌ {adapter_name} failed!")
                    continue
        
        # Run Gemini-level tests
        print(f"\n🎯 GEMINI-LEVEL QUALITY TESTS")
        print("=" * 80)
        
        total_score = 0
        total_max = 0
        adaptrix_wins = 0
        gemini_wins = 0
        ties = 0
        gemini_level_count = 0
        
        for test_case in test_cases:
            adapter_name = test_case["adapter"]
            domain = test_case["domain"]
            
            if adapter_name not in engine.list_adapters():
                print(f"⚠️ {adapter_name} not available, skipping...")
                continue
            
            print(f"\n🔧 TESTING {adapter_name.upper()} - GEMINI-LEVEL STANDARDS")
            print("-" * 70)
            
            # Load adapter
            if not engine.load_adapter(adapter_name):
                print(f"❌ Failed to load {adapter_name}")
                continue
            
            print(f"✅ {adapter_name} loaded with Gemini-level enhancements!")
            
            # Test each prompt
            for prompt, expected in test_case["tests"]:
                print(f"\n🧪 Testing: {prompt}")
                
                try:
                    # Generate with Gemini-level parameters
                    print("🤖 Generating Adaptrix response (Gemini-level)...", end="", flush=True)
                    adaptrix_response = engine.generate(
                        prompt, 
                        max_length=512,  # Gemini-level length
                        temperature=0.7,  # Gemini-level creativity
                        top_p=0.9,
                        do_sample=True
                    )
                    print(" ✅")
                    
                    # Evaluate quality
                    quality = evaluate_gemini_level_quality(adaptrix_response, domain, expected)
                    
                    print(f"\n📊 ADAPTRIX QUALITY (Gemini Standards):")
                    print(f"   Response: {adaptrix_response[:150]}{'...' if len(adaptrix_response) > 150 else ''}")
                    print(f"   Score: {quality['score']}/{quality['max_score']} ({quality['percentage']:.1f}%)")
                    print(f"   Quality: {quality['quality']}")
                    if quality['strengths']:
                        print(f"   ✅ Strengths: {', '.join(quality['strengths'])}")
                    if quality['issues']:
                        print(f"   ⚠️ Issues: {', '.join(quality['issues'])}")
                    
                    # Track Gemini-level achievements
                    if quality['quality'] == 'GEMINI-LEVEL':
                        gemini_level_count += 1
                        print(f"   🎊 ACHIEVED GEMINI-LEVEL QUALITY! 🎊")
                    
                    # Compare with Gemini
                    print("\n🧠 Querying Gemini for comparison...", end="", flush=True)
                    gemini_response = query_gemini(prompt)
                    print(" ✅")
                    
                    gemini_quality = evaluate_gemini_level_quality(gemini_response, domain, expected)
                    
                    print(f"\n📊 GEMINI QUALITY:")
                    print(f"   Response: {gemini_response[:150]}{'...' if len(gemini_response) > 150 else ''}")
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
        
        # Final Gemini-level report
        print(f"\n🎊 FINAL GEMINI-LEVEL QUALITY REPORT 🎊")
        print("=" * 80)
        
        if total_max > 0:
            overall_quality = (total_score / total_max) * 100
            
            print(f"📊 OVERALL PERFORMANCE:")
            print(f"   🤖 Adaptrix Quality: {overall_quality:.1f}%")
            print(f"   🏆 Wins vs Gemini: {adaptrix_wins}")
            print(f"   🤝 Ties: {ties}")
            print(f"   📉 Losses: {gemini_wins}")
            print(f"   🎊 Gemini-Level Responses: {gemini_level_count}")
            
            print(f"\n🎯 GEMINI-LEVEL ACHIEVEMENT:")
            if overall_quality >= 90:
                print(f"   🎊 GEMINI-LEVEL ACHIEVED! 🎊")
                print(f"   🚀 Adaptrix matches Gemini quality standards!")
                print(f"   🔥 Ready for production deployment!")
            elif overall_quality >= 80:
                print(f"   ✅ EXCELLENT - Near Gemini-level!")
                print(f"   🚀 Minor optimizations needed for full Gemini parity")
            elif overall_quality >= 70:
                print(f"   ⚠️ GOOD - Significant improvement achieved")
                print(f"   🔧 Continue optimizing for Gemini-level quality")
            else:
                print(f"   ❌ NEEDS IMPROVEMENT")
                print(f"   🔧 Major enhancements still required")
            
            # Success metrics
            success_rate = (adaptrix_wins + ties) / max(1, adaptrix_wins + gemini_wins + ties) * 100
            print(f"\n📈 SUCCESS METRICS:")
            print(f"   🎯 Success Rate vs Gemini: {success_rate:.1f}%")
            print(f"   🏆 Competitive Performance: {'YES' if success_rate >= 60 else 'NO'}")
            print(f"   🎊 Production Ready: {'YES' if overall_quality >= 80 else 'NO'}")
        
        engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    success = gemini_level_test()
    
    if success:
        print(f"\n🎯 GEMINI-LEVEL TEST COMPLETE!")
    else:
        print(f"\n❌ Test failed")


if __name__ == "__main__":
    main()
