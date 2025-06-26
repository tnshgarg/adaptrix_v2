#!/usr/bin/env python3
"""
🚀 QUICK GEMINI COMPARISON - FOCUSED QUALITY TEST

Quick but comprehensive test comparing Adaptrix vs Gemini 2.0 Flash:
- 3 key test cases per domain
- Accuracy and quality comparison
- Performance report
"""

import sys
import os
import requests
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

GEMINI_API_KEY = "AIzaSyAA-4qYJmlNtzO6gR-L5-pSEWPfuSl_xEA"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def query_gemini(prompt: str) -> str:
    """Query Gemini 2.0 Flash API."""
    try:
        headers = {'Content-Type': 'application/json'}
        data = {
            'contents': [{
                'parts': [{'text': prompt}]
            }]
        }
        
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text'].strip()
            return "Error: No response from Gemini"
        else:
            return f"Error: API returned {response.status_code}"
    except Exception as e:
        return f"Error: {e}"


def compare_responses(prompt: str, adaptrix_resp: str, gemini_resp: str, expected_answer: str = ""):
    """Compare responses and score them."""
    print(f"\n{'🔥' * 80}")
    print(f"🎯 PROMPT: {prompt}")
    print(f"{'🔥' * 80}")
    
    print(f"\n🤖 ADAPTRIX:")
    print(f"   {adaptrix_resp}")
    
    print(f"\n🧠 GEMINI 2.0 FLASH:")
    print(f"   {gemini_resp}")
    
    if expected_answer:
        print(f"\n✅ EXPECTED: {expected_answer}")
    
    # Scoring
    adaptrix_score = 0
    gemini_score = 0
    
    # Accuracy check
    if expected_answer:
        if expected_answer.lower() in adaptrix_resp.lower():
            adaptrix_score += 3
            print(f"   ✅ Adaptrix: CORRECT answer")
        else:
            print(f"   ❌ Adaptrix: INCORRECT answer")
            
        if expected_answer.lower() in gemini_resp.lower():
            gemini_score += 3
            print(f"   ✅ Gemini: CORRECT answer")
        else:
            print(f"   ❌ Gemini: INCORRECT answer")
    
    # Quality checks
    if 20 <= len(adaptrix_resp) <= 200:
        adaptrix_score += 1
    if 20 <= len(gemini_resp) <= 200:
        gemini_score += 1
    
    # Relevance
    prompt_words = set(prompt.lower().split())
    if len(prompt_words.intersection(set(adaptrix_resp.lower().split()))) >= 2:
        adaptrix_score += 1
    if len(prompt_words.intersection(set(gemini_resp.lower().split()))) >= 2:
        gemini_score += 1
    
    print(f"\n📊 SCORES: Adaptrix {adaptrix_score}/5, Gemini {gemini_score}/5")
    
    if adaptrix_score > gemini_score:
        print(f"🏆 WINNER: ADAPTRIX")
        winner = "Adaptrix"
    elif gemini_score > adaptrix_score:
        print(f"🏆 WINNER: GEMINI")
        winner = "Gemini"
    else:
        print(f"🤝 TIE")
        winner = "Tie"
    
    return adaptrix_score, gemini_score, winner


def quick_gemini_test():
    """Run quick focused comparison test."""
    
    print("🚀" * 80)
    print("🚀 QUICK GEMINI COMPARISON - FOCUSED QUALITY TEST 🚀")
    print("🚀" * 80)
    
    # Key test cases with expected answers
    test_cases = [
        {
            "adapter": "math_specialist",
            "tests": [
                ("What is 25 times 8?", "200"),
                ("If I have 100 dollars and spend 35 dollars, how much do I have left?", "65"),
                ("Calculate 15% of 240", "36")
            ]
        },
        {
            "adapter": "news_specialist", 
            "tests": [
                ("Write a news headline about AI breakthrough", ""),
                ("Report on renewable energy progress", ""),
                ("Create a technology news update", "")
            ]
        },
        {
            "adapter": "code_specialist",
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
        
        # Initialize
        print("\n🚀 INITIALIZING SYSTEM...")
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
        
        # Run tests
        total_adaptrix = 0
        total_gemini = 0
        total_tests = 0
        adaptrix_wins = 0
        gemini_wins = 0
        ties = 0
        
        for test_case in test_cases:
            adapter_name = test_case["adapter"]
            
            print(f"\n{'='*60}")
            print(f"🔧 TESTING {adapter_name.upper()}")
            print(f"{'='*60}")
            
            if adapter_name not in engine.list_adapters():
                print(f"⚠️ {adapter_name} not available, skipping...")
                continue
            
            # Load adapter
            if not engine.load_adapter(adapter_name):
                print(f"❌ Failed to load {adapter_name}")
                continue
            
            print(f"✅ {adapter_name} loaded!")
            
            # Run tests
            for prompt, expected in test_case["tests"]:
                print(f"\n🧪 Testing: {prompt}")
                
                try:
                    # Get responses
                    print("🤖 Generating Adaptrix response...", end="", flush=True)
                    adaptrix_resp = engine.generate(prompt, max_length=100, temperature=0.3)
                    print(" ✅")
                    
                    print("🧠 Querying Gemini...", end="", flush=True)
                    gemini_resp = query_gemini(prompt)
                    print(" ✅")
                    
                    # Compare
                    a_score, g_score, winner = compare_responses(prompt, adaptrix_resp, gemini_resp, expected)
                    
                    total_adaptrix += a_score
                    total_gemini += g_score
                    total_tests += 1
                    
                    if winner == "Adaptrix":
                        adaptrix_wins += 1
                    elif winner == "Gemini":
                        gemini_wins += 1
                    else:
                        ties += 1
                        
                except Exception as e:
                    print(f"❌ Test failed: {e}")
            
            # Unload
            engine.unload_adapter(adapter_name)
            print(f"✅ {adapter_name} unloaded!")
        
        # Final report
        print(f"\n{'🎊' * 80}")
        print(f"🎊 FINAL COMPARISON REPORT 🎊")
        print(f"{'🎊' * 80}")
        
        if total_tests > 0:
            adaptrix_avg = (total_adaptrix / (total_tests * 5)) * 100
            gemini_avg = (total_gemini / (total_tests * 5)) * 100
            
            print(f"\n📊 OVERALL PERFORMANCE:")
            print(f"   🤖 Adaptrix Average: {adaptrix_avg:.1f}%")
            print(f"   🧠 Gemini Average:   {gemini_avg:.1f}%")
            print(f"   📈 Total Tests:      {total_tests}")
            
            print(f"\n🏆 WIN STATISTICS:")
            print(f"   🤖 Adaptrix Wins: {adaptrix_wins}")
            print(f"   🧠 Gemini Wins:   {gemini_wins}")
            print(f"   🤝 Ties:          {ties}")
            
            print(f"\n📋 PERFORMANCE ANALYSIS:")
            
            if adaptrix_wins > gemini_wins:
                print(f"   🎊 OVERALL WINNER: ADAPTRIX! 🎊")
                print(f"   🚀 Adaptrix outperformed Gemini 2.0 Flash!")
            elif gemini_wins > adaptrix_wins:
                print(f"   🏆 OVERALL WINNER: GEMINI 2.0 FLASH")
                print(f"   📈 Gemini showed superior performance")
            else:
                print(f"   🤝 RESULT: COMPETITIVE TIE")
                print(f"   ⚖️ Both models performed equally well")
            
            # Quality assessment
            if adaptrix_avg >= 80:
                print(f"   ✅ Adaptrix: EXCELLENT quality (≥80%)")
            elif adaptrix_avg >= 60:
                print(f"   ⚠️ Adaptrix: GOOD quality (≥60%)")
            else:
                print(f"   ❌ Adaptrix: NEEDS IMPROVEMENT (<60%)")
            
            if gemini_avg >= 80:
                print(f"   ✅ Gemini: EXCELLENT quality (≥80%)")
            elif gemini_avg >= 60:
                print(f"   ⚠️ Gemini: GOOD quality (≥60%)")
            else:
                print(f"   ❌ Gemini: NEEDS IMPROVEMENT (<60%)")
            
            # Conclusion
            print(f"\n🎯 CONCLUSION:")
            if adaptrix_avg >= 70:
                print(f"   🎊 Adaptrix is PRODUCTION READY!")
                print(f"   🚀 High-quality responses with domain specialization")
                print(f"   🔥 Competitive with state-of-the-art models")
            else:
                print(f"   ⚠️ Adaptrix needs further optimization")
                print(f"   🔧 Focus on improving response accuracy")
        
        engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    success = quick_gemini_test()
    
    if success:
        print(f"\n🎯 GEMINI COMPARISON COMPLETE!")
    else:
        print(f"\n❌ Test failed")


if __name__ == "__main__":
    main()
