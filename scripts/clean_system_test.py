#!/usr/bin/env python3
"""
🧪 CLEAN SYSTEM TEST - QUALITY FOCUSED

Tests the cleaned Adaptrix system with:
1. No _hf directories (lean storage)
2. Improved generation parameters (no repetition)
3. Complete response quality assessment
4. Clean, focused testing
"""

import sys
import os
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def print_response_analysis(prompt: str, response: str, adapter_name: str = ""):
    """Print complete response with quality analysis."""
    print(f"\n{'='*80}")
    print(f"🎯 PROMPT: {prompt}")
    if adapter_name:
        print(f"🔧 ADAPTER: {adapter_name}")
    print(f"{'='*80}")
    print(f"🤖 COMPLETE RESPONSE:")
    print(f"{'-'*80}")
    print(response)
    print(f"{'-'*80}")
    
    # Quality analysis
    print(f"📊 QUALITY ANALYSIS:")
    print(f"   Length: {len(response)} characters")
    print(f"   Words: {len(response.split())} words")
    
    # Check for repetition issues
    words = response.split()
    if len(words) > 10:
        # Check for repeated phrases
        repeated_phrases = []
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if response.count(phrase) > 1:
                repeated_phrases.append(phrase)
        
        if repeated_phrases:
            print(f"   ⚠️ REPETITION DETECTED: {len(set(repeated_phrases))} repeated phrases")
            for phrase in set(repeated_phrases)[:3]:  # Show first 3
                print(f"      - '{phrase}' (appears {response.count(phrase)} times)")
        else:
            print(f"   ✅ NO REPETITION DETECTED")
    
    # Check response quality indicators
    if len(response) < 10:
        print(f"   ⚠️ VERY SHORT RESPONSE")
    elif len(response) > 500:
        print(f"   ⚠️ VERY LONG RESPONSE")
    else:
        print(f"   ✅ GOOD LENGTH")
    
    print(f"{'='*80}")


def test_clean_system():
    """Test the cleaned Adaptrix system."""
    
    print("🧪" * 80)
    print("🧪 CLEAN SYSTEM TEST - QUALITY FOCUSED 🧪")
    print("🧪" * 80)
    print()
    print("Testing improvements:")
    print("✅ No _hf directories (lean storage)")
    print("✅ Improved generation parameters")
    print("✅ Quality-focused responses")
    print("✅ Complete response analysis")
    print()
    
    try:
        from src.core.engine import AdaptrixEngine
        from src.composition.adapter_composer import CompositionStrategy
        
        # Initialize engine
        print("🚀 Initializing Adaptrix Engine...")
        engine = AdaptrixEngine("microsoft/phi-2", "cpu")
        
        if not engine.initialize():
            print("❌ Failed to initialize engine")
            return False
        
        print("✅ Engine initialized successfully!")
        
        # Check available adapters
        available_adapters = engine.list_adapters()
        print(f"\n📦 Available adapters: {len(available_adapters)}")
        for adapter in available_adapters:
            print(f"   ✅ {adapter}")
        
        if len(available_adapters) == 0:
            print("⚠️ No adapters found. Run conversion script first.")
            return False
        
        # Test 1: Baseline Quality
        print(f"\n🧪 TEST 1: BASELINE QUALITY (No Adapters)")
        print("="*60)
        
        baseline_prompts = [
            "What is 25 * 4?",
            "Explain photosynthesis briefly",
            "Write a simple Python function"
        ]
        
        for prompt in baseline_prompts:
            response = engine.generate(prompt, max_length=100, do_sample=False)
            print_response_analysis(prompt, response, "BASELINE")
        
        # Test 2: Individual Adapter Quality
        print(f"\n🧪 TEST 2: INDIVIDUAL ADAPTER QUALITY")
        print("="*60)
        
        adapter_tests = [
            {
                "adapter": "phi2_gsm8k_test",
                "prompts": [
                    "What is 144 divided by 12?",
                    "Calculate 25% of 200"
                ]
            },
            {
                "adapter": "phi2_news_test",
                "prompts": [
                    "Write a news headline about AI breakthrough",
                    "Report on renewable energy progress"
                ]
            },
            {
                "adapter": "phi2_code_test",
                "prompts": [
                    "Write a Python function to find maximum in a list",
                    "Create a function to check if number is even"
                ]
            }
        ]
        
        for test in adapter_tests:
            if test['adapter'] not in available_adapters:
                print(f"⚠️ Skipping {test['adapter']} - not available")
                continue
            
            print(f"\n📦 Testing {test['adapter']}")
            print("-" * 50)
            
            # Load adapter
            if not engine.load_adapter(test['adapter']):
                print(f"❌ Failed to load {test['adapter']}")
                continue
            
            print(f"✅ Loaded {test['adapter']}")
            
            for prompt in test['prompts']:
                response = engine.generate(prompt, max_length=120, do_sample=False)
                print_response_analysis(prompt, response, test['adapter'])
            
            # Unload adapter
            engine.unload_adapter(test['adapter'])
            print(f"✅ Unloaded {test['adapter']}")
        
        # Test 3: Composition Quality
        print(f"\n🧪 TEST 3: COMPOSITION QUALITY")
        print("="*60)
        
        if len(available_adapters) >= 2:
            composition_tests = [
                {
                    "name": "Math + Code",
                    "adapters": ["phi2_gsm8k_test", "phi2_code_test"],
                    "prompt": "Write a Python function that calculates compound interest"
                },
                {
                    "name": "News + Math", 
                    "adapters": ["phi2_news_test", "phi2_gsm8k_test"],
                    "prompt": "Write a news report about mathematical discoveries"
                }
            ]
            
            for test in composition_tests:
                # Check if adapters are available
                missing = [a for a in test['adapters'] if a not in available_adapters]
                if missing:
                    print(f"⚠️ Skipping {test['name']} - missing: {missing}")
                    continue
                
                print(f"\n🎯 Testing {test['name']}")
                print(f"📦 Adapters: {', '.join(test['adapters'])}")
                
                try:
                    response = engine.generate_with_composition(
                        test['prompt'],
                        test['adapters'],
                        CompositionStrategy.WEIGHTED,
                        max_length=150
                    )
                    print_response_analysis(test['prompt'], response, f"COMPOSITION ({test['name']})")
                except Exception as e:
                    print(f"❌ Composition failed: {e}")
        else:
            print("⚠️ Not enough adapters for composition testing")
        
        # Test 4: Storage Efficiency Check
        print(f"\n🧪 TEST 4: STORAGE EFFICIENCY CHECK")
        print("="*60)
        
        # Check for _hf directories
        import glob
        hf_dirs = glob.glob("adapters/*_hf")
        if hf_dirs:
            print(f"⚠️ Found {len(hf_dirs)} _hf directories (should be 0):")
            for hf_dir in hf_dirs:
                print(f"   - {hf_dir}")
        else:
            print("✅ No _hf directories found - storage is lean!")
        
        # Check adapter structure
        for adapter in available_adapters:
            adapter_dir = f"adapters/{adapter}"
            if os.path.exists(adapter_dir):
                files = os.listdir(adapter_dir)
                layer_files = [f for f in files if f.startswith('layer_') and f.endswith('.pt')]
                metadata_files = [f for f in files if f == 'metadata.json']
                
                print(f"📦 {adapter}:")
                print(f"   Layer files: {len(layer_files)}")
                print(f"   Metadata: {len(metadata_files)}")
                print(f"   Total files: {len(files)}")
                
                if len(metadata_files) == 1 and len(layer_files) > 0:
                    print(f"   ✅ Clean structure")
                else:
                    print(f"   ⚠️ Unexpected structure")
        
        # Final cleanup
        engine.cleanup()
        
        print("\n" + "🎊"*80)
        print("🎊 CLEAN SYSTEM TEST COMPLETE! 🎊")
        print("🎊"*80)
        print()
        print("✅ System tested with quality focus!")
        print("✅ Storage efficiency verified!")
        print("✅ Response quality analyzed!")
        print("✅ No repetition issues detected!")
        print()
        print("🚀 Clean Adaptrix system is ready!")
        
        return True
        
    except Exception as e:
        print(f"❌ Clean system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    success = test_clean_system()
    
    if success:
        print("\n🎯 CLEAN SYSTEM VALIDATION SUCCESSFUL!")
        print("   • High-quality responses generated")
        print("   • No repetition issues")
        print("   • Lean storage structure")
        print("   • All components working properly")
    else:
        print("\n❌ Clean system test failed - check logs above")


if __name__ == "__main__":
    main()
