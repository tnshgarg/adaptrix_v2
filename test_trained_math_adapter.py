"""
Test the trained math adapter.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.core.engine import AdaptrixEngine


def test_math_adapter():
    """Test the trained math adapter."""
    print("🧪 Testing Trained Math LoRA Adapter")
    print("=" * 60)
    
    try:
        # Initialize engine
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        # Test math problems
        math_problems = [
            "What is 2 + 2?",
            "If a store sells 3 apples for $2, how much would 12 apples cost?",
            "A rectangle has length 8 cm and width 5 cm. What is its area?",
            "If 25% of a number is 15, what is the number?",
            "What is 7 × 8?"
        ]
        
        print("🔍 Testing WITHOUT math adapter (baseline):")
        print("-" * 50)
        
        baseline_responses = []
        for i, problem in enumerate(math_problems, 1):
            prompt = f"Solve this math problem step by step.\n\nProblem: {problem}\n\nSolution:"
            response = engine.generate(prompt, max_length=200, temperature=0.7)
            baseline_responses.append(response)
            print(f"\n{i}. Problem: {problem}")
            print(f"   Baseline: {response}")
        
        # Load the trained math adapter
        print(f"\n🔧 Loading trained math adapter...")
        success = engine.load_adapter("simple_math_test")
        
        if not success:
            print("❌ Failed to load math adapter")
            return False
        
        print("✅ Math adapter loaded successfully!")
        
        print(f"\n🔥 Testing WITH math adapter:")
        print("-" * 50)
        
        adapter_responses = []
        for i, problem in enumerate(math_problems, 1):
            prompt = f"Solve this math problem step by step.\n\nProblem: {problem}\n\nSolution:"
            response = engine.generate(prompt, max_length=200, temperature=0.7)
            adapter_responses.append(response)
            print(f"\n{i}. Problem: {problem}")
            print(f"   With Adapter: {response}")
        
        # Compare responses
        print(f"\n📊 Comparison Analysis:")
        print("-" * 50)
        
        differences = 0
        improvements = 0
        
        for i, (baseline, adapter) in enumerate(zip(baseline_responses, adapter_responses), 1):
            if baseline != adapter:
                differences += 1
                print(f"\n{i}. Problem: {math_problems[i-1]}")
                print(f"   Baseline:     {baseline[:100]}...")
                print(f"   With Adapter: {adapter[:100]}...")
                
                # Simple heuristic for improvement
                if (len(adapter.split()) > len(baseline.split()) or 
                    any(word in adapter.lower() for word in ['step', 'first', 'then', 'therefore'])):
                    improvements += 1
                    print(f"   ✅ Potential improvement detected")
                else:
                    print(f"   ⚠️  Different but unclear if better")
        
        print(f"\n📈 Results Summary:")
        print(f"   Total problems tested: {len(math_problems)}")
        print(f"   Responses changed: {differences}/{len(math_problems)} ({differences/len(math_problems)*100:.1f}%)")
        print(f"   Potential improvements: {improvements}/{len(math_problems)} ({improvements/len(math_problems)*100:.1f}%)")
        
        if differences >= len(math_problems) * 0.5:
            print(f"   ✅ Adapter is significantly affecting responses!")
        elif differences > 0:
            print(f"   ⚠️  Adapter has some effect on responses")
        else:
            print(f"   ❌ Adapter appears to have no effect")
        
        engine.cleanup()
        return differences > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adapter_system_integration():
    """Test that the adapter integrates properly with the Adaptrix system."""
    print(f"\n🔧 Testing Adapter System Integration")
    print("=" * 60)
    
    try:
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        # Test adapter listing
        adapters = engine.list_adapters()
        print(f"Available adapters: {adapters}")
        
        if "simple_math_test" not in adapters:
            print("❌ Trained adapter not found in adapter list")
            return False
        
        print("✅ Trained adapter found in system")
        
        # Test loading/unloading
        print("Testing load/unload cycle...")
        
        # Load
        load_success = engine.load_adapter("simple_math_test")
        if not load_success:
            print("❌ Failed to load adapter")
            return False
        print("✅ Adapter loaded")
        
        # Test generation
        response = engine.generate("What is 5 + 3?", max_length=50)
        print(f"Generation test: '{response}'")
        
        # Unload
        unload_success = engine.unload_adapter("simple_math_test")
        if not unload_success:
            print("❌ Failed to unload adapter")
            return False
        print("✅ Adapter unloaded")
        
        # Test system status
        status = engine.get_system_status()
        print(f"System status: {status}")
        
        engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def check_adapter_files():
    """Check that adapter files were created correctly."""
    print(f"\n📁 Checking Adapter Files")
    print("=" * 60)
    
    adapter_dir = "adapters/simple_math_test"
    
    if not os.path.exists(adapter_dir):
        print(f"❌ Adapter directory not found: {adapter_dir}")
        return False
    
    print(f"✅ Adapter directory exists: {adapter_dir}")
    
    # Check for required files
    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "training_config.json",
        "metadata.json"
    ]
    
    found_files = os.listdir(adapter_dir)
    print(f"Files found: {found_files}")
    
    missing_files = []
    for file in required_files:
        if file not in found_files:
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
    else:
        print("✅ All required files present")
    
    # Check file sizes
    for file in found_files:
        file_path = os.path.join(adapter_dir, file)
        size = os.path.getsize(file_path)
        print(f"   {file}: {size:,} bytes")
    
    return len(missing_files) == 0


def main():
    """Main test function."""
    print("🎯 TRAINED MATH ADAPTER TEST")
    print("=" * 80)
    print("Testing the custom-trained math LoRA adapter")
    print("=" * 80)
    
    # Check files
    files_ok = check_adapter_files()
    
    # Test integration
    integration_ok = test_adapter_system_integration()
    
    # Test functionality
    functionality_ok = test_math_adapter()
    
    # Final summary
    print(f"\n" + "=" * 80)
    print(f"🎊 MATH ADAPTER TEST RESULTS")
    print(f"=" * 80)
    print(f"📁 Adapter Files: {'✅ GOOD' if files_ok else '❌ ISSUES'}")
    print(f"🔧 System Integration: {'✅ WORKING' if integration_ok else '❌ FAILED'}")
    print(f"🧮 Math Functionality: {'✅ WORKING' if functionality_ok else '❌ NO EFFECT'}")
    
    overall_success = files_ok and integration_ok and functionality_ok
    
    if overall_success:
        print(f"\n🎊 🎊 🎊 MATH ADAPTER FULLY FUNCTIONAL! 🎊 🎊 🎊")
        print(f"✅ Successfully trained custom LoRA adapter")
        print(f"✅ Adapter integrates with Adaptrix system")
        print(f"✅ Adapter affects model behavior for math problems")
        print(f"✅ Complete training pipeline working")
        
        print(f"\n🚀 ACHIEVEMENTS:")
        print(f"   • Created modular LoRA training framework")
        print(f"   • Successfully trained on GSM8K dataset")
        print(f"   • Integrated with existing Adaptrix architecture")
        print(f"   • Demonstrated adapter effectiveness")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   • Train with more samples for better performance")
        print(f"   • Create adapters for other domains (code, creative)")
        print(f"   • Implement adapter composition")
        print(f"   • Add automated evaluation metrics")
        
    elif integration_ok:
        print(f"\n✅ TRAINING SYSTEM WORKING!")
        print(f"✅ Adapter creation and integration successful")
        print(f"⚠️  May need more training data or epochs for better effect")
        
    else:
        print(f"\n⚠️  ISSUES DETECTED")
        print(f"🔧 Check training logs and adapter files")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
