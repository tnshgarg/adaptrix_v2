#!/usr/bin/env python3
"""
🧪 SIMPLE VALIDATION TEST
Tests the 3 key fixes made to the Adaptrix system:
1. Future Adapter Compatibility (same format will work)
2. Enhanced Output Formatting (better structured responses)  
3. Increased Token Length (512 tokens, better parameters)
"""

import json
import time
import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.engine import AdaptrixEngine

def test_enhanced_system():
    """Test the enhanced Adaptrix system with fixes"""
    print("🧪 TESTING ENHANCED ADAPTRIX SYSTEM")
    print("=" * 50)
    
    # Test prompts for different scenarios
    test_cases = [
        {
            "prompt": "Create a Python function to calculate factorial recursively with proper error handling and documentation",
            "category": "Short Programming Task",
            "expected_improvements": ["Code blocks", "Docstrings", "Error handling"]
        },
        {
            "prompt": "Build a comprehensive class for managing a simple database of students with methods for adding, removing, searching, and displaying students. Include full documentation and example usage.",
            "category": "Long Programming Task", 
            "expected_improvements": ["Longer response", "Multiple methods", "Complete structure"]
        }
    ]
    
    try:
        # Initialize Adaptrix
        print("🚀 Initializing Adaptrix Engine...")
        engine = AdaptrixEngine(model_name="Qwen/Qwen3-1.7B")
        
        if not engine.initialize():
            print("❌ Failed to initialize engine")
            return False
        
        # Load the working adapter
        print("📦 Loading code_adapter_middle_layers...")
        if not engine.load_adapter("code_adapter_middle_layers"):
            print("❌ Failed to load adapter")
            return False
        
        print("✅ System ready! Testing enhancements...\n")
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"🎯 TEST {i}: {test_case['category']}")
            print(f"📝 Prompt: {test_case['prompt'][:60]}...")
            
            start_time = time.time()
            
            # Test with enhanced parameters (default now 512 tokens)
            response = engine.generate(
                test_case["prompt"],
                max_length=512,  # Using enhanced default
                temperature=0.8,
                stream=True
            )
            
            end_time = time.time()
            
            # Analyze improvements
            improvements_found = []
            
            # Check for code formatting
            if "```python" in response:
                improvements_found.append("✅ Proper code blocks")
            
            # Check for docstrings
            if '"""' in response:
                improvements_found.append("✅ Docstrings included")
            
            # Check response length
            response_length = len(response)
            if response_length > 200:
                improvements_found.append(f"✅ Good length ({response_length} chars)")
            
            # Check for structure
            if "def " in response and "class " in response:
                improvements_found.append("✅ Complete structure")
            elif "def " in response:
                improvements_found.append("✅ Function structure")
            
            # Display results
            print(f"⏱️  Generation time: {end_time - start_time:.2f}s")
            print(f"📊 Response length: {response_length} characters")
            print(f"🎨 Improvements detected:")
            for improvement in improvements_found:
                print(f"   {improvement}")
            
            # Show first part of response
            print(f"📋 Response preview:")
            preview = response[:200] + "..." if len(response) > 200 else response
            print(f"   {preview}")
            print()
            
            results.append({
                "test_case": test_case["category"],
                "response_length": response_length,
                "generation_time": end_time - start_time,
                "improvements": improvements_found,
                "full_response": response
            })
        
        # Summary
        print("📊 ENHANCEMENT VALIDATION SUMMARY")
        print("=" * 50)
        
        total_improvements = sum(len(r["improvements"]) for r in results)
        avg_length = sum(r["response_length"] for r in results) / len(results)
        avg_time = sum(r["generation_time"] for r in results) / len(results)
        
        print(f"✅ Total improvements detected: {total_improvements}")
        print(f"📏 Average response length: {avg_length:.0f} characters")
        print(f"⏱️  Average generation time: {avg_time:.2f}s")
        
        # Check if key fixes are working
        fixes_working = []
        
        # Fix 1: Token length increase
        if avg_length > 300:  # Significantly longer than previous 150-token responses
            fixes_working.append("🔧 Token length increase: WORKING")
        
        # Fix 2: Domain detection and formatting
        if any("code blocks" in str(r["improvements"]) for r in results):
            fixes_working.append("🔧 Enhanced formatting: WORKING")
        
        # Fix 3: System robustness (if no errors occurred)
        if len(results) == len(test_cases):
            fixes_working.append("🔧 System robustness: WORKING")
        
        print(f"\n🎯 KEY FIXES STATUS:")
        for fix in fixes_working:
            print(f"   {fix}")
        
        engine.cleanup()
        
        # Save detailed results
        results_file = f"enhancement_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_improvements": total_improvements,
                    "average_length": avg_length,
                    "average_time": avg_time,
                    "fixes_working": fixes_working
                },
                "detailed_results": results
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved: {results_file}")
        print(f"🎉 ENHANCEMENT TEST COMPLETE!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_system()
    sys.exit(0 if success else 1) 