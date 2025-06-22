"""
Simple training script for math LoRA adapter.
Uses a minimal setup for quick testing.
"""

import sys
import os
import logging
import torch
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.training.config import TrainingConfig
from src.training.trainer import LoRATrainer


def create_simple_math_config():
    """Create a simple configuration for quick math training."""
    config = TrainingConfig(
        # Model and dataset
        model_name="deepseek-ai/deepseek-r1-distill-qwen-1.5b",
        dataset_name="gsm8k",
        
        # Training parameters (minimal for testing)
        num_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        max_length=256,  # Shorter for quick training
        max_new_tokens=128,
        
        # Use very small subset for testing
        max_train_samples=10,  # Only 10 samples for quick test
        max_eval_samples=5,
        
        # Output
        adapter_name="simple_math_test",
        
        # Hardware settings
        fp16=False,  # Disable for CPU
        device="cpu",
        dataloader_num_workers=0,
        
        # Disable evaluation for simplicity
        evaluation_strategy="no",
        save_strategy="epoch",
        
        # Logging
        logging_steps=1,
        report_to=[],  # No external logging
        
        # Math-specific prompt
        prompt_template="Solve this math problem step by step.\n\nProblem: {instruction}\n\nSolution: {response}",
        instruction_key="question",
        response_key="answer"
    )
    
    return config


def simple_train():
    """Simple training function."""
    print("🚀 Simple Math LoRA Training")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create config
    config = create_simple_math_config()
    print(f"📋 Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Dataset: {config.dataset_name}")
    print(f"   Samples: {config.max_train_samples}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Output: {config.adapter_name}")
    
    # Create trainer
    trainer = LoRATrainer(config)
    
    try:
        print(f"\n1️⃣ Setting up trainer...")
        trainer.setup()
        
        print(f"\n2️⃣ Starting training...")
        results = trainer.train()
        
        print(f"\n3️⃣ Testing generation...")
        test_prompts = [
            "Solve this math problem step by step.\n\nProblem: What is 2 + 2?\n\nSolution:",
            "Solve this math problem step by step.\n\nProblem: If I have 5 apples and eat 2, how many do I have left?\n\nSolution:"
        ]
        
        generation_results = trainer.test_generation(test_prompts)
        
        print(f"\n✅ Training completed!")
        print(f"📁 Adapter saved to: {results['adapter_path']}")
        
        print(f"\n💬 Generation Examples:")
        for i, example in enumerate(generation_results, 1):
            print(f"\nExample {i}:")
            print(f"Prompt: {example['prompt'][:100]}...")
            print(f"Response: {example['generated_response']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        trainer.cleanup()


def test_adapter_integration():
    """Test the trained adapter with Adaptrix system."""
    print(f"\n🧪 Testing Adapter Integration")
    print("=" * 60)
    
    try:
        from src.core.engine import AdaptrixEngine
        
        # Initialize engine
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        # Test without adapter
        print("Testing without adapter:")
        baseline_response = engine.generate(
            "Solve this math problem step by step.\n\nProblem: What is 3 + 4?\n\nSolution:",
            max_length=150,
            temperature=0.7
        )
        print(f"Baseline: {baseline_response}")
        
        # Load the trained adapter
        print(f"\nLoading trained adapter...")
        success = engine.load_adapter("simple_math_test")
        
        if success:
            print("✅ Adapter loaded successfully!")
            
            # Test with adapter
            print("Testing with adapter:")
            adapter_response = engine.generate(
                "Solve this math problem step by step.\n\nProblem: What is 3 + 4?\n\nSolution:",
                max_length=150,
                temperature=0.7
            )
            print(f"With adapter: {adapter_response}")
            
            # Compare responses
            if adapter_response != baseline_response:
                print("✅ Adapter is affecting model behavior!")
            else:
                print("⚠️  Adapter response is similar to baseline")
            
        else:
            print("❌ Failed to load adapter")
        
        engine.cleanup()
        return success
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("🎯 SIMPLE MATH LORA ADAPTER TRAINING")
    print("=" * 80)
    print("Quick training test with minimal configuration")
    print("=" * 80)
    
    # Train the adapter
    results = simple_train()
    
    if results is None:
        print("❌ Training failed. Cannot proceed.")
        return
    
    # Test integration
    integration_success = test_adapter_integration()
    
    # Final summary
    print(f"\n" + "=" * 80)
    print(f"🎊 TRAINING SUMMARY")
    print(f"=" * 80)
    print(f"✅ Training: {'COMPLETED' if results else 'FAILED'}")
    print(f"✅ Integration: {'WORKING' if integration_success else 'FAILED'}")
    
    if results and integration_success:
        print(f"\n🎊 SUCCESS! Math adapter training pipeline is working!")
        print(f"📁 Adapter location: {results['adapter_path']}")
        print(f"🔧 System integration: FUNCTIONAL")
        print(f"💡 Ready for full-scale training with larger datasets")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"1. Increase training samples (currently using only 10)")
        print(f"2. Add more epochs for better learning")
        print(f"3. Experiment with different LoRA parameters")
        print(f"4. Create adapters for other domains (code, creative writing)")
        print(f"5. Implement adapter composition and switching")
        
    else:
        print(f"\n⚠️  Issues detected. Check logs for details.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
