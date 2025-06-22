"""
Train a math reasoning LoRA adapter using GSM8K dataset.
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.training.trainer import train_adapter, quick_train_math_adapter
from src.training.config import MATH_CONFIG, TrainingConfig
from src.core.engine import AdaptrixEngine


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )


def test_base_model():
    """Test the base model before training."""
    print("🧪 Testing Base Model Performance")
    print("=" * 60)
    
    try:
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        math_problems = [
            "If a store sells 3 apples for $2, how much would 12 apples cost?",
            "A rectangle has length 8 cm and width 5 cm. What is its area?",
            "If 25% of a number is 15, what is the number?"
        ]
        
        print("Base model responses to math problems:")
        for i, problem in enumerate(math_problems, 1):
            response = engine.generate(
                f"Solve this math problem step by step.\n\nProblem: {problem}\n\nSolution:",
                max_length=200,
                temperature=0.7
            )
            print(f"\n{i}. Problem: {problem}")
            print(f"   Response: {response[:150]}...")
        
        engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Base model test failed: {e}")
        return False


def train_math_adapter_full():
    """Train a full math adapter with custom configuration."""
    print("🚀 Training Math Reasoning LoRA Adapter")
    print("=" * 60)
    
    # Create custom config for quick training
    config = TrainingConfig(
        # Model and dataset
        model_name="deepseek-ai/deepseek-r1-distill-qwen-1.5b",
        dataset_name="gsm8k",
        
        # Training parameters (reduced for quick testing)
        num_epochs=1,  # Reduced for quick test
        batch_size=1,  # Small batch size for MacBook Air
        gradient_accumulation_steps=8,  # Compensate with accumulation
        learning_rate=1e-4,
        max_length=512,
        max_new_tokens=256,
        
        # Sampling (use subset for quick training)
        max_train_samples=100,  # Use only 100 samples for quick test
        max_eval_samples=20,
        
        # Output
        adapter_name="math_reasoning_test",
        
        # Hardware optimization
        fp16=False,  # Disable FP16 for CPU training
        device="cpu",  # Force CPU for compatibility
        dataloader_num_workers=0,
        
        # Prompt template for math
        prompt_template="Solve this math problem step by step.\n\nProblem: {instruction}\n\nSolution: {response}",
        instruction_key="question",
        response_key="answer"
    )
    
    try:
        print(f"📋 Training Configuration:")
        print(f"   Model: {config.model_name}")
        print(f"   Dataset: {config.dataset_name}")
        print(f"   Epochs: {config.num_epochs}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Learning rate: {config.learning_rate}")
        print(f"   Max train samples: {config.max_train_samples}")
        print(f"   Output: {config.adapter_name}")
        
        # Train the adapter
        print(f"\n🔥 Starting training...")
        results = train_adapter(config)
        
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Results:")
        print(f"   Adapter saved to: {results['adapter_path']}")
        
        if 'eval_results' in results:
            eval_results = results['eval_results']
            print(f"   Evaluation metrics:")
            for key, value in eval_results.items():
                if isinstance(value, float):
                    print(f"     {key}: {value:.4f}")
                else:
                    print(f"     {key}: {value}")
        
        if 'generation_examples' in results:
            print(f"\n💬 Generation Examples:")
            for i, example in enumerate(results['generation_examples'][:2], 1):
                print(f"   Example {i}:")
                print(f"     Prompt: {example['prompt'][:100]}...")
                print(f"     Response: {example['generated_response'][:150]}...")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_trained_adapter(adapter_path: str):
    """Test the trained adapter."""
    print(f"\n🧪 Testing Trained Adapter")
    print("=" * 60)
    
    try:
        # Load the engine with the trained adapter
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        # Load the trained adapter
        adapter_name = os.path.basename(adapter_path)
        success = engine.load_adapter(adapter_name)
        
        if not success:
            print(f"❌ Failed to load adapter: {adapter_name}")
            return False
        
        print(f"✅ Loaded adapter: {adapter_name}")
        
        # Test with math problems
        math_problems = [
            "If a store sells 3 apples for $2, how much would 12 apples cost?",
            "A rectangle has length 8 cm and width 5 cm. What is its area?",
            "If 25% of a number is 15, what is the number?"
        ]
        
        print(f"\n💬 Testing adapter responses:")
        for i, problem in enumerate(math_problems, 1):
            prompt = f"Solve this math problem step by step.\n\nProblem: {problem}\n\nSolution:"
            response = engine.generate(
                prompt,
                max_length=200,
                temperature=0.7
            )
            print(f"\n{i}. Problem: {problem}")
            print(f"   Adapter Response: {response}")
        
        engine.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Adapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main training pipeline."""
    print("🎯 ADAPTRIX MATH LORA TRAINING")
    print("=" * 80)
    print("Training custom math reasoning adapter for DeepSeek model")
    print("=" * 80)
    
    # Setup logging
    setup_logging()
    
    # Test base model first
    print("\n1️⃣ Testing base model...")
    base_test = test_base_model()
    
    if not base_test:
        print("❌ Base model test failed. Cannot proceed with training.")
        return
    
    # Train the adapter
    print("\n2️⃣ Training math adapter...")
    results = train_math_adapter_full()
    
    if results is None:
        print("❌ Training failed. Cannot proceed with testing.")
        return
    
    # Test the trained adapter
    print("\n3️⃣ Testing trained adapter...")
    adapter_test = test_trained_adapter(results['adapter_path'])
    
    # Final summary
    print(f"\n" + "=" * 80)
    print(f"🎊 TRAINING PIPELINE COMPLETE")
    print(f"=" * 80)
    print(f"✅ Base model test: {'PASSED' if base_test else 'FAILED'}")
    print(f"✅ Adapter training: {'COMPLETED' if results else 'FAILED'}")
    print(f"✅ Adapter testing: {'PASSED' if adapter_test else 'FAILED'}")
    
    if results and adapter_test:
        print(f"\n🎊 SUCCESS! Math reasoning adapter is ready!")
        print(f"📁 Adapter location: {results['adapter_path']}")
        print(f"🔧 Integration: Adapter is compatible with Adaptrix system")
        print(f"💡 Usage: Load with engine.load_adapter('math_reasoning_test')")
    else:
        print(f"\n⚠️  Training completed with issues. Check logs for details.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
