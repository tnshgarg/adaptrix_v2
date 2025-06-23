#!/usr/bin/env python3
"""
Train a REAL GSM8K math adapter for Adaptrix.

This script trains a proper LoRA adapter on 1000 GSM8K samples
using the existing training infrastructure.
"""

import sys
import os
import torch
import json
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.training.trainer import train_adapter
from src.training.config import TrainingConfig, LoRAConfig


def create_gsm8k_training_config():
    """Create optimized training config for GSM8K math adapter."""
    
    # Create LoRA config optimized for math reasoning
    lora_config = LoRAConfig(
        r=32,  # Higher rank for better math capacity
        alpha=64,  # Higher alpha for stronger effect
        dropout=0.05,  # Lower dropout for stability
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "dense",
            "fc1",
            "fc2"
        ],  # Phi-2 module names
        bias="none"
    )
    
    # Create training config
    config = TrainingConfig(
        # Model and dataset
        model_name="microsoft/phi-2",
        dataset_name="gsm8k",
        dataset_config="main",
        
        # Adapter info
        adapter_name="gsm8k_math",
        output_dir="adapters",
        
        # Training parameters - optimized for quality
        num_epochs=3,
        batch_size=1,  # Small batch for memory efficiency
        gradient_accumulation_steps=16,  # Compensate with accumulation
        learning_rate=5e-5,  # Lower learning rate for stability
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        
        # Data limits
        max_train_samples=1000,  # Exactly 1000 samples as requested
        max_eval_samples=100,
        
        # Generation parameters
        max_length=512,
        max_new_tokens=256,
        
        # LoRA configuration
        lora=lora_config,
        
        # Optimization
        optimizer="adamw_torch",
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_steps=10,
        
        # Hardware
        device="cpu",  # Use CPU for compatibility
        fp16=False,  # Disable fp16 for CPU
        dataloader_num_workers=0,
        
        # Prompt template optimized for math
        prompt_template="Solve this math problem step by step.\n\nProblem: {instruction}\n\nSolution: {response}",
        instruction_key="question",
        response_key="answer",
        
        # Evaluation
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
    )
    
    return config


def train_gsm8k_adapter():
    """Train the GSM8K math adapter."""
    
    print("🧮" * 60)
    print("🧮 TRAINING REAL GSM8K MATH ADAPTER 🧮")
    print("🧮" * 60)
    print()
    print("Training specifications:")
    print("✅ Dataset: GSM8K (1000 samples)")
    print("✅ Model: DeepSeek R1 Distill 1.5B")
    print("✅ Method: LoRA fine-tuning")
    print("✅ Target: Mathematical reasoning")
    print("✅ Epochs: 3")
    print("✅ Batch size: 1 (with 16x accumulation)")
    print()
    
    # Create training config
    config = create_gsm8k_training_config()
    
    print(f"📁 Output directory: {config.adapter_output_dir}")
    print(f"🎯 LoRA rank: {config.lora.r}, alpha: {config.lora.alpha}")
    print(f"📊 Training samples: {config.max_train_samples}")
    print()
    
    try:
        # Start training
        print("🚀 Starting training...")
        results = train_adapter(config)
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Adapter saved to: {results['adapter_path']}")
        
        # Print training results
        if 'train_results' in results:
            train_loss = results['train_results'].training_loss
            print(f"📊 Final training loss: {train_loss:.4f}")
        
        if 'eval_results' in results:
            eval_loss = results['eval_results'].get('eval_loss', 'N/A')
            print(f"📊 Final eval loss: {eval_loss}")
        
        # Print generation examples
        if 'generation_examples' in results:
            print("\n🧪 Generation examples:")
            for i, example in enumerate(results['generation_examples'][:3]):
                print(f"\nExample {i+1}:")
                print(f"Prompt: {example['prompt'][:100]}...")
                print(f"Generated: {example['generated_response'][:200]}...")
        
        print("\n🎊 GSM8K MATH ADAPTER TRAINING COMPLETE! 🎊")
        print(f"Use 'gsm8k_math' adapter in Adaptrix for mathematical reasoning!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_trained_adapter():
    """Test the trained adapter with sample math problems."""
    print("\n🧪 Testing trained adapter...")
    
    try:
        from src.core.engine import AdaptrixEngine
        
        # Initialize engine
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        if not engine.initialize():
            print("❌ Failed to initialize engine")
            return
        
        # Load the trained adapter
        if not engine.load_adapter("gsm8k_math"):
            print("❌ Failed to load gsm8k_math adapter")
            return
        
        # Test with sample problems
        test_problems = [
            "What is 25 * 4?",
            "If John has 15 apples and gives away 7, how many does he have left?",
            "A rectangle has length 8 and width 5. What is its area?",
            "What is 144 divided by 12?",
        ]
        
        print("\n📝 Testing with sample problems:")
        for problem in test_problems:
            print(f"\n❓ {problem}")
            
            # Format as the adapter expects
            prompt = f"Solve this math problem step by step.\n\nProblem: {problem}\n\nSolution:"
            response = engine.generate(prompt, max_length=100, do_sample=False)
            
            print(f"🤖 {response}")
        
        engine.cleanup()
        print("\n✅ Testing completed!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")


def main():
    """Main function."""
    print("Starting GSM8K adapter training...")
    
    # Check if adapters directory exists
    os.makedirs("adapters", exist_ok=True)
    
    # Train the adapter
    results = train_gsm8k_adapter()
    
    if results:
        # Test the trained adapter
        test_trained_adapter()
    
    print("\n🎊 PROCESS COMPLETE! 🎊")


if __name__ == "__main__":
    main()
