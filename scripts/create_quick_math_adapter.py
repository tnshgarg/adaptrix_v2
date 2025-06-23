#!/usr/bin/env python3
"""
Create a quick working math adapter while the full GSM8K training runs.

This creates a smaller, faster adapter for immediate testing.
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


def create_quick_math_config():
    """Create a quick training config for immediate results."""
    
    # Create minimal LoRA config
    lora_config = LoRAConfig(
        r=16,  # Smaller rank for speed
        alpha=32,
        dropout=0.1,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.o_proj", 
            "mlp.gate_proj"
        ],  # Only 3 modules for speed
        bias="none"
    )
    
    # Create quick training config
    config = TrainingConfig(
        # Model and dataset
        model_name="deepseek-ai/deepseek-r1-distill-qwen-1.5b",
        dataset_name="gsm8k",
        dataset_config="main",
        
        # Adapter info
        adapter_name="quick_math",
        output_dir="adapters",
        
        # Training parameters - optimized for SPEED
        num_epochs=1,  # Just 1 epoch for speed
        batch_size=4,  # Larger batch
        gradient_accumulation_steps=4,  # Less accumulation
        learning_rate=1e-4,  # Higher learning rate
        weight_decay=0.01,
        warmup_ratio=0.05,  # Less warmup
        max_grad_norm=1.0,
        
        # Data limits - MUCH SMALLER
        max_train_samples=100,  # Only 100 samples for speed
        max_eval_samples=20,
        
        # Generation parameters
        max_length=256,  # Shorter sequences
        max_new_tokens=128,
        
        # LoRA configuration
        lora=lora_config,
        
        # Optimization
        optimizer="adamw_torch",
        lr_scheduler_type="linear",
        save_strategy="no",  # Don't save intermediate checkpoints
        evaluation_strategy="no",  # Skip evaluation for speed
        logging_steps=5,
        
        # Hardware
        device="cpu",
        fp16=False,
        dataloader_num_workers=0,
        
        # Prompt template - SIMPLE
        prompt_template="{instruction} = {response}",
        instruction_key="question",
        response_key="answer",
        
        # No evaluation for speed
        load_best_model_at_end=False,
        save_total_limit=1,
    )
    
    return config


def train_quick_adapter():
    """Train a quick math adapter."""
    
    print("⚡" * 60)
    print("⚡ QUICK MATH ADAPTER TRAINING ⚡")
    print("⚡" * 60)
    print()
    print("Quick training specifications:")
    print("✅ Dataset: GSM8K (100 samples)")
    print("✅ Model: DeepSeek R1 Distill 1.5B")
    print("✅ Method: LoRA fine-tuning")
    print("✅ Target: Quick math demo")
    print("✅ Epochs: 1")
    print("✅ Batch size: 4")
    print("✅ Expected time: ~10 minutes")
    print()
    
    # Create training config
    config = create_quick_math_config()
    
    print(f"📁 Output directory: {config.adapter_output_dir}")
    print(f"🎯 LoRA rank: {config.lora.r}, alpha: {config.lora.alpha}")
    print(f"📊 Training samples: {config.max_train_samples}")
    print()
    
    try:
        # Start training
        print("🚀 Starting quick training...")
        results = train_adapter(config)
        
        print("\n✅ Quick training completed!")
        print(f"📁 Adapter saved to: {results['adapter_path']}")
        
        # Print training results
        if 'train_results' in results:
            train_loss = results['train_results'].training_loss
            print(f"📊 Final training loss: {train_loss:.4f}")
        
        # Print generation examples
        if 'generation_examples' in results:
            print("\n🧪 Generation examples:")
            for i, example in enumerate(results['generation_examples'][:2]):
                print(f"\nExample {i+1}:")
                print(f"Prompt: {example['prompt'][:100]}...")
                print(f"Generated: {example['generated_response'][:100]}...")
        
        print("\n⚡ QUICK MATH ADAPTER READY! ⚡")
        print(f"Use 'quick_math' adapter in Adaptrix for immediate testing!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Quick training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_quick_adapter():
    """Test the quick adapter."""
    print("\n🧪 Testing quick adapter...")
    
    try:
        from src.core.engine import AdaptrixEngine
        
        # Initialize engine
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        if not engine.initialize():
            print("❌ Failed to initialize engine")
            return
        
        # Test baseline first
        print("\n📝 BASELINE (no adapter):")
        test_problems = ["5*12=", "7*8=", "25*4="]
        
        for problem in test_problems:
            response = engine.generate(problem, max_length=10, do_sample=False)
            answer = response.split(',')[0].strip() if ',' in response else response.strip()
            print(f"{problem} → {answer}")
        
        # Load the quick adapter
        if not engine.load_adapter("quick_math"):
            print("❌ Failed to load quick_math adapter")
            return
        
        print("\n📝 WITH QUICK ADAPTER:")
        for problem in test_problems:
            response = engine.generate(problem, max_length=10, do_sample=False)
            answer = response.split(',')[0].strip() if ',' in response else response.strip()
            print(f"{problem} → {answer}")
        
        engine.cleanup()
        print("\n✅ Quick testing completed!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")


def main():
    """Main function."""
    print("Starting quick math adapter training...")
    
    # Check if adapters directory exists
    os.makedirs("adapters", exist_ok=True)
    
    # Train the quick adapter
    results = train_quick_adapter()
    
    if results:
        # Test the quick adapter
        test_quick_adapter()
    
    print("\n⚡ QUICK PROCESS COMPLETE! ⚡")
    print("Note: Full GSM8K training is still running in the background.")


if __name__ == "__main__":
    main()
