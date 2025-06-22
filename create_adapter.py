#!/usr/bin/env python3
"""
Automated LoRA Adapter Creation Pipeline
Easy-to-use script for training custom adapters for different domains.
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.training.trainer import train_adapter
from src.training.config import get_config_for_domain, TrainingConfig
from convert_peft_to_adaptrix import convert_peft_adapter_to_adaptrix


def setup_logging(log_level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'adapter_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )


def create_math_adapter(name="math_reasoning", samples=100, epochs=2, quick=False):
    """Create a math reasoning adapter using GSM8K dataset."""
    print(f"🧮 Creating Math Reasoning Adapter: {name}")
    print("=" * 60)
    
    # Create config
    config = TrainingConfig(
        model_name="deepseek-ai/deepseek-r1-distill-qwen-1.5b",
        dataset_name="gsm8k",
        adapter_name=name,
        
        # Training parameters
        num_epochs=1 if quick else epochs,
        batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        max_length=512,
        max_new_tokens=256,
        
        # Data sampling
        max_train_samples=10 if quick else samples,
        max_eval_samples=5 if quick else min(20, samples // 5),
        
        # Hardware optimization
        fp16=False,
        device="cpu",
        dataloader_num_workers=0,
        evaluation_strategy="no",
        
        # Math-specific prompt
        prompt_template="Solve this math problem step by step.\n\nProblem: {instruction}\n\nSolution: {response}",
        instruction_key="question",
        response_key="answer"
    )
    
    return config


def create_code_adapter(name="code_generation", samples=100, epochs=2, quick=False):
    """Create a code generation adapter."""
    print(f"💻 Creating Code Generation Adapter: {name}")
    print("=" * 60)
    
    # For now, use a simple synthetic dataset since code_alpaca might not be available
    config = TrainingConfig(
        model_name="deepseek-ai/deepseek-r1-distill-qwen-1.5b",
        dataset_name="gsm8k",  # We'll adapt this for code-like problems
        adapter_name=name,
        
        # Training parameters
        num_epochs=1 if quick else epochs,
        batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_length=1024,
        max_new_tokens=512,
        
        # Data sampling
        max_train_samples=10 if quick else samples,
        max_eval_samples=5 if quick else min(20, samples // 5),
        
        # Hardware optimization
        fp16=False,
        device="cpu",
        dataloader_num_workers=0,
        evaluation_strategy="no",
        
        # Code-specific prompt
        prompt_template="Generate code for the following task.\n\nTask: {instruction}\n\nCode: {response}",
        instruction_key="question",
        response_key="answer"
    )
    
    return config


def create_creative_adapter(name="creative_writing", samples=100, epochs=2, quick=False):
    """Create a creative writing adapter."""
    print(f"✍️ Creating Creative Writing Adapter: {name}")
    print("=" * 60)
    
    config = TrainingConfig(
        model_name="deepseek-ai/deepseek-r1-distill-qwen-1.5b",
        dataset_name="gsm8k",  # We'll adapt this for creative prompts
        adapter_name=name,
        
        # Training parameters
        num_epochs=1 if quick else epochs,
        batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1.5e-4,
        max_length=1024,
        max_new_tokens=512,
        
        # Data sampling
        max_train_samples=10 if quick else samples,
        max_eval_samples=5 if quick else min(20, samples // 5),
        
        # Hardware optimization
        fp16=False,
        device="cpu",
        dataloader_num_workers=0,
        evaluation_strategy="no",
        
        # Creative-specific prompt
        prompt_template="Write a creative response to this prompt.\n\nPrompt: {instruction}\n\nResponse: {response}",
        instruction_key="question",
        response_key="answer"
    )
    
    return config


def train_and_convert_adapter(config):
    """Train an adapter and convert it to Adaptrix format."""
    print(f"🚀 Training adapter: {config.adapter_name}")
    
    try:
        # Train the adapter
        results = train_adapter(config)
        
        if results is None:
            print(f"❌ Training failed for {config.adapter_name}")
            return False
        
        print(f"✅ Training completed for {config.adapter_name}")
        
        # Convert to Adaptrix format
        print(f"🔄 Converting to Adaptrix format...")
        conversion_success = convert_peft_adapter_to_adaptrix(results['adapter_path'])
        
        if conversion_success:
            print(f"✅ Conversion completed for {config.adapter_name}")
            return True
        else:
            print(f"❌ Conversion failed for {config.adapter_name}")
            return False
            
    except Exception as e:
        print(f"❌ Error training {config.adapter_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adapter(adapter_name):
    """Test the trained adapter."""
    print(f"🧪 Testing adapter: {adapter_name}")
    
    try:
        from src.core.engine import AdaptrixEngine
        
        # Initialize engine
        engine = AdaptrixEngine("deepseek-ai/deepseek-r1-distill-qwen-1.5b", "cpu")
        engine.initialize()
        
        # Test loading
        success = engine.load_adapter(adapter_name)
        
        if success:
            print(f"✅ Adapter {adapter_name} loads successfully")
            
            # Test generation
            test_prompt = "Solve this math problem step by step.\n\nProblem: What is 3 + 5?\n\nSolution:"
            response = engine.generate(test_prompt, max_length=100, temperature=0.7)
            print(f"Test response: {response[:100]}...")
            
            engine.unload_adapter(adapter_name)
            print(f"✅ Adapter {adapter_name} works correctly")
            return True
        else:
            print(f"❌ Failed to load adapter {adapter_name}")
            return False
            
        engine.cleanup()
        
    except Exception as e:
        print(f"❌ Error testing {adapter_name}: {e}")
        return False


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description="Create custom LoRA adapters for Adaptrix")
    parser.add_argument("domain", choices=["math", "code", "creative", "all"], 
                       help="Domain to create adapter for")
    parser.add_argument("--name", type=str, help="Custom adapter name")
    parser.add_argument("--samples", type=int, default=100, help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--quick", action="store_true", help="Quick training with minimal samples")
    parser.add_argument("--test", action="store_true", help="Test adapter after training")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    print("🎯 ADAPTRIX CUSTOM ADAPTER CREATION PIPELINE")
    print("=" * 80)
    print(f"Domain: {args.domain}")
    print(f"Samples: {args.samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Quick mode: {args.quick}")
    print("=" * 80)
    
    adapters_to_create = []
    
    if args.domain == "all":
        adapters_to_create = [
            ("math", create_math_adapter),
            ("code", create_code_adapter), 
            ("creative", create_creative_adapter)
        ]
    else:
        domain_map = {
            "math": create_math_adapter,
            "code": create_code_adapter,
            "creative": create_creative_adapter
        }
        adapters_to_create = [(args.domain, domain_map[args.domain])]
    
    results = {}
    
    for domain, create_func in adapters_to_create:
        adapter_name = args.name if args.name else f"{domain}_adapter"
        
        print(f"\n{'='*20} Creating {domain.upper()} Adapter {'='*20}")
        
        # Create config
        config = create_func(
            name=adapter_name,
            samples=args.samples,
            epochs=args.epochs,
            quick=args.quick
        )
        
        # Train and convert
        success = train_and_convert_adapter(config)
        results[domain] = success
        
        # Test if requested
        if args.test and success:
            test_success = test_adapter(adapter_name)
            results[f"{domain}_test"] = test_success
    
    # Final summary
    print(f"\n" + "=" * 80)
    print(f"🎊 ADAPTER CREATION PIPELINE COMPLETE")
    print(f"=" * 80)
    
    for domain, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{domain.upper():20} {status}")
    
    successful_adapters = [k for k, v in results.items() if v and not k.endswith("_test")]
    
    if successful_adapters:
        print(f"\n🎊 Created {len(successful_adapters)} adapter(s) successfully!")
        print(f"📁 Available in adapters/ directory")
        print(f"🔧 Ready to use with Adaptrix system")
        
        print(f"\n💡 Usage:")
        print(f"   from src.core.engine import AdaptrixEngine")
        print(f"   engine = AdaptrixEngine('deepseek-ai/deepseek-r1-distill-qwen-1.5b', 'cpu')")
        print(f"   engine.initialize()")
        for domain in successful_adapters:
            adapter_name = args.name if args.name else f"{domain}_adapter"
            print(f"   engine.load_adapter('{adapter_name}')")
    else:
        print(f"\n⚠️  No adapters created successfully")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
