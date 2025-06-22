"""
LoRA trainer for creating custom adapters.
"""

import logging
import os
import torch
from typing import Dict, List, Optional, Any
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import json
from datetime import datetime

from .config import TrainingConfig
from .data_handler import get_dataset_handler
from .evaluator import AdapterEvaluator

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Main trainer class for LoRA adapters."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the LoRA trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset_handler = None
        self.trainer = None
        self.evaluator = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger.info(f"Initializing LoRA trainer for {config.adapter_name}")
    
    def setup(self):
        """Setup model, tokenizer, and dataset."""
        logger.info("Setting up trainer components...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        logger.info(f"Loading base model: {self.config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            device_map="auto" if self.config.device == "auto" else None,
            trust_remote_code=True
        )
        
        # Setup LoRA
        self._setup_lora()
        
        # Setup dataset
        self.dataset_handler = get_dataset_handler(self.config)
        self.dataset_handler.set_tokenizer(self.tokenizer)
        
        # Setup evaluator
        self.evaluator = AdapterEvaluator(self.config, self.model, self.tokenizer)
        
        logger.info("Setup complete!")
    
    def _setup_lora(self):
        """Setup LoRA configuration and apply to model."""
        logger.info("Setting up LoRA configuration...")

        # Get LoRA config dict and ensure no conflicts
        lora_config_dict = self.config.get_lora_config()

        # Create LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config_dict['r'],
            lora_alpha=lora_config_dict['lora_alpha'],
            lora_dropout=lora_config_dict['lora_dropout'],
            target_modules=lora_config_dict['target_modules'],
            bias=lora_config_dict['bias']
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("LoRA setup complete!")
    
    def prepare_data(self):
        """Load and prepare training data."""
        logger.info("Preparing training data...")
        
        # Load dataset
        dataset = self.dataset_handler.load_dataset()
        
        # Tokenize dataset
        tokenized_dataset = self.dataset_handler.tokenize_dataset()
        
        logger.info("Data preparation complete!")
        return tokenized_dataset
    
    def train(self) -> Dict[str, Any]:
        """Train the LoRA adapter."""
        logger.info(f"Starting training for {self.config.adapter_name}")
        
        # Prepare data
        tokenized_dataset = self.prepare_data()
        
        # Setup training arguments
        training_args = TrainingArguments(**self.config.get_training_args())
        
        # Get data collator
        data_collator = self.dataset_handler.get_data_collator()
        
        # Setup callbacks
        callbacks = []
        if (self.config.evaluation_strategy != "no" and
            self.config.test_split in tokenized_dataset and
            tokenized_dataset[self.config.test_split] is not None):
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset[self.config.train_split],
            eval_dataset=tokenized_dataset.get(self.config.test_split) if self.config.evaluation_strategy != "no" else None,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=callbacks if callbacks else None
        )
        
        # Train
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save the adapter
        self.save_adapter()
        
        # Evaluate (skip if evaluation strategy is "no")
        eval_results = {}
        if self.config.evaluation_strategy != "no":
            eval_results = self.evaluate()
        
        # Combine results
        results = {
            'train_results': train_result,
            'eval_results': eval_results,
            'adapter_path': self.config.adapter_output_dir
        }
        
        logger.info("Training complete!")
        return results
    
    def save_adapter(self):
        """Save the trained LoRA adapter."""
        logger.info(f"Saving adapter to {self.config.adapter_output_dir}")
        
        # Create output directory
        os.makedirs(self.config.adapter_output_dir, exist_ok=True)
        
        # Save the adapter
        self.model.save_pretrained(self.config.adapter_output_dir)
        self.tokenizer.save_pretrained(self.config.adapter_output_dir)
        
        # Save training config
        config_path = os.path.join(self.config.adapter_output_dir, "training_config.json")
        self.config.save(config_path)
        
        # Create metadata in Adaptrix-compatible format
        lora_config = self.config.get_lora_config()

        # Map DeepSeek layers to target layers (use a subset for efficiency)
        # DeepSeek-R1-1.5B has 24 layers, we'll target layers spread across the model
        target_layers = [6, 12, 18]  # Early, middle, late layers

        metadata = {
            # Required fields for Adaptrix system
            'name': self.config.adapter_name,
            'version': '1.0.0',
            'target_layers': target_layers,
            'rank': lora_config['r'],
            'alpha': lora_config['lora_alpha'],

            # Additional metadata
            'description': f"LoRA adapter for {self.config.dataset_name} trained on {self.config.model_name}",
            'base_model': self.config.model_name,
            'dataset': self.config.dataset_name,
            'task_type': 'causal_lm',
            'target_modules': lora_config['target_modules'],
            'created_at': datetime.now().isoformat(),
            'source': 'custom_training',

            # Training details
            'training_config': {
                'epochs': self.config.num_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'dataset_samples': self.config.max_train_samples,
                'lora_dropout': lora_config['lora_dropout']
            }
        }
        
        metadata_path = os.path.join(self.config.adapter_output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Adapter saved successfully!")
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the trained adapter."""
        if self.evaluator is None:
            logger.warning("No evaluator available")
            return {}
        
        logger.info("Evaluating adapter...")
        return self.evaluator.evaluate()
    
    def test_generation(self, prompts: Optional[List[str]] = None, num_samples: int = 3) -> List[Dict[str, str]]:
        """Test generation with the trained adapter."""
        if prompts is None:
            prompts = self.dataset_handler.get_sample_prompts(num_samples)
        
        logger.info(f"Testing generation with {len(prompts)} prompts...")
        
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating for prompt {i+1}/{len(prompts)}")
            
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = full_response[len(prompt):].strip()
            
            results.append({
                'prompt': prompt,
                'generated_response': generated_part,
                'full_response': full_response
            })
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("Cleanup complete!")


def train_adapter(config: TrainingConfig) -> Dict[str, Any]:
    """
    Convenience function to train a LoRA adapter.
    
    Args:
        config: Training configuration
        
    Returns:
        Training results
    """
    trainer = LoRATrainer(config)
    
    try:
        # Setup
        trainer.setup()
        
        # Train
        results = trainer.train()
        
        # Test generation
        generation_results = trainer.test_generation()
        results['generation_examples'] = generation_results
        
        return results
    
    finally:
        # Always cleanup
        trainer.cleanup()


def quick_train_math_adapter(
    num_epochs: int = 2,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    max_train_samples: Optional[int] = 1000
) -> Dict[str, Any]:
    """
    Quick function to train a math reasoning adapter with default settings.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_train_samples: Maximum training samples (None for all)
        
    Returns:
        Training results
    """
    from .config import MATH_CONFIG
    
    # Create config with custom parameters
    config = MATH_CONFIG
    config.num_epochs = num_epochs
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.max_train_samples = max_train_samples
    
    logger.info(f"Quick training math adapter with {num_epochs} epochs, {batch_size} batch size")
    
    return train_adapter(config)
