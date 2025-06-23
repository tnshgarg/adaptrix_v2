"""
Training configuration for LoRA adapters.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json
import os


@dataclass
class LoRAConfig:
    """Configuration for LoRA parameters."""
    r: int = 16  # LoRA rank
    alpha: int = 32  # LoRA alpha (scaling factor)
    dropout: float = 0.1  # LoRA dropout
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",
        "v_proj",
        "k_proj",
        "dense",
        "fc1",
        "fc2"
    ])
    bias: str = "none"  # Bias type: "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"  # Task type for PEFT


@dataclass
class TrainingConfig:
    """Comprehensive training configuration."""
    
    # Model and data
    model_name: str = "microsoft/phi-2"
    dataset_name: str = "gsm8k"
    dataset_config: Optional[str] = "main"
    
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Generation parameters
    max_length: int = 512
    max_new_tokens: int = 256
    
    # LoRA configuration
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    
    # Optimization
    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    logging_steps: int = 10
    
    # Output and saving
    output_dir: str = "adapters"
    adapter_name: str = "math_reasoning"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Hardware and performance
    device: str = "auto"
    fp16: bool = True
    dataloader_num_workers: int = 0
    remove_unused_columns: bool = False
    
    # Logging and monitoring
    report_to: List[str] = field(default_factory=lambda: [])  # ["wandb"] if you want to use wandb
    run_name: Optional[str] = None
    
    # Data processing
    train_split: str = "train"
    test_split: str = "test"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    
    # Prompt formatting
    prompt_template: str = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"
    instruction_key: str = "question"
    response_key: str = "answer"
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.run_name is None:
            self.run_name = f"{self.adapter_name}_{self.dataset_name}"
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set adapter output path
        self.adapter_output_dir = os.path.join(self.output_dir, self.adapter_name)
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            # Skip computed properties that are set in __post_init__
            if key == 'adapter_output_dir':
                continue
            if isinstance(value, LoRAConfig):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create from dictionary."""
        if 'lora' in config_dict and isinstance(config_dict['lora'], dict):
            config_dict['lora'] = LoRAConfig(**config_dict['lora'])
        return cls(**config_dict)
    
    def get_training_args(self) -> Dict[str, Any]:
        """Get arguments for Transformers TrainingArguments."""
        args = {
            'output_dir': self.adapter_output_dir,
            'num_train_epochs': self.num_epochs,
            'per_device_train_batch_size': self.batch_size,
            'per_device_eval_batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_ratio': self.warmup_ratio,
            'max_grad_norm': self.max_grad_norm,
            'save_strategy': self.save_strategy,
            'logging_steps': self.logging_steps,
            'save_total_limit': self.save_total_limit,
            'fp16': self.fp16,
            'dataloader_num_workers': self.dataloader_num_workers,
            'remove_unused_columns': self.remove_unused_columns,
            'report_to': self.report_to,
            'run_name': self.run_name,
        }

        # Add evaluation strategy only if we have eval data
        if self.evaluation_strategy != "no":
            args['eval_strategy'] = self.evaluation_strategy
            args['load_best_model_at_end'] = self.load_best_model_at_end
            args['metric_for_best_model'] = self.metric_for_best_model
            args['greater_is_better'] = self.greater_is_better

        return args
    
    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration for PEFT."""
        return {
            'r': self.lora.r,
            'lora_alpha': self.lora.alpha,
            'lora_dropout': self.lora.dropout,
            'target_modules': self.lora.target_modules,
            'bias': self.lora.bias,
            'task_type': self.lora.task_type,
        }


# Predefined configurations for different domains
MATH_CONFIG = TrainingConfig(
    adapter_name="math_reasoning",
    dataset_name="gsm8k",
    num_epochs=3,
    batch_size=2,  # Smaller batch size for math problems
    gradient_accumulation_steps=8,  # Compensate with more accumulation
    learning_rate=1e-4,  # Lower learning rate for math
    max_length=512,
    max_new_tokens=256,
    prompt_template="Solve this math problem step by step.\n\nProblem: {instruction}\n\nSolution: {response}",
    lora=LoRAConfig(r=16, alpha=32, dropout=0.1)
)

CODE_CONFIG = TrainingConfig(
    adapter_name="code_generation",
    dataset_name="code_alpaca",
    num_epochs=2,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=1024,  # Longer for code
    max_new_tokens=512,
    prompt_template="Generate code for the following task.\n\nTask: {instruction}\n\nCode: {response}",
    lora=LoRAConfig(r=32, alpha=64, dropout=0.05)  # Higher rank for code
)

CREATIVE_CONFIG = TrainingConfig(
    adapter_name="creative_writing",
    dataset_name="writing_prompts",
    num_epochs=2,
    batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1.5e-4,
    max_length=1024,
    max_new_tokens=512,
    prompt_template="Write a creative response to this prompt.\n\nPrompt: {instruction}\n\nResponse: {response}",
    lora=LoRAConfig(r=24, alpha=48, dropout=0.1)
)


def get_config_for_domain(domain: str) -> TrainingConfig:
    """Get predefined configuration for a domain."""
    configs = {
        'math': MATH_CONFIG,
        'code': CODE_CONFIG,
        'creative': CREATIVE_CONFIG
    }
    
    if domain not in configs:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(configs.keys())}")
    
    return configs[domain]
