"""
Dataset handling for LoRA training.
Supports GSM8K and other datasets with modular preprocessing.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer
import re
import json

logger = logging.getLogger(__name__)


class DatasetHandler:
    """Handles dataset loading, preprocessing, and formatting for LoRA training."""
    
    def __init__(self, config):
        """
        Initialize dataset handler.
        
        Args:
            config: TrainingConfig object with dataset parameters
        """
        self.config = config
        self.tokenizer = None
        self.dataset = None
        
    def set_tokenizer(self, tokenizer: PreTrainedTokenizer):
        """Set the tokenizer for text processing."""
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_dataset(self) -> DatasetDict:
        """Load and preprocess the dataset."""
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        # Load dataset based on name
        if self.config.dataset_name == "gsm8k":
            self.dataset = self._load_gsm8k()
        elif self.config.dataset_name == "code_alpaca":
            self.dataset = self._load_code_alpaca()
        elif self.config.dataset_name == "writing_prompts":
            self.dataset = self._load_writing_prompts()
        else:
            # Try to load as a generic HuggingFace dataset
            self.dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config,
                trust_remote_code=True
            )
        
        # Apply preprocessing
        self.dataset = self._preprocess_dataset(self.dataset)
        
        # Apply sampling if specified
        if self.config.max_train_samples:
            train_size = min(self.config.max_train_samples, len(self.dataset[self.config.train_split]))
            self.dataset[self.config.train_split] = self.dataset[self.config.train_split].select(range(train_size))
        
        if self.config.max_eval_samples and self.config.test_split in self.dataset:
            eval_size = min(self.config.max_eval_samples, len(self.dataset[self.config.test_split]))
            self.dataset[self.config.test_split] = self.dataset[self.config.test_split].select(range(eval_size))
        
        logger.info(f"Dataset loaded: {len(self.dataset[self.config.train_split])} train samples")
        if self.config.test_split in self.dataset:
            logger.info(f"Dataset loaded: {len(self.dataset[self.config.test_split])} eval samples")
        
        return self.dataset
    
    def _load_gsm8k(self) -> DatasetDict:
        """Load and format GSM8K dataset."""
        dataset = load_dataset("gsm8k", "main")
        
        def format_gsm8k(example):
            # Extract the final answer from the solution
            answer = self._extract_answer_from_solution(example['answer'])
            
            return {
                'instruction': example['question'],
                'response': example['answer'],
                'answer_only': answer
            }
        
        return dataset.map(format_gsm8k)
    
    def _load_code_alpaca(self) -> DatasetDict:
        """Load and format Code Alpaca dataset."""
        # This is a placeholder - you would implement based on actual dataset structure
        dataset = load_dataset("sahil2801/CodeAlpaca-20k")
        
        def format_code_alpaca(example):
            return {
                'instruction': example.get('instruction', ''),
                'response': example.get('output', ''),
                'input': example.get('input', '')
            }
        
        return dataset.map(format_code_alpaca)
    
    def _load_writing_prompts(self) -> DatasetDict:
        """Load and format writing prompts dataset."""
        # This is a placeholder - you would implement based on actual dataset structure
        dataset = load_dataset("writing_prompts")
        
        def format_writing_prompts(example):
            return {
                'instruction': example.get('prompt', ''),
                'response': example.get('story', '')
            }
        
        return dataset.map(format_writing_prompts)
    
    def _extract_answer_from_solution(self, solution: str) -> str:
        """Extract the final numerical answer from GSM8K solution."""
        # GSM8K answers typically end with "#### <number>"
        match = re.search(r'####\s*([0-9,]+)', solution)
        if match:
            return match.group(1).replace(',', '')
        
        # Fallback: look for numbers at the end
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', solution)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return "Unknown"
    
    def _preprocess_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Apply general preprocessing to the dataset."""
        def preprocess_function(examples):
            # Format using the prompt template
            instructions = examples[self.config.instruction_key]
            responses = examples[self.config.response_key]
            
            formatted_texts = []
            for instruction, response in zip(instructions, responses):
                formatted_text = self.config.prompt_template.format(
                    instruction=instruction,
                    response=response
                )
                formatted_texts.append(formatted_text)
            
            return {'formatted_text': formatted_texts}
        
        return dataset.map(preprocess_function, batched=True)
    
    def tokenize_dataset(self) -> DatasetDict:
        """Tokenize the dataset for training."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set. Call set_tokenizer() first.")
        
        def tokenize_function(examples):
            # Tokenize the formatted text
            tokenized = self.tokenizer(
                examples['formatted_text'],
                truncation=True,
                padding=False,  # We'll pad dynamically during training
                max_length=self.config.max_length,
                return_tensors=None
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            return tokenized
        
        tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.dataset[self.config.train_split].column_names
        )
        
        return tokenized_dataset
    
    def get_data_collator(self):
        """Get data collator for training."""
        from transformers import DataCollatorForLanguageModeling
        
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8 if self.config.fp16 else None
        )
    
    def create_evaluation_dataset(self, num_samples: int = 100) -> Dataset:
        """Create a smaller evaluation dataset for quick testing."""
        if self.config.test_split not in self.dataset:
            # Use a subset of training data for evaluation
            eval_dataset = self.dataset[self.config.train_split].select(range(num_samples))
        else:
            eval_dataset = self.dataset[self.config.test_split].select(range(min(num_samples, len(self.dataset[self.config.test_split]))))
        
        return eval_dataset
    
    def get_sample_prompts(self, num_samples: int = 5) -> List[str]:
        """Get sample prompts for testing generation."""
        if self.config.test_split in self.dataset:
            samples = self.dataset[self.config.test_split].select(range(num_samples))
        else:
            samples = self.dataset[self.config.train_split].select(range(num_samples))
        
        prompts = []
        for sample in samples:
            # Create prompt without the response
            prompt = self.config.prompt_template.format(
                instruction=sample[self.config.instruction_key],
                response=""
            ).rstrip()
            prompts.append(prompt)
        
        return prompts
    
    def format_for_inference(self, instruction: str) -> str:
        """Format a single instruction for inference."""
        return self.config.prompt_template.format(
            instruction=instruction,
            response=""
        ).rstrip()


class GSM8KHandler(DatasetHandler):
    """Specialized handler for GSM8K mathematical reasoning dataset."""
    
    def __init__(self, config):
        super().__init__(config)
        # Override config for GSM8K specific settings
        self.config.instruction_key = "question"
        self.config.response_key = "answer"
    
    def evaluate_math_accuracy(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate mathematical accuracy by comparing final answers."""
        correct = 0
        total = len(predictions)
        
        for pred, ref in zip(predictions, references):
            pred_answer = self._extract_answer_from_solution(pred)
            ref_answer = self._extract_answer_from_solution(ref)
            
            try:
                # Try to compare as numbers
                if float(pred_answer.replace(',', '')) == float(ref_answer.replace(',', '')):
                    correct += 1
            except (ValueError, AttributeError):
                # Fallback to string comparison
                if pred_answer.strip() == ref_answer.strip():
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'math_accuracy': accuracy,
            'correct_answers': correct,
            'total_answers': total
        }


def get_dataset_handler(config) -> DatasetHandler:
    """Factory function to get appropriate dataset handler."""
    if config.dataset_name == "gsm8k":
        return GSM8KHandler(config)
    else:
        return DatasetHandler(config)
