"""
Evaluation utilities for LoRA adapters.
"""

import logging
import torch
from typing import Dict, List, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
import re
import numpy as np
from datasets import Dataset

logger = logging.getLogger(__name__)


class AdapterEvaluator:
    """Evaluates LoRA adapter performance on various metrics."""
    
    def __init__(self, config, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initialize evaluator.
        
        Args:
            config: Training configuration
            model: The model with LoRA adapter
            tokenizer: Tokenizer
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
    def evaluate(self) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        logger.info("Starting adapter evaluation...")
        
        results = {}
        
        # Basic generation quality
        results.update(self._evaluate_generation_quality())
        
        # Domain-specific evaluation
        if self.config.dataset_name == "gsm8k":
            results.update(self._evaluate_math_reasoning())
        elif "code" in self.config.dataset_name.lower():
            results.update(self._evaluate_code_generation())
        
        # General metrics
        results.update(self._evaluate_general_metrics())
        
        logger.info("Evaluation complete!")
        return results
    
    def _evaluate_generation_quality(self) -> Dict[str, float]:
        """Evaluate basic generation quality metrics."""
        logger.info("Evaluating generation quality...")
        
        # Get sample prompts
        from .data_handler import get_dataset_handler
        dataset_handler = get_dataset_handler(self.config)
        dataset_handler.set_tokenizer(self.tokenizer)
        dataset_handler.load_dataset()
        
        # Use smaller number of samples for evaluation
        max_samples = min(50, len(dataset_handler.dataset[dataset_handler.config.test_split]) if dataset_handler.config.test_split in dataset_handler.dataset else 10)
        prompts = dataset_handler.get_sample_prompts(max_samples)
        
        responses = []
        for prompt in prompts:
            response = self._generate_response(prompt)
            responses.append(response)
        
        # Calculate metrics
        avg_length = np.mean([len(r.split()) for r in responses])
        avg_char_length = np.mean([len(r) for r in responses])
        
        # Check for repetition
        repetition_scores = [self._calculate_repetition_score(r) for r in responses]
        avg_repetition = np.mean(repetition_scores)
        
        # Check for coherence (simple heuristic)
        coherence_scores = [self._calculate_coherence_score(r) for r in responses]
        avg_coherence = np.mean(coherence_scores)
        
        return {
            'avg_response_length_words': avg_length,
            'avg_response_length_chars': avg_char_length,
            'avg_repetition_score': avg_repetition,
            'avg_coherence_score': avg_coherence,
            'num_evaluated_samples': len(responses)
        }
    
    def _evaluate_math_reasoning(self) -> Dict[str, float]:
        """Evaluate mathematical reasoning capabilities."""
        logger.info("Evaluating math reasoning...")
        
        # Test with some math problems
        math_prompts = [
            "Solve this math problem step by step.\n\nProblem: If a store sells 3 apples for $2, how much would 12 apples cost?\n\nSolution:",
            "Solve this math problem step by step.\n\nProblem: A rectangle has length 8 cm and width 5 cm. What is its area?\n\nSolution:",
            "Solve this math problem step by step.\n\nProblem: If 25% of a number is 15, what is the number?\n\nSolution:",
            "Solve this math problem step by step.\n\nProblem: A train travels 120 km in 2 hours. What is its average speed?\n\nSolution:",
            "Solve this math problem step by step.\n\nProblem: If x + 5 = 12, what is x?\n\nSolution:"
        ]
        
        correct_answers = ["$8", "40", "60", "60", "7"]
        
        responses = []
        for prompt in math_prompts:
            response = self._generate_response(prompt, max_new_tokens=150)
            responses.append(response)
        
        # Check for mathematical accuracy
        correct_count = 0
        for response, correct in zip(responses, correct_answers):
            if self._check_math_answer(response, correct):
                correct_count += 1
        
        # Check for step-by-step reasoning
        step_by_step_count = 0
        for response in responses:
            if self._has_step_by_step_reasoning(response):
                step_by_step_count += 1
        
        return {
            'math_accuracy': correct_count / len(math_prompts),
            'step_by_step_reasoning_rate': step_by_step_count / len(math_prompts),
            'math_problems_evaluated': len(math_prompts)
        }
    
    def _evaluate_code_generation(self) -> Dict[str, float]:
        """Evaluate code generation capabilities."""
        logger.info("Evaluating code generation...")
        
        code_prompts = [
            "Generate code for the following task.\n\nTask: Write a Python function to calculate factorial\n\nCode:",
            "Generate code for the following task.\n\nTask: Write a Python function to check if a number is prime\n\nCode:",
            "Generate code for the following task.\n\nTask: Write a Python function to reverse a string\n\nCode:"
        ]
        
        responses = []
        for prompt in code_prompts:
            response = self._generate_response(prompt, max_new_tokens=200)
            responses.append(response)
        
        # Check for code quality indicators
        has_function_def = sum(1 for r in responses if 'def ' in r) / len(responses)
        has_return_statement = sum(1 for r in responses if 'return' in r) / len(responses)
        has_proper_indentation = sum(1 for r in responses if self._check_python_indentation(r)) / len(responses)
        
        return {
            'code_function_definition_rate': has_function_def,
            'code_return_statement_rate': has_return_statement,
            'code_proper_indentation_rate': has_proper_indentation,
            'code_problems_evaluated': len(code_prompts)
        }
    
    def _evaluate_general_metrics(self) -> Dict[str, float]:
        """Evaluate general language model metrics."""
        logger.info("Evaluating general metrics...")
        
        # Test instruction following
        instruction_prompts = [
            "Write exactly 3 sentences about the weather.",
            "List 5 benefits of exercise.",
            "Explain photosynthesis in simple terms."
        ]
        
        responses = []
        for prompt in instruction_prompts:
            response = self._generate_response(prompt, max_new_tokens=100)
            responses.append(response)
        
        # Check instruction following (simple heuristics)
        follows_instructions = 0
        for i, response in enumerate(responses):
            if i == 0:  # 3 sentences
                sentence_count = len([s for s in response.split('.') if s.strip()])
                if 2 <= sentence_count <= 4:  # Allow some flexibility
                    follows_instructions += 1
            elif i == 1:  # 5 benefits
                if '1.' in response or 'first' in response.lower() or len(response.split('\n')) >= 3:
                    follows_instructions += 1
            elif i == 2:  # Simple explanation
                if len(response.split()) >= 20 and any(word in response.lower() for word in ['plant', 'light', 'energy', 'oxygen']):
                    follows_instructions += 1
        
        instruction_following_rate = follows_instructions / len(instruction_prompts)
        
        return {
            'instruction_following_rate': instruction_following_rate,
            'general_prompts_evaluated': len(instruction_prompts)
        }
    
    def _generate_response(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate response for a given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to appropriate device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract only the generated part
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = full_response[len(prompt):].strip()
        
        return generated_part
    
    def _calculate_repetition_score(self, text: str) -> float:
        """Calculate repetition score (lower is better)."""
        words = text.split()
        if len(words) < 4:
            return 0.0
        
        # Check for repeated 3-grams
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        unique_trigrams = set(trigrams)
        
        repetition_rate = 1 - (len(unique_trigrams) / len(trigrams)) if trigrams else 0
        return repetition_rate
    
    def _calculate_coherence_score(self, text: str) -> float:
        """Calculate coherence score (simple heuristic)."""
        # Simple coherence indicators
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.5
        
        # Check for logical connectors
        connectors = ['therefore', 'because', 'however', 'first', 'then', 'finally', 'next', 'also']
        has_connectors = any(conn in text.lower() for conn in connectors)
        
        # Check for consistent tense (simple check)
        past_tense_words = len(re.findall(r'\w+ed\b', text))
        present_tense_words = len(re.findall(r'\w+s\b', text))
        total_words = len(text.split())
        
        coherence_score = 0.5  # Base score
        if has_connectors:
            coherence_score += 0.3
        if total_words > 20:  # Sufficient length
            coherence_score += 0.2
        
        return min(coherence_score, 1.0)
    
    def _check_math_answer(self, response: str, correct_answer: str) -> bool:
        """Check if math response contains correct answer."""
        # Extract numbers from response
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
        
        # Check if correct answer appears
        correct_num = re.findall(r'\b\d+(?:\.\d+)?\b', correct_answer)
        if correct_num and numbers:
            return correct_num[0] in numbers
        
        return correct_answer.lower() in response.lower()
    
    def _has_step_by_step_reasoning(self, response: str) -> bool:
        """Check if response shows step-by-step reasoning."""
        indicators = [
            'step', 'first', 'then', 'next', 'finally',
            '1.', '2.', '3.', 'therefore', 'so'
        ]
        return any(indicator in response.lower() for indicator in indicators)
    
    def _check_python_indentation(self, code: str) -> bool:
        """Check if code has proper Python indentation."""
        lines = code.split('\n')
        has_indentation = any(line.startswith('    ') or line.startswith('\t') for line in lines)
        has_function = 'def ' in code
        return has_indentation and has_function
