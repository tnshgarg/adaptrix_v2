"""
Robust Adapter Manager with validation and error handling.
"""

import torch
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class RobustAdapterManager:
    """Enhanced adapter manager with validation and robust error handling."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.active_adapter = None
        self.adapter_states = {}  # Track adapter states
        self.original_weights = {}  # Store original weights for validation
        
    def validate_adapter_application(self, adapter_name: str) -> bool:
        """Validate that adapter is actually modifying model behavior."""
        try:
            # Simple test prompt
            test_prompt = "What is 2+2?"
            test_inputs = self.tokenizer(test_prompt, return_tensors="pt")
            
            # Get output with adapter
            with torch.no_grad():
                output_with_adapter = self.model(**test_inputs)
                logits_with = output_with_adapter.logits
            
            # Check if we have stored baseline (without adapter)
            if adapter_name not in self.adapter_states:
                logger.warning(f"No baseline stored for {adapter_name}")
                return True  # Assume valid if no baseline
            
            baseline_logits = self.adapter_states[adapter_name].get('baseline_logits')
            if baseline_logits is not None:
                # Compare logits - they should be different if adapter is working
                diff = torch.abs(logits_with - baseline_logits).mean().item()
                is_different = diff > 1e-6  # Threshold for meaningful difference
                
                logger.debug(f"Adapter {adapter_name} logits difference: {diff}")
                return is_different
            
            return True
            
        except Exception as e:
            logger.error(f"Adapter validation failed: {e}")
            return False
    
    def store_baseline(self, adapter_name: str):
        """Store baseline model behavior before loading adapter."""
        try:
            test_prompt = "What is 2+2?"
            test_inputs = self.tokenizer(test_prompt, return_tensors="pt")
            
            with torch.no_grad():
                baseline_output = self.model(**test_inputs)
                
            if adapter_name not in self.adapter_states:
                self.adapter_states[adapter_name] = {}
                
            self.adapter_states[adapter_name]['baseline_logits'] = baseline_output.logits.clone()
            logger.debug(f"Stored baseline for {adapter_name}")
            
        except Exception as e:
            logger.error(f"Failed to store baseline for {adapter_name}: {e}")
    
    def load_adapter(self, adapter_name: str, adapter_path: str) -> bool:
        """Load adapter with validation and error handling."""
        try:
            # Store baseline before loading
            if self.active_adapter is None:
                self.store_baseline(adapter_name)
            
            # Unload current adapter if any
            if self.active_adapter and self.active_adapter != adapter_name:
                self.unload_adapter()
            
            # Load the adapter
            logger.info(f"Loading adapter: {adapter_name}")
            
            # Check if adapter path exists
            if not Path(adapter_path).exists():
                logger.error(f"Adapter path does not exist: {adapter_path}")
                return False
            
            # Load adapter weights (implementation depends on your adapter system)
            success = self._load_adapter_weights(adapter_name, adapter_path)
            
            if success:
                # Validate adapter is working
                if self.validate_adapter_application(adapter_name):
                    self.active_adapter = adapter_name
                    self.adapter_states[adapter_name]['loaded'] = True
                    logger.info(f"✅ Adapter {adapter_name} loaded and validated successfully")
                    return True
                else:
                    logger.error(f"❌ Adapter {adapter_name} failed validation")
                    self._unload_adapter_weights(adapter_name)
                    return False
            else:
                logger.error(f"❌ Failed to load adapter weights for {adapter_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Adapter loading failed: {e}")
            return False
    
    def unload_adapter(self) -> bool:
        """Safely unload current adapter."""
        if not self.active_adapter:
            logger.debug("No adapter to unload")
            return True
        
        try:
            adapter_name = self.active_adapter
            logger.info(f"Unloading adapter: {adapter_name}")
            
            # Unload adapter weights
            success = self._unload_adapter_weights(adapter_name)
            
            if success:
                self.active_adapter = None
                if adapter_name in self.adapter_states:
                    self.adapter_states[adapter_name]['loaded'] = False
                logger.info(f"✅ Adapter {adapter_name} unloaded successfully")
                return True
            else:
                logger.error(f"❌ Failed to unload adapter {adapter_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Adapter unloading failed: {e}")
            # Force reset state
            self.active_adapter = None
            return False
    
    def _load_adapter_weights(self, adapter_name: str, adapter_path: str) -> bool:
        """Load adapter weights into model."""
        try:
            # This would integrate with your existing dynamic loader
            from .dynamic_loader import DynamicLoader
            
            loader = DynamicLoader(self.model, None)  # Pass appropriate injector
            return loader.load_adapter(adapter_name, adapter_path)
            
        except Exception as e:
            logger.error(f"Failed to load adapter weights: {e}")
            return False
    
    def _unload_adapter_weights(self, adapter_name: str) -> bool:
        """Unload adapter weights from model."""
        try:
            # This would integrate with your existing dynamic loader
            from .dynamic_loader import DynamicLoader
            
            loader = DynamicLoader(self.model, None)  # Pass appropriate injector
            return loader.unload_adapter(adapter_name)
            
        except Exception as e:
            logger.error(f"Failed to unload adapter weights: {e}")
            return False
    
    def get_adapter_status(self) -> Dict[str, Any]:
        """Get current adapter status."""
        return {
            'active_adapter': self.active_adapter,
            'adapter_states': self.adapter_states,
            'total_adapters': len(self.adapter_states)
        }
    
    def cleanup(self):
        """Clean up all adapter states."""
        try:
            if self.active_adapter:
                self.unload_adapter()
            
            self.adapter_states.clear()
            self.original_weights.clear()
            logger.info("✅ Adapter manager cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


class DomainPromptManager:
    """Manages domain-specific prompts and templates."""
    
    DOMAIN_TEMPLATES = {
        'mathematics': {
            'template': "Solve this mathematical problem step by step: {prompt}",
            'indicators': ['calculate', 'solve', 'what is', 'times', 'plus', 'minus', 'divided', '%']
        },
        'journalism': {
            'template': "Write a professional news article about: {prompt}",
            'indicators': ['news', 'report', 'headline', 'article', 'breaking', 'announced']
        },
        'programming': {
            'template': "Write a complete Python function with docstring to: {prompt}",
            'indicators': ['function', 'code', 'python', 'write', 'def', 'algorithm', 'program']
        }
    }
    
    @classmethod
    def classify_domain(cls, prompt: str) -> str:
        """Classify prompt into domain based on keywords."""
        prompt_lower = prompt.lower()
        
        domain_scores = {}
        for domain, config in cls.DOMAIN_TEMPLATES.items():
            score = sum(1 for indicator in config['indicators'] if indicator in prompt_lower)
            domain_scores[domain] = score
        
        # Return domain with highest score, or 'general' if no clear match
        if max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        return 'general'
    
    @classmethod
    def format_prompt(cls, prompt: str, domain: str = None) -> str:
        """Format prompt with domain-specific template."""
        if domain is None:
            domain = cls.classify_domain(prompt)
        
        if domain in cls.DOMAIN_TEMPLATES:
            template = cls.DOMAIN_TEMPLATES[domain]['template']
            return template.format(prompt=prompt)
        
        return prompt  # Return original if no template found
