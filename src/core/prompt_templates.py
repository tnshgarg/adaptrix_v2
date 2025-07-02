"""
Clean, configurable prompt templates for production-quality responses.
"""

from typing import Dict, Any
import os
import json


class PromptTemplateManager:
    """Manages configurable, clean prompt templates."""
    
    # Clean, non-instruction-heavy templates
    CLEAN_TEMPLATES = {
        'programming': {
            'template': "{task}",
            'indicators': ['function', 'code', 'python', 'write', 'def', 'algorithm', 'program', 'script'],
            'context': "code"
        },
        
        'mathematics': {
            'template': "{task}",
            'indicators': ['calculate', 'solve', 'what is', 'times', 'plus', 'minus', 'divided', '%', 'math', 'equation'],
            'context': "math"
        },
        
        'journalism': {
            'template': "{task}",
            'indicators': ['news', 'report', 'headline', 'article', 'breaking', 'announced', 'journalism', 'write about'],
            'context': "news"
        },
        
        'general': {
            'template': "{task}",
            'indicators': [],
            'context': "general"
        }
    }
    
    @classmethod
    def load_custom_templates(cls, config_path: str = None) -> Dict[str, Any]:
        """Load custom templates from configuration file."""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return cls.CLEAN_TEMPLATES
    
    @classmethod
    def classify_domain(cls, prompt: str) -> str:
        """Classify prompt into domain based on keywords."""
        prompt_lower = prompt.lower()
        templates = cls.load_custom_templates()
        
        domain_scores = {}
        for domain, config in templates.items():
            if domain == 'general':
                continue
            score = sum(1 for indicator in config['indicators'] if indicator in prompt_lower)
            domain_scores[domain] = score
        
        if max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        return 'general'
    
    @classmethod
    def get_structured_prompt(cls, task: str, domain: str = None, adapter_name: str = None) -> str:
        """Get clean, structured prompt without heavy instructions."""
        
        # Determine domain
        if domain is None:
            if adapter_name:
                # Flexible adapter mapping
                if any(kw in adapter_name.lower() for kw in ['code', 'programming', 'python']):
                    domain = 'programming'
                elif any(kw in adapter_name.lower() for kw in ['math', 'calculator']):
                    domain = 'mathematics'
                elif any(kw in adapter_name.lower() for kw in ['news', 'journalism']):
                    domain = 'journalism'
                else:
                    domain = 'general'
            else:
                domain = cls.classify_domain(task)
        
        # Get clean template
        templates = cls.load_custom_templates()
        template_config = templates.get(domain, templates['general'])
        
        # Return clean prompt - just the task
        return template_config['template'].format(task=task)


class ResponseFormatter:
    """Clean, configurable response formatting without hardcoded artifacts."""
    
    @staticmethod
    def format_code_response(response: str) -> str:
        """Format code responses cleanly without injecting artifacts."""
        response = response.strip()
        
        # Only add code blocks if they're missing and response contains code
        if ('def ' in response or 'import ' in response or 'class ' in response) and '```' not in response:
            # Simple code block wrapping
            response = f"```python\n{response}\n```"
        
        return response
    
    @staticmethod
    def format_news_response(response: str) -> str:
        """Format news responses with clean structure."""
        response = response.strip()
        
        # Simple headline formatting
        lines = response.split('\n')
        if lines and not lines[0].startswith('#') and len(lines[0]) < 100:
            lines[0] = f"# {lines[0]}"
        
        return '\n'.join(lines)
    
    @staticmethod
    def format_math_response(response: str) -> str:
        """Format mathematical responses cleanly."""
        return response.strip()
    
    @staticmethod
    def format_response(response: str, domain: str) -> str:
        """Format response based on domain using configurable rules."""
        
        # Load formatting rules from config if available
        formatting_rules = ResponseFormatter._load_formatting_config()
        
        if domain in formatting_rules:
            formatter_method = getattr(ResponseFormatter, formatting_rules[domain], None)
            if formatter_method:
                return formatter_method(response)
        
        # Default formatting by domain
        if domain == 'programming':
            return ResponseFormatter.format_code_response(response)
        elif domain == 'journalism':
            return ResponseFormatter.format_news_response(response)
        elif domain == 'mathematics':
            return ResponseFormatter.format_math_response(response)
        else:
            return response.strip()
    
    @staticmethod
    def _load_formatting_config() -> Dict[str, str]:
        """Load formatting configuration from file or return defaults."""
        default_config = {
            'programming': 'format_code_response',
            'journalism': 'format_news_response', 
            'mathematics': 'format_math_response',
            'general': None
        }
        
        # Could load from configs/formatting.json in the future
        return default_config
