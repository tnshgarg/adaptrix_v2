"""
Training data generation for MoE task classifier.

This module provides utilities to generate and manage training data
for the task classifier used in adapter selection.
"""

import logging
from typing import List, Tuple, Dict, Any
import random

logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """Generates training data for different adapter domains."""
    
    def __init__(self):
        self.domains = {
            "code": self._generate_code_samples,
            "legal": self._generate_legal_samples,
            "general": self._generate_general_samples,
            "math": self._generate_math_samples
        }
    
    def generate_training_data(
        self, 
        samples_per_domain: int = 100,
        domains: List[str] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Generate training data for specified domains.
        
        Args:
            samples_per_domain: Number of samples to generate per domain
            domains: List of domains to generate data for
            
        Returns:
            Tuple of (texts, labels)
        """
        if domains is None:
            domains = list(self.domains.keys())
        
        all_texts = []
        all_labels = []
        
        for domain in domains:
            if domain not in self.domains:
                logger.warning(f"Unknown domain: {domain}")
                continue
            
            logger.info(f"Generating {samples_per_domain} samples for domain: {domain}")
            
            texts = self.domains[domain](samples_per_domain)
            labels = [domain] * len(texts)
            
            all_texts.extend(texts)
            all_labels.extend(labels)
        
        # Shuffle the data
        combined = list(zip(all_texts, all_labels))
        random.shuffle(combined)
        all_texts, all_labels = zip(*combined)
        
        logger.info(f"Generated {len(all_texts)} total training samples")
        return list(all_texts), list(all_labels)
    
    def _generate_code_samples(self, count: int) -> List[str]:
        """Generate code-related training samples."""
        
        code_templates = [
            # Python function requests
            "Write a Python function to {}",
            "Create a function that {}",
            "Implement a Python method to {}",
            "How do I write code to {}?",
            "Show me Python code for {}",
            
            # Algorithm requests
            "Implement {} algorithm",
            "Write code for {} sorting",
            "Create a {} data structure",
            "How to implement {}?",
            
            # Debugging and optimization
            "Debug this code: {}",
            "Optimize this function: {}",
            "Fix the error in: {}",
            "Improve performance of: {}",
            
            # Web development
            "Create a {} API endpoint",
            "Write JavaScript for {}",
            "Build a React component for {}",
            "Implement {} in Node.js",
            
            # Data processing
            "Process {} data with pandas",
            "Parse {} file format",
            "Extract {} from text",
            "Convert {} to {}",
        ]
        
        code_tasks = [
            "calculate factorial", "sort a list", "find prime numbers", "reverse a string",
            "binary search", "merge sort", "quick sort", "linked list",
            "binary tree", "hash table", "graph traversal", "dynamic programming",
            "file I/O", "database connection", "API request", "JSON parsing",
            "regular expressions", "web scraping", "data visualization", "machine learning",
            "authentication", "encryption", "unit testing", "error handling",
            "async programming", "multithreading", "memory optimization", "caching",
            "REST API", "GraphQL", "microservices", "containerization",
            "CSV files", "XML documents", "log files", "configuration files"
        ]
        
        samples = []
        for _ in range(count):
            template = random.choice(code_templates)
            task = random.choice(code_tasks)

            # Count the number of {} placeholders
            placeholder_count = template.count("{}")

            if placeholder_count == 1:
                sample = template.format(task)
            elif placeholder_count == 2:
                task2 = random.choice(code_tasks)
                while task2 == task:
                    task2 = random.choice(code_tasks)
                sample = template.format(task, task2)
            else:
                sample = template + " " + task

            samples.append(sample)
        
        return samples
    
    def _generate_legal_samples(self, count: int) -> List[str]:
        """Generate legal-related training samples."""
        
        legal_templates = [
            # Contract analysis
            "Analyze this contract for {}",
            "Review the {} clause in this agreement",
            "Summarize the {} terms",
            "Identify {} risks in this document",
            
            # Legal research
            "Research {} law",
            "Find precedents for {}",
            "Explain {} regulations",
            "What are the {} requirements?",
            
            # Document drafting
            "Draft a {} agreement",
            "Create a {} clause",
            "Write a {} notice",
            "Prepare {} documentation",
            
            # Compliance
            "Ensure compliance with {}",
            "Check {} regulations",
            "Verify {} requirements",
            "Audit {} procedures",
            
            # Legal advice
            "Advise on {} matters",
            "Recommend {} strategy",
            "Assess {} liability",
            "Evaluate {} options",
        ]
        
        legal_topics = [
            "liability", "intellectual property", "employment", "contract",
            "privacy", "data protection", "securities", "corporate governance",
            "merger", "acquisition", "licensing", "trademark",
            "copyright", "patent", "trade secret", "non-disclosure",
            "termination", "breach", "damages", "indemnification",
            "jurisdiction", "arbitration", "mediation", "litigation",
            "compliance", "regulatory", "antitrust", "competition",
            "tax", "immigration", "real estate", "environmental"
        ]
        
        samples = []
        for _ in range(count):
            template = random.choice(legal_templates)
            topic = random.choice(legal_topics)

            # Count the number of {} placeholders
            placeholder_count = template.count("{}")

            if placeholder_count == 1:
                sample = template.format(topic)
            elif placeholder_count == 2:
                topic2 = random.choice(legal_topics)
                while topic2 == topic:
                    topic2 = random.choice(legal_topics)
                sample = template.format(topic, topic2)
            else:
                sample = template + " " + topic

            samples.append(sample)
        
        return samples
    
    def _generate_general_samples(self, count: int) -> List[str]:
        """Generate general conversation samples."""
        
        general_templates = [
            "What is {}?",
            "Explain {} to me",
            "Tell me about {}",
            "How does {} work?",
            "Why is {} important?",
            "What are the benefits of {}?",
            "Describe {} in simple terms",
            "Give me an overview of {}",
            "What should I know about {}?",
            "Help me understand {}",
            "Can you explain {}?",
            "What's the difference between {} and {}?",
            "How can I improve my {}?",
            "What are some examples of {}?",
            "When should I use {}?",
        ]
        
        general_topics = [
            "artificial intelligence", "machine learning", "climate change", "renewable energy",
            "healthy eating", "exercise", "meditation", "productivity",
            "time management", "communication skills", "leadership", "teamwork",
            "creativity", "innovation", "entrepreneurship", "investing",
            "personal finance", "budgeting", "saving money", "career development",
            "education", "learning", "reading", "writing",
            "travel", "culture", "history", "science",
            "technology", "internet", "social media", "privacy"
        ]
        
        samples = []
        for _ in range(count):
            template = random.choice(general_templates)
            
            if template.count("{}") == 2:
                topic1 = random.choice(general_topics)
                topic2 = random.choice(general_topics)
                while topic2 == topic1:
                    topic2 = random.choice(general_topics)
                sample = template.format(topic1, topic2)
            else:
                topic = random.choice(general_topics)
                sample = template.format(topic)
            
            samples.append(sample)
        
        return samples
    
    def _generate_math_samples(self, count: int) -> List[str]:
        """Generate math-related training samples."""
        
        math_templates = [
            "Solve: {}",
            "Calculate {}",
            "Find the {} of {}",
            "What is {} + {}?",
            "What is {} - {}?",
            "What is {} × {}?",
            "What is {} ÷ {}?",
            "Simplify {}",
            "Factor {}",
            "Expand {}",
            "Derive {}",
            "Integrate {}",
            "Find the limit of {}",
            "Solve the equation: {}",
            "Graph the function: {}",
            "Find the {} of the triangle",
            "Calculate the {} of the circle",
            "What is {}% of {}?",
            "Convert {} to {}",
            "Round {} to {} decimal places",
        ]
        
        math_expressions = [
            "2x + 5 = 15", "x² - 4x + 3 = 0", "3x + 2y = 10",
            "sin(x) + cos(x)", "e^x", "ln(x)", "√(x² + 1)",
            "x³ - 2x² + x - 1", "2x + 3", "x² + 5x + 6",
            "area", "perimeter", "volume", "surface area",
            "derivative", "integral", "slope", "intercept"
        ]
        
        numbers = ["2", "5", "10", "15", "25", "50", "100", "0.5", "1.5", "3.14"]
        
        samples = []
        for _ in range(count):
            template = random.choice(math_templates)

            placeholder_count = template.count("{}")

            if placeholder_count == 2:
                if "%" in template:
                    num1 = random.choice(["10", "25", "50", "75"])
                    num2 = random.choice(numbers)
                    sample = template.format(num1, num2)
                else:
                    expr1 = random.choice(math_expressions)
                    expr2 = random.choice(math_expressions)
                    sample = template.format(expr1, expr2)
            elif placeholder_count == 1:
                expr = random.choice(math_expressions)
                sample = template.format(expr)
            else:
                # No placeholders, use as is
                sample = template

            samples.append(sample)
        
        return samples
    
    def save_training_data(
        self, 
        texts: List[str], 
        labels: List[str], 
        filepath: str
    ):
        """Save training data to a file."""
        import json
        
        data = {
            "texts": texts,
            "labels": labels,
            "num_samples": len(texts),
            "domains": list(set(labels))
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Training data saved to {filepath}")
    
    def load_training_data(self, filepath: str) -> Tuple[List[str], List[str]]:
        """Load training data from a file."""
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {data['num_samples']} samples from {filepath}")
        logger.info(f"Domains: {data['domains']}")
        
        return data["texts"], data["labels"]


# Convenience function for backward compatibility
def generate_training_data(samples_per_domain: int = 200) -> List[Tuple[str, str]]:
    """
    Generate training data for the task classifier.

    Args:
        samples_per_domain: Number of samples per domain

    Returns:
        List of (text, label) tuples
    """
    generator = TrainingDataGenerator()
    texts, labels = generator.generate_training_data(samples_per_domain)
    return list(zip(texts, labels))
