"""
Gemini-style structured prompt templates for high-quality responses.
"""

from typing import Dict, Any


class PromptTemplateManager:
    TEMPLATES = {
    """Manages domain-specific prompt templates for Gemini-level quality."""
    
        'mathematics': {
            'template': """You are a helpful and expert mathematics tutor.
Task: {task}

Instructions:
- Solve the problem step by step with clear explanations.
- Show all mathematical work and calculations.
- Provide the final numerical answer clearly.
- Use proper mathematical notation and terminology.
- Be precise and accurate in all calculations.

Now, solve this mathematical problem:
""",
            'indicators': ['calculate', 'solve', 'what is', 'times', 'plus', 'minus', 'divided', '%', 'math', 'equation']
        },
        
        'journalism': {
            'template': """You are a professional journalist and news writer.
Task: {task}

Instructions:
- Write in clear, professional journalistic style.
- Structure your response with proper headlines and paragraphs.
- Include relevant details and context.
- Use active voice and engaging language.
- Maintain objectivity and factual accuracy.
- Format as a complete news article or report.

Now, write the news content:
""",
            'indicators': ['news', 'report', 'headline', 'article', 'breaking', 'announced', 'journalism', 'write about']
        },
        
        'programming': {
            'template': """You are a helpful and expert Python programmer.
Task: {task}

Instructions:
- Write clean, complete Python code.
- Include a function with proper docstrings.
- Add comments explaining each step.
- Use descriptive variable names.
- Follow Python best practices and PEP 8.
- Provide working, executable code.
- Include example usage if appropriate.

Now, generate the Python code:
""",
            'indicators': ['function', 'code', 'python', 'write', 'def', 'algorithm', 'program', 'script']
        },
        
        'general': {
            'template': """You are a helpful and knowledgeable assistant.
Task: {task}

Instructions:
- Provide a complete, well-structured response.
- Be clear, accurate, and informative.
- Use proper formatting and organization.
- Include relevant details and examples.
- Maintain a professional and helpful tone.

Now, provide your response:
""",
            'indicators': []
        }
    }
    
    @classmethod
    def classify_domain(cls, prompt: str) -> str:
        """Classify prompt into domain based on keywords."""
        prompt_lower = prompt.lower()
        
        domain_scores = {}
        for domain, config in cls.TEMPLATES.items():
            if domain == 'general':
                continue
            score = sum(1 for indicator in config['indicators'] if indicator in prompt_lower)
            domain_scores[domain] = score
        
        # Return domain with highest score, or 'general' if no clear match
        if max(domain_scores.values()) > 0:
            return max(domain_scores, key=domain_scores.get)
        return 'general'
    
    @classmethod
    def get_structured_prompt(cls, task: str, domain: str = None, adapter_name: str = None) -> str:
        """Get structured prompt for the given task and domain."""
        
        # Determine domain
        if domain is None:
            if adapter_name:
                # Map adapter names to domains
                adapter_domain_map = {
                    'math_specialist': 'mathematics',
                    'news_specialist': 'journalism', 
                    'code_specialist': 'programming'
                }
                domain = adapter_domain_map.get(adapter_name, 'general')
            else:
                domain = cls.classify_domain(task)
        
        # Get template
        template_config = cls.TEMPLATES.get(domain, cls.TEMPLATES['general'])
        template = template_config['template']
        
        # Format with task
        structured_prompt = template.format(task=task)
        
        return structured_prompt
    
    @classmethod
    def get_completion_anchor(cls, domain: str) -> str:
        """Get completion anchor phrase to help model start properly."""
        anchors = {
            'mathematics': "Here is the step-by-step solution:\n\n",
            'journalism': "Here is the news report:\n\n",
            'programming': "Here is the Python code:\n\n```python\n",
            'general': "Here is the response:\n\n"
        }
        return anchors.get(domain, anchors['general'])


class ResponseFormatter:
    """Formats responses for Gemini-level quality and structure."""
    
    @staticmethod
    def format_code_response(response: str) -> str:
        """Format code responses with proper markdown and structure."""
        response = response.strip()
        
        # If it contains code but no markdown, wrap it
        if ('def ' in response or 'import ' in response or 'class ' in response) and '```' not in response:
            # Extract just the code part
            lines = response.split('\n')
            code_lines = []
            in_code = False
            
            for line in lines:
                if any(keyword in line for keyword in ['def ', 'import ', 'class ', 'for ', 'if ', 'while ']):
                    in_code = True
                if in_code:
                    code_lines.append(line)
            
            if code_lines:
                code_block = '\n'.join(code_lines)
                response = f"```python\n{code_block}\n```"
        
        # Add docstring if missing
        if 'def ' in response and '"""' not in response and "'''" not in response:
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if 'def ' in line and '(' in line and ')' in line:
                    # Insert docstring after function definition
                    func_name = line.split('def ')[1].split('(')[0]
                    docstring = f'    """Generated function: {func_name}."""'
                    lines.insert(i + 1, docstring)
                    break
            response = '\n'.join(lines)
        
        return response
    
    @staticmethod
    def format_news_response(response: str) -> str:
        """Format news responses with proper journalistic structure."""
        response = response.strip()
        
        # Ensure proper headline format
        lines = response.split('\n')
        if lines and not lines[0].startswith('#') and len(lines[0]) < 100:
            # First line might be a headline
            lines[0] = f"# {lines[0]}"
        
        # Add proper paragraph breaks
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if line:
                formatted_lines.append(line)
                # Add spacing after headlines
                if line.startswith('#'):
                    formatted_lines.append('')
        
        return '\n'.join(formatted_lines)
    
    @staticmethod
    def format_math_response(response: str) -> str:
        """Format mathematical responses with clear structure."""
        response = response.strip()
        
        # Ensure step-by-step format
        if 'step' not in response.lower() and '=' in response:
            # Try to add structure to mathematical content
            lines = response.split('\n')
            formatted_lines = []
            step_count = 1
            
            for line in lines:
                line = line.strip()
                if '=' in line and not line.startswith('Step'):
                    formatted_lines.append(f"Step {step_count}: {line}")
                    step_count += 1
                else:
                    formatted_lines.append(line)
            
            response = '\n'.join(formatted_lines)
        
        return response
    
    @staticmethod
    def format_response(response: str, domain: str) -> str:
        """Format response based on domain."""
        if domain == 'programming':
            return ResponseFormatter.format_code_response(response)
        elif domain == 'journalism':
            return ResponseFormatter.format_news_response(response)
        elif domain == 'mathematics':
            return ResponseFormatter.format_math_response(response)
        else:
            return response.strip()
