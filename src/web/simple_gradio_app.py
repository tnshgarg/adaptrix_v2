#!/usr/bin/env python3
"""
🚀 SIMPLIFIED ADAPTRIX WEB INTERFACE

A working, simplified version of the revolutionary Adaptrix web interface
that demonstrates the core multi-adapter composition capabilities.
"""

import gradio as gr
import logging
import time
from typing import Dict, List, Optional, Any, Tuple

# Import Adaptrix components
from ..core.engine import AdaptrixEngine
from ..composition.adapter_composer import CompositionStrategy

logger = logging.getLogger(__name__)


class SimpleAdaptrixInterface:
    """Simplified Adaptrix web interface."""
    
    def __init__(self):
        self.engine = None
        self.composition_history = []
        
    def initialize_engine(self, model_name: str = "microsoft/phi-2") -> str:
        """Initialize the Adaptrix engine."""
        try:
            self.engine = AdaptrixEngine(model_name, "cpu")
            if self.engine.initialize():
                return "✅ Adaptrix Engine initialized successfully!"
            else:
                return "❌ Failed to initialize Adaptrix Engine."
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def get_available_adapters(self) -> List[str]:
        """Get list of available adapters."""
        if not self.engine:
            return []
        return self.engine.list_adapters()
    
    def compose_adapters(self, selected_adapters: List[str], strategy: str) -> str:
        """Compose selected adapters."""
        if not self.engine:
            return "❌ Engine not initialized"
        
        if not selected_adapters:
            return "❌ No adapters selected"
        
        try:
            strategy_enum = CompositionStrategy(strategy.lower())
            result = self.engine.compose_adapters(selected_adapters, strategy_enum)
            
            if result.get('success'):
                return f"✅ Composition successful!\nStrategy: {result['strategy']}\nAdapters: {', '.join(result['adapters_used'])}"
            else:
                return f"❌ Composition failed: {result.get('error')}"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def generate_text(self, prompt: str, selected_adapters: List[str], strategy: str) -> str:
        """Generate text with composition - ULTRA-OPTIMIZED FOR SPEED AND QUALITY."""
        if not self.engine:
            return "❌ Engine not initialized"

        if not prompt.strip():
            return "❌ Please enter a prompt"

        try:
            # ULTRA-OPTIMIZATION: Better prompting for math questions
            enhanced_prompt = self._enhance_math_prompt(prompt)

            # OPTIMIZED GENERATION: Longer for complete answers
            if selected_adapters:
                strategy_enum = CompositionStrategy(strategy.lower())
                response = self.engine.generate_with_composition(
                    enhanced_prompt,
                    selected_adapters,
                    strategy_enum,
                    max_length=15,  # Longer for complete numbers
                    temperature=0.05,  # ULTRA-LOW temperature for focus
                    do_sample=False  # Deterministic for math
                )
            else:
                response = self.engine.generate(
                    enhanced_prompt,
                    max_length=15,  # Longer for complete numbers
                    temperature=0.05,  # ULTRA-FOCUSED
                    do_sample=False  # Deterministic
                )

            # QUALITY FILTER: Extract clean answer
            response = self._extract_clean_answer(response, prompt)

            return response

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _enhance_math_prompt(self, prompt: str) -> str:
        """Enhance prompts for better math responses - OPTIMIZED FORMAT."""
        # Check if it's a math question
        if any(op in prompt.lower() for op in ['*', '+', '-', '/', 'what is', 'calculate', 'multiply', 'add', 'subtract', 'divide']):
            # Extract the math expression
            import re

            # Look for patterns like "What is 5*12?" or "Calculate 5+3"
            math_pattern = r'(?:what is|calculate)?\s*(\d+\s*[+\-*/]\s*\d+)'
            match = re.search(math_pattern, prompt.lower())

            if match:
                # Use the simple format that works: "5*12="
                math_expr = match.group(1).replace(' ', '')
                return f"{math_expr}="

            # Fallback: try to extract any math expression
            simple_pattern = r'(\d+\s*[+\-*/]\s*\d+)'
            match = re.search(simple_pattern, prompt)
            if match:
                math_expr = match.group(1).replace(' ', '')
                return f"{math_expr}="

        return prompt

    def _extract_clean_answer(self, response: str, original_prompt: str) -> str:
        """Extract clean answer from response - OPTIMIZED FOR MATH."""
        # Remove composition metadata
        if "[Composed using" in response:
            response = response.split("[Composed using")[0].strip()

        # For math questions, extract the numerical answer
        if any(op in original_prompt.lower() for op in ['*', '+', '-', '/', 'what is', 'calculate']):
            # The response should be in format like "60, 3*4=12." or just "60"
            # Extract the first number (which is the answer)
            import re

            # Look for the first number in the response
            numbers = re.findall(r'\d+(?:\.\d+)?', response)
            if numbers:
                return numbers[0]  # Return just the number

            # If no number found, return the first few characters
            return response.split(',')[0].strip()

        # Clean up the response for non-math questions
        response = response.strip()

        # Remove common unwanted prefixes
        unwanted_prefixes = ["</think>", "**Solution:**", "To find", "Let me", "I need to"]
        for prefix in unwanted_prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()

        # Return first sentence or 50 characters, whichever is shorter
        sentences = response.split('.')
        if sentences and len(sentences[0]) < 50:
            return sentences[0].strip()

        return response[:50].strip()


def create_simple_interface():
    """Create the simplified Adaptrix interface."""
    
    interface = SimpleAdaptrixInterface()
    
    with gr.Blocks(title="🚀 Adaptrix - Revolutionary AI Composition") as demo:
        
        gr.HTML("""
        <div style="text-align: center; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0;">🚀 ADAPTRIX - REVOLUTIONARY AI COMPOSITION</h1>
            <p style="color: white; margin: 10px 0 0 0;">World's First Middle-Layer Multi-Adapter Composition System</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 System Control")
                
                init_btn = gr.Button("Initialize Adaptrix Engine", variant="primary")
                init_status = gr.Textbox(label="Status", interactive=False)
                
                gr.Markdown("### 📦 Adapter Selection")
                
                refresh_btn = gr.Button("Refresh Adapters")
                adapter_choices = gr.CheckboxGroup(
                    label="Available Adapters",
                    choices=[],
                    value=[]
                )
                
                strategy_choice = gr.Dropdown(
                    label="Composition Strategy",
                    choices=[
                        ("🔄 Parallel", "parallel"),
                        ("⛓️ Sequential", "sequential"),
                        ("🏗️ Hierarchical", "hierarchical"),
                        ("🎯 Attention", "attention")
                    ],
                    value="parallel"
                )
                
                compose_btn = gr.Button("🚀 Compose Adapters", variant="primary")
                composition_result = gr.Textbox(label="Composition Result", interactive=False)
            
            with gr.Column(scale=2):
                gr.Markdown("### ✨ Enhanced Text Generation")
                
                prompt_input = gr.Textbox(
                    label="Enter your prompt",
                    placeholder="Ask anything! The composed adapters will enhance the response...",
                    lines=3
                )
                
                generate_btn = gr.Button("🚀 Generate with Composition", variant="primary")
                
                output_text = gr.Textbox(
                    label="Generated Response",
                    lines=15,
                    interactive=False
                )
        
        # Event handlers
        def initialize():
            return interface.initialize_engine()
        
        def refresh_adapters():
            adapters = interface.get_available_adapters()
            return gr.CheckboxGroup(choices=adapters, value=[])
        
        def compose(adapters, strategy):
            return interface.compose_adapters(adapters, strategy)
        
        def generate(prompt, adapters, strategy):
            return interface.generate_text(prompt, adapters, strategy)
        
        # Wire up events
        init_btn.click(initialize, outputs=init_status)
        refresh_btn.click(refresh_adapters, outputs=adapter_choices)
        compose_btn.click(compose, inputs=[adapter_choices, strategy_choice], outputs=composition_result)
        generate_btn.click(
            generate, 
            inputs=[prompt_input, adapter_choices, strategy_choice], 
            outputs=output_text
        )
    
    return demo


def launch_simple_interface(port: int = 7861):
    """Launch the simplified interface."""
    print("🚀" * 50)
    print("🚀 LAUNCHING SIMPLIFIED ADAPTRIX INTERFACE 🚀")
    print("🚀" * 50)
    print()
    print("🌟 Features:")
    print("   • Multi-adapter composition")
    print("   • Enhanced text generation")
    print("   • Real-time adapter management")
    print("   • Revolutionary AI capabilities")
    print()
    
    demo = create_simple_interface()
    
    print(f"🚀 Starting server on http://127.0.0.1:{port}")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        debug=True,
        show_error=True
    )


if __name__ == "__main__":
    launch_simple_interface()
