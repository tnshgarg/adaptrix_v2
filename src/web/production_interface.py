#!/usr/bin/env python3
"""
🚀 PRODUCTION-READY ADAPTRIX WEB INTERFACE

A comprehensive, production-ready web interface for Adaptrix that showcases
all the revolutionary multi-adapter composition capabilities with real-time
management, performance metrics, and user-friendly design.
"""

import gradio as gr
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import threading

# Import Adaptrix components
from ..core.engine import AdaptrixEngine
from ..composition.adapter_composer import CompositionStrategy

logger = logging.getLogger(__name__)


class ProductionAdaptrixInterface:
    """Production-ready Adaptrix web interface with advanced features."""
    
    def __init__(self):
        self.engine = None
        self.composition_history = []
        self.performance_metrics = {
            'total_generations': 0,
            'total_compositions': 0,
            'avg_response_time': 0.0,
            'adapter_usage': {}
        }
        self.is_initialized = False
        
    def initialize_engine(self, model_name: str = "microsoft/phi-2") -> Tuple[str, str]:
        """Initialize the Adaptrix engine with status updates."""
        try:
            start_time = time.time()
            self.engine = AdaptrixEngine(model_name, "cpu")
            
            if self.engine.initialize():
                init_time = time.time() - start_time
                self.is_initialized = True
                
                # Get system info
                status = self.engine.get_system_status()
                available_adapters = self.engine.list_adapters()
                
                status_msg = f"✅ Adaptrix Engine initialized successfully!\n"
                status_msg += f"⏱️ Initialization time: {init_time:.2f}s\n"
                status_msg += f"🤖 Model: {status['model_name']}\n"
                status_msg += f"💾 Device: {status['device']}\n"
                status_msg += f"📦 Available adapters: {len(available_adapters)}"
                
                adapters_info = f"📦 Available Adapters ({len(available_adapters)}):\n"
                for adapter in available_adapters:
                    adapters_info += f"   • {adapter}\n"
                
                return status_msg, adapters_info
            else:
                return "❌ Failed to initialize Adaptrix Engine.", ""
                
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return f"❌ Error: {str(e)}", ""
    
    def get_available_adapters(self) -> List[str]:
        """Get list of available adapters."""
        if not self.engine:
            return []
        return self.engine.list_adapters()
    
    def get_loaded_adapters(self) -> str:
        """Get currently loaded adapters."""
        if not self.engine:
            return "Engine not initialized"
        
        loaded = self.engine.get_loaded_adapters()
        if loaded:
            return f"🔄 Loaded: {', '.join(loaded)}"
        else:
            return "📭 No adapters currently loaded"
    
    def load_single_adapter(self, adapter_name: str) -> str:
        """Load a single adapter."""
        if not self.engine:
            return "❌ Engine not initialized"
        
        if not adapter_name:
            return "❌ No adapter selected"
        
        try:
            start_time = time.time()
            success = self.engine.load_adapter(adapter_name)
            load_time = time.time() - start_time
            
            if success:
                return f"✅ Loaded '{adapter_name}' in {load_time:.2f}s"
            else:
                return f"❌ Failed to load '{adapter_name}'"
                
        except Exception as e:
            return f"❌ Error loading adapter: {str(e)}"
    
    def unload_adapter(self, adapter_name: str) -> str:
        """Unload an adapter."""
        if not self.engine:
            return "❌ Engine not initialized"
        
        if not adapter_name:
            return "❌ No adapter specified"
        
        try:
            success = self.engine.unload_adapter(adapter_name)
            if success:
                return f"✅ Unloaded '{adapter_name}'"
            else:
                return f"❌ Failed to unload '{adapter_name}'"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def switch_adapters(self, old_adapter: str, new_adapter: str) -> str:
        """Switch from one adapter to another."""
        if not self.engine:
            return "❌ Engine not initialized"
        
        if not old_adapter or not new_adapter:
            return "❌ Both adapters must be specified"
        
        try:
            start_time = time.time()
            success = self.engine.switch_adapter(old_adapter, new_adapter)
            switch_time = time.time() - start_time
            
            if success:
                return f"✅ Switched '{old_adapter}' → '{new_adapter}' in {switch_time:.2f}s"
            else:
                return f"❌ Failed to switch adapters"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def compose_adapters(self, selected_adapters: List[str], strategy: str) -> str:
        """Compose selected adapters with detailed feedback."""
        if not self.engine:
            return "❌ Engine not initialized"
        
        if not selected_adapters:
            return "❌ No adapters selected"
        
        try:
            start_time = time.time()
            strategy_enum = CompositionStrategy(strategy.lower())
            result = self.engine.compose_adapters(selected_adapters, strategy_enum)
            compose_time = time.time() - start_time
            
            self.performance_metrics['total_compositions'] += 1
            
            if result.get('success'):
                # Update adapter usage stats
                for adapter in selected_adapters:
                    self.performance_metrics['adapter_usage'][adapter] = \
                        self.performance_metrics['adapter_usage'].get(adapter, 0) + 1
                
                feedback = f"✅ Composition successful!\n"
                feedback += f"⏱️ Composition time: {compose_time:.2f}s\n"
                feedback += f"🎯 Strategy: {result['strategy']}\n"
                feedback += f"📦 Adapters: {', '.join(result['adapters_used'])}\n"
                feedback += f"🎚️ Weights: {json.dumps(result.get('weights', {}), indent=2)}\n"
                feedback += f"📊 Confidence: {json.dumps(result.get('confidence', {}), indent=2)}"
                
                return feedback
            else:
                return f"❌ Composition failed: {result.get('error')}"
                
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def generate_text(self, prompt: str, selected_adapters: List[str], strategy: str, 
                     max_length: int = 100, temperature: float = 0.7) -> Tuple[str, str]:
        """Generate text with comprehensive feedback."""
        if not self.engine:
            return "❌ Engine not initialized", ""
        
        if not prompt.strip():
            return "❌ Please enter a prompt", ""
        
        try:
            start_time = time.time()
            self.performance_metrics['total_generations'] += 1
            
            # Generate response
            if selected_adapters:
                strategy_enum = CompositionStrategy(strategy.lower())
                response = self.engine.generate_with_composition(
                    prompt,
                    selected_adapters,
                    strategy_enum,
                    max_length=max_length,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None
                )
                generation_type = f"Composition ({strategy})"
            else:
                response = self.engine.generate(
                    prompt,
                    max_length=max_length,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None
                )
                generation_type = "Base model"
            
            generation_time = time.time() - start_time
            
            # Update performance metrics
            self.performance_metrics['avg_response_time'] = (
                (self.performance_metrics['avg_response_time'] * (self.performance_metrics['total_generations'] - 1) + 
                 generation_time) / self.performance_metrics['total_generations']
            )
            
            # Create generation info
            gen_info = f"🎯 Type: {generation_type}\n"
            gen_info += f"⏱️ Generation time: {generation_time:.2f}s\n"
            gen_info += f"📏 Response length: {len(response)} chars\n"
            gen_info += f"🎚️ Temperature: {temperature}\n"
            gen_info += f"📊 Max length: {max_length}"
            
            if selected_adapters:
                gen_info += f"\n📦 Adapters used: {', '.join(selected_adapters)}"
            
            # Add to history
            self.composition_history.append({
                'timestamp': datetime.now().isoformat(),
                'prompt': prompt,
                'response': response,
                'adapters': selected_adapters,
                'strategy': strategy,
                'generation_time': generation_time
            })
            
            return response, gen_info
            
        except Exception as e:
            return f"❌ Error: {str(e)}", ""
    
    def get_performance_metrics(self) -> str:
        """Get current performance metrics."""
        if not self.is_initialized:
            return "Engine not initialized"
        
        metrics = f"📊 PERFORMANCE METRICS\n"
        metrics += f"{'='*30}\n"
        metrics += f"🎯 Total generations: {self.performance_metrics['total_generations']}\n"
        metrics += f"🔄 Total compositions: {self.performance_metrics['total_compositions']}\n"
        metrics += f"⏱️ Avg response time: {self.performance_metrics['avg_response_time']:.2f}s\n"
        metrics += f"📈 History entries: {len(self.composition_history)}\n\n"
        
        if self.performance_metrics['adapter_usage']:
            metrics += f"📦 ADAPTER USAGE:\n"
            for adapter, count in self.performance_metrics['adapter_usage'].items():
                metrics += f"   • {adapter}: {count} times\n"
        
        return metrics
    
    def get_system_status(self) -> str:
        """Get comprehensive system status."""
        if not self.engine:
            return "Engine not initialized"
        
        try:
            status = self.engine.get_system_status()
            loaded_adapters = self.engine.get_loaded_adapters()
            
            status_info = f"🖥️ SYSTEM STATUS\n"
            status_info += f"{'='*30}\n"
            status_info += f"🤖 Model: {status['model_name']}\n"
            status_info += f"💾 Device: {status['device']}\n"
            status_info += f"📦 Available adapters: {len(status['available_adapters'])}\n"
            status_info += f"🔄 Loaded adapters: {len(loaded_adapters)}\n"
            
            if loaded_adapters:
                status_info += f"\nCurrently loaded:\n"
                for adapter in loaded_adapters:
                    status_info += f"   • {adapter}\n"
            
            return status_info
            
        except Exception as e:
            return f"❌ Error getting status: {str(e)}"


def create_production_interface():
    """Create the production-ready Adaptrix interface."""

    interface = ProductionAdaptrixInterface()

    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .metric-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    """

    with gr.Blocks(title="🚀 Adaptrix Production Interface", css=custom_css) as demo:

        # Header
        gr.HTML("""
        <div class="header-gradient">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">🚀 ADAPTRIX</h1>
            <h2 style="color: white; margin: 10px 0; font-size: 1.2em;">Production Multi-Adapter Composition System</h2>
            <p style="color: white; margin: 0; opacity: 0.9;">Real HuggingFace LoRA Adapters • Dynamic Composition • Production Ready</p>
        </div>
        """)

        # Main interface
        with gr.Tabs():

            # Tab 1: System Control
            with gr.Tab("🎯 System Control"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🚀 Engine Initialization")
                        init_btn = gr.Button("Initialize Adaptrix Engine", variant="primary", size="lg")

                        with gr.Row():
                            init_status = gr.Textbox(label="Initialization Status", lines=6, interactive=False)
                            adapters_info = gr.Textbox(label="Available Adapters", lines=6, interactive=False)

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 System Status")
                        refresh_status_btn = gr.Button("Refresh Status")
                        system_status = gr.Textbox(label="System Status", lines=8, interactive=False)

                        gr.Markdown("### 📈 Performance Metrics")
                        refresh_metrics_btn = gr.Button("Refresh Metrics")
                        performance_metrics = gr.Textbox(label="Performance Metrics", lines=8, interactive=False)

            # Tab 2: Adapter Management
            with gr.Tab("📦 Adapter Management"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🔄 Single Adapter Operations")

                        refresh_adapters_btn = gr.Button("Refresh Adapter List")
                        available_adapters = gr.Dropdown(
                            label="Available Adapters",
                            choices=[],
                            interactive=True
                        )

                        with gr.Row():
                            load_btn = gr.Button("Load Adapter", variant="primary")
                            unload_btn = gr.Button("Unload Adapter", variant="secondary")

                        adapter_operation_result = gr.Textbox(label="Operation Result", lines=3, interactive=False)

                        gr.Markdown("### 🔄 Adapter Switching")
                        with gr.Row():
                            old_adapter = gr.Dropdown(label="From Adapter", choices=[], interactive=True)
                            new_adapter = gr.Dropdown(label="To Adapter", choices=[], interactive=True)

                        switch_btn = gr.Button("Switch Adapters", variant="primary")
                        switch_result = gr.Textbox(label="Switch Result", lines=2, interactive=False)

                    with gr.Column():
                        gr.Markdown("### 📊 Adapter Status")
                        refresh_loaded_btn = gr.Button("Refresh Loaded Adapters")
                        loaded_adapters_status = gr.Textbox(label="Currently Loaded Adapters", lines=4, interactive=False)

                        gr.Markdown("### 🎯 Quick Actions")
                        gr.Markdown("Load commonly used adapters:")

                        with gr.Row():
                            load_gsm8k_btn = gr.Button("Load GSM8K Math", variant="secondary")
                            load_instruct_btn = gr.Button("Load Instruction", variant="secondary")

                        quick_action_result = gr.Textbox(label="Quick Action Result", lines=2, interactive=False)

            # Tab 3: Text Generation
            with gr.Tab("✨ Text Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Input Configuration")

                        prompt_input = gr.Textbox(
                            label="Enter your prompt",
                            placeholder="Ask anything! Try math problems, instructions, or creative tasks...",
                            lines=4
                        )

                        with gr.Row():
                            max_length_slider = gr.Slider(
                                minimum=10,
                                maximum=200,
                                value=100,
                                step=10,
                                label="Max Length"
                            )
                            temperature_slider = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.7,
                                step=0.1,
                                label="Temperature"
                            )

                        gr.Markdown("### 📦 Adapter Selection")
                        adapter_checkboxes = gr.CheckboxGroup(
                            label="Select Adapters for Composition",
                            choices=[],
                            value=[]
                        )

                        strategy_dropdown = gr.Dropdown(
                            label="Composition Strategy",
                            choices=[
                                ("🔄 Parallel", "parallel"),
                                ("⛓️ Sequential", "sequential"),
                                ("🏗️ Hierarchical", "hierarchical"),
                                ("🎯 Weighted", "weighted")
                            ],
                            value="parallel"
                        )

                        generate_btn = gr.Button("🚀 Generate Text", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        gr.Markdown("### 🎯 Generated Response")

                        output_text = gr.Textbox(
                            label="Generated Text",
                            lines=12,
                            interactive=False,
                            placeholder="Generated text will appear here..."
                        )

                        generation_info = gr.Textbox(
                            label="Generation Information",
                            lines=8,
                            interactive=False,
                            placeholder="Generation details will appear here..."
                        )

            # Tab 4: Multi-Adapter Composition
            with gr.Tab("🚀 Advanced Composition"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 🎯 Composition Setup")

                        composition_adapters = gr.CheckboxGroup(
                            label="Select Adapters for Composition",
                            choices=[],
                            value=[]
                        )

                        composition_strategy = gr.Dropdown(
                            label="Composition Strategy",
                            choices=[
                                ("🔄 Parallel", "parallel"),
                                ("⛓️ Sequential", "sequential"),
                                ("🏗️ Hierarchical", "hierarchical"),
                                ("🎯 Weighted", "weighted")
                            ],
                            value="weighted"
                        )

                        compose_btn = gr.Button("🚀 Compose Adapters", variant="primary")
                        composition_result = gr.Textbox(label="Composition Result", lines=10, interactive=False)

                    with gr.Column():
                        gr.Markdown("### 🧪 Test Composition")

                        test_prompts = gr.Dropdown(
                            label="Quick Test Prompts",
                            choices=[
                                "What is 25 * 4?",
                                "Please explain how to solve 15% of 240 step by step",
                                "Write a short story about a robot learning math",
                                "Calculate the area of a rectangle with length 8 and width 5",
                                "Explain the benefits of exercise in simple terms"
                            ],
                            value="What is 25 * 4?"
                        )

                        custom_test_prompt = gr.Textbox(
                            label="Or enter custom test prompt",
                            placeholder="Enter your own test prompt...",
                            lines=2
                        )

                        test_btn = gr.Button("🧪 Test Composition", variant="secondary")
                        test_result = gr.Textbox(label="Test Result", lines=8, interactive=False)

        # Event handlers
        def initialize():
            return interface.initialize_engine()

        def refresh_adapters():
            adapters = interface.get_available_adapters()
            return (
                gr.Dropdown(choices=adapters),  # available_adapters
                gr.Dropdown(choices=adapters),  # old_adapter
                gr.Dropdown(choices=adapters),  # new_adapter
                gr.CheckboxGroup(choices=adapters),  # adapter_checkboxes
                gr.CheckboxGroup(choices=adapters)   # composition_adapters
            )

        def load_adapter(adapter_name):
            return interface.load_single_adapter(adapter_name)

        def unload_adapter_func(adapter_name):
            return interface.unload_adapter(adapter_name)

        def switch_adapters_func(old, new):
            return interface.switch_adapters(old, new)

        def get_loaded_status():
            return interface.get_loaded_adapters()

        def get_system_status():
            return interface.get_system_status()

        def get_performance_metrics():
            return interface.get_performance_metrics()

        def compose_adapters_func(adapters, strategy):
            return interface.compose_adapters(adapters, strategy)

        def generate_text_func(prompt, adapters, strategy, max_length, temperature):
            return interface.generate_text(prompt, adapters, strategy, max_length, temperature)

        def test_composition(test_prompt, custom_prompt, adapters, strategy):
            prompt = custom_prompt.strip() if custom_prompt.strip() else test_prompt
            response, info = interface.generate_text(prompt, adapters, strategy, 100, 0.7)
            return f"Prompt: {prompt}\n\nResponse: {response}\n\n{info}"

        def quick_load_gsm8k():
            return interface.load_single_adapter("phi2_gsm8k_converted")

        def quick_load_instruct():
            return interface.load_single_adapter("phi2_instruct_converted")

        # Wire up all events
        init_btn.click(initialize, outputs=[init_status, adapters_info])
        refresh_adapters_btn.click(
            refresh_adapters,
            outputs=[available_adapters, old_adapter, new_adapter, adapter_checkboxes, composition_adapters]
        )
        load_btn.click(load_adapter, inputs=available_adapters, outputs=adapter_operation_result)
        unload_btn.click(unload_adapter_func, inputs=available_adapters, outputs=adapter_operation_result)
        switch_btn.click(switch_adapters_func, inputs=[old_adapter, new_adapter], outputs=switch_result)
        refresh_loaded_btn.click(get_loaded_status, outputs=loaded_adapters_status)
        refresh_status_btn.click(get_system_status, outputs=system_status)
        refresh_metrics_btn.click(get_performance_metrics, outputs=performance_metrics)
        compose_btn.click(compose_adapters_func, inputs=[composition_adapters, composition_strategy], outputs=composition_result)
        generate_btn.click(
            generate_text_func,
            inputs=[prompt_input, adapter_checkboxes, strategy_dropdown, max_length_slider, temperature_slider],
            outputs=[output_text, generation_info]
        )
        test_btn.click(
            test_composition,
            inputs=[test_prompts, custom_test_prompt, composition_adapters, composition_strategy],
            outputs=test_result
        )
        load_gsm8k_btn.click(quick_load_gsm8k, outputs=quick_action_result)
        load_instruct_btn.click(quick_load_instruct, outputs=quick_action_result)

    return demo


def launch_production_interface(port: int = 7862, share: bool = False):
    """Launch the production interface."""
    print("🚀" * 80)
    print("🚀 LAUNCHING PRODUCTION ADAPTRIX INTERFACE 🚀")
    print("🚀" * 80)
    print()
    print("🌟 PRODUCTION FEATURES:")
    print("   ✅ Real-time adapter management")
    print("   ✅ Multi-adapter composition with strategies")
    print("   ✅ Performance metrics and monitoring")
    print("   ✅ Advanced text generation controls")
    print("   ✅ System status and health monitoring")
    print("   ✅ Quick actions and presets")
    print("   ✅ Comprehensive testing interface")
    print()

    demo = create_production_interface()

    print(f"🚀 Starting production server on http://127.0.0.1:{port}")
    if share:
        print("🌐 Creating public share link...")

    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        debug=True,
        show_error=True,
        share=share
    )


if __name__ == "__main__":
    launch_production_interface()
