"""
🚀 REVOLUTIONARY ADAPTRIX WEB INTERFACE

This is the world's first web interface for middle-layer multi-adapter composition!
Showcases the revolutionary capabilities of Adaptrix through an intuitive,
interactive interface.

Features:
- Real-time adapter composition
- Live performance monitoring  
- Side-by-side comparisons
- Adapter marketplace
- Interactive composition strategies
- Performance visualization
"""

import gradio as gr
import logging
import time
import json
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime

# Import Adaptrix components
from ..core.engine import AdaptrixEngine
from ..composition.adapter_composer import CompositionStrategy
from ..utils.config import config

logger = logging.getLogger(__name__)


class AdaptrixWebInterface:
    """
    Revolutionary web interface for Adaptrix multi-adapter composition system.
    """
    
    def __init__(self):
        """Initialize the web interface."""
        self.engine = None
        self.composition_history = []
        self.performance_data = []
        
        # Interface state
        self.current_adapters = []
        self.current_strategy = CompositionStrategy.PARALLEL
        
        logger.info("AdaptrixWebInterface initialized")
    
    def initialize_engine(self, model_name: str = "deepseek-ai/deepseek-r1-distill-qwen-1.5b") -> str:
        """Initialize the Adaptrix engine."""
        try:
            self.engine = AdaptrixEngine(model_name, "cpu")
            if self.engine.initialize():
                return "✅ Adaptrix Engine initialized successfully! Ready for revolutionary AI composition."
            else:
                return "❌ Failed to initialize Adaptrix Engine. Please check the logs."
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            return f"❌ Error: {str(e)}"
    
    def get_available_adapters(self) -> List[str]:
        """Get list of available adapters."""
        if not self.engine:
            return []
        return self.engine.list_adapters()
    
    def get_composition_recommendations(self) -> str:
        """Get intelligent composition recommendations."""
        if not self.engine:
            return "❌ Engine not initialized"
        
        try:
            recommendations = self.engine.get_composition_recommendations()
            if not recommendations.get('success'):
                return f"❌ Failed to get recommendations: {recommendations.get('error')}"
            
            result = "🧠 **AI-Powered Composition Recommendations:**\n\n"
            
            for config_name, config in recommendations['recommendations'].items():
                result += f"**{config_name.replace('_', ' ').title()}:**\n"
                result += f"- Strategy: {config['strategy']}\n"
                result += f"- Adapters: {', '.join(config['adapters'])}\n"
                result += f"- Benefits:\n"
                for benefit in config['expected_benefits']:
                    result += f"  • {benefit}\n"
                result += "\n"
            
            stats = recommendations.get('composition_stats', {})
            result += f"**System Stats:**\n"
            result += f"- Total Compositions: {stats.get('total_compositions', 0)}\n"
            result += f"- Success Rate: {stats.get('success_rate', 0):.1%}\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return f"❌ Error: {str(e)}"
    
    def compose_adapters(self, 
                        selected_adapters: List[str], 
                        strategy: str,
                        temperature: float = 1.0,
                        confidence_threshold: float = 0.7) -> Tuple[str, str]:
        """
        Compose selected adapters with the given strategy.
        
        Returns:
            Tuple of (status_message, composition_details)
        """
        if not self.engine:
            return "❌ Engine not initialized", ""
        
        if not selected_adapters:
            return "❌ No adapters selected", ""
        
        try:
            # Convert strategy string to enum
            strategy_enum = CompositionStrategy(strategy.lower())
            
            # Perform composition
            start_time = time.time()
            result = self.engine.compose_adapters(
                selected_adapters, 
                strategy_enum,
                temperature=temperature,
                confidence_threshold=confidence_threshold
            )
            processing_time = time.time() - start_time
            
            if not result.get('success'):
                return f"❌ Composition failed: {result.get('error')}", ""
            
            # Store in history
            composition_record = {
                'timestamp': datetime.now().isoformat(),
                'adapters': selected_adapters,
                'strategy': strategy,
                'processing_time': processing_time,
                'result': result
            }
            self.composition_history.append(composition_record)
            
            # Update performance data
            self.performance_data.append({
                'timestamp': datetime.now(),
                'adapters_count': len(selected_adapters),
                'strategy': strategy,
                'processing_time': processing_time,
                'success': True
            })
            
            # Create status message
            status = f"✅ **Composition Successful!**\n"
            status += f"Strategy: {result['strategy']}\n"
            status += f"Adapters: {', '.join(result['adapters_used'])}\n"
            status += f"Processing Time: {processing_time:.3f}s"
            
            # Create detailed results
            details = f"**🚀 Revolutionary Multi-Adapter Composition Results**\n\n"
            details += f"**Strategy Used:** {result['strategy']}\n"
            details += f"**Adapters Composed:** {', '.join(result['adapters_used'])}\n"
            details += f"**Processing Time:** {processing_time:.3f} seconds\n\n"
            
            details += f"**Adapter Weights:**\n"
            for adapter, weight in result.get('weights', {}).items():
                details += f"- {adapter}: {weight:.3f}\n"
            
            details += f"\n**Confidence Scores:**\n"
            for adapter, confidence in result.get('confidence', {}).items():
                details += f"- {adapter}: {confidence:.3f}\n"
            
            details += f"\n**Metadata:**\n"
            for key, value in result.get('metadata', {}).items():
                details += f"- {key}: {value}\n"
            
            return status, details
            
        except Exception as e:
            logger.error(f"Composition failed: {e}")
            error_msg = f"❌ Error: {str(e)}"
            return error_msg, ""
    
    def generate_with_composition(self,
                                 prompt: str,
                                 selected_adapters: List[str],
                                 strategy: str,
                                 max_length: int = 150,
                                 temperature: float = 0.7) -> str:
        """Generate text using multi-adapter composition."""
        if not self.engine:
            return "❌ Engine not initialized"
        
        if not prompt.strip():
            return "❌ Please enter a prompt"
        
        if not selected_adapters:
            return "❌ Please select at least one adapter"
        
        try:
            # Convert strategy string to enum
            strategy_enum = CompositionStrategy(strategy.lower())
            
            # Generate with composition
            start_time = time.time()
            response = self.engine.generate_with_composition(
                prompt,
                selected_adapters,
                strategy_enum,
                max_length=max_length,
                temperature=temperature
            )
            generation_time = time.time() - start_time
            
            # Add generation metadata
            metadata = f"\n\n---\n**Generation Info:**\n"
            metadata += f"- Adapters: {', '.join(selected_adapters)}\n"
            metadata += f"- Strategy: {strategy}\n"
            metadata += f"- Generation Time: {generation_time:.3f}s\n"
            metadata += f"- Max Length: {max_length}\n"
            metadata += f"- Temperature: {temperature}"
            
            return response + metadata
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"❌ Generation Error: {str(e)}"
    
    def compare_strategies(self,
                          prompt: str,
                          selected_adapters: List[str],
                          max_length: int = 100) -> str:
        """Compare different composition strategies side-by-side."""
        if not self.engine or not prompt.strip() or not selected_adapters:
            return "❌ Please ensure engine is initialized, prompt is provided, and adapters are selected"
        
        strategies = [
            CompositionStrategy.PARALLEL,
            CompositionStrategy.SEQUENTIAL,
            CompositionStrategy.HIERARCHICAL,
            CompositionStrategy.ATTENTION
        ]
        
        results = "🔬 **Strategy Comparison Results**\n\n"
        
        for strategy in strategies:
            try:
                start_time = time.time()
                response = self.engine.generate_with_composition(
                    prompt,
                    selected_adapters,
                    strategy,
                    max_length=max_length,
                    temperature=0.7
                )
                processing_time = time.time() - start_time
                
                results += f"**{strategy.value.title()} Strategy** ({processing_time:.3f}s):\n"
                results += f"{response[:200]}...\n\n"
                results += "---\n\n"
                
            except Exception as e:
                results += f"**{strategy.value.title()} Strategy**: ❌ Error: {str(e)}\n\n"
        
        return results
    
    def get_performance_chart(self) -> go.Figure:
        """Create performance visualization chart."""
        if not self.performance_data:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No performance data available yet.<br>Run some compositions to see charts!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(title="Performance Monitoring")
            return fig
        
        # Create DataFrame from performance data
        df = pd.DataFrame(self.performance_data)
        
        # Create performance chart
        fig = go.Figure()
        
        # Add processing time trace
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['processing_time'],
            mode='lines+markers',
            name='Processing Time (s)',
            line=dict(color='blue'),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         'Time: %{x}<br>' +
                         'Processing Time: %{y:.3f}s<br>' +
                         '<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title="🚀 Adaptrix Performance Monitoring",
            xaxis_title="Time",
            yaxis_title="Processing Time (seconds)",
            hovermode='closest',
            showlegend=True
        )
        
        return fig
    
    def get_system_status(self) -> str:
        """Get current system status."""
        if not self.engine:
            return "❌ Engine not initialized"
        
        try:
            status = self.engine.get_system_status()
            
            result = "🔧 **System Status**\n\n"
            result += f"**Model:** {status.get('model_name', 'Unknown')}\n"
            result += f"**Device:** {status.get('device', 'Unknown')}\n"
            result += f"**Initialized:** {'✅' if status.get('initialized') else '❌'}\n"
            result += f"**Available Adapters:** {len(status.get('available_adapters', []))}\n"
            result += f"**Loaded Adapters:** {len(status.get('loaded_adapters', []))}\n"
            
            if 'composition_stats' in status:
                comp_stats = status['composition_stats']
                result += f"\n**Composition Statistics:**\n"
                result += f"- Total Compositions: {comp_stats.get('total_compositions', 0)}\n"
                result += f"- Success Rate: {comp_stats.get('success_rate', 0):.1%}\n"
                result += f"- Avg Processing Time: {comp_stats.get('avg_processing_time', 0):.3f}s\n"
            
            # Memory info
            if 'system_memory' in status:
                mem = status['system_memory']
                result += f"\n**Memory Usage:**\n"
                result += f"- Available: {mem.get('available_gb', 0):.1f} GB\n"
                result += f"- Used: {mem.get('used_gb', 0):.1f} GB\n"
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return f"❌ Error getting status: {str(e)}"


def create_adaptrix_interface() -> gr.Blocks:
    """
    Create the revolutionary Adaptrix web interface.

    Returns:
        Gradio Blocks interface
    """
    # Initialize the interface
    interface = AdaptrixWebInterface()

    # Custom CSS for revolutionary styling
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        border: none;
        color: white;
        font-weight: bold;
    }
    .gr-button-primary:hover {
        background: linear-gradient(45deg, #FF5252, #26C6DA);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .revolutionary-header {
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        margin: 20px 0;
    }
    """

    with gr.Blocks(css=custom_css, title="🚀 Adaptrix - Revolutionary AI Composition") as demo:

        # Header
        gr.HTML("""
        <div class="revolutionary-header">
            🚀 ADAPTRIX - REVOLUTIONARY AI COMPOSITION SYSTEM 🚀
        </div>
        <div style="text-align: center; margin-bottom: 30px;">
            <h3>World's First Middle-Layer Multi-Adapter Composition Platform</h3>
            <p>Experience the future of AI through dynamic adapter composition</p>
        </div>
        """)

        # Main tabs
        with gr.Tabs():

            # Tab 1: Quick Start & Composition
            with gr.TabItem("🚀 Quick Start & Composition"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 System Initialization")
                        init_btn = gr.Button("Initialize Adaptrix Engine", variant="primary")
                        init_status = gr.Textbox(label="Status", interactive=False)

                        gr.Markdown("### 📦 Adapter Selection")
                        adapter_refresh_btn = gr.Button("Refresh Adapters")
                        available_adapters = gr.CheckboxGroup(
                            label="Available Adapters",
                            choices=[],
                            value=[]
                        )

                        gr.Markdown("### ⚙️ Composition Settings")
                        strategy_dropdown = gr.Dropdown(
                            label="Composition Strategy",
                            choices=[
                                ("🔄 Parallel - Simultaneous Processing", "parallel"),
                                ("⛓️ Sequential - Pipeline Processing", "sequential"),
                                ("🏗️ Hierarchical - Structured Stages", "hierarchical"),
                                ("🎯 Attention - Dynamic Weighting", "attention"),
                                ("⚖️ Weighted - Custom Weights", "weighted")
                            ],
                            value="parallel"
                        )

                        with gr.Row():
                            temperature_slider = gr.Slider(
                                label="Temperature",
                                minimum=0.1,
                                maximum=2.0,
                                value=1.0,
                                step=0.1
                            )
                            confidence_slider = gr.Slider(
                                label="Confidence Threshold",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.7,
                                step=0.1
                            )

                        compose_btn = gr.Button("🚀 Compose Adapters", variant="primary")

                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Composition Results")
                        composition_status = gr.Textbox(label="Composition Status", interactive=False)
                        composition_details = gr.Textbox(
                            label="Detailed Results",
                            lines=10,
                            interactive=False
                        )

                        gr.Markdown("### 🧠 AI Recommendations")
                        recommendations_btn = gr.Button("Get AI Recommendations")
                        recommendations_output = gr.Textbox(
                            label="Intelligent Recommendations",
                            lines=8,
                            interactive=False
                        )

            # Tab 2: Text Generation
            with gr.TabItem("✨ Enhanced Text Generation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Input")
                        prompt_input = gr.Textbox(
                            label="Enter your prompt",
                            placeholder="Ask anything! The composed adapters will enhance the response...",
                            lines=3
                        )

                        gen_adapters = gr.CheckboxGroup(
                            label="Select Adapters for Generation",
                            choices=[],
                            value=[]
                        )

                        gen_strategy = gr.Dropdown(
                            label="Generation Strategy",
                            choices=[
                                ("🔄 Parallel", "parallel"),
                                ("⛓️ Sequential", "sequential"),
                                ("🏗️ Hierarchical", "hierarchical"),
                                ("🎯 Attention", "attention")
                            ],
                            value="parallel"
                        )

                        with gr.Row():
                            max_length_slider = gr.Slider(
                                label="Max Length",
                                minimum=50,
                                maximum=500,
                                value=150,
                                step=10
                            )
                            gen_temperature_slider = gr.Slider(
                                label="Temperature",
                                minimum=0.1,
                                maximum=2.0,
                                value=0.7,
                                step=0.1
                            )

                        generate_btn = gr.Button("🚀 Generate with Composition", variant="primary")
                        compare_btn = gr.Button("🔬 Compare All Strategies")

                    with gr.Column(scale=2):
                        gr.Markdown("### 🎯 Enhanced Output")
                        generation_output = gr.Textbox(
                            label="Generated Response",
                            lines=15,
                            interactive=False
                        )

            # Tab 3: Performance Monitoring
            with gr.TabItem("📊 Performance Monitoring"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔧 System Controls")
                        status_refresh_btn = gr.Button("Refresh System Status")
                        system_status_output = gr.Textbox(
                            label="System Status",
                            lines=15,
                            interactive=False
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 📈 Performance Charts")
                        performance_chart = gr.Plot(label="Performance Metrics")
                        chart_refresh_btn = gr.Button("Refresh Charts")

            # Tab 4: Adapter Marketplace
            with gr.TabItem("🏪 Adapter Marketplace"):
                gr.Markdown("### 🏪 Adapter Marketplace")
                gr.Markdown("""
                **Coming Soon!** The Adapter Marketplace will feature:
                - 📦 Browse community-created adapters
                - ⭐ Rate and review adapters
                - 🔄 One-click adapter installation
                - 🏷️ Categorized adapter collections
                - 🎯 Personalized recommendations
                - 🔒 Verified and secure adapters
                """)

                marketplace_placeholder = gr.HTML("""
                <div style="text-align: center; padding: 50px; background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%); border-radius: 10px; color: white;">
                    <h2>🚀 Revolutionary Adapter Marketplace</h2>
                    <p>The future of AI customization is coming soon!</p>
                    <p>Imagine thousands of specialized adapters at your fingertips...</p>
                </div>
                """)

        # Event handlers

        def initialize_engine():
            return interface.initialize_engine()

        def compose_adapters_handler(adapters, strategy, temp, conf):
            return interface.compose_adapters(adapters, strategy, temp, conf)

        def generate_handler(prompt, adapters, strategy, max_len, temp):
            return interface.generate_with_composition(prompt, adapters, strategy, max_len, temp)

        def compare_handler(prompt, adapters, max_len):
            return interface.compare_strategies(prompt, adapters, max_len)

        def get_recommendations():
            return interface.get_composition_recommendations()

        def get_status():
            return interface.get_system_status()

        def get_performance_chart():
            return interface.get_performance_chart()

        # Wire up events
        init_btn.click(initialize_engine, outputs=init_status)

        def update_adapter_choices():
            adapters = interface.get_available_adapters()
            return gr.CheckboxGroup(choices=adapters, value=[]), gr.CheckboxGroup(choices=adapters, value=[])

        adapter_refresh_btn.click(update_adapter_choices, outputs=[available_adapters, gen_adapters])
        compose_btn.click(
            compose_adapters_handler,
            inputs=[available_adapters, strategy_dropdown, temperature_slider, confidence_slider],
            outputs=[composition_status, composition_details]
        )
        generate_btn.click(
            generate_handler,
            inputs=[prompt_input, gen_adapters, gen_strategy, max_length_slider, gen_temperature_slider],
            outputs=generation_output
        )
        compare_btn.click(
            compare_handler,
            inputs=[prompt_input, gen_adapters, max_length_slider],
            outputs=generation_output
        )
        recommendations_btn.click(get_recommendations, outputs=recommendations_output)
        status_refresh_btn.click(get_status, outputs=system_status_output)
        chart_refresh_btn.click(get_performance_chart, outputs=performance_chart)

    return demo


def launch_adaptrix_interface(
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    debug: bool = False
) -> None:
    """
    Launch the revolutionary Adaptrix web interface.

    Args:
        share: Whether to create a public link
        server_name: Server hostname
        server_port: Server port
        debug: Enable debug mode
    """
    print("🚀" * 50)
    print("🚀 LAUNCHING ADAPTRIX REVOLUTIONARY WEB INTERFACE 🚀")
    print("🚀" * 50)
    print()
    print("🌟 Features:")
    print("   • Multi-adapter composition")
    print("   • Real-time performance monitoring")
    print("   • Interactive strategy comparison")
    print("   • AI-powered recommendations")
    print("   • Revolutionary user experience")
    print()

    # Create and launch interface
    demo = create_adaptrix_interface()

    print(f"🚀 Starting server on {server_name}:{server_port}")
    if share:
        print("🌍 Creating public link for sharing...")

    demo.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        debug=debug,
        show_error=True
    )


if __name__ == "__main__":
    # Launch with default settings
    launch_adaptrix_interface(debug=True)
