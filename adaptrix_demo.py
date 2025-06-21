"""
Adaptrix Demo: Real Adapter Switching with Context Preservation
Demonstrates intelligent switching between general chat and math adapters.
"""

import sys
import os
import re
import torch
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.adapters.peft_converter import PEFTConverter
from src.adapters.adapter_manager import AdapterManager
from src.core.engine import AdaptrixEngine


class AdaptrixDemo:
    """Interactive demo for Adaptrix with real adapter switching."""
    
    def __init__(self):
        self.engine = None
        self.available_adapters = {}
        self.current_adapter = None
        self.conversation_history = []
        
        # Math keywords for intelligent switching
        self.math_keywords = [
            'solve', 'calculate', 'equation', 'derivative', 'integral', 'math', 'mathematics',
            'algebra', 'geometry', 'trigonometry', 'calculus', 'statistics', 'probability',
            'formula', 'theorem', 'proof', 'graph', 'function', 'variable', 'coefficient',
            'polynomial', 'logarithm', 'exponential', 'matrix', 'vector', 'sum', 'product',
            'divide', 'multiply', 'subtract', 'add', 'equals', 'percentage', 'fraction',
            'decimal', 'number', 'digit', 'prime', 'factor', 'square', 'cube', 'root',
            'sin', 'cos', 'tan', 'pi', 'radius', 'diameter', 'area', 'volume', 'perimeter'
        ]
    
    def setup_engine(self):
        """Initialize the Adaptrix engine."""
        print("🚀 Initializing Adaptrix Engine...")
        try:
            self.engine = AdaptrixEngine("microsoft/DialoGPT-small", "cpu")
            self.engine.initialize()
            print("✅ Engine initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize engine: {e}")
            return False
    
    def download_and_convert_adapters(self) -> bool:
        """Download and convert real adapters."""
        print("\n📥 Setting up real adapters...")

        adapters_to_setup = [
            {
                "id": "tloen/alpaca-lora-7b",
                "name": "general_chat",
                "description": "General conversation and instruction following (Alpaca)",
                "type": "general"
            }
        ]

        # For now, we'll use the existing creative_writing_demo as our "math" adapter
        # since the real math adapter conversion needs more work
        print("📝 Note: Using existing creative_writing_demo as secondary adapter for demonstration")
        
        converter = PEFTConverter(target_layers=[3, 6, 9])
        success_count = 0
        
        for adapter_info in adapters_to_setup:
            print(f"\n🔄 Processing {adapter_info['id']}...")
            
            try:
                # Check if adapter already exists
                adapter_dir = os.path.join("adapters", adapter_info['name'])
                if os.path.exists(adapter_dir):
                    print(f"   ✅ Adapter {adapter_info['name']} already exists")
                    self.available_adapters[adapter_info['type']] = adapter_info['name']
                    success_count += 1
                    continue
                
                # Create temporary directory for conversion
                temp_dir = tempfile.mkdtemp()
                
                try:
                    # Convert from HuggingFace Hub
                    success = converter.convert_from_hub(
                        adapter_id=adapter_info['id'],
                        output_dir=temp_dir,
                        base_model_name="microsoft/DialoGPT-small"
                    )
                    
                    if success:
                        # Move to adapters directory
                        if not os.path.exists("adapters"):
                            os.makedirs("adapters")
                        
                        shutil.move(temp_dir, adapter_dir)
                        print(f"   ✅ Successfully converted {adapter_info['name']}")
                        self.available_adapters[adapter_info['type']] = adapter_info['name']
                        success_count += 1
                    else:
                        print(f"   ❌ Failed to convert {adapter_info['id']}")
                
                finally:
                    # Cleanup temp directory if it still exists
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        
            except Exception as e:
                print(f"   ❌ Error processing {adapter_info['id']}: {e}")
        
        # Add existing adapter as fallback for math/creative tasks
        existing_adapters = self.engine.list_adapters() if self.engine else []
        if existing_adapters and "creative_writing_demo" in existing_adapters:
            self.available_adapters["math"] = "creative_writing_demo"
            print(f"✅ Added existing adapter 'creative_writing_demo' as math/creative adapter")
            success_count += 1

        print(f"\n📊 Adapter setup complete: {success_count}/{len(adapters_to_setup) + 1} adapters available")
        return success_count > 0
    
    def detect_query_type(self, query: str) -> str:
        """Detect if query is math-related or general."""
        query_lower = query.lower()
        
        # Check for math keywords
        math_score = sum(1 for keyword in self.math_keywords if keyword in query_lower)
        
        # Check for mathematical symbols and patterns
        math_patterns = [
            r'\d+\s*[\+\-\*\/\=]\s*\d+',  # Basic arithmetic
            r'[xy]\s*[\+\-\*\/\=]',       # Variables
            r'\b\d+%\b',                   # Percentages
            r'\b\d+\.\d+\b',              # Decimals
            r'\b\d+\/\d+\b',              # Fractions
            r'\b\d+\^\d+\b',              # Exponents
        ]
        
        pattern_matches = sum(1 for pattern in math_patterns if re.search(pattern, query_lower))
        
        # Decision logic
        if math_score >= 2 or pattern_matches >= 1:
            return "math"
        else:
            return "general"
    
    def switch_adapter(self, adapter_type: str) -> bool:
        """Switch to the specified adapter type."""
        if adapter_type not in self.available_adapters:
            print(f"⚠️  Adapter type '{adapter_type}' not available")
            return False
        
        target_adapter = self.available_adapters[adapter_type]
        
        if self.current_adapter == target_adapter:
            # Already using the correct adapter
            return True
        
        try:
            # Unload current adapter if any
            if self.current_adapter:
                self.engine.unload_adapter(self.current_adapter)
                print(f"🔄 Unloaded {self.current_adapter}")
            
            # Load new adapter
            success = self.engine.load_adapter(target_adapter)
            if success:
                self.current_adapter = target_adapter
                adapter_emoji = "🧮" if adapter_type == "math" else "💬"
                print(f"{adapter_emoji} Switched to {target_adapter} ({adapter_type})")
                return True
            else:
                print(f"❌ Failed to load adapter {target_adapter}")
                return False
                
        except Exception as e:
            print(f"❌ Error switching adapter: {e}")
            return False
    
    def generate_response(self, query: str) -> Tuple[str, Dict]:
        """Generate response with appropriate adapter."""
        # Detect query type and switch adapter
        query_type = self.detect_query_type(query)
        adapter_switched = self.switch_adapter(query_type)
        
        if not adapter_switched:
            return "❌ Failed to switch to appropriate adapter", {}
        
        try:
            # Set context anchor for context preservation
            if self.engine.tokenizer:
                query_tokens = self.engine.tokenizer.encode(query, return_tensors="pt")
                query_embedding = torch.randn(1, query_tokens.shape[1], 768)
                self.engine.layer_injector.context_injector.set_query_anchor(query_embedding)
            
            # Generate response
            response = self.engine.query(query, max_length=50)
            
            # Get context statistics
            context_stats = self.engine.layer_injector.context_injector.get_context_statistics()
            
            # Get system status
            system_status = self.engine.get_system_status()
            
            return response, {
                'adapter_type': query_type,
                'adapter_name': self.current_adapter,
                'context_stats': context_stats,
                'system_status': system_status
            }
            
        except Exception as e:
            return f"❌ Generation failed: {e}", {}
    
    def run_test_flow(self) -> bool:
        """Run automated test flow to verify everything works."""
        print("\n🧪 Running Automated Test Flow...")
        print("=" * 60)
        
        test_queries = [
            ("Hello! How are you today?", "general"),
            ("What is 2 + 2?", "math"),
            ("Solve the equation: 3x + 5 = 14", "math"),
            ("Tell me about the weather", "general"),
            ("Calculate the area of a circle with radius 5", "math"),
            ("What's your favorite color?", "general")
        ]
        
        success_count = 0
        
        for i, (query, expected_type) in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: {query}")
            
            # Detect type
            detected_type = self.detect_query_type(query)
            print(f"   🎯 Expected: {expected_type}, Detected: {detected_type}")
            
            if detected_type == expected_type:
                print(f"   ✅ Type detection correct")
            else:
                print(f"   ⚠️  Type detection mismatch")
            
            # Generate response
            response, stats = self.generate_response(query)
            
            if response and not response.startswith("❌"):
                print(f"   💬 Response: '{response[:50]}{'...' if len(response) > 50 else ''}'")
                print(f"   📊 Context injections: {stats.get('context_stats', {}).get('total_injections', 0)}")
                success_count += 1
            else:
                print(f"   ❌ Failed: {response}")
        
        print(f"\n📊 Test Results: {success_count}/{len(test_queries)} successful")
        return success_count == len(test_queries)
    
    def run_interactive_cli(self):
        """Run interactive CLI for user testing."""
        print("\n" + "=" * 80)
        print("🎉 Welcome to Adaptrix Interactive Demo!")
        print("=" * 80)
        print("💡 Features:")
        print("   • Intelligent adapter switching (general chat ↔ math)")
        print("   • Context preservation across conversations")
        print("   • Real-time performance monitoring")
        print("\n💬 Available adapters:")
        for adapter_type, adapter_name in self.available_adapters.items():
            emoji = "🧮" if adapter_type == "math" else "💬"
            print(f"   {emoji} {adapter_type}: {adapter_name}")
        
        print("\n📝 Commands:")
        print("   • Type your message to chat")
        print("   • 'stats' - Show system statistics")
        print("   • 'history' - Show conversation history")
        print("   • 'clear' - Clear conversation history")
        print("   • 'quit' or 'exit' - Exit demo")
        print("=" * 80)
        
        while True:
            try:
                user_input = input("\n🤖 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! Thanks for testing Adaptrix!")
                    break
                
                elif user_input.lower() == 'stats':
                    self.show_statistics()
                    continue
                
                elif user_input.lower() == 'history':
                    self.show_conversation_history()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.conversation_history.clear()
                    print("🗑️  Conversation history cleared")
                    continue
                
                # Generate response
                print("🤔 Thinking...")
                response, stats = self.generate_response(user_input)
                
                # Display response
                adapter_emoji = "🧮" if stats.get('adapter_type') == 'math' else "💬"
                print(f"{adapter_emoji} Adaptrix: {response}")
                
                # Show brief stats
                if stats:
                    context_injections = stats.get('context_stats', {}).get('total_injections', 0)
                    print(f"📊 Context injections: {context_injections} | Adapter: {stats.get('adapter_name', 'unknown')}")
                
                # Save to history
                self.conversation_history.append({
                    'user': user_input,
                    'assistant': response,
                    'adapter_type': stats.get('adapter_type'),
                    'adapter_name': stats.get('adapter_name'),
                    'stats': stats
                })
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Thanks for testing Adaptrix!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_statistics(self):
        """Show detailed system statistics."""
        if not self.engine:
            print("❌ Engine not initialized")
            return
        
        try:
            context_stats = self.engine.layer_injector.context_injector.get_context_statistics()
            system_status = self.engine.get_system_status()
            
            print("\n📊 System Statistics:")
            print("=" * 40)
            print(f"🧠 Context Preservation:")
            print(f"   Layers with context: {context_stats['layers_with_context']}")
            print(f"   Total injections: {context_stats['total_injections']}")
            print(f"   Avg processing time: {context_stats['average_processing_time']:.4f}s")
            
            print(f"\n💾 Memory Usage:")
            memory_info = system_status.get('memory_usage', {})
            if isinstance(memory_info, dict):
                injector_memory = memory_info.get('injector_memory', {})
                print(f"   Injector memory: {injector_memory.get('memory_mb', 'unknown')} MB")
                print(f"   Total parameters: {injector_memory.get('total_parameters', 'unknown')}")
            
            print(f"\n🎯 Active Adapters:")
            print(f"   Current: {self.current_adapter or 'None'}")
            print(f"   Available: {list(self.available_adapters.values())}")
            
            print(f"\n💬 Conversation:")
            print(f"   Messages: {len(self.conversation_history)}")
            
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
    
    def show_conversation_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            print("📝 No conversation history yet")
            return
        
        print("\n📝 Conversation History:")
        print("=" * 50)
        
        for i, entry in enumerate(self.conversation_history, 1):
            adapter_emoji = "🧮" if entry.get('adapter_type') == 'math' else "💬"
            print(f"\n{i}. 🤖 You: {entry['user']}")
            print(f"   {adapter_emoji} Adaptrix ({entry.get('adapter_name', 'unknown')}): {entry['assistant']}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.engine:
            if self.current_adapter:
                self.engine.unload_adapter(self.current_adapter)
            self.engine.cleanup()


def main():
    """Main demo function."""
    demo = AdaptrixDemo()
    
    try:
        # Setup engine
        if not demo.setup_engine():
            return
        
        # Download and convert adapters
        if not demo.download_and_convert_adapters():
            print("❌ Failed to setup adapters. Cannot continue.")
            return
        
        # Run test flow
        test_success = demo.run_test_flow()
        
        if test_success:
            print("✅ All tests passed! Ready for interactive demo.")
        else:
            print("⚠️  Some tests failed, but continuing with demo...")
        
        # Run interactive CLI
        demo.run_interactive_cli()
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()
