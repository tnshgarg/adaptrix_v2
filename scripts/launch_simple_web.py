#!/usr/bin/env python3
"""
🚀 SIMPLIFIED ADAPTRIX WEB INTERFACE LAUNCHER

Launch the working, simplified version of the revolutionary Adaptrix web interface.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.web.simple_gradio_app import launch_simple_interface


def main():
    """Main launcher function."""
    print("🚀" * 60)
    print("🚀 SIMPLIFIED ADAPTRIX WEB INTERFACE LAUNCHER 🚀")
    print("🚀" * 60)
    print()
    print("🌟 This is a working, simplified version that demonstrates:")
    print("   ✅ Multi-Adapter Composition")
    print("   ✅ Enhanced Text Generation")
    print("   ✅ Real-time Adapter Management")
    print("   ✅ Revolutionary AI Capabilities")
    print()
    
    try:
        launch_simple_interface(port=7861)
        
    except KeyboardInterrupt:
        print("\n🛑 Interface stopped by user")
        print("Thank you for using Adaptrix! 🚀")
        
    except Exception as e:
        print(f"\n❌ Failed to launch interface: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
