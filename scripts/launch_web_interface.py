#!/usr/bin/env python3
"""
🚀 ADAPTRIX WEB INTERFACE LAUNCHER

Launch the revolutionary Adaptrix web interface with all its groundbreaking features:
- Multi-adapter composition
- Real-time performance monitoring
- Interactive strategy comparison
- AI-powered recommendations
- Revolutionary user experience
"""

import sys
import os
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.web.gradio_app import launch_adaptrix_interface


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="🚀 Launch the Revolutionary Adaptrix Web Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/launch_web_interface.py                    # Local interface
  python scripts/launch_web_interface.py --share            # Public sharing
  python scripts/launch_web_interface.py --port 8080        # Custom port
  python scripts/launch_web_interface.py --debug            # Debug mode

🚀 REVOLUTIONARY FEATURES:
  • Multi-adapter composition with 5 strategies
  • Real-time performance monitoring and analytics
  • Interactive strategy comparison
  • AI-powered composition recommendations
  • Beautiful, responsive web interface
  • Live adapter marketplace (coming soon)
        """
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link for sharing the interface"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server hostname (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Server port (default: 7860)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    print("🚀" * 60)
    print("🚀 ADAPTRIX REVOLUTIONARY WEB INTERFACE LAUNCHER 🚀")
    print("🚀" * 60)
    print()
    print("🌟 Welcome to the future of AI composition!")
    print()
    print("📋 Configuration:")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Share: {'Yes' if args.share else 'No'}")
    print(f"   Debug: {'Yes' if args.debug else 'No'}")
    print()
    
    if args.share:
        print("🌍 SHARING ENABLED:")
        print("   A public link will be created for sharing")
        print("   ⚠️  Warning: Anyone with the link can access your interface")
        print()
    
    print("🚀 REVOLUTIONARY FEATURES AVAILABLE:")
    print("   ✅ Multi-Adapter Composition (5 strategies)")
    print("   ✅ Real-time Performance Monitoring")
    print("   ✅ Interactive Strategy Comparison")
    print("   ✅ AI-Powered Recommendations")
    print("   ✅ Beautiful Responsive Interface")
    print("   ✅ Live System Status")
    print("   🔜 Adapter Marketplace (coming soon)")
    print()
    
    print("🎯 USAGE INSTRUCTIONS:")
    print("   1. Click 'Initialize Adaptrix Engine' to start")
    print("   2. Refresh adapters to see available options")
    print("   3. Select adapters and composition strategy")
    print("   4. Compose adapters or generate enhanced text")
    print("   5. Monitor performance in real-time")
    print()
    
    try:
        # Launch the interface
        launch_adaptrix_interface(
            share=args.share,
            server_name=args.host,
            server_port=args.port,
            debug=args.debug
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Interface stopped by user")
        print("Thank you for using Adaptrix! 🚀")
        
    except Exception as e:
        print(f"\n❌ Failed to launch interface: {e}")
        print("Please check the error details above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
