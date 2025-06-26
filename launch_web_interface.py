#!/usr/bin/env python3
"""
🚀 ADAPTRIX WEB INTERFACE LAUNCHER

Easy launcher for the production-ready Adaptrix web interface.
"""

import sys
import os
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="Launch Adaptrix Web Interface")
    parser.add_argument("--port", type=int, default=7862, help="Port to run the interface on")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--simple", action="store_true", help="Use the simple interface instead of production")
    
    args = parser.parse_args()
    
    print("🚀" * 60)
    print("🚀 ADAPTRIX WEB INTERFACE LAUNCHER 🚀")
    print("🚀" * 60)
    print()
    
    if args.simple:
        print("📱 Launching Simple Interface...")
        from src.web.simple_gradio_app import launch_simple_interface
        launch_simple_interface(port=args.port)
    else:
        print("🏭 Launching Production Interface...")
        from src.web.production_interface import launch_production_interface
        launch_production_interface(port=args.port, share=args.share)


if __name__ == "__main__":
    main()
