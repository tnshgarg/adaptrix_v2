#!/usr/bin/env python3
"""
Setup script for the Adaptrix CLI.

This script sets up the CLI for development and installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="adaptrix-cli",
    version="1.0.0",
    description="Command Line Interface for Adaptrix Modular AI System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Adaptrix Team",
    author_email="team@adaptrix.ai",
    url="https://github.com/adaptrix/adaptrix-cli",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "adaptrix=src.cli.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    keywords="ai, machine learning, adapters, lora, cli, llm",
    project_urls={
        "Bug Reports": "https://github.com/adaptrix/adaptrix-cli/issues",
        "Source": "https://github.com/adaptrix/adaptrix-cli",
        "Documentation": "https://docs.adaptrix.ai/cli",
    },
)
