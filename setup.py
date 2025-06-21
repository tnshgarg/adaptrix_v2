"""
Setup script for Adaptrix.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adaptrix",
    version="0.1.0",
    author="Adaptrix Team",
    author_email="team@adaptrix.ai",
    description="Middle-Layer LoRA Injection System for Enhanced Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adaptrix/adaptrix",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
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
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.7.0",
        ],
        "web": [
            "gradio>=4.0.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
        ],
        "monitoring": [
            "wandb>=0.16.0",
            "tensorboard>=2.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "adaptrix=src.cli.main:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "adaptrix": ["configs/*.yaml"],
    },
)
