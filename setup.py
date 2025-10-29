#!/usr/bin/env python3
"""
Pac-Man RL - Language Model Strategy Training
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read the README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("git+")
        ]
else:
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "trl>=0.8.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.28.0",
        "flask>=3.0.0",
        "flask-socketio>=5.3.0",
        "numpy>=1.24.0",
    ]

setup(
    name="pacman-rl",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Training language models to play Pac-Man by generating strategy code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cpich3g/pacman-rl",
    packages=find_packages(exclude=["notebooks", "play_the_game"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=24.0.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "pre-commit>=3.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pacman-train=train_pacman_docker_grpo_v2:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
