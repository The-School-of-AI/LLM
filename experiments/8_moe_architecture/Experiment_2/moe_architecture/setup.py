"""
MoE Architecture Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="moe_architecture",
    version="1.0.0",
    author="Team 8 - MoE Architecture",
    description="Mixture of Experts Architecture for LLM Development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "cuda": ["triton>=2.0.0"],
        "dev": ["pytest", "black", "isort", "mypy"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "moe-architecture=moe_architecture.__main__:main",
        ],
    },
)
