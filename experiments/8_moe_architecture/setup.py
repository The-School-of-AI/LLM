from setuptools import setup, find_packages

setup(
    name="coreset-engineering",
    version="0.1.0",
    description="Coreset selection pipeline for LLM training data",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "pandas>=2.0.0",
        "datasketch>=1.6.0",
        "xxhash>=3.2.0",
        "datasets>=2.14.0",
        "tokenizers>=0.13.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "jsonlines>=3.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "ruff>=0.1.0",
            "mypy>=1.5.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "jupyter>=1.0.0",
        ],
    },
)
