"""
NES-LLM — Neural-Entropic Steganography for Large Language Models.
"""

from setuptools import setup, find_packages

setup(
    name="nes-llm",
    version="0.1.0",
    description="Hide AES-encrypted payloads inside LLM quantization residuals",
    author="NES Research",
    python_requires=">=3.10",
    packages=find_packages(where="."),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "bitsandbytes>=0.43.0",
        "cryptography>=41.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "eval": [
            "datasets>=2.18.0",
            "lm-eval>=0.4.0",
        ],
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nes=src.cli:main",
        ],
    },
)