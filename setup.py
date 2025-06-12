#!/usr/bin/env python3
"""
Setup script for unified_gut_polymerization package.
"""

from setuptools import setup, find_packages

setup(
    name="unified_gut_polymerization",
    version="0.1.0",
    description="Framework for polymer quantization of grand unified theories",
    author="Unified LQG Collaboration",
    author_email="contact@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.6.0",
        "matplotlib>=3.3.0",
        "sympy>=1.8.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
)
