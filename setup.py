#!/usr/bin/env python3
"""
Setup script for TabTune package.
This is provided as an alternative to pyproject.toml for compatibility.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "An Advanced Library for Tabular Model Training and Adaptation"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            reqs = []
            for line in f:
                line = line.strip()
                # Skip comments and pip options like --extra-index-url
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                reqs.append(line)
            return reqs
    return []

setup(
    name="tabtune",
    version="0.1.0",
    author="Aditya Tanna, Pratinav Seth, Mohamed Bouadi, Utsav Avaiya, Vinay Kumar Sankarapu",
    author_email="contact@lexsi.ai",
    maintainer="Aditya Tanna",
    maintainer_email="contact@lexsi.ai",
    description="An Advanced Library for Tabular Model Training and Adaptation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Lexsi-Labs/TabTune",
    project_urls={
        "Homepage": "https://github.com/Lexsi-Labs/TabTune",
        "Repository": "https://github.com/Lexsi-Labs/TabTune",
        "Issues": "https://github.com/Lexsi-Labs/TabTune/issues",
    },
    packages=find_packages(include=["tabtune*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
        ],
    },
    keywords=["tabular-data", "machine-learning", "tabular-foundation-models", "pytorch", "tabular-deep-learning"],
    include_package_data=True,
    zip_safe=False,
)
