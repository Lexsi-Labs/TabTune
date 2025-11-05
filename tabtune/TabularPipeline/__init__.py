"""
TabularPipeline module for TabTune.

This module contains the main TabularPipeline class that provides a unified interface
for training and fine-tuning tabular models.
"""

from .pipeline import TabularPipeline

# Make TabularPipeline directly importable
__all__ = ["TabularPipeline"]
