"""Student models for knowledge distillation."""
from .mlp_student import MLPStudent
from .gbdt_student import GBDTStudent

__all__ = ["MLPStudent", "GBDTStudent"]
