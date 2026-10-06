"""Fixed per-stage precision shared by direct inference, caching, and restoration."""

import torch

from .memory import Stage


def stage_dtype(task: str, stage: Stage, device: torch.device) -> torch.dtype:
    """Keep CPU and regression column stages in FP32; use CUDA FP16 elsewhere."""
    if device.type != "cuda" or (task == "regression" and stage == Stage.COLUMN):
        return torch.float32
    return torch.float16


def stage_autocast(task: str, stage: Stage, device: torch.device):
    """Apply the stage policy without changing the stored pretrained parameters."""
    enabled = stage_dtype(task, stage, device) == torch.float16
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)
