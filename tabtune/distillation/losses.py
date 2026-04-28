"""
Distillation Loss Functions 
================================
Standalone loss functions for knowledge distillation.
Usable independently of TabDistiller for custom pipelines.
"""

import numpy as np
import torch
import torch.nn.functional as F


def kd_classification_loss(student_logits, teacher_probs, hard_labels,
                           temperature=3.0, alpha=0.7,
                           sample_weights=None, sample_temperatures=None):
    """Combined KD + hard-label loss for classification.

    Supports per-sample temperature and confidence weighting.

    Parameters
    ----------
    student_logits : Tensor (B, C)
    teacher_probs : Tensor (B, C)
    hard_labels : Tensor (B,)
    temperature : float
        Base temperature (used if sample_temperatures is None).
    alpha : float
    sample_weights : Tensor (B,) or None
    sample_temperatures : Tensor (B,) or None
    """
    B = student_logits.shape[0]
    device = student_logits.device

    if sample_temperatures is not None:
        T = sample_temperatures.unsqueeze(-1)  # (B, 1)
    else:
        T = torch.full((B, 1), temperature, device=device)

    if sample_weights is None:
        w = torch.ones(B, device=device)
    else:
        w = sample_weights

    # KL divergence with per-sample T
    s_lp = F.log_softmax(student_logits / T, dim=-1)
    t_p = teacher_probs.clamp(min=1e-8)
    t_p = t_p / t_p.sum(dim=-1, keepdim=True)

    kl = F.kl_div(s_lp, t_p, reduction="none").sum(dim=-1)
    T_sq = T.squeeze(-1) ** 2
    kd = (kl * T_sq * w).sum() / w.sum()

    # Hard label CE
    ce = F.cross_entropy(student_logits, hard_labels, reduction="none")
    hard = (ce * w).sum() / w.sum()

    return alpha * kd + (1 - alpha) * hard


def kd_regression_loss(student_preds, teacher_preds, hard_targets,
                       alpha=0.7, sample_weights=None):
    """Combined KD + hard-label loss for regression."""
    if sample_weights is None:
        w = torch.ones_like(student_preds)
    else:
        w = sample_weights

    t_loss = ((student_preds - teacher_preds) ** 2 * w).sum() / w.sum()
    h_loss = ((student_preds - hard_targets) ** 2 * w).sum() / w.sum()
    return alpha * t_loss + (1 - alpha) * h_loss


def compute_adaptive_temperatures(soft_labels, T_base=3.0):
    """Per-sample temperature based on teacher entropy.

    Parameters
    ----------
    soft_labels : ndarray (N, C)
    T_base : float

    Returns
    -------
    ndarray (N,)
    """
    eps = 1e-10
    probs = np.clip(soft_labels, eps, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    max_entropy = np.log(soft_labels.shape[1]) + eps
    norm = entropy / max_entropy

    T_min = max(1.0, T_base * 0.5)
    T_max = T_base * 2.0
    return (T_min + (T_max - T_min) * norm).astype(np.float32)


def compute_confidence_weights(soft_labels):
    """Bell-shaped confidence weighting — peaks at moderate confidence.

    Returns weights with mean=1.
    """
    eps = 1e-10
    probs = np.clip(soft_labels, eps, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    max_entropy = np.log(soft_labels.shape[1]) + eps
    norm = entropy / max_entropy

    weights = 4.0 * norm * (1.0 - norm) + 0.5
    return (weights / weights.mean()).astype(np.float32)
