"""
MLP Student  —Knowledge Distillation with Robust Training
=============================================================
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class _ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Add."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.norm(x)
        out = self.act(self.fc1(out))
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        return residual + out


class _MLPNetwork(nn.Module):
    """MLP with input normalization, projection, residual blocks, and output head."""

    def __init__(self, n_features, n_output, embed_dim=256, n_blocks=3,
                 dropout=0.1):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(n_features)
        self.input_proj = nn.Linear(n_features, embed_dim)
        self.input_act = nn.GELU()
        self.input_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            _ResidualBlock(embed_dim, dropout) for _ in range(n_blocks)
        ])

        self.output_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_output)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_act(self.input_proj(x))
        x = self.input_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        return self.head(x)

    def get_embeddings(self, x):
        """Penultimate representations for feature distillation."""
        x = self.input_norm(x)
        x = self.input_act(self.input_proj(x))
        x = self.input_drop(x)
        for block in self.blocks:
            x = block(x)
        return self.output_norm(x)


def _init_weights(module):
    """Xavier uniform init -- more stable than Kaiming for KD soft targets."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class MLPStudent:
    """MLP student with knowledge distillation training.

    Parameters
    ----------
    task_type : str
    n_features : int
    n_classes : int or None
    embed_dim : int or None
        Auto-scaled if None. Capped at 8x n_features.
    n_blocks : int or None
        Auto-scaled if None.
    temperature : float, default=3.0
    alpha : float, default=0.7
    epochs : int, default=300
    learning_rate : float or None
        Auto-scaled if None based on dataset size.
    batch_size : int or None
        Auto-scaled if None based on dataset size.
    patience : int, default=30
    dropout : float, default=0.1
    weight_decay : float, default=1e-4
    warmup_epochs : int, default=10
    label_smoothing : float, default=0.05
    use_amp : bool, default=True
    use_swa : bool, default=True
    max_restarts : int, default=2
    device : str, default='cpu'
    random_state : int, default=42
    """

    def __init__(
        self,
        task_type="classification",
        n_features=10,
        n_classes=None,
        embed_dim=None,
        n_blocks=None,
        temperature=3.0,
        alpha=0.7,
        epochs=300,
        learning_rate=None,
        batch_size=None,
        patience=30,
        dropout=0.1,
        weight_decay=1e-4,
        warmup_epochs=10,
        label_smoothing=0.05,
        use_amp=True,
        use_swa=True,
        max_restarts=2,
        device="cpu",
        random_state=42,
        **kwargs,
    ):
        self.task_type = task_type
        self.n_features = n_features
        self.n_classes = n_classes
        self.temperature = temperature
        self.alpha = alpha
        self.epochs = epochs
        self.patience = patience
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.label_smoothing = label_smoothing
        self.use_amp = use_amp and device != "cpu"
        self.use_swa = use_swa
        self.max_restarts = max_restarts
        self.device = device
        self.random_state = random_state

        # === Auto-scale architecture (conservative for small feature counts) ===
        if embed_dim is None:
            raw = 64 if n_features <= 8 else 128 if n_features <= 20 else 256 if n_features <= 100 else 512
            self.embed_dim = min(raw, max(32, n_features * 8))
        else:
            self.embed_dim = embed_dim

        if n_blocks is None:
            self.n_blocks = 1 if n_features <= 8 else 2 if n_features <= 20 else 3 if n_features <= 100 else 4
        else:
            self.n_blocks = n_blocks

        # Auto-scale flags
        self._lr_auto = learning_rate is None
        self.learning_rate = learning_rate if learning_rate is not None else 5e-4

        self._bs_auto = batch_size is None
        self.batch_size = batch_size if batch_size is not None else 512

        self.model_: Optional[_MLPNetwork] = None
        self._label_map: Optional[dict] = None
        self._label_inv_map: Optional[dict] = None

    def fit(self, X, y, soft_labels, X_val=None, y_val=None,
            sample_weights=None, sample_temperatures=None):
        """Train with KD loss, supporting adaptive temperature and weighting."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_np = self._to_numpy(X)
        y_np = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        soft_np = np.asarray(soft_labels, dtype=np.float32)
        n_samples = len(X_np)

        # === Dataset-aware hyperparameter adjustment ===
        if self._lr_auto:
            if n_samples < 300:
                self.learning_rate = 1e-4
            elif n_samples < 1000:
                self.learning_rate = 3e-4
            else:
                self.learning_rate = 5e-4

        if self._bs_auto:
            # At least 4 batches per epoch for SGD noise
            self.batch_size = min(256, max(32, n_samples // 4))

        # Increase dropout for small datasets
        effective_dropout = self.dropout
        if n_samples < 500:
            effective_dropout = min(0.3, self.dropout + 0.1)

        logger.info(f"[MLPStudent] n={n_samples}, f={self.n_features}, "
                    f"embed={self.embed_dim}, blocks={self.n_blocks}, "
                    f"lr={self.learning_rate:.1e}, bs={self.batch_size}, "
                    f"dropout={effective_dropout:.2f}")

        # Label encoding
        if self.task_type == "classification":
            unique_labels = np.unique(y_np)
            self._label_map = {lab: i for i, lab in enumerate(unique_labels)}
            self._label_inv_map = {i: lab for lab, i in self._label_map.items()}
            y_encoded = np.array([self._label_map[lab] for lab in y_np], dtype=np.int64)
            n_output = soft_np.shape[1] if soft_np.ndim == 2 else len(unique_labels)
            self.n_classes = n_output
        else:
            y_encoded = y_np.astype(np.float32).ravel()
            n_output = 1

        # Train with collapse detection and restart
        for attempt in range(self.max_restarts + 1):
            seed = self.random_state + attempt * 1000
            torch.manual_seed(seed)
            np.random.seed(seed)

            self.model_ = _MLPNetwork(
                n_features=self.n_features, n_output=n_output,
                embed_dim=self.embed_dim, n_blocks=self.n_blocks,
                dropout=effective_dropout,
            ).to(self.device)
            self.model_.apply(_init_weights)

            self._train_loop(
                X_np, y_encoded, soft_np, X_val, y_val,
                sample_weights, sample_temperatures,
            )

            if self.task_type == "classification" and self._check_collapsed(X_np[:min(200, len(X_np))]):
                if attempt < self.max_restarts:
                    logger.warning(f"[MLPStudent] Collapse detected (attempt {attempt + 1}), "
                                   f"restarting with more regularization")
                    effective_dropout = min(0.4, effective_dropout + 0.05)
                    self.learning_rate *= 0.7
                else:
                    logger.warning(f"[MLPStudent] All {self.max_restarts + 1} attempts showed collapse.")
            else:
                logger.info(f"[MLPStudent] Training succeeded (attempt {attempt + 1})")
                break

        return self

    def _train_loop(self, X_np, y_encoded, soft_np, X_val, y_val,
                    sample_weights, sample_temperatures):
        """Core training loop."""
        X_t = torch.from_numpy(X_np).to(self.device)
        y_t = torch.from_numpy(y_encoded).to(self.device)
        soft_t = torch.from_numpy(soft_np).to(self.device)

        if sample_weights is not None:
            weights_t = torch.from_numpy(np.asarray(sample_weights, dtype=np.float32)).to(self.device)
        else:
            weights_t = torch.ones(len(X_np), device=self.device)

        if sample_temperatures is not None:
            temps_t = torch.from_numpy(np.asarray(sample_temperatures, dtype=np.float32)).to(self.device)
        else:
            temps_t = torch.full((len(X_np),), self.temperature, device=self.device)

        dataset = TensorDataset(X_t, y_t, soft_t, weights_t, temps_t)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            drop_last=len(X_np) > self.batch_size * 2,
        )

        # Validation
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_np = self._to_numpy(X_val)
            y_val_np = y_val.to_numpy() if hasattr(y_val, "to_numpy") else np.asarray(y_val)
            X_val_t = torch.from_numpy(X_val_np).to(self.device)
            if self.task_type == "classification":
                y_val_enc = np.array([self._label_map.get(l, 0) for l in y_val_np], dtype=np.int64)
            else:
                y_val_enc = y_val_np.astype(np.float32).ravel()
            y_val_t = torch.from_numpy(y_val_enc).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            progress = (epoch - self.warmup_epochs) / max(1, self.epochs - self.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = torch.amp.GradScaler(enabled=self.use_amp)

        # SWA
        swa_model = None
        swa_start = int(self.epochs * 0.8) if self.use_swa else self.epochs + 1
        if self.use_swa:
            try:
                swa_model = torch.optim.swa_utils.AveragedModel(self.model_)
            except Exception:
                swa_model = None
                swa_start = self.epochs + 1

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_b, y_b, soft_b, w_b, t_b in loader:
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits = self.model_(X_b)
                    loss = self._compute_loss(logits, y_b, soft_b, w_b, t_b)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            if swa_model is not None and epoch >= swa_start:
                swa_model.update_parameters(self.model_)

            avg_loss = epoch_loss / max(n_batches, 1)

            if has_val:
                self.model_.eval()
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
                    val_logits = self.model_(X_val_t)
                    if self.task_type == "classification":
                        val_loss = F.cross_entropy(val_logits, y_val_t).item()
                    else:
                        val_loss = F.mse_loss(val_logits.squeeze(-1), y_val_t).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        logger.info(f"[MLPStudent] Early stopping at epoch {epoch}")
                        break

            if epoch % 50 == 0 or epoch == 1:
                msg = f"[MLPStudent] Epoch {epoch}/{self.epochs}: loss={avg_loss:.5f}"
                if has_val:
                    msg += f", val_loss={val_loss:.5f}"
                msg += f", lr={optimizer.param_groups[0]['lr']:.2e}"
                logger.info(msg)

        # Load best or SWA model
        if swa_model is not None and epoch >= swa_start:
            try:
                torch.optim.swa_utils.update_bn(loader, swa_model, device=self.device)
                # Copy SWA weights back to self.model_
                self.model_.load_state_dict(swa_model.module.state_dict())
                logger.info("[MLPStudent] Using SWA averaged model")
            except Exception:
                if best_state is not None:
                    self.model_.load_state_dict(best_state)
        elif best_state is not None:
            self.model_.load_state_dict(best_state)

        self.model_.eval()
        logger.info("[MLPStudent] Training complete")

    def _check_collapsed(self, X_sample: np.ndarray) -> bool:
        """Return True if model predictions have collapsed."""
        if self.model_ is None or self.task_type != "classification":
            return False
        try:
            X_t = torch.from_numpy(self._to_numpy(X_sample)).to(self.device)
            self.model_.eval()
            with torch.no_grad():
                logits = self.model_(X_t)
                probs = F.softmax(logits, dim=-1)

            entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)
            max_entropy = np.log(probs.shape[1])
            mean_ratio = entropy.mean().item() / max_entropy
            pred_diversity = len(torch.unique(probs.argmax(dim=-1)))

            is_collapsed = (pred_diversity <= 1) or (mean_ratio > 0.98)
            if is_collapsed:
                logger.warning(f"[MLPStudent] Collapse: entropy_ratio={mean_ratio:.3f}, "
                               f"pred_classes={pred_diversity}/{probs.shape[1]}")
            return is_collapsed
        except Exception:
            return False

    def predict(self, X):
        logits = self._forward(X)
        if self.task_type == "classification":
            preds = logits.argmax(dim=-1).cpu().numpy()
            if self._label_inv_map:
                preds = np.array([self._label_inv_map.get(p, p) for p in preds])
            return preds
        return logits.squeeze(-1).cpu().numpy()

    def predict_proba(self, X):
        logits = self._forward(X)
        return F.softmax(logits, dim=-1).cpu().numpy()

    def _forward(self, X):
        X_np = self._to_numpy(X)
        X_t = torch.from_numpy(X_np).to(self.device)
        self.model_.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
            return self.model_(X_t)

    def _compute_loss(self, logits, y_hard, soft_targets, sample_weights, sample_temps):
        """KD loss with per-sample temperature, confidence weighting, and label smoothing."""
        alpha = self.alpha

        if self.task_type == "classification":
            T = sample_temps.unsqueeze(-1)

            student_log_probs = F.log_softmax(logits / T, dim=-1)

            if soft_targets.dim() == 2:
                teacher_probs = soft_targets.clamp(min=1e-8)
                teacher_probs = teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)
            else:
                teacher_probs = soft_targets

            min_c = min(student_log_probs.shape[-1], teacher_probs.shape[-1])
            s_lp = student_log_probs[..., :min_c]
            t_p = teacher_probs[..., :min_c]
            t_p = t_p / t_p.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            kl_per_sample = F.kl_div(s_lp, t_p, reduction="none").sum(dim=-1)
            T_sq = sample_temps ** 2
            kd_loss_per_sample = kl_per_sample * T_sq

            kd_loss = (kd_loss_per_sample * sample_weights).sum() / sample_weights.sum()

            hard_loss_per_sample = F.cross_entropy(
                logits, y_hard, reduction="none",
                label_smoothing=self.label_smoothing,
            )
            hard_loss = (hard_loss_per_sample * sample_weights).sum() / sample_weights.sum()

            return alpha * kd_loss + (1 - alpha) * hard_loss

        else:
            pred = logits.squeeze(-1)
            teacher_loss = F.mse_loss(pred, soft_targets, reduction="none")
            hard_loss = F.mse_loss(pred, y_hard, reduction="none")

            teacher_loss = (teacher_loss * sample_weights).sum() / sample_weights.sum()
            hard_loss = (hard_loss * sample_weights).sum() / sample_weights.sum()

            return alpha * teacher_loss + (1 - alpha) * hard_loss

    @staticmethod
    def _to_numpy(X):
        X_np = X.to_numpy().astype(np.float32) if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float32)
        return np.nan_to_num(X_np, nan=0.0, posinf=1e6, neginf=-1e6)