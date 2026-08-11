"""Tensor encoders used by the classification model."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _positive_dimension(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _support_size(value: object, rows: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("support_size must be an integer")
    if value < 1 or value > rows:
        raise ValueError("support_size is outside the row range")
    return value


def _floating_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must use float32 or float64")
    return value


def _nonfinite_codes(values: torch.Tensor) -> torch.Tensor:
    codes = torch.zeros_like(values)
    codes = torch.where(torch.isnan(values), codes.new_tensor(-2.0), codes)
    codes = torch.where(torch.isposinf(values), codes.new_tensor(2.0), codes)
    return torch.where(torch.isneginf(values), codes.new_tensor(4.0), codes)


def _finite_support_mean(values: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(values)
    if not bool(finite.any()):
        return values.new_zeros(())
    return values[finite].mean()


class _PublicAffine(nn.Module):
    """A small affine module whose parameters intentionally live at top level."""

    def _make_parameters(self, output_size: int, input_size: int, bias: bool) -> None:
        if not isinstance(bias, bool):
            raise TypeError("bias must be boolean")
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            limit = 1 / math.sqrt(input_size)
            nn.init.uniform_(self.bias, -limit, limit)

    def _project(self, values: torch.Tensor) -> torch.Tensor:
        if values.device != self.weight.device:
            raise RuntimeError("input and parameters must be on the same device")
        projected = F.linear(values.to(dtype=self.weight.dtype), self.weight, self.bias)
        # Affine overflow is possible for finite but extreme parameter values.
        return torch.nan_to_num(projected)


class FeatureEncoder(_PublicAffine):
    """Encode grouped numeric coordinates into fixed-width embeddings."""

    def __init__(self, max_features: int, embedding_dim: int, *, bias: bool = False) -> None:
        super().__init__()
        self.max_features = _positive_dimension(max_features, "max_features")
        self.embedding_dim = _positive_dimension(embedding_dim, "embedding_dim")
        self._make_parameters(self.embedding_dim, 2 * self.max_features, bias)

    def forward(self, features: torch.Tensor, support_size: int) -> torch.Tensor:
        features = _floating_tensor(features, "features")
        if features.ndim != 3:
            raise ValueError("features must have three axes")
        rows, groups, width = features.shape
        if rows == 0 or groups == 0:
            raise ValueError("row and group axes must be nonempty")
        if width == 0 or width > self.max_features:
            raise ValueError("feature width is outside the configured range")
        support_size = _support_size(support_size, rows)

        if self.max_features == 1:
            return self._forward_single_coordinate(features, support_size)

        values_out = features.new_zeros((rows, groups, self.max_features))
        codes_out = features.new_zeros((rows, groups, self.max_features))

        for group in range(groups):
            source = features[:, group, :]
            if rows == 1:
                retained = torch.zeros(width, dtype=torch.bool, device=features.device)
            else:
                retained = (source[1:] != source[0]).any(dim=0)
            selected = source[:, retained]
            count = selected.shape[1]
            if count == 0:
                continue

            codes = _nonfinite_codes(selected)
            replaced_columns = []
            for coordinate in range(count):
                column = selected[:, coordinate]
                replacement = _finite_support_mean(column[:support_size])
                replaced_columns.append(torch.where(torch.isfinite(column), column, replacement))
            replaced = torch.stack(replaced_columns, dim=1)

            support = replaced[:support_size]
            means = support.mean(dim=0)
            if support_size == 1:
                deviations = torch.ones_like(means)
            else:
                deviations = support.std(dim=0, correction=1)
                deviations = torch.where(deviations == 0, torch.ones_like(deviations), deviations)
            normalized = ((replaced - means) / (deviations + 1e-16)).clamp(-100, 100)
            normalized = torch.nan_to_num(normalized, nan=0.0, posinf=100.0, neginf=-100.0)

            if rows == 1:
                active_count = 1
            else:
                active_count = max(int((normalized[1:] != normalized[0]).any(dim=0).sum()), 1)
            # The coordinate-width rescaling is defined by the model's scalar
            # (float32) path, then promoted when the feature tensor is float64.
            scale = torch.sqrt(
                torch.tensor(
                    self.max_features / active_count,
                    dtype=torch.float32,
                    device=features.device,
                )
            )
            normalized = normalized * scale
            values_out[:, group, :count] = normalized
            codes_out[:, group, :count] = codes

        channels = torch.cat((values_out, codes_out), dim=-1)
        return self._project(channels)

    def _forward_single_coordinate(
        self, features: torch.Tensor, support_size: int
    ) -> torch.Tensor:
        """Vectorized equivalent of the one-coordinate group path."""
        rows = features.shape[0]
        source = features[..., 0]
        if rows == 1:
            retained = torch.zeros(
                source.shape[1], dtype=torch.bool, device=features.device
            )
        else:
            retained = (source[1:] != source[0]).any(dim=0)

        finite = torch.isfinite(source)
        finite_support = finite[:support_size]
        support_values = source[:support_size]
        counts = finite_support.sum(dim=0)
        sums = torch.where(finite_support, support_values, 0).sum(dim=0)
        replacements = torch.where(
            counts > 0,
            sums / counts.clamp_min(1),
            torch.zeros_like(sums),
        )
        replaced = torch.where(finite, source, replacements.unsqueeze(0))

        support = replaced[:support_size]
        means = support.mean(dim=0)
        if support_size == 1:
            deviations = torch.ones_like(means)
        else:
            deviations = support.std(dim=0, correction=1)
            deviations = torch.where(
                deviations == 0, torch.ones_like(deviations), deviations
            )
        normalized = ((replaced - means) / (deviations + 1e-16)).clamp(-100, 100)
        normalized = torch.nan_to_num(
            normalized, nan=0.0, posinf=100.0, neginf=-100.0
        )

        # With one coordinate per group active_count is clamped to one and the
        # feature-count scale is exactly one.
        keep = retained.unsqueeze(0)
        values = torch.where(keep, normalized, 0).unsqueeze(-1)
        codes = torch.where(keep, _nonfinite_codes(source), 0).unsqueeze(-1)
        return self._project(torch.cat((values, codes), dim=-1))


class LabelEncoder(_PublicAffine):
    """Encode scalar labels by their support-set ordinal positions."""

    def __init__(self, embedding_dim: int, *, bias: bool = True) -> None:
        super().__init__()
        self.embedding_dim = _positive_dimension(embedding_dim, "embedding_dim")
        self._make_parameters(self.embedding_dim, 2, bias)

    def forward(self, labels: torch.Tensor, support_size: int) -> torch.Tensor:
        labels = _floating_tensor(labels, "labels")
        if labels.ndim != 3:
            raise ValueError("labels must have three axes")
        rows, batches, width = labels.shape
        if rows == 0 or batches == 0:
            raise ValueError("row and batch axes must be nonempty")
        if width != 1:
            raise ValueError("labels must contain one scalar per row and batch")
        support_size = _support_size(support_size, rows)

        scalar_labels = labels[..., 0]
        codes = _nonfinite_codes(scalar_labels)
        ordinals = labels.new_empty((rows, batches))
        for batch in range(batches):
            column = scalar_labels[:, batch]
            replacement = _finite_support_mean(column[:support_size])
            replaced = torch.where(torch.isfinite(column), column, replacement)
            support_values = torch.unique(replaced[:support_size], sorted=True)
            ordinals[:, batch] = (replaced[:, None] > support_values[None, :]).sum(dim=1)

        channels = torch.stack((ordinals, codes), dim=-1)
        return self._project(channels)


class RegressionLabelEncoder(_PublicAffine):
    """Encode normalized continuous targets and non-finite indicators."""

    def __init__(self, embedding_dim: int, *, bias: bool = True) -> None:
        super().__init__()
        self.embedding_dim = _positive_dimension(embedding_dim, "embedding_dim")
        self._make_parameters(self.embedding_dim, 2, bias)

    def forward(self, labels: torch.Tensor, support_size: int) -> torch.Tensor:
        labels = _floating_tensor(labels, "labels")
        if labels.ndim != 3:
            raise ValueError("labels must have three axes")
        rows, batches, width = labels.shape
        if rows == 0 or batches == 0:
            raise ValueError("row and batch axes must be nonempty")
        if width != 1:
            raise ValueError("labels must contain one scalar per row and batch")
        support_size = _support_size(support_size, rows)

        scalar_labels = labels[..., 0]
        codes = _nonfinite_codes(scalar_labels)
        replaced_columns = []
        for batch in range(batches):
            column = scalar_labels[:, batch]
            replacement = _finite_support_mean(column[:support_size])
            replaced_columns.append(
                torch.where(torch.isfinite(column), column, replacement)
            )
        replaced = torch.stack(replaced_columns, dim=1)
        return self._project(torch.stack((replaced, codes), dim=-1))
