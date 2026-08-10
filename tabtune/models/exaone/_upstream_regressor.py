"""Public fitted regression estimator."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from sklearn.feature_selection import f_regression
from sklearn.utils import InputTags, RegressorTags, Tags, TargetTags

from ._execution import _InferenceExecutor
from .checkpoint import load_regressor_checkpoint
from .config import InferenceManifest
from .ensemble import EnsemblePlan, build_ensemble_inputs
from .model.heads import RegressionModel
from .preprocessing import TabularPreprocessor
from .presets import released_checkpoint
from .weights import resolve_weights


_logger = logging.getLogger(__name__)

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class EXAONETabularRegressor:
    def __init__(self, manifest: InferenceManifest, *, device: torch.device | str = "cpu", model: nn.Module | None = None, max_vram_bytes: int | None = None) -> None:
        if not isinstance(manifest, InferenceManifest):
            raise TypeError("manifest must be an InferenceManifest")
        if manifest.task != "regression":
            raise ValueError("manifest task must be regression")
        if model is not None and not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module or None")
        if max_vram_bytes is not None and (isinstance(max_vram_bytes, bool) or not isinstance(max_vram_bytes, int)):
            raise TypeError("max_vram_bytes must be an integer or None")
        if max_vram_bytes is not None and max_vram_bytes <= 0:
            raise ValueError("max_vram_bytes must be positive")
        if max_vram_bytes is not None and torch.device(device).type != "cuda":
            raise ValueError("max_vram_bytes requires a CUDA device")
        self.manifest, self.device, self.max_vram_bytes = manifest, torch.device(device), max_vram_bytes
        assert manifest.regression is not None
        if model is None:
            with torch.random.fork_rng(devices=[], enabled=True):
                torch.manual_seed(manifest.runtime.seed)
                model = RegressionModel(
                    manifest.model,
                    manifest.regression,
                    device=self.device,
                    dtype=_DTYPES[manifest.runtime.compute_dtype],
                )
        self.model = model.to(self.device)
        self._fitted_state: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(
        cls,
        weights: str | None = None,
        *,
        device: torch.device | str = "cpu",
        revision: str | None = None,
        cache_dir: str | None = None,
        filename: str | None = None,
        manifest: InferenceManifest | None = None,
        ensemble_count: int | None = None,
        compute_dtype: str | None = None,
        seed: int | None = None,
        max_vram_bytes: int | None = None,
    ) -> "EXAONETabularRegressor":
        """Build a regressor from the released checkpoint (or your own weights).

        See :meth:`EXAONETabularClassifier.from_pretrained`; the only difference is
        that this loads the released regression checkpoint and its quantile head.
        """
        checkpoint = released_checkpoint("regression")
        base_manifest = checkpoint.manifest if manifest is None else manifest
        if not isinstance(base_manifest, InferenceManifest):
            raise TypeError("manifest must be an InferenceManifest or None")
        if base_manifest.task != "regression":
            raise ValueError("manifest task must be regression")
        resolved_manifest = base_manifest.with_overrides(
            ensemble_count=ensemble_count, compute_dtype=compute_dtype, seed=seed
        )
        source = resolve_weights(
            checkpoint, weights, filename=filename, revision=revision, cache_dir=cache_dir
        )
        estimator = cls(resolved_manifest, device=device, max_vram_bytes=max_vram_bytes)
        if not source.is_default:
            _logger.warning(
                "using user-supplied weights at %s; released-checkpoint integrity "
                "pin not enforced (SHA-256 not checked)",
                source.path,
            )
        load_regressor_checkpoint(
            source.path,
            estimator.model,
            resolved_manifest,
            verify_checksum=source.is_default,
        )
        return estimator

    @property
    def is_fitted(self) -> bool:
        return self._fitted_state is not None

    def __sklearn_is_fitted__(self) -> bool:
        """Answer scikit-learn's fitted check; see the classifier's counterpart."""
        return self.is_fitted

    def __sklearn_tags__(self) -> Tags:
        """Describe this estimator to scikit-learn; see the classifier's counterpart."""
        return Tags(
            estimator_type="regressor",
            target_tags=TargetTags(required=True),
            regressor_tags=RegressorTags(),
            input_tags=InputTags(allow_nan=True),
        )

    @property
    def n_features_in_(self) -> int:
        return int(self._state()["n_features"])

    def _state(self) -> dict[str, Any]:
        if self._fitted_state is None:
            raise RuntimeError("regressor is not fitted")
        return self._fitted_state

    def _executor(self) -> _InferenceExecutor:
        return _InferenceExecutor(
            self.manifest,
            device=self.device,
            max_vram_bytes=self.max_vram_bytes,
        )

    def _forward_chunked(
        self,
        support_batch: torch.Tensor,
        label_batch: torch.Tensor,
        query_batch: torch.Tensor,
    ) -> torch.Tensor:
        return self._executor().forward(
            self.model, support_batch, label_batch, query_batch
        )

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "EXAONETabularRegressor":
        if not isinstance(features, np.ndarray):
            raise TypeError("features must be a NumPy array")
        if not isinstance(targets, np.ndarray):
            raise TypeError("targets must be a NumPy array")
        if features.ndim != 2 or targets.ndim != 1:
            raise ValueError("features must be rank two and targets rank one")
        if features.shape[0] == 0 or features.shape[0] != targets.shape[0]:
            raise ValueError("training rows must be nonempty and aligned")
        if not np.issubdtype(targets.dtype, np.number) or np.issubdtype(targets.dtype, np.complexfloating):
            raise TypeError("targets must have a real numeric dtype")
        y = np.asarray(targets, dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError("targets must be finite")
        limit = self.manifest.runtime.feature_limit
        columns = np.arange(features.shape[1], dtype=np.int64)
        if columns.size > limit:
            numeric = np.asarray(features, dtype=np.float64).copy()
            numeric[np.isnan(numeric)] = np.take(np.nanmean(numeric, axis=0), np.where(np.isnan(numeric))[1])
            numeric = np.nan_to_num(numeric)
            scores, _ = f_regression(numeric, y, force_finite=True)
            columns = np.sort(np.argsort(np.nan_to_num(scores), kind="stable")[-limit:]).astype(np.int64)
        preprocessor = TabularPreprocessor(limit, gaussianize=self.manifest.preprocessing.use_quantile_map, feature_count_rescale=self.manifest.preprocessing.rescale_for_column_count, fixed_columns=columns)
        prepared = preprocessor.fit_transform(features, np.arange(features.shape[0]))
        row_limit = self.manifest.runtime.support_row_limit
        if y.size > row_limit:
            chosen = np.random.default_rng(self.manifest.runtime.seed).choice(y.size, row_limit, replace=False)
        else:
            chosen = np.arange(y.size)
        selected_y = torch.as_tensor(y[chosen], dtype=torch.float32)
        center_tensor = selected_y.mean()
        scale_tensor = (
            selected_y.std(correction=1)
            if selected_y.numel() > 1
            else selected_y.new_zeros(())
        )
        center = float(center_tensor)
        scale = float(scale_tensor)
        transformed = (selected_y - center_tensor) / (scale_tensor + 1e-8)
        self._fitted_state = {
            "preprocessor": preprocessor,
            "n_features": features.shape[1],
            "support_x": np.array(prepared.values[chosen], copy=True),
            "support_y": transformed.numpy().copy(),
            "center": center,
            "scale": scale,
        }
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        state = self._state()
        if not isinstance(features, np.ndarray):
            raise TypeError("features must be a NumPy array")
        if features.ndim != 2 or features.shape[1] != state["n_features"]:
            raise ValueError("query features have an invalid shape")
        query = state["preprocessor"].transform(features).values
        if query.shape[0] == 0:
            return np.empty(0, dtype=np.float64)

        xs = torch.as_tensor(
            state["support_x"], dtype=torch.float32, device=self.device
        )
        ys = torch.as_tensor(
            state["support_y"], dtype=torch.float32, device=self.device
        )
        xq = torch.as_tensor(query, dtype=torch.float32, device=self.device)
        members = self.manifest.runtime.ensemble_count
        plan = EnsemblePlan(
            members=members,
            seed=self.manifest.runtime.seed,
            task="regression",
        )
        batch_xs, batch_y, batch_xq, _fitted_plan = build_ensemble_inputs(
            xs, ys, xq, plan
        )
        self.model.eval()
        with torch.inference_mode():
            output = self._forward_chunked(batch_xs, batch_y, batch_xq)
        expected = (members, query.shape[0], self.manifest.output_width)
        if (
            not isinstance(output, torch.Tensor)
            or tuple(output.shape) != expected
            or not bool(torch.isfinite(output).all())
        ):
            raise RuntimeError("model returned invalid regression quantiles")

        assert self.manifest.regression is not None
        median_index = self.manifest.regression.quantile_count // 2
        # Slice before widening so the unused 998 quantiles are not copied to
        # float32 after the memory-planned model execution has completed.
        member_predictions = output[..., median_index].float()
        member_predictions = (
            member_predictions * state["scale"] + state["center"]
        )
        predictions = member_predictions.mean(dim=0).cpu().numpy()
        return np.array(predictions, dtype=np.float64, order="C", copy=True)


__all__ = ["EXAONETabularRegressor"]
