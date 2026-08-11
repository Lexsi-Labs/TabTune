"""Public fitted classification estimator."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from sklearn.utils import ClassifierTags, InputTags, Tags, TargetTags
from torch import nn

from ._execution import _InferenceExecutor
from .checkpoint import load_classifier_checkpoint
from .config import FEATURE_SELECTION, InferenceManifest
from .ecoc import ECOCCodec
from .ensemble import EnsemblePlan, aggregate_probabilities, build_ensemble_inputs
from .feature_selection import select_features
from .model.heads import ClassificationModel
from .preprocessing import TabularPreprocessor
from .presets import released_checkpoint
from .weights import resolve_weights


_logger = logging.getLogger(__name__)

_COMPUTE_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class EXAONETabularClassifier:
    """A classification-only, manifest-configured fitted estimator."""

    def __init__(
        self,
        manifest: InferenceManifest,
        *,
        device: torch.device | str = "cpu",
        model: nn.Module | None = None,
        max_vram_bytes: int | None = None,
    ) -> None:
        if not isinstance(manifest, InferenceManifest):
            raise TypeError("manifest must be an InferenceManifest")
        if manifest.task != "classification":
            raise ValueError("manifest task must be classification")
        if model is not None and not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module or None")
        self.manifest = manifest
        self.device = torch.device(device)
        if max_vram_bytes is not None:
            if isinstance(max_vram_bytes, bool) or not isinstance(max_vram_bytes, int):
                raise TypeError("max_vram_bytes must be an integer or None")
            if max_vram_bytes <= 0:
                raise ValueError("max_vram_bytes must be positive")
            if self.device.type != "cuda":
                raise ValueError("max_vram_bytes requires a CUDA device")
        self.max_vram_bytes = max_vram_bytes
        self.feature_selection = FEATURE_SELECTION
        # feature_limit is the model's hard input width, so it wins where it is the
        # tighter of the two; selecting past it would only hand the preprocessor more
        # columns than it can keep, and f_classif would re-truncate them.
        self._selection_target = min(
            self.feature_selection.target_feature_count,
            manifest.runtime.feature_limit,
        )
        if model is None:
            dtype = _COMPUTE_DTYPES[manifest.runtime.compute_dtype]
            with torch.random.fork_rng(devices=[], enabled=True):
                torch.manual_seed(manifest.runtime.seed)
                model = ClassificationModel(
                    manifest.model, device=self.device, dtype=dtype
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
    ) -> "EXAONETabularClassifier":
        """Build a classifier from the released checkpoint (or your own weights).

        With no arguments this downloads the released classifier from the Hub,
        builds the model from its pinned manifest, and loads the weights, checking
        them against the manifest's SHA-256 pin if it has one. Pass ``weights=`` (a
        local file or a Hub repo id) to load your own checkpoint of the same
        architecture; the pin is then not enforced and a warning is emitted.
        ``ensemble_count``/``compute_dtype``/``seed`` override the matching runtime
        knobs; ``manifest=`` replaces the whole configuration for a fully custom
        checkpoint.

        Tables wider than the feature limit are narrowed by attention feature
        selection during ``fit``; see :attr:`selected_feature_indices_`.
        """
        checkpoint = released_checkpoint("classification")
        base_manifest = checkpoint.manifest if manifest is None else manifest
        if not isinstance(base_manifest, InferenceManifest):
            raise TypeError("manifest must be an InferenceManifest or None")
        if base_manifest.task != "classification":
            raise ValueError("manifest task must be classification")
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
        load_classifier_checkpoint(
            source.path,
            estimator.model,
            resolved_manifest,
            verify_checksum=source.is_default,
        )
        return estimator

    def _executor(self) -> _InferenceExecutor:
        return _InferenceExecutor(
            self.manifest,
            device=self.device,
            max_vram_bytes=self.max_vram_bytes,
        )

    def _empty_cuda_cache(self) -> None:
        self._executor().empty_cuda_cache()

    def _cuda_budget(self) -> tuple[int, int, int, int]:
        return self._executor().cuda_budget()

    _safe_query_chunk = staticmethod(_InferenceExecutor.safe_query_chunk)

    def _plan_cuda_execution(
        self, support_batch, query_batch, compute_dtype, estimate,
        hard_estimate=None, **chunked_estimates,
    ):
        return self._executor().plan_cuda_execution(
            self.model, support_batch, query_batch, compute_dtype, estimate,
            hard_estimate, **chunked_estimates,
        )

    @property
    def is_fitted(self) -> bool:
        return self._fitted_state is not None

    def __sklearn_is_fitted__(self) -> bool:
        """Answer scikit-learn's fitted check.

        The fitted state lives behind a private attribute and public properties,
        so sklearn's default scan for trailing-underscore *instance* attributes
        finds none and reads a fitted estimator as unfitted -- a FutureWarning in
        scikit-learn 1.7 that becomes an error in 1.8.
        """
        return self.is_fitted

    def __sklearn_tags__(self) -> Tags:
        """Describe this estimator to scikit-learn's tag machinery.

        Estimators that do not inherit ``BaseEstimator`` also have to answer this
        one: sklearn 1.7 warns and substitutes defaults, 1.8 raises. The declared
        surface is what ``fit`` accepts -- a dense 2-D float table that may carry
        NaNs, and a required 1-D target.
        """
        return Tags(
            estimator_type="classifier",
            target_tags=TargetTags(required=True),
            classifier_tags=ClassifierTags(),
            input_tags=InputTags(allow_nan=True),
        )

    def _state(self) -> dict[str, Any]:
        if self._fitted_state is None:
            raise RuntimeError("classifier is not fitted")
        return self._fitted_state

    @property
    def classes_(self) -> np.ndarray:
        return self._state()["classes"].copy()

    @property
    def n_classes_(self) -> int:
        return int(self._state()["classes"].size)

    @property
    def n_features_in_(self) -> int:
        return int(self._state()["n_features"])

    @property
    def selected_feature_indices_(self) -> np.ndarray | None:
        """Columns kept by feature selection, or None when the table was narrow."""
        columns = self._state()["selected_columns"]
        return None if columns is None else columns.copy()

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "EXAONETabularClassifier":
        if not isinstance(features, np.ndarray):
            raise TypeError("features must be a NumPy array")
        if not isinstance(targets, np.ndarray):
            raise TypeError("targets must be a NumPy array")
        if features.ndim != 2:
            raise ValueError("features must have rank two")
        if targets.ndim != 1:
            raise ValueError("targets must have rank one")
        if features.shape[0] == 0:
            raise ValueError("training data must be nonempty")
        if targets.shape[0] != features.shape[0]:
            raise ValueError("features and targets must have equal row counts")
        if self._contains_missing(targets):
            raise ValueError("targets must not contain missing values")

        # All work is local until the final assignment, preserving a previous
        # successful fit if validation or preprocessing fails.
        try:
            classes, ordinal = np.unique(targets, return_inverse=True)
        except (TypeError, ValueError) as exc:
            raise TypeError("target classes must be mutually orderable") from exc
        if classes.size < 2:
            raise ValueError("at least two target classes are required")
        ordinal = np.asarray(ordinal, dtype=np.int64)

        # Selection runs on the raw table, before preprocessing, and hands the
        # preprocessor its columns; narrow tables skip the pre-pass entirely.
        selected_columns = None
        if features.shape[1] > self._selection_target:
            selected_columns = select_features(
                model=self.model,
                features=features,
                targets=ordinal,
                config=replace(
                    self.feature_selection,
                    target_feature_count=self._selection_target,
                ),
                max_vram_bytes=self.max_vram_bytes,
            )
        preprocessor = TabularPreprocessor(
            self.manifest.runtime.feature_limit,
            gaussianize=self.manifest.preprocessing.use_quantile_map,
            feature_count_rescale=self.manifest.preprocessing.rescale_for_column_count,
            fixed_columns=selected_columns,
        )
        prepared = preprocessor.fit_transform(features, ordinal)
        values = np.array(prepared.values, copy=True)

        row_limit = self.manifest.runtime.support_row_limit
        if values.shape[0] > row_limit:
            indices = np.random.default_rng(self.manifest.runtime.seed).choice(
                values.shape[0], size=row_limit, replace=False
            )
            values = np.array(values[indices], copy=True)
            ordinal = np.array(ordinal[indices], dtype=np.int64, copy=True)
        else:
            ordinal = np.array(ordinal, dtype=np.int64, copy=True)

        codec = None
        if classes.size > self.manifest.model.class_capacity:
            codec = ECOCCodec(
                int(classes.size),
                self.manifest.model.class_capacity,
                redundancy=4,
                strategy="rest",
                aggregation="log_likelihood",
                retries=50,
                seed=self.manifest.runtime.seed,
            )
        new_state = {
            "preprocessor": preprocessor,
            "classes": np.array(classes, copy=True),
            "n_features": features.shape[1],
            "selected_columns": selected_columns,
            "support_x": values,
            "support_y": ordinal,
            "codec": codec,
        }
        self._fitted_state = new_state
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        state = self._state()
        if not isinstance(features, np.ndarray):
            raise TypeError("features must be a NumPy array")
        if features.ndim != 2:
            raise ValueError("features must have rank two")
        if features.shape[1] != state["n_features"]:
            raise ValueError("query feature width differs from fitted width")
        if features.shape[0] == 0:
            # Validate the feature domain through the fitted boundary while
            # deliberately avoiding model execution.
            state["preprocessor"].transform(features)
            return np.empty((0, state["classes"].size), dtype=np.float64)

        query = np.array(state["preprocessor"].transform(features).values, copy=True)
        self.model.eval()
        with torch.inference_mode():
            if state["codec"] is None:
                probabilities = self._infer_problem(
                    state["support_x"], state["support_y"], query,
                    int(state["classes"].size),
                )
            else:
                encoded = state["codec"].encode(state["support_y"])
                rows = [
                    self._infer_ecoc_row(
                        state["support_x"], encoded[index], query
                    )
                    for index in range(encoded.shape[0])
                ]
                probabilities = state["codec"].decode(np.stack(rows, axis=0))
        return self._finalize_probabilities(probabilities, features.shape[0])

    def predict(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)
        classes = self._state()["classes"]
        if probabilities.shape[0] == 0:
            return np.empty((0,), dtype=classes.dtype)
        return np.array(classes[np.argmax(probabilities, axis=1)], copy=True)

    def _infer_problem(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        query_x: np.ndarray,
        class_count: int,
    ) -> np.ndarray:
        represented, remapped_targets = np.unique(support_y, return_inverse=True)
        represented = np.asarray(represented, dtype=np.int64)
        contiguous = (
            represented.size == class_count
            and np.array_equal(represented, np.arange(class_count, dtype=np.int64))
        )
        if not contiguous:
            if represented.size == 0:
                raise RuntimeError("support labels are empty")
            if np.any(represented < 0) or np.any(represented >= class_count):
                raise RuntimeError("support labels are outside the class range")
            if represented.size == 1:
                probabilities = np.zeros(
                    (query_x.shape[0], class_count), dtype=np.float64
                )
                probabilities[:, int(represented[0])] = 1.0
                return probabilities

            local = self._infer_problem(
                support_x,
                np.asarray(remapped_targets, dtype=np.int64),
                query_x,
                int(represented.size),
            )
            probabilities = np.zeros((query_x.shape[0], class_count), dtype=np.float64)
            probabilities[:, represented] = np.asarray(local, dtype=np.float64)
            return probabilities

        x_support = torch.as_tensor(support_x, dtype=torch.float32, device=self.device)
        y_support = torch.as_tensor(support_y, dtype=torch.float32, device=self.device)
        x_query = torch.as_tensor(query_x, dtype=torch.float32, device=self.device)
        plan = EnsemblePlan(
            members=self.manifest.runtime.ensemble_count,
            seed=self.manifest.runtime.seed,
        )
        ens_xs, ens_ys, ens_xq, fitted_plan = build_ensemble_inputs(
            x_support, y_support, x_query, plan
        )
        logits = self._forward_chunked(ens_xs, ens_ys, ens_xq)
        expected = (
            self.manifest.runtime.ensemble_count,
            query_x.shape[0],
            self.manifest.model.class_capacity,
        )
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected:
            raise RuntimeError("model returned logits with an invalid shape")
        selected_logits = logits[..., :class_count]
        result = aggregate_probabilities(selected_logits, fitted_plan)
        if not isinstance(result, torch.Tensor):
            raise RuntimeError("probability aggregation returned an invalid result")
        result = result.detach().to(device="cpu")
        if result.dtype not in (torch.float32, torch.float64):
            result = result.to(dtype=torch.float32)
        return result.numpy()

    def _forward_chunked(
        self,
        support_batch: torch.Tensor,
        label_batch: torch.Tensor,
        query_batch: torch.Tensor,
    ) -> torch.Tensor:
        return self._executor().forward(
            self.model, support_batch, label_batch, query_batch
        )

    def _infer_ecoc_row(
        self,
        support_x: np.ndarray,
        encoded_targets: np.ndarray,
        query_x: np.ndarray,
    ) -> np.ndarray:
        """Infer one code row while preserving the codec's symbol columns.

        Support sampling can remove every example carrying a particular code
        symbol.  The ensemble input boundary accepts only a contiguous label
        range, so the symbols still represented in the retained support are
        mapped locally to that range.  Their probabilities are then restored
        to the original codec columns; absent symbols receive zero mass.
        """
        symbols, remapped_targets = np.unique(
            encoded_targets, return_inverse=True
        )
        represented = self._infer_problem(
            support_x,
            np.asarray(remapped_targets, dtype=np.int64),
            query_x,
            int(symbols.size),
        )
        capacity = self.manifest.model.class_capacity
        symbol_probabilities = np.zeros(
            (query_x.shape[0], capacity), dtype=np.float64
        )
        symbol_probabilities[:, np.asarray(symbols, dtype=np.int64)] = np.asarray(
            represented, dtype=np.float64
        )
        totals = symbol_probabilities.sum(axis=1, keepdims=True)
        if (
            not np.isfinite(symbol_probabilities).all()
            or np.any(symbol_probabilities < 0)
            or not np.isfinite(totals).all()
            or np.any(totals <= 0)
        ):
            raise RuntimeError("ECOC symbol probabilities are invalid")
        needs_normalization = np.abs(totals[:, 0] - 1.0) > 1e-7
        symbol_probabilities[needs_normalization] /= totals[needs_normalization]
        return symbol_probabilities

    def _finalize_probabilities(self, values: object, row_count: int) -> np.ndarray:
        probabilities = np.array(values, dtype=np.float64, order="C", copy=True)
        expected = (row_count, self.n_classes_)
        if probabilities.shape != expected:
            raise RuntimeError("model probabilities have an invalid shape")
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise RuntimeError("model probabilities must be finite and non-negative")
        totals = probabilities.sum(axis=1, keepdims=True)
        if np.any(totals <= 0) or not np.isfinite(totals).all():
            raise RuntimeError("model probability rows must have positive sums")
        probabilities /= totals
        return np.ascontiguousarray(probabilities, dtype=np.float64)

    @staticmethod
    def _contains_missing(values: np.ndarray) -> bool:
        if values.dtype.kind in "fc":
            return bool(np.isnan(values).any())
        if values.dtype.kind in "mM":
            return bool(np.isnat(values).any())
        if values.dtype.kind != "O":
            return False
        for value in values:
            if value is None:
                return True
            try:
                unequal = value != value
                if isinstance(unequal, (bool, np.bool_)) and unequal:
                    return True
            except Exception:
                continue
        return False
