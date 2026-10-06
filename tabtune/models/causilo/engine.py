"""Connect fixed pretrained weights to fitted data, caches, and ensemble outputs."""

from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
import torch
from sklearn.exceptions import NotFittedError

from . import checkpoints
from .data.dataset import PreparedDataset
from .data.ensemble import EnsembleMember
from .ecoc import ECOCCodec
from .execution.cached import cached_predictions
from .execution.direct import direct_predictions
from .execution.runner import ModelCache, ModelRunner


def resolve_device(request: str) -> torch.device:
    """Resolve auto placement once; explicit unavailable devices fail instead of falling back."""
    if request == "auto":
        request = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(request)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Causilo supports CPU or a single CUDA device")
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device {index} is unavailable")
        device = torch.device("cuda", index)
    return device


@dataclass
class FitState:
    """Successful fit only; caches follow ``dataset.members_by_normalization()``.

    Keep the stored tuple layout stable for fitted-state restoration. Consumers
    obtain member/cache pairs through ``cached_members`` rather than rebuilding
    the positional association themselves.
    """

    dataset: PreparedDataset
    caches: tuple[ModelCache, ...] | None
    codec: ECOCCodec | None = None
    code_caches: tuple[tuple[ModelCache, ...], ...] | None = None
    code_datasets: tuple[PreparedDataset, ...] | None = None

    def cached_members(
        self, caches: tuple[ModelCache, ...] | None = None, dataset: PreparedDataset | None = None
    ) -> tuple[tuple[EnsembleMember, ModelCache], ...]:
        """Pair every cache with its input permutation, rejecting incomplete state."""
        caches = self.caches if caches is None else caches
        if caches is None:
            raise ValueError("This fit does not contain K/V caches")
        dataset = self.dataset if dataset is None else dataset
        members = dataset.members_by_normalization()
        if len(members) != len(caches):
            raise ValueError("Cache count does not match the fitted ensemble")
        return tuple(zip(members, caches))


class Engine:
    """Own one task/device model and replace its training context on each fit."""

    def __init__(self, task: str, device: str) -> None:
        self.task = task
        self.device_request = device
        self.device = resolve_device(device)
        self.model = None
        self.state: FitState | None = None

    def fit(
        self,
        table,
        targets,
        *,
        n_estimators: int,
        use_kv_cache: bool,
        retain_preprocessing: bool,
        random_state: int,
    ) -> None:
        """Prepare context without optimizing weights; a failed fit stays unfitted."""
        self.state = None
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, Integral) or n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if isinstance(random_state, bool) or not isinstance(random_state, Integral) or random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")
        for name, value in (("use_kv_cache", use_kv_cache), ("retain_preprocessing", retain_preprocessing)):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a Boolean")
        if self.model is None:
            self.model = checkpoints.load_pretrained_model(self.task).to(self.device)
        prepared = PreparedDataset.prepare(
            table,
            targets,
            task=self.task,
            n_estimators=int(n_estimators),
            retain_preprocessing=retain_preprocessing,
            random_state=int(random_state),
            class_permutation_size=self.model.config.outputs,
        )
        codec = None
        code_datasets = None
        if (
            self.task == "classification"
            and len(prepared.target_encoder.classes_) > self.model.config.outputs
        ):
            codec = ECOCCodec(
                len(prepared.target_encoder.classes_),
                self.model.config.outputs,
                int(random_state),
            )
            encoded = codec.encode(prepared.targets)
            # Share preprocessing and retain all members for every target context.
            code_datasets = tuple(replace(prepared, targets=row) for row in encoded)
        caches = None
        code_caches = None
        if use_kv_cache:
            runner = ModelRunner(self.model)
            with torch.inference_mode():
                if codec is None:
                    caches = self._build_caches(runner, prepared)
                else:
                    code_caches = tuple(self._build_caches(runner, dataset) for dataset in code_datasets)
        # Publish only a complete context; partially built caches stay local.
        self.state = FitState(
            dataset=prepared,
            caches=caches,
            codec=codec,
            code_caches=code_caches,
            code_datasets=code_datasets,
        )

    def _build_caches(self, runner: ModelRunner, fitted: PreparedDataset) -> tuple[ModelCache, ...]:
        collected = []
        for member in fitted.members_by_normalization():
            features = fitted.training_table(member.normalization)[:, member.feature_order]
            collected.append(
                runner.build_cache(self._tensor(features), self._tensor(fitted.targets_for(member)))
            )
        return tuple(collected)

    def _tensor(self, value: np.ndarray) -> torch.Tensor:
        """Place one member on the device and prepend its singleton batch axis."""
        return torch.as_tensor(
            np.ascontiguousarray(value), device=self.device, dtype=torch.float32
        ).unsqueeze(0)

    def require_state(self) -> FitState:
        """Expose fitted state only after all preparation has succeeded."""
        if self.state is None:
            raise NotFittedError("Call fit successfully before prediction")
        return self.state

    def _member_predictions(self, transformed, state, reduce_output, *, fitted=None, caches=None):
        """Select direct or cached execution using already transformed queries."""
        fitted = state.dataset if fitted is None else fitted
        caches = state.caches if caches is None else caches
        if caches is None:
            return direct_predictions(self.model, fitted, transformed, self.device, reduce_output)
        return cached_predictions(
            self.model, fitted, transformed, state.cached_members(caches, fitted), self.device, reduce_output
        )

    def _classification_probabilities(self, transformed, state, *, fitted=None, caches=None) -> np.ndarray:
        logits = []
        for member, result in self._member_predictions(
            transformed, state, lambda value: value, fitted=fitted, caches=caches
        ):
            logits.append(result[:, member.class_order].numpy())
        combined = np.mean(logits, axis=0)
        probabilities = np.exp(combined - combined.max(axis=-1, keepdims=True))
        return probabilities / probabilities.sum(axis=-1, keepdims=True)

    def predict(self, table) -> np.ndarray:
        """Return class probabilities or regression point predictions."""
        state = self.require_state()
        transformed = state.dataset.query_tables(table)
        if self.task == "classification" and state.codec is not None:
            with torch.inference_mode():

                def row_probabilities():
                    for index, fitted in enumerate(state.code_datasets):
                        caches = None if state.code_caches is None else state.code_caches[index]
                        yield self._classification_probabilities(
                            transformed, state, fitted=fitted, caches=caches
                        )

                return state.codec.decode_rows(row_probabilities())
        if self.task == "classification":
            with torch.inference_mode():
                return self._classification_probabilities(transformed, state)
        outputs = []

        def reduce_output(result):
            # Sorting fixes the floating-point summation order for regression.
            return result.sort(dim=-1).values.mean(dim=-1)

        with torch.inference_mode():
            for _, result in self._member_predictions(transformed, state, reduce_output):
                point = result.numpy().astype(np.float64)
                outputs.append(state.dataset.target_encoder.inverse_transform(point[:, None])[:, 0])
        combined = np.mean(outputs, axis=0)
        return combined

    def predict_raw(self, table) -> np.ndarray:
        """Sort each member, restore target units, then average matching quantiles."""
        state = self.require_state()
        if self.task != "regression":
            raise ValueError("Quantile prediction requires a regression model")
        transformed = state.dataset.query_tables(table)
        outputs = []

        def reduce_output(result):
            return result.sort(dim=-1).values

        with torch.inference_mode():
            for _, result in self._member_predictions(transformed, state, reduce_output):
                values = result.numpy().astype(np.float64)
                outputs.append(
                    state.dataset.target_encoder.inverse_transform(values.reshape(-1, 1)).reshape(
                        values.shape
                    )
                )
        return np.mean(outputs, axis=0)
