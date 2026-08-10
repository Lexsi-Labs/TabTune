"""Model-aware feature scoring and selection.

The public boundary is classification-focused and NumPy-based.  Internally the
attention pre-pass evaluates feature attention in bounded flat-batch chunks and
retains only the reduced statistics the configured scorers need, so the full
attention matrix is never materialized.
"""

from __future__ import annotations

from contextlib import contextmanager
import gc
import logging
import math
import types
from typing import Iterator

import numpy as np

from ..config import FeatureSelectionConfig, AttentionScorer
from ..model.memory import PeakMemoryMode, feedforward_token_counts


logger = logging.getLogger(__name__)
_EXACT_CAPTURE_MODULES = 12
_CAPTURE_CHUNK_TARGET_BYTES = 2 * 1024 * 1024 * 1024


def _public_controls(model, config) -> None:
    if not isinstance(config, FeatureSelectionConfig):
        raise TypeError("config must be FeatureSelectionConfig")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for model-aware selection") from error
    if not isinstance(model, torch.nn.Module) or not any(
        cls.__name__ == "ClassificationModel" for cls in type(model).__mro__
    ):
        raise TypeError("model must be a ClassificationModel")


def _validate_feature_matrix(features, training: bool = True):
    if not isinstance(features, np.ndarray):
        raise TypeError("features must be a NumPy array")
    if features.ndim != 2 or features.shape[1] == 0 or (
        training and features.shape[0] == 0
    ):
        raise ValueError("features must be a nonempty rank-two array")
    if not np.issubdtype(features.dtype, np.number):
        raise TypeError("features must be numeric")
    if np.isinf(features).any():
        raise ValueError("features contain infinity")
    return features


def _attention_feedforward_token_chunk(
    model, support, query, max_vram_bytes: int | None
) -> int:
    """Plan C for the feature-attention pre-pass's fixed Nq=1, E=1 call."""
    import torch

    members, support_rows, feature_count = support.shape
    query_rows = query.shape[1]
    token_counts = feedforward_token_counts(
        model.config,
        support_rows=support_rows,
        query_rows=query_rows,
        feature_count=feature_count,
        members=members,
        mode=PeakMemoryMode.JOINED,
    )
    maximum_useful_chunk = max(token_counts)
    parameter = next(model.parameters())
    if parameter.device.type != "cuda":
        return maximum_useful_chunk

    device = parameter.device
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated_bytes = int(torch.cuda.memory_allocated(device))
    reserved_bytes = int(torch.cuda.memory_reserved(device))
    target_vram_bytes = min(
        int(total_bytes), int(free_bytes) + reserved_bytes
    )
    if max_vram_bytes is not None:
        target_vram_bytes = min(target_vram_bytes, int(max_vram_bytes))

    def estimate(token_chunk: int):
        return model.estimated_peak(
            support_rows=support_rows,
            query_rows=query_rows,
            feature_count=feature_count,
            members=members,
            feedforward_token_chunk=token_chunk,
            mode=PeakMemoryMode.JOINED,
            max_vram_bytes=target_vram_bytes,
            baseline_allocated_bytes=allocated_bytes,
            baseline_reserved_bytes=reserved_bytes,
            free_bytes=int(free_bytes),
        )

    minimum = estimate(1)
    if not minimum.fits:
        raise RuntimeError(
            "feature-selection attention pre-pass cannot fit the minimum "
            "feedforward chunk; "
            f"support_rows={support_rows}, query_rows={query_rows}, "
            f"feature_count={feature_count}, members={members}, "
            f"estimated_reserved_bytes={minimum.projected_reserved_bytes}, "
            f"target_vram_bytes={minimum.target_vram_bytes}"
        )

    low, high = 1, maximum_useful_chunk
    while low < high:
        middle = (low + high + 1) // 2
        if estimate(middle).fits:
            low = middle
        else:
            high = middle - 1
    maximum_safe_chunk = low

    def feedforward_calls(token_chunk: int) -> int:
        return sum(
            (tokens + token_chunk - 1) // token_chunk
            for tokens in token_counts
        )

    minimum_calls = feedforward_calls(maximum_safe_chunk)
    low, high = 1, maximum_safe_chunk
    while low < high:
        middle = (low + high) // 2
        if feedforward_calls(middle) <= minimum_calls:
            high = middle
        else:
            low = middle + 1
    return low


def _inputs(features, targets):
    x = _validate_feature_matrix(features)
    if not isinstance(targets, np.ndarray):
        raise TypeError("targets must be a NumPy array")
    if targets.ndim != 1 or len(targets) != len(x):
        raise ValueError("targets must be a rank-one aligned array")
    missing = np.equal(targets, None)
    if np.issubdtype(targets.dtype, np.floating):
        missing = np.logical_or(missing, np.isnan(targets))
    if len(targets) == 0 or np.any(missing):
        raise ValueError("targets contain missing values")
    try:
        classes = np.unique(targets)
    except TypeError as error:
        raise ValueError("targets must contain orderable classes") from error
    if len(classes) < 2:
        raise ValueError("targets must contain at least two classes")
    z = x.copy() if np.issubdtype(x.dtype, np.floating) else x.astype(np.float64)
    means = np.nan_to_num(np.nanmean(z, axis=0))
    return x, np.where(np.isnan(z), means, z), targets


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    return np.sort(top).astype(np.int64)


def _probability_mean(scores: list[np.ndarray]) -> np.ndarray:
    """Min-max normalize each scorer's vector, then average across scorers."""
    arr = np.asarray(scores, dtype=np.float64)
    lo = arr.min(axis=1, keepdims=True)
    hi = arr.max(axis=1, keepdims=True)
    normalized = (arr - lo) / np.maximum(hi - lo, 1e-12)
    return normalized.mean(axis=0)


def _subsample_support(
    features: np.ndarray,
    targets: np.ndarray,
    cap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if features.shape[0] <= cap:
        return features, targets
    indices = rng.choice(features.shape[0], size=cap, replace=False)
    return features[indices], targets[indices]


def _accumulate(previous, new):
    import torch

    new = new.detach().float()
    return new if previous is None else torch.cat((previous, new), dim=0)


class _AttentionCapture:
    """Reduced feature-attention state: summary rows, target row, and value norms."""

    def __init__(self, module_count: int, item_summary_count: int) -> None:
        self.item_summary_count = item_summary_count
        self.per_layer = [{} for _ in range(module_count)]

    def consume_selected_rows(self, attention, values, slot: int) -> None:
        group_count = values.size(-2) - self.item_summary_count - 1
        feature_slice = slice(
            self.item_summary_count, self.item_summary_count + group_count
        )
        entry = self.per_layer[slot]
        summary_sum = attention[
            :, :, : self.item_summary_count, feature_slice
        ].sum(dim=-2)
        entry["sumi_to_feat"] = _accumulate(entry.get("sumi_to_feat"), summary_sum)
        target_row = attention[:, :, -1, feature_slice]
        entry["y_attn"] = _accumulate(entry.get("y_attn"), target_row)
        value_norm = values.float().norm(dim=-1)[:, :, feature_slice]
        entry["v_norm_feat"] = _accumulate(entry.get("v_norm_feat"), value_norm)


def _feature_attention_modules(model):
    modules = []
    for name, module in model.named_modules():
        if ".feature_attentions." not in name:
            continue
        suffix = name.rsplit(".", 1)[-1]
        if not suffix.isdigit():
            continue
        modules.append(module)
    if not modules:
        raise RuntimeError("model has no feature-attention modules")
    first_layer = next(iter(model.transformer.layers), None)
    item_summary_count = getattr(first_layer, "item_summary_count", None)
    if not isinstance(item_summary_count, int):
        raise RuntimeError("model item-summary metadata is unavailable")
    return modules, item_summary_count


def _captured_attention_forward(module, capture, slot: int, chunk_target_bytes: int):
    """Build a temporary forward that captures attention instead of using SDPA."""
    import torch
    from torch.nn import functional as functional

    def forward(
        self,
        query,
        context=None,
        *,
        cache_context=False,
        use_cached_context=False,
        reuse_first_context_head=False,
    ):
        if (
            context is not None
            or cache_context
            or use_cached_context
            or reuse_first_context_head
        ):
            raise RuntimeError("feature-attention capture requires uncached self-attention")
        query = self._validate_tensor("query", query)
        query_heads = self._as_heads(functional.linear(query, self.query_weight))
        key_heads = self._as_heads(functional.linear(query, self.key_weight))
        value_heads = self._as_heads(functional.linear(query, self.value_weight))
        prefix = query_heads.shape[:-3]
        head_count, token_count, head_width = query_heads.shape[-3:]
        query_heads = query_heads.reshape(-1, head_count, token_count, head_width)
        key_heads = key_heads.reshape(-1, head_count, token_count, head_width)
        value_heads = value_heads.reshape(-1, head_count, token_count, head_width)

        base_scale = self.score_scale
        if base_scale is None:
            base_scale = 1.0 / math.sqrt(head_width)
        # On GPU, Q/K are scaled by SSMax in reduced precision and QK^T and its
        # softmax are then formed explicitly in float32.  The CPU path keeps the
        # model's native reduced-precision order instead.  The order is part of
        # the captured scores, so neither side may adopt the other's.
        explicit_float32_softmax = query_heads.device.type == "cuda"
        if explicit_float32_softmax:
            ss_factor = self.head_scale.to(
                dtype=query_heads.dtype, device=query_heads.device
            ).reshape(1, head_count, 1, 1)
            scaled_query = query_heads * ss_factor
            scaled_key = key_heads * math.log(token_count)
            element_size = 4
        else:
            head_multiplier = self.head_scale * (
                math.log(token_count) * float(base_scale)
            )
            head_multiplier = head_multiplier.reshape(1, head_count, 1, 1)
            element_size = query_heads.element_size()
        bytes_per_row = (
            head_count
            * token_count
            * token_count
            * element_size
        )
        chunk_size = max(
            1,
            min(len(query_heads), chunk_target_bytes // max(bytes_per_row, 1)),
        )

        outputs = []
        if explicit_float32_softmax and slot >= _EXACT_CAPTURE_MODULES:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                attended = functional.scaled_dot_product_attention(
                    scaled_query,
                    scaled_key,
                    value_heads,
                    dropout_p=0.0,
                    scale=base_scale,
                )
            selected_query = torch.cat(
                (
                    scaled_query[:, :, : capture.item_summary_count],
                    scaled_query[:, :, -1:],
                ),
                dim=-2,
            )
            selected_scores = torch.matmul(
                selected_query.float(), scaled_key.float().transpose(-2, -1)
            ) * base_scale
            torch.softmax(selected_scores, dim=-1, out=selected_scores)
            selected_probabilities = selected_scores
            capture.consume_selected_rows(
                selected_probabilities, value_heads, slot
            )
            del selected_query, selected_scores, selected_probabilities
            outputs.append(attended)
        else:
            for start in range(0, len(query_heads), chunk_size):
                stop = start + chunk_size
                if explicit_float32_softmax:
                    scores = torch.matmul(
                        scaled_query[start:stop].float(),
                        scaled_key[start:stop].float().transpose(-2, -1),
                    ) * base_scale
                else:
                    scores = torch.matmul(
                        query_heads[start:stop],
                        key_heads[start:stop].transpose(-2, -1),
                    )
                    scores = scores * head_multiplier
                torch.softmax(scores, dim=-1, out=scores)
                probabilities = scores
                capture.consume_selected_rows(
                    probabilities, value_heads[start:stop], slot
                )
                outputs.append(
                    torch.matmul(
                        probabilities.to(value_heads.dtype), value_heads[start:stop]
                    )
                )
                del scores, probabilities

        attended = torch.cat(outputs, dim=0).reshape(
            *prefix, head_count, token_count, head_width
        )
        joined = attended.transpose(-3, -2).flatten(-2)
        return functional.linear(joined, self.output_weight)

    return types.MethodType(forward, module)


@contextmanager
def _capture_feature_attention(model) -> Iterator[_AttentionCapture]:
    modules, item_summary_count = _feature_attention_modules(model)
    capture = _AttentionCapture(len(modules), item_summary_count)
    originals = []
    try:
        for slot, module in enumerate(modules):
            had_own_forward = "forward" in module.__dict__
            original = module.__dict__.get("forward")
            originals.append((module, had_own_forward, original))
            module.forward = _captured_attention_forward(
                module, capture, slot, _CAPTURE_CHUNK_TARGET_BYTES
            )
        yield capture
    finally:
        for module, had_own_forward, original in originals:
            if had_own_forward:
                module.forward = original
            else:
                delattr(module, "forward")


def _compute_capture_score(
    capture: _AttentionCapture, spec: AttentionScorer
) -> np.ndarray:
    import torch

    per_layer = [entry.get(spec.aggregation) for entry in capture.per_layer]
    per_layer = [value for value in per_layer if value is not None]
    if not per_layer:
        raise RuntimeError(f"No captures found for aggregation={spec.aggregation}")
    value_norms = [entry.get("v_norm_feat") for entry in capture.per_layer]
    value_norms = [value for value in value_norms if value is not None]
    if len(value_norms) != len(per_layer):
        raise RuntimeError("ALTI requires a value norm for every attention capture")
    per_layer = [value * norm for value, norm in zip(per_layer, value_norms)]
    stacked = torch.stack(per_layer, dim=0).mean(dim=2).mean(dim=1)
    if spec.layer_reduce == "mean":
        score = stacked.mean(dim=0)
    else:
        score = stacked[-1]
    return score.detach().cpu().float().numpy().astype(np.float64, copy=False)


def _attention_scores(model, features, targets, config, max_vram_bytes) -> np.ndarray:
    import torch

    rng = np.random.default_rng(config.seed)
    support_x, support_y = _subsample_support(
        features, targets, config.pre_pass_row_limit, rng
    )
    was_training = model.training
    model.eval()
    try:
        parameter = next(model.parameters())
        device = parameter.device
        tensor_x = torch.as_tensor(
            support_x, device=device, dtype=torch.float32
        ).unsqueeze(0)
        _, encoded_y = np.unique(support_y, return_inverse=True)
        tensor_y = torch.as_tensor(
            encoded_y, device=device, dtype=torch.float32
        ).unsqueeze(0)
        query = torch.as_tensor(
            support_x[:1], device=device, dtype=torch.float32
        ).unsqueeze(0)
        try:
            feedforward_token_chunk = _attention_feedforward_token_chunk(
                model, tensor_x, query, max_vram_bytes
            )
            try:
                with _capture_feature_attention(model) as capture:
                    with torch.no_grad():
                        model(
                            tensor_x,
                            tensor_y,
                            query,
                            feedforward_token_chunk=feedforward_token_chunk,
                            trusted_internal_inputs=True,
                        )
            except (AttributeError, ValueError) as error:
                raise RuntimeError(
                    "required feature-attention metadata is unavailable"
                ) from error
        finally:
            del tensor_x, tensor_y, query
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        scorer_values = [
            _compute_capture_score(capture, spec) for spec in config.scorers
        ]
        combined = _probability_mean(scorer_values)
        if combined.shape != (features.shape[1],):
            raise RuntimeError(
                "captured score width does not match the input feature width"
            )
        return combined
    finally:
        model.train(was_training)


def select_features(
    *, model, features, targets, config, max_vram_bytes: int | None = None
) -> np.ndarray:
    """Return the sorted indices of the columns to keep, as int64.

    Yields the full column range unchanged when the table is already within
    ``config.target_feature_count``.
    """
    _public_controls(model, config)
    original, prepared, checked_targets = _inputs(features, targets)
    target = config.target_feature_count
    feature_count = original.shape[1]
    if feature_count <= target:
        return np.arange(feature_count, dtype=np.int64)

    import torch

    was_training = model.training
    model.eval()
    try:
        scores = _attention_scores(
            model, prepared, checked_targets, config, max_vram_bytes
        )
        return _topk_indices(scores, target)
    except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError) as error:
        logger.error(
            "[FS] attention selection failed: %s: %s", type(error).__name__, error
        )
        try:
            parameter = next(model.parameters())
            if parameter.device.type == "cuda":
                torch.cuda.empty_cache()
        except StopIteration:
            pass
        raise
    finally:
        model.train(was_training)
