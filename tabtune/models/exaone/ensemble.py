"""Deterministic test-time ensembles for tabular inference."""

from dataclasses import dataclass, replace
import math

import torch


_FLOAT_DTYPES = (torch.float32, torch.float64)
_LOGIT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_LABEL_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float32,
    torch.float64,
)
_FLOOR = 1e-6


@dataclass(frozen=True, eq=False, kw_only=True)
class EnsemblePlan:
    members: int
    seed: int
    task: str = "classification"
    feature_permutations: torch.Tensor | None = None
    class_permutations: torch.Tensor | None = None
    class_count: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.members, bool) or not isinstance(self.members, int):
            raise TypeError("members must be an integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.members < 1:
            raise ValueError("members must be positive")
        if not 0 <= self.seed <= 2**63 - 1:
            raise ValueError("seed is outside the supported range")
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression")


def _percentile(sorted_values: torch.Tensor, fraction: float) -> torch.Tensor:
    position = (sorted_values.numel() - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _support_quartiles(observed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ordered = torch.sort(observed).values
    return _percentile(ordered, 0.25), _percentile(ordered, 0.75)


def _yeo_johnson(values: torch.Tensor, lam_index: int) -> torch.Tensor:
    lam = lam_index / 8.0
    nonnegative = values >= 0
    result = torch.empty_like(values)
    positive = values[nonnegative]
    negative = values[~nonnegative]
    if lam_index == 0:
        result[nonnegative] = torch.log1p(positive)
    else:
        result[nonnegative] = (torch.pow(positive + 1, lam) - 1) / lam
    if lam_index == 16:
        result[~nonnegative] = -torch.log1p(-negative)
    else:
        exponent = 2.0 - lam
        result[~nonnegative] = -(torch.pow(1 - negative, exponent) - 1) / exponent
    return result


def _yeo_johnson_candidates(values: torch.Tensor) -> torch.Tensor:
    """Evaluate all 17 fitted lambda candidates without host synchronization."""
    indices = torch.arange(17, dtype=values.dtype, device=values.device).unsqueeze(1)
    lam = indices / 8.0
    expanded = values.unsqueeze(0)
    nonnegative = expanded >= 0

    safe_lam = torch.where(indices == 0, torch.ones_like(lam), lam)
    positive_power = (torch.pow(expanded + 1, lam) - 1) / safe_lam
    positive = torch.where(indices == 0, torch.log1p(expanded), positive_power)

    exponent = 2.0 - lam
    safe_exponent = torch.where(indices == 16, torch.ones_like(exponent), exponent)
    negative_power = -(torch.pow(1 - expanded, exponent) - 1) / safe_exponent
    negative = torch.where(indices == 16, -torch.log1p(-expanded), negative_power)
    return torch.where(nonnegative, positive, negative)


def _yeo_johnson_selected(
    values: torch.Tensor, lam_index: torch.Tensor
) -> torch.Tensor:
    """Evaluate one device-selected lambda without converting it to Python."""
    index = lam_index.to(device=values.device)
    lam = index.to(dtype=values.dtype) / 8.0
    nonnegative = values >= 0
    safe_lam = torch.where(index == 0, torch.ones_like(lam), lam)
    positive_power = (torch.pow(values + 1, lam) - 1) / safe_lam
    positive = torch.where(index == 0, torch.log1p(values), positive_power)
    exponent = 2.0 - lam
    safe_exponent = torch.where(index == 16, torch.ones_like(exponent), exponent)
    negative_power = -(torch.pow(1 - values, exponent) - 1) / safe_exponent
    negative = torch.where(index == 16, -torch.log1p(-values), negative_power)
    return torch.where(nonnegative, positive, negative)


def _numeric_column(
    support: torch.Tensor,
    query: torch.Tensor,
    rule: int,
    generator: torch.Generator,
    *,
    regression: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rule == 0:
        return support.clone(), query.clone()

    support_mask = ~torch.isnan(support)
    query_mask = ~torch.isnan(query)
    observed = support[support_mask]
    if rule == 1:
        ordered = torch.sort(observed).values
        median = ordered[(ordered.numel() - 1) // 2]
        q1, q3 = _support_quartiles(observed)
        scale = torch.clamp(q3 - q1, min=_FLOOR)
        if regression:
            return (support - median) / scale, (query - median) / scale
        # Classification retains its independently validated nonlinear rule.
        return (
            torch.asinh((support - median) / scale),
            torch.asinh((query - median) / scale),
        )

    if observed.numel() <= 1:
        if rule == 2:
            return support.clone(), query.clone()
        support_result = torch.zeros_like(support)
        query_result = torch.zeros_like(query)
        support_result[~support_mask] = torch.nan
        query_result[~query_mask] = torch.nan
        return support_result, query_result

    if rule == 2:
        candidates = _yeo_johnson_candidates(observed)
        candidate_means = candidates.mean(dim=1)
        candidate_stds = torch.clamp(
            candidates.std(dim=1, correction=0), min=_FLOOR
        )
        candidate_scores = torch.abs(
            ((candidates - candidate_means[:, None]) ** 3).mean(dim=1)
            / candidate_stds**3
        )
        best_index = torch.argmin(candidate_scores)
        fitted = candidates[best_index]
        mean = fitted.mean()
        scale = torch.clamp(fitted.std(unbiased=False), min=_FLOOR)
        support_result = support.clone()
        query_result = query.clone()
        support_result[support_mask] = (fitted - mean) / scale
        query_result[query_mask] = (
            _yeo_johnson_selected(query[query_mask], best_index) - mean
        ) / scale
        return support_result, query_result

    q1, q3 = _support_quartiles(observed)
    noise_scale = 1e-4 * torch.clamp(q3 - q1, min=_FLOOR)
    reference_noise = torch.randn(
        observed.shape,
        dtype=support.dtype,
        device=support.device,
        generator=generator,
    )
    support_noise = torch.randn(
        observed.shape,
        dtype=support.dtype,
        device=support.device,
        generator=generator,
    )
    observed_query = query[query_mask]
    query_noise = torch.randn(
        observed_query.shape,
        dtype=query.dtype,
        device=query.device,
        generator=generator,
    )
    perturbed_support = observed + reference_noise * noise_scale
    reference = torch.sort(perturbed_support).values

    def gaussianize(
        values: torch.Tensor, mask: torch.Tensor, observed_noise: torch.Tensor
    ) -> torch.Tensor:
        output = values.clone()
        perturbed = values[mask] + observed_noise * noise_scale
        ranks = torch.searchsorted(reference, perturbed, right=False)
        probabilities = (ranks.to(values.dtype) + 0.5) / (reference.numel() + 1)
        probabilities = torch.clamp(probabilities, _FLOOR, 1.0 - _FLOOR)
        output[mask] = torch.erfinv(2 * probabilities - 1) * math.sqrt(2.0)
        return output

    return (
        gaussianize(support, support_mask, support_noise),
        gaussianize(query, query_mask, query_noise),
    )


def _categorical_column(
    support: torch.Tensor,
    query: torch.Tensor,
    generator: torch.Generator,
    *,
    forbidden_assignment: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    support_mask = ~torch.isnan(support)
    query_mask = ~torch.isnan(query)
    categories = torch.unique(support[support_mask], sorted=True)
    if categories.numel() == 0:
        return (
            torch.zeros_like(support),
            torch.zeros_like(query),
            torch.empty(0, dtype=torch.int64, device=support.device),
        )
    assignment = _random_permutation(
        categories.numel(),
        device=support.device,
        generator=generator,
        forbidden=forbidden_assignment,
    )
    support_positions = torch.searchsorted(categories, support[support_mask])
    support_codes = assignment[support_positions].to(support.dtype)
    mean = support_codes.mean()
    scale = torch.clamp(support_codes.std(unbiased=False), min=_FLOOR)
    support_result = support.clone()
    support_result[support_mask] = (support_codes - mean) / scale

    query_result = query.clone()
    query_positions = torch.searchsorted(categories, query[query_mask])
    bounded = torch.clamp(query_positions, max=categories.numel() - 1)
    known = (query_positions < categories.numel()) & (
        categories[bounded] == query[query_mask]
    )
    query_codes = torch.full(
        query_positions.shape,
        assignment[0],
        dtype=torch.int64,
        device=query.device,
    )
    query_codes[known] = assignment[bounded[known]]
    query_result[query_mask] = (query_codes.to(query.dtype) - mean) / scale
    return support_result, query_result, assignment


def _random_permutation(
    width: int,
    *,
    device: torch.device,
    generator: torch.Generator,
    forbidden: torch.Tensor | None = None,
) -> torch.Tensor:
    """Draw uniformly, excluding one paired-member permutation when possible."""
    candidate = torch.randperm(
        width, dtype=torch.int64, device=device, generator=generator
    )
    if forbidden is None or width < 2:
        return candidate
    if forbidden.shape != (width,) or forbidden.dtype != torch.int64:
        raise ValueError("forbidden permutation is invalid")
    if forbidden.device != device:
        raise ValueError("forbidden permutation is on the wrong device")
    for _ in range(31):
        if not torch.equal(candidate, forbidden):
            return candidate
        candidate = torch.randperm(
            width, dtype=torch.int64, device=device, generator=generator
        )
    if not torch.equal(candidate, forbidden):
        return candidate
    # The rejection probability after 32 draws is negligible even for binary
    # labels. Keep the path total and deterministic if a pathological random
    # stream nevertheless repeats the forbidden value every time.
    candidate = forbidden.clone()
    candidate[[0, 1]] = candidate[[1, 0]]
    return candidate


def _validate_build(
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    x_query: torch.Tensor,
    plan: EnsemblePlan,
) -> None:
    if not isinstance(plan, EnsemblePlan):
        raise TypeError("plan must be an EnsemblePlan")
    if not all(isinstance(value, torch.Tensor) for value in (x_support, y_support, x_query)):
        raise TypeError("all data inputs must be tensors")
    if x_support.dtype not in _FLOAT_DTYPES or x_query.dtype not in _FLOAT_DTYPES:
        raise TypeError("features must use float32 or float64")
    if y_support.dtype not in _LABEL_DTYPES:
        raise TypeError("labels must use a real numeric dtype")
    if x_support.ndim != 2 or x_query.ndim != 2 or y_support.ndim != 1:
        raise ValueError("unexpected tensor rank")
    if x_support.shape[0] != y_support.shape[0]:
        raise ValueError("support features and labels have different lengths")
    if x_support.shape[1] != x_query.shape[1]:
        raise ValueError("support and query feature counts differ")
    if x_support.dtype != x_query.dtype:
        raise TypeError("feature dtypes differ")
    if not (x_support.device == x_query.device == y_support.device):
        raise ValueError("all inputs must share a device")
    if any(size == 0 for size in (x_support.shape[0], x_query.shape[0], x_support.shape[1])):
        raise ValueError("empty data axes are unsupported")
    if any(
        value is not None
        for value in (
            plan.feature_permutations,
            plan.class_permutations,
            plan.class_count,
        )
    ):
        raise ValueError("plan is already fitted")
    labels_as_float = y_support.to(torch.float64)
    if not bool(torch.isfinite(labels_as_float).all()):
        raise ValueError("labels must be finite")
    if plan.task == "regression":
        if not y_support.is_floating_point():
            raise TypeError("regression targets must use a floating dtype")
        return
    if not bool((labels_as_float == torch.floor(labels_as_float)).all()):
        raise ValueError("labels must be integral")
    if bool((labels_as_float < 0).any()):
        raise ValueError("labels must be non-negative")
    distinct = torch.unique(y_support.to(torch.int64), sorted=True)
    expected = torch.arange(distinct.numel(), dtype=torch.int64, device=y_support.device)
    if distinct.numel() == 0 or not torch.equal(distinct, expected):
        raise ValueError("labels must form a contiguous zero-based range")


def build_ensemble_inputs(
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    x_query: torch.Tensor,
    plan: EnsemblePlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, EnsemblePlan]:
    """Construct deterministic ensemble members and their fitted metadata."""
    _validate_build(x_support, y_support, x_query, plan)
    device = x_support.device
    generator = torch.Generator(device=device)
    generator.manual_seed(plan.seed)
    category_columns = []
    numeric_columns = []
    for column in range(x_support.shape[1]):
        observed = x_support[:, column][~torch.isnan(x_support[:, column])]
        target = category_columns if torch.unique(observed).numel() < 10 else numeric_columns
        target.append(column)

    support_members = []
    query_members = []
    label_members = []
    feature_permutations = []
    class_permutations = []
    categorical_assignments: dict[tuple[int, int], torch.Tensor] = {}
    numeric_transform_cache: dict[
        tuple[int, int], tuple[torch.Tensor, torch.Tensor]
    ] = {}
    class_count = (
        int(torch.unique(y_support.to(torch.int64)).numel())
        if plan.task == "classification"
        else None
    )
    for member in range(plan.members):
        support_member = x_support.clone()
        query_member = x_query.clone()
        for column in category_columns:
            if member == 0:
                category_count = torch.unique(
                    x_support[:, column][~torch.isnan(x_support[:, column])]
                ).numel()
                categorical_assignments[(member, column)] = torch.arange(
                    category_count, dtype=torch.int64, device=device
                )
                continue
            paired = categorical_assignments.get((member - 4, column))
            transformed_support, transformed_query, assignment = _categorical_column(
                x_support[:, column],
                x_query[:, column],
                generator,
                forbidden_assignment=(
                    paired if plan.task == "classification" else None
                ),
            )
            support_member[:, column] = transformed_support
            query_member[:, column] = transformed_query
            categorical_assignments[(member, column)] = assignment
        rule = member % 4
        if rule != 0:
            for column in numeric_columns:
                cache_key = (rule, column)
                transformed = numeric_transform_cache.get(cache_key)
                if transformed is None:
                    transformed = _numeric_column(
                        x_support[:, column],
                        x_query[:, column],
                        rule,
                        generator,
                        regression=plan.task == "regression",
                    )
                    if rule in (1, 2):
                        numeric_transform_cache[cache_key] = transformed
                support_member[:, column], query_member[:, column] = transformed
        feature_order = torch.randperm(
            x_support.shape[1], dtype=torch.int64, device=device, generator=generator
        )
        if plan.task == "regression":
            encoded_labels = y_support.clone()
            class_order = None
        elif member == 0:
            assert class_count is not None
            class_order = torch.arange(class_count, dtype=torch.int64, device=device)
            encoded_labels = class_order[y_support.to(torch.int64)].to(y_support.dtype)
        else:
            assert class_count is not None
            class_order = _random_permutation(
                class_count,
                device=device,
                generator=generator,
                forbidden=(
                    class_permutations[member - 4] if member >= 4 else None
                ),
            )
            encoded_labels = class_order[y_support.to(torch.int64)].to(y_support.dtype)
        support_members.append(support_member[:, feature_order])
        query_members.append(query_member[:, feature_order])
        label_members.append(encoded_labels)
        feature_permutations.append(feature_order)
        if class_order is not None:
            class_permutations.append(class_order)

    fitted = replace(
        plan,
        feature_permutations=torch.stack(feature_permutations),
        class_permutations=(
            torch.stack(class_permutations)
            if plan.task == "classification"
            else None
        ),
        class_count=class_count,
    )
    return (
        torch.stack(support_members),
        torch.stack(label_members),
        torch.stack(query_members),
        fitted,
    )


def _is_permutation_rows(values: torch.Tensor, width: int) -> bool:
    expected = torch.arange(width, dtype=torch.int64, device=values.device).expand_as(values)
    return torch.equal(torch.sort(values, dim=1).values, expected)


def aggregate_probabilities(
    member_logits: torch.Tensor, fitted_plan: EnsemblePlan
) -> torch.Tensor:
    """Decode member class order, average logits, and normalize once."""
    if not isinstance(fitted_plan, EnsemblePlan):
        raise TypeError("fitted_plan must be an EnsemblePlan")
    if fitted_plan.task != "classification":
        raise ValueError("probability aggregation requires a classification plan")
    if not isinstance(member_logits, torch.Tensor):
        raise TypeError("member_logits must be a tensor")
    if member_logits.dtype not in _LOGIT_DTYPES:
        raise TypeError("logits must use a floating dtype")
    if member_logits.ndim != 3:
        raise ValueError("logits must have three axes")
    if member_logits.shape[0] != fitted_plan.members:
        raise ValueError("member count does not match plan")
    if member_logits.shape[1] == 0 or member_logits.shape[2] == 0:
        raise ValueError("empty logits axes are unsupported")
    features = fitted_plan.feature_permutations
    classes = fitted_plan.class_permutations
    count = fitted_plan.class_count
    if features is None or classes is None or count is None:
        raise RuntimeError("plan has not been fitted")
    if not isinstance(features, torch.Tensor) or not isinstance(classes, torch.Tensor):
        raise RuntimeError("fitted metadata is invalid")
    if features.device != member_logits.device or classes.device != member_logits.device:
        raise ValueError("logits and plan metadata must share a device")
    consistent = (
        isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
        and features.dtype == torch.int64
        and classes.dtype == torch.int64
        and features.ndim == 2
        and classes.ndim == 2
        and features.shape[0] == fitted_plan.members
        and classes.shape == (fitted_plan.members, count)
        and features.shape[1] > 0
    )
    if not consistent:
        raise RuntimeError("fitted metadata is inconsistent")
    if not _is_permutation_rows(features, features.shape[1]) or not _is_permutation_rows(
        classes, count
    ):
        raise RuntimeError("fitted metadata does not contain permutations")
    if member_logits.shape[2] < count:
        raise ValueError("too few class-score columns")
    restored = []
    for member in range(fitted_plan.members):
        leading = member_logits[member, :, :count].index_select(1, classes[member])
        restored.append(torch.cat((leading, member_logits[member, :, count:]), dim=1))
    decoded = torch.stack(restored)
    if decoded.dtype in (torch.float16, torch.bfloat16):
        decoded = decoded.to(torch.float32)
    return torch.softmax(decoded.mean(dim=0), dim=1)
