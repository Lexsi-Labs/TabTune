"""
Hyperparameter search space for iLTM.

This module exposes a description of the recommended search
space and a helper to draw random configurations from it.

High level API
--------------

- get_hyperparameter_search_space(available_checkpoints=None)

    Returns a dictionary that describes the distribution of every hyperparameter
    that iLTM recommends tuning. The returned structure is deliberately simple
    so it can be re-expressed with any hyperparameter optimization library or
    custom search procedure.

- sample_hyperparameters(rng, available_checkpoints=None)

    Draws a single random configuration from the same space using NumPy.
    This is intended for quick baselines, smoke tests, or simple schedulers
    that accept externally sampled configurations. For more advanced or
    adaptive search strategies you will usually want your tuning framework
    to sample configurations directly using `get_hyperparameter_search_space`.

Notes
-----

The selected checkpoint automatically determines several parameters via
`model_checkpoints.py`: `tree_embedding`, `tree_model`,
`concat_tree_with_orig_features`, and `preprocessing`. Tree embedding
parameters in this search space are only used when the selected checkpoint
enables `tree_embedding`.

Device is fixed to `"cuda:0"` because CPU execution is very slow and not
recommended for typical workloads.
"""

from typing import Dict, Any
import numpy as np


AVAILABLE_CHECKPOINTS = [
    "xgbrconcat",
    "cbrconcat",
    "r128bn",
    "rnobn",
    "xgb",
    "catb",
    "rtr",
    "rtrcb",
]


# A single hyperparameter specification and the full search space.
HyperparamSpec = Dict[str, Any]
SearchSpace = Dict[str, HyperparamSpec]


def _rand_log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """
    Sample from a log-uniform distribution on [low, high].

    Parameters
    ----------
    rng : np.random.Generator
        NumPy random number generator.
    low : float
        Lower bound (inclusive).
    high : float
        Upper bound (inclusive).

    Returns
    -------
    float
        A value sampled from a log-uniform distribution between low and high.
    """
    log_low = np.log(low)
    log_high = np.log(high)
    return float(np.exp(rng.uniform(log_low, log_high)))


def _sample_from_spec(rng: np.random.Generator, spec: HyperparamSpec) -> Any:
    """
    Sample a single value from a HyperparamSpec entry.

    The spec format is intentionally simple:

        {"type": "constant", "value": ...}
        {"type": "categorical", "choices": [...], "probs": [...]}   # probs optional
        {"type": "float_uniform", "low": float, "high": float}
        {"type": "log_uniform",   "low": float, "high": float}
    """
    kind = spec["type"]

    if kind == "constant":
        return spec["value"]

    if kind == "categorical":
        choices = spec["choices"]
        probs = spec.get("probs")
        value = rng.choice(choices, p=probs)
        # Ensure we return plain Python scalars rather than NumPy scalars
        if isinstance(value, np.generic):
            return value.item()
        return value

    if kind == "float_uniform":
        return float(rng.uniform(spec["low"], spec["high"]))

    if kind == "log_uniform":
        return _rand_log_uniform(rng, spec["low"], spec["high"])

    raise ValueError(f"Unknown hyperparameter type {kind!r}")


def get_hyperparameter_search_space(
    available_checkpoints: list[str] | None = None,
) -> SearchSpace:
    """
    Return the canonical hyperparameter search space for iLTM.

    Parameters
    ----------
    available_checkpoints : list[str] | None, optional
        List of checkpoint names to choose from. If None, uses all
        `AVAILABLE_CHECKPOINTS`.

    Returns
    -------
    SearchSpace
        A dictionary mapping hyperparameter names to small spec dictionaries.
        Each spec describes the recommended distribution for that parameter.

    The specification format is intentionally minimal so that it can be
    re-expressed in any hyperparameter optimization library (e.g., 
    Optuna, Hyperopt, etc.) or custom search procedure.
    """
    if available_checkpoints is None:
        available_checkpoints = AVAILABLE_CHECKPOINTS

    space: SearchSpace = {
        # Checkpoint selection: determines tree_embedding, tree_model, concat_tree_with_orig_features, and preprocessing via model_checkpoints.py.
        "checkpoint": {"type": "categorical", "choices": available_checkpoints},
        # Device: fixed to CUDA because CPU execution is very slow.
        "device": {"type": "constant", "value": "cuda:0"},
        "n_ensemble": {"type": "categorical", "choices": [4, 8, 12, 16, 32, 64]},
        "batch_size": {"type": "categorical", "choices": [2048, 4096]},
        # Finetuning switches that are almost always beneficial.
        "finetuning": {"type": "constant", "value": True},
        "finetuning_dropout": {"type": "categorical", "choices": [0.0, 0.15]},
        "finetuning_max_steps": {"type": "categorical", "choices": [2048, 4096]},
        "finetuning_batch_size": {"type": "categorical", "choices": [64, 128, 256, 512, 1024, 2048, 4096]},
        "finetuning_data": {"type": "constant", "value": "entire_dataset"},
        "finetuning_lr": {"type": "log_uniform", "low": 1e-4, "high": 3e-3},
        "gradient_clip_norm": {"type": "float_uniform", "low": 0.5, "high": 1.5},
        "finetuning_optimizer": {"type": "categorical", "choices": ["adamw", "lion"]},
        "tree_data_split": {"type": "categorical", "choices": ["dynamic", "all"]},
        "tree_for_each_predictor": {"type": "constant", "value": True},
        # Tree embedding parameters (used only when checkpoint enables tree_embedding).
        "tree_n_estimators": {"type": "categorical", "choices": [100, 125, 150, 200, 300]},
        "tree_lr": {"type": "log_uniform", "low": 1e-3, "high": 1.0},
        "tree_max_depth": {"type": "categorical", "choices": [4, 5, 6], "probs": [0.20, 0.65, 0.15]},
        "tree_min_samples_leaf": {"type": "categorical", "choices": [1, 2, 4, 8, 12, 16]},
        "tree_subsample": {"type": "float_uniform", "low": 0.5, "high": 1.0},
        "tree_feature_fraction": {"type": "float_uniform", "low": 0.6, "high": 1.0},
        "tree_gamma": {"type": "categorical", "choices": [0.0, 0.05, 0.1, 0.25, 0.5], "probs": [0.6, 0.1, 0.1, 0.1, 0.1]},
        "tree_l2_leaf_reg": {"type": "categorical", "choices": [0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0]},
        "tree_bagging_temperature": {"type": "float_uniform", "low": 0.1, "high": 1.0},
        # Retrieval related parameters.
        "do_retrieval": {"type": "categorical", "choices": [True, False], "probs": [0.65, 0.35]},
        "retrieval_alpha": {"type": "float_uniform", "low": 0.0, "high": 1.0},
        "retrieval_temperature": {"type": "float_uniform", "low": 1.0, "high": 2.5},
        "retrieval_distance": {"type": "categorical", "choices": ["cosine", "euclidean"]},
        "retrieval_alpha_finetuning": {"type": "constant", "value": False},
        "retrieval_temperature_finetuning": {"type": "constant", "value": False},
        # Misc preprocessing and scheduler settings.
        "clip_data_value": {"type": "constant", "value": 1_000_000},
        "rf_size": {"type": "constant", "value": 32_768},
        "pca_sampling": {"type": "constant", "value": "zeropad"},
        "scheduler_min_lr": {"type": "log_uniform", "low": 1e-7, "high": 3e-4},
        "clip_predictions": {"type": "categorical", "choices": [False, True]},
        "corr_select_k": {
            "type": "categorical",
            "choices": [0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 512, 1024, 2048, 4096],
            "probs": [0.20, 0.02, 0.02, 0.02, 0.03, 0.03, 0.05, 0.10, 0.15, 0.15, 0.08, 0.08, 0.03, 0.02, 0.02],
        },
    }

    return space


def sample_hyperparameters(
    rng: np.random.Generator,
    available_checkpoints: list[str] | None = None,
) -> Dict[str, Any]:
    """
    Sample a single random hyperparameter configuration from the recommended space.

    This is a convenience wrapper around `get_hyperparameter_search_space` for
    quick experiments, simple random search, and schedulers that accept external
    configurations.

    For tuning methods that adapt sampling based on previous evaluations you
    should instead call `get_hyperparameter_search_space` and translate the
    returned specs into your own search space representation.
    """
    space = get_hyperparameter_search_space(available_checkpoints)
    cfg: Dict[str, Any] = {}

    for name, spec in space.items():
        cfg[name] = _sample_from_spec(rng, spec)

    return cfg


__all__ = [
    "AVAILABLE_CHECKPOINTS",
    "HyperparamSpec",
    "SearchSpace",
    "get_hyperparameter_search_space",
    "sample_hyperparameters",
]