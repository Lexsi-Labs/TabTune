"""The catalog of time series foundation models TabTune ships support for.

Adding a TSFM starts here with a :class:`TimeSeriesModelSpec` and an adapter,
and the pipeline picks it up without changes.
"""

from __future__ import annotations

from .catalog import MODEL_SPECS as _TABULAR_SPECS
from .spec import LicenseSpec
from .TimeSeriesSpec import FORECAST_TASKS, TimeSeriesModelSpec

__all__ = [
    "TS_MODEL_SPECS",
    "BASELINE_SPECS",
    "TABULAR_FORECASTER_SPECS",
    "NATIVE_QUANTILE_TABULAR_MODELS",
]

_EMBEDDING_TASKS = FORECAST_TASKS | {"embedding"}
_TRAINABLE = frozenset({"inference", "finetune", "peft"})

_CHRONOS_T5_CHECKPOINTS = tuple(
    f"amazon/chronos-t5-{size}" for size in ("tiny", "mini", "small", "base", "large")
)

_CHRONOS_BOLT_CHECKPOINTS = tuple(
    f"amazon/chronos-bolt-{size}" for size in ("tiny", "mini", "small", "base")
)

_TOTO2_CHECKPOINTS = tuple(
    f"Datadog/Toto-2.0-{size}" for size in ("4m", "22m", "313m", "1B", "2.5B")
)

def _tabular_license(name: str) -> LicenseSpec:
    return next(spec.license for spec in _TABULAR_SPECS if spec.name == name)


TS_MODEL_SPECS: tuple[TimeSeriesModelSpec, ...] = (

    # Chronos
    TimeSeriesModelSpec(
        name="Chronos",
        family="chronos",
        adapter="tabtune.models.TimeSeries.chronos:ChronosAdapter",
        aliases=("ChronosT5", "Chronos-v1", "Chronos-T5"),
        tasks=_EMBEDDING_TASKS,
        strategies=_TRAINABLE,
        default_checkpoint="amazon/chronos-t5-small",
        checkpoints=_CHRONOS_T5_CHECKPOINTS,
        max_context=512,
        max_horizon=64,
        native_missing=True,
        license=LicenseSpec(
            name="Apache-2.0",
            commercial_use_ok=True,
            url="https://huggingface.co/amazon/chronos-t5-small",
        ),
        paper="https://arxiv.org/abs/2403.07815",
        summary="T5 language model trained on scaled and quantised time series tokens.",
    ),

    # Chronos Bolt
    TimeSeriesModelSpec(
        name="ChronosBolt",
        family="chronos",
        adapter="tabtune.models.TimeSeries.chronos_bolt:ChronosBoltAdapter",
        aliases=("Chronos-Bolt", "Bolt"),
        tasks=_EMBEDDING_TASKS,
        strategies=_TRAINABLE,
        default_checkpoint="amazon/chronos-bolt-small",
        checkpoints=_CHRONOS_BOLT_CHECKPOINTS,
        max_context=2048,
        max_horizon=64,
        native_missing=True,
        license=LicenseSpec(
            name="Apache-2.0",
            commercial_use_ok=True,
            url="https://huggingface.co/amazon/chronos-bolt-small",
        ),
        paper="https://arxiv.org/abs/2403.07815",
        summary=(
            "Patch-based encoder-decoder predicting all quantiles in one pass: "
            "deterministic and much faster than Chronos v1."
        ),
    ),

    # Chronos 2
    TimeSeriesModelSpec(
        name="Chronos2",
        family="chronos",
        adapter="tabtune.models.TimeSeries.chronos2:Chronos2Adapter",
        aliases=("Chronos-2", "ChronosV2", "Chronos-v2"),
        tasks=_EMBEDDING_TASKS,
        strategies=_TRAINABLE,
        default_checkpoint="amazon/chronos-2",
        checkpoints=("amazon/chronos-2",),
        max_context=8192,
        max_horizon=1024,
        native_missing=True,
        supports_multivariate=True,
        supports_covariates=True,
        supports_categorical_covariates=True,
        license=LicenseSpec(
            name="Apache-2.0",
            commercial_use_ok=True,
            url="https://huggingface.co/amazon/chronos-2",
        ),
        paper="https://arxiv.org/abs/2510.15821",
        summary=(
            "Encoder-only group-attention model: multivariate targets, past and "
            "known-future covariates, in-context learning across related series."
        ),
    ),

    # TimesFM v3
    TimeSeriesModelSpec(
        name="TimesFM3",
        family="timesfm",
        adapter="tabtune.models.TimeSeries.timesfm3:TimesFM3Adapter",
        aliases=("TimesFM-3", "TimesFM3.0", "timesfm-3.0-pytorch"),
        tasks=_EMBEDDING_TASKS,
        strategies=_TRAINABLE,
        default_checkpoint="google/timesfm-3.0-pytorch",
        checkpoints=("google/timesfm-3.0-pytorch",),
        max_context=15360,
        max_horizon=None,
        native_missing=True,
        supports_multivariate=True,
        supports_covariates=True,
        supports_categorical_covariates=False,
        license=LicenseSpec(
            name="TimesFM Non-Commercial License v1.0",
            commercial_use_ok=False,
            url="https://huggingface.co/google/timesfm-3.0-pytorch",
            notes=(
                "Code and weights are licensed separately. The vendored code is "
                "Apache-2.0; the published weights are granted for "
                "Non-Commercial Purposes only and the license prohibits "
                "revenue-generating activity, production systems and end-user "
                "interactions. LicenseSpec describes the weights, hence "
                "commercial_use_ok=False."
            ),
        ),
        commercial_alternatives=("Chronos2", "ChronosBolt"),
        paper="https://arxiv.org/abs/2310.10688",
        summary=(
            "Single-pass patched transformer with variate attention: multivariate "
            "targets, past and known-future numeric covariates, 15360-step context."
        ),
    ),

    # Toto 2.0
    TimeSeriesModelSpec(
        name="Toto2",
        family="toto",
        adapter="tabtune.models.TimeSeries.toto2:Toto2Adapter",
        aliases=("Toto-2", "Toto2.0", "TotoV2"),
        tasks=_EMBEDDING_TASKS,
        strategies=_TRAINABLE,
        default_checkpoint="Datadog/Toto-2.0-22m",
        checkpoints=_TOTO2_CHECKPOINTS,
        max_context=4096,
        max_horizon=None,
        native_missing=True,
        supports_multivariate=True,
        supports_covariates=False,
        supports_categorical_covariates=False,
        dependency_extra="toto2",
        license=LicenseSpec(
            name="Apache-2.0",
            commercial_use_ok=True,
            url="https://huggingface.co/Datadog/Toto-2.0-22m",
        ),
        paper="https://arxiv.org/abs/2605.20119",
        summary=(
            "Observability-oriented decoder with alternating time/variate attention "
            "and u-microP scaling: multivariate targets, five sizes from 4m to 2.5B."
        ),
    ),
    # Toto 1.0
    TimeSeriesModelSpec(
        name="Toto1",
        family="toto",
        adapter="tabtune.models.TimeSeries.toto1:Toto1Adapter",
        aliases=("Toto-1", "Toto1.0", "TotoV1", "Toto-Open-Base-1.0"),
        tasks=_EMBEDDING_TASKS,
        strategies=_TRAINABLE,
        default_checkpoint="Datadog/Toto-Open-Base-1.0",
        checkpoints=("Datadog/Toto-Open-Base-1.0",),
        max_context=4096,
        max_horizon=None,
        native_missing=True,
        supports_multivariate=True,
        supports_covariates=True,
        supports_categorical_covariates=False,
        license=LicenseSpec(
            name="Apache-2.0",
            commercial_use_ok=True,
            url="https://huggingface.co/Datadog/Toto-Open-Base-1.0",
        ),
        paper="https://openreview.net/forum?id=1jDAYXfcS2",
        summary=(
            "Observability-oriented decoder with proportional factorised space-time "
            "attention: sample-based, so any quantile level and a mean forecast."
        ),
    ),

    # TiRex
    TimeSeriesModelSpec(
        name="TiRex",
        family="xlstm",
        adapter="tabtune.models.TimeSeries.tirex:TiRexAdapter",
        aliases=("TiRex-1", "TiRex-1.1"),
        tasks=_EMBEDDING_TASKS,
        strategies=_TRAINABLE,
        default_checkpoint="NX-AI/TiRex",
        checkpoints=("NX-AI/TiRex", "NX-AI/TiRex-1.1-gifteval"),
        max_context=2048,
        native_missing=True,
        license=LicenseSpec(
            name="NXAI Community License",
            commercial_use_ok=None,
            requires_attribution=True,
            url="https://github.com/NX-AI/tirex/blob/main/LICENSE",
            notes=(
                "Conditional: covers the code and the weights. Commercial use is allowed for "
                "licensees with consolidated annual revenue up to EUR 100M, with the notice "
                "'Built with technology from NXAI'; above that a separate NXAI license is needed."
            ),
        ),
        commercial_alternatives=("Chronos2", "ChronosBolt"),
        paper="https://arxiv.org/abs/2505.23719",
        summary="Small xLSTM forecaster for short and long horizons, with missing-value support.",
    ),

    # TiRex-2
    TimeSeriesModelSpec(
        name="TiRex2",
        family="xlstm",
        adapter="tabtune.models.TimeSeries.tirex2:TiRex2Adapter",
        aliases=("TiRex-2",),
        tasks=_EMBEDDING_TASKS,
        strategies=_TRAINABLE,
        default_checkpoint="NX-AI/TiRex-2",
        checkpoints=(
            "NX-AI/TiRex-2",
            "NX-AI/TiRex-2-fevbench",
            "NX-AI/TiRex-2-gifteval-pretrain",
            "NX-AI/TiRex-2-gifteval-zs",
        ),
        native_missing=True,
        supports_multivariate=True,
        supports_covariates=True,
        license=LicenseSpec(
            name="Apache-2.0 (code); weight license not verified",
            commercial_use_ok=None,
            url="https://github.com/NX-AI/tirex-2",
            notes=(
                "The code is Apache-2.0 and the README states the model is licensed under "
                "Apache-2.0; confirm the weight license on the model card before commercial use."
            ),
        ),
        commercial_alternatives=("Chronos2", "ChronosBolt"),
        paper="https://arxiv.org/abs/2607.01204",
        summary=(
            "Patch-based xLSTM forecaster with variate attention: multivariate targets and "
            "past and known-future numeric covariates."
        ),
    ),

    # Time-MoE
    TimeSeriesModelSpec(
        name="TimeMoE",
        family="time-moe",
        adapter="tabtune.models.TimeSeries.timemoe:TimeMoEAdapter",
        aliases=("Time-MoE",),
        tasks=_EMBEDDING_TASKS,
        strategies=_TRAINABLE,
        default_checkpoint="Maple728/TimeMoE-50M",
        checkpoints=("Maple728/TimeMoE-50M", "Maple728/TimeMoE-200M"),
        max_context=4096,
        native_missing=False,
        license=LicenseSpec(
            name="Apache-2.0 (code); weight license not verified",
            commercial_use_ok=None,
            url="https://github.com/Time-MoE/Time-MoE",
            notes=(
                "The repository is Apache-2.0; confirm the weight license on the model card "
                "before commercial use."
            ),
        ),
        commercial_alternatives=("Chronos2", "ChronosBolt"),
        paper="https://arxiv.org/abs/2409.16040",
        summary="Decoder-only mixture-of-experts point forecaster (univariate).",
    ),

    # TabPFN-TS
    TimeSeriesModelSpec(
        name="TabPFN-TS",
        family="tabular-icl",
        adapter="tabtune.models.TimeSeries.tabular:TabPFNTSAdapter",
        aliases=("TabPFNTS", "tabpfn-time-series"),
        default_checkpoint="tabpfn-ts-3.5",
        checkpoints=("tabpfn-ts-3.5", "tabpfn-ts-3", "tabpfn-ts-2"),
        max_context=32768,
        native_missing=True,
        supports_covariates=True,
        supports_categorical_covariates=True,
        license=LicenseSpec(
            name="tabpfn-3-5-license-v1.0 (weights, non-commercial); Apache-2.0 (code)",
            commercial_use_ok=False,
            url="https://github.com/PriorLabs/tabpfn-time-series",
            notes=(
                "The default and tabpfn-ts-3 checkpoints use non-commercial TabPFN-3.x weights. "
                "tabpfn-ts-2 uses TabPFN v2 weights under the Prior Labs License (Apache-2.0 "
                "with an attribution clause), which permits commercial use."
            ),
        ),
        checkpoint_licenses=(("tabpfn-ts-2", _tabular_license("TabPFN")),),
        commercial_alternatives=("Chronos2", "ChronosBolt"),
        paper="https://arxiv.org/abs/2501.02945",
        summary=(
            "Forecasting as per-series TabPFN regression on time features; uses known "
            "covariates only."
        ),
    ),
)

_BASELINE_LICENSE = LicenseSpec(name="MIT (TabTune code, no weights)", commercial_use_ok=True)


def _baseline(name: str, adapter: str, aliases: tuple[str, ...], summary: str) -> TimeSeriesModelSpec:
    return TimeSeriesModelSpec(
        name=name,
        family="baseline",
        adapter=f"tabtune.models.TimeSeries.baselines:{adapter}",
        aliases=aliases,
        native_missing=True,
        license=_BASELINE_LICENSE,
        paper="https://otexts.com/fpp3/prediction-intervals.html",
        summary=summary,
    )


BASELINE_SPECS: tuple[TimeSeriesModelSpec, ...] = (
    _baseline(
        "SeasonalNaive",
        "SeasonalNaiveAdapter",
        ("seasonal-naive", "snaive"),
        "Repeats the last seasonal cycle; the reference of fev-bench and GIFT-Eval skill scores.",
    ),
    _baseline("Naive", "NaiveAdapter", ("random-walk",), "Repeats the last observation."),
    _baseline("Mean", "MeanAdapter", ("historic-average",), "Forecasts the historical mean."),
    _baseline(
        "Drift",
        "DriftAdapter",
        ("random-walk-drift",),
        "Extrapolates the line from the first to the last observation.",
    ),
    _baseline(
        "WindowAverage",
        "WindowAverageAdapter",
        ("moving-average",),
        "Forecasts the mean of the last window (one season by default).",
    ),
    TimeSeriesModelSpec(
        name="TabularTS-GBM",
        family="baseline",
        adapter="tabtune.models.TimeSeries.tabular:TabularForecasterAdapter",
        aliases=("tabular-gbm",),
        default_checkpoint="sklearn-gbm",
        checkpoints=("sklearn-gbm",),
        native_missing=True,
        supports_covariates=True,
        supports_categorical_covariates=True,
        license=LicenseSpec(name="BSD-3-Clause (scikit-learn; no weights)", commercial_use_ok=True),
        paper="https://doi.org/10.1016/j.ijforecast.2021.11.013",
        summary=(
            "Gradient boosting on a pooled direct multi-horizon lag design (M5 style), "
            "one quantile-loss model per level."
        ),
    ),
)

NATIVE_QUANTILE_TABULAR_MODELS: frozenset[str] = frozenset(
    {"TabPFN", "TabPFNv26", "TabPFNv3", "TabPFNv35", "TabPFNv35Fast", "TabICLv2"}
)


def _tabular_forecaster(tabular) -> TimeSeriesModelSpec:
    alternatives = tuple(
        f"TabularTS-{name}"
        for name in tabular.commercial_alternatives
        if any(name == s.name and s.regression_strategies for s in _TABULAR_SPECS)
    )
    quantiles = tabular.name in NATIVE_QUANTILE_TABULAR_MODELS
    return TimeSeriesModelSpec(
        name=f"TabularTS-{tabular.name}",
        family="tabular-icl",
        adapter="tabtune.models.TimeSeries.tabular:TabularForecasterAdapter",
        default_checkpoint=tabular.name,
        checkpoints=(tabular.name,),
        native_missing=True,
        supports_covariates=True,
        supports_categorical_covariates=True,
        license=tabular.license,
        commercial_alternatives=alternatives,
        paper=tabular.paper,
        summary=(
            f"{tabular.name} regression on TabPFN-TS time features per series, or on a pooled "
            f"lag design; {'native quantiles' if quantiles else 'point forecasts'}."
        ),
    )


TABULAR_FORECASTER_SPECS: tuple[TimeSeriesModelSpec, ...] = tuple(
    _tabular_forecaster(s) for s in _TABULAR_SPECS if "inference" in s.regression_strategies
)

TS_MODEL_SPECS = TS_MODEL_SPECS + BASELINE_SPECS + TABULAR_FORECASTER_SPECS
