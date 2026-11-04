# Time Series Models

Eleven time series foundation models, vendored. No model package is installed: the upstream
code lives under `tabtune/models/` with its licence file and is imported only when a
pipeline is fitted. All eleven run zero-shot, fine-tune fully or with LoRA, and serve
embeddings. The tables were generated from the registry; query it with
`list_time_series_models()` or `tabtune timeseries list-models` rather than copying them.

## Foundation models

| Model | `model_name` | Default checkpoint | Multivariate | Covariates | NaN in history | Max context | Weight licence | Commercial |
|---|---|---|---|---|---|---|---|---|
| Chronos | `Chronos` | `amazon/chronos-t5-small` | no | no | yes | 512 | Apache-2.0 | yes |
| Chronos-Bolt | `ChronosBolt` | `amazon/chronos-bolt-small` | no | no | yes | 2,048 | Apache-2.0 | yes |
| Chronos-2 | `Chronos2` | `amazon/chronos-2` | yes | past + known, incl. categorical | yes | 8,192 | Apache-2.0 | yes |
| TimesFM 3 | `TimesFM3` | `google/timesfm-3.0-pytorch` | yes | past + known, numeric | yes | 15,360 | TimesFM Non-Commercial License v1.0 | no |
| Toto 1 | `Toto1` | `Datadog/Toto-Open-Base-1.0` | yes | past + known, numeric | yes | 4,096 | Apache-2.0 | yes |
| Toto 2 | `Toto2` | `Datadog/Toto-2.0-22m` | yes | no | yes | 4,096 | Apache-2.0 (`tabtune[toto2]`) | yes |
| TiRex | `TiRex` | `NX-AI/TiRex` | no | no | yes | 2,048 | NXAI Community License | unverified |
| TiRex-2 | `TiRex2` | `NX-AI/TiRex-2` | yes | past + known, numeric | yes | from checkpoint | Apache-2.0 (code); weight license not verified | unverified |
| Time-MoE | `TimeMoE` | `Maple728/TimeMoE-50M` | no | no | no | 4,096 | Apache-2.0 (code); weight license not verified | unverified |
| [Timestral](../models/timestral.md) | `Timestral` | `mistralai/Timestral-209M` | yes | past + known, incl. categorical | yes | 32,768 | Apache-2.0 (code) / Modified MIT (weights) | unverified |
| [T0](../models/t0.md) | `T0` | `theforecastingcompany/t0-gamma` | yes | known, numeric | yes | not declared | Apache-2.0 (code); weight license not verified | unverified |

"Max context" is what the checkpoint was documented for. Longer histories are truncated to
it with a warning (`envelope_mode="warn"`) or rejected (`"error"`). TiRex is under the NXAI
Community License, which allows commercial use under conditions; the registry records it as
unverified so `license_mode="commercial"` warns and lets you decide.

## Checkpoints and papers

Pass another checkpoint with `model_params={"checkpoint": ...}`. Only ids the registry
validated are accepted unless `validate=False`; local directories are always allowed.

| Model | Checkpoints | Paper | Summary |
|---|---|---|---|
| Chronos | `amazon/chronos-t5-tiny`, `amazon/chronos-t5-mini`, `amazon/chronos-t5-small`, `amazon/chronos-t5-base`, `amazon/chronos-t5-large` | [paper](https://arxiv.org/abs/2403.07815) | T5 language model trained on scaled and quantised time series tokens. |
| Chronos-Bolt | `amazon/chronos-bolt-tiny`, `amazon/chronos-bolt-mini`, `amazon/chronos-bolt-small`, `amazon/chronos-bolt-base` | [paper](https://arxiv.org/abs/2403.07815) | Patch-based encoder-decoder predicting all quantiles in one pass: deterministic and much faster than Chronos v1. |
| Chronos-2 | `amazon/chronos-2` | [paper](https://arxiv.org/abs/2510.15821) | Encoder-only group-attention model: multivariate targets, past and known-future covariates, in-context learning across related series. |
| TimesFM 3 | `google/timesfm-3.0-pytorch` | [paper](https://arxiv.org/abs/2310.10688) | Single-pass patched transformer with variate attention: multivariate targets, past and known-future numeric covariates, 15360-step context. |
| Toto 1 | `Datadog/Toto-Open-Base-1.0` | [paper](https://openreview.net/forum?id=1jDAYXfcS2) | Observability-oriented decoder with proportional factorised space-time attention: sample-based, so any quantile level and a mean forecast. |
| Toto 2 | `Datadog/Toto-2.0-4m`, `Datadog/Toto-2.0-22m`, `Datadog/Toto-2.0-313m`, `Datadog/Toto-2.0-1B`, `Datadog/Toto-2.0-2.5B` | [paper](https://arxiv.org/abs/2605.20119) | Observability-oriented decoder with alternating time/variate attention and u-microP scaling: multivariate targets, five sizes from 4m to 2.5B. |
| TiRex | `NX-AI/TiRex`, `NX-AI/TiRex-1.1-gifteval` | [paper](https://arxiv.org/abs/2505.23719) | Small xLSTM forecaster for short and long horizons, with missing-value support. |
| TiRex-2 | `NX-AI/TiRex-2`, `NX-AI/TiRex-2-fevbench`, `NX-AI/TiRex-2-gifteval-pretrain`, `NX-AI/TiRex-2-gifteval-zs` | [paper](https://arxiv.org/abs/2607.01204) | Patch-based xLSTM forecaster with variate attention: multivariate targets and past and known-future numeric covariates. |
| Time-MoE | `Maple728/TimeMoE-50M`, `Maple728/TimeMoE-200M` | [paper](https://arxiv.org/abs/2409.16040) | Decoder-only mixture-of-experts point forecaster (univariate). |
| Timestral | `mistralai/Timestral-5M`, `mistralai/Timestral-28M`, `mistralai/Timestral-123M`, `mistralai/Timestral-209M` | [announcement](https://mistral.ai/news/timestral) | Causal patch transformer trained with Decoupled Horizon Forecasting: multivariate targets, past-only, known-future and categorical covariates, 21 quantiles, 32,768-step context. |
| T0 | `theforecastingcompany/t0-alpha`, `theforecastingcompany/t0-beta`, `theforecastingcompany/t0-gamma` | [paper](https://arxiv.org/abs/2609.24559) | Patch transformer with factorised time and group attention (Toto and Chronos-2 lineage): multivariate targets and known-future covariates, quantiles at any level. |

## Tabular foundation models as forecasters

`TabPFN-TS` re-implements the TabPFN-TS recipe on TabTune's vendored TabPFN v3.5: per-series
regression on calendar and Fourier time features, known covariates included, native
quantiles. Every tabular model with a regression head is also registered as
`TabularTS-<name>`, with TabPFN-TS's time features or a pooled lag design
(`model_params={"features": "time" | "lags"}`). Today that is 20 models:
`TabularTS-TabPFN`, `TabularTS-TabPFNv26`, `TabularTS-TabPFNv3`, `TabularTS-TabPFNv35`, `TabularTS-TabPFNv35Fast`, `TabularTS-Causilo`, `TabularTS-TabLDM`, `TabularTS-TabICLv2`, `TabularTS-Mitra`, `TabularTS-MitraV2`, `TabularTS-ContextTab`, `TabularTS-TabDPT`, `TabularTS-Limix`, `TabularTS-KumoTabular`, `TabularTS-LimiXv2`, `TabularTS-TabAbax`, `TabularTS-TabFM`, `TabularTS-XRFM`, `TabularTS-ILTM`, `TabularTS-EXAONETabular`. They inherit the tabular model's weight licence. `TabularTS-GBM` is a weight-free
gradient-boosting forecaster on the same lag design, and any scikit-learn regressor drops
in through `SklearnBackend`. All of these are inference-only as forecasters; see
[Tabular Models as Forecasters](tabular-forecasters.md).

## Statistical baselines

Every leaderboard needs a floor. These have no weights and run anywhere.

| Baseline | `model_name` | Aliases | What it does |
|---|---|---|---|
| SeasonalNaive | `SeasonalNaive` | `seasonal-naive`, `snaive` | Repeats the last seasonal cycle; the reference of fev-bench and GIFT-Eval skill scores. |
| Naive | `Naive` | `random-walk` | Repeats the last observation. |
| Mean | `Mean` | `historic-average` | Forecasts the historical mean. |
| Drift | `Drift` | `random-walk-drift` | Extrapolates the line from the first to the last observation. |
| WindowAverage | `WindowAverage` | `moving-average` | Forecasts the mean of the last window (one season by default). |
| TabularTS-GBM | `TabularTS-GBM` | `tabular-gbm` | Gradient boosting on a pooled direct multi-horizon lag design (M5 style), one quantile-loss model per level. |

## Picking one

- **Covariates or several targets?** Chronos-2 and Timestral (numeric and categorical
  covariates), TimesFM 3, T0, Toto 1, TiRex-2 (numeric covariates), or TabPFN-TS (known covariates only).
- **Long histories?** Timestral (32,768 steps), TimesFM 3 (15,360), Chronos-2 (8,192), Toto (4,096).
- **Fast and cheap?** Chronos-Bolt predicts all quantiles in one pass and is the usual
  default. TiRex is small and strong on both short and long horizons.
- **Commercial use?** Chronos family and Toto are Apache-2.0. TimesFM 3 is non-commercial.
  TiRex, TiRex-2, Time-MoE, Timestral and T0 are unverified; read their terms.
- **Observability metrics?** Toto was trained for them.
- **Any quantile level?** T0 interpolates between its trained levels and extrapolates on
  exponential tails, so any level in (0, 1) is served. Toto samples, which does the same.
- **Not sure?** `TimeSeriesPipeline.select(...)` backtests the candidates and returns the
  best one, fitted. See [Compare, Combine, Select](comparison.md).
