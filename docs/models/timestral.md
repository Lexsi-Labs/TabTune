# Timestral

Timestral is Mistral AI's family of time series foundation models for multivariate probabilistic forecasting. It is a causal patch transformer trained with Decoupled Horizon Forecasting: separate prediction tokens read the causally encoded context, and a monotone head emits 21 non-crossing quantiles per step. Each variate is a target, a past-only covariate or a known-future covariate; covariates condition the targets but never read them. TabTune vendors the `timestral` 1.0.0 package in `tabtune/models/timestral` and serves it through `TimeSeriesPipeline`.

| | |
|---|---|
| `model_name` | `"Timestral"` (aliases `"Timestral-209M"`, `"Mistral-Timestral"`) |
| Tasks | forecasting, anomaly detection, imputation, embedding |
| Strategies | `inference`, `finetune`, `peft` |
| Checkpoints | `mistralai/Timestral-5M`, `-28M`, `-123M`, `-209M` (default) |
| Multivariate / covariates | yes; past-only, known-future and categorical |
| Context | up to 32,768 steps |
| Quantiles | 0.01, 0.05, 0.1 ... 0.9, 0.95, 0.99 |

## Usage

```python
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema

schema = TimeSeriesSchema(target="sales", item_id="store", past_covariates=["temp"], known_covariates=["promo"])
pipe = TimeSeriesPipeline(
    "Timestral",
    model_params={"checkpoint": "mistralai/Timestral-28M"},
    forecast_params={"prediction_length": 96, "quantile_levels": [0.1, 0.5, 0.9]},
)
forecast = pipe.fit(history, schema, future_df=future_promo).predict()
```

Notebook 18 on [Notebooks](../tutorials/notebooks.md) (`examples/notebooks/18_timestral.ipynb`) runs every task, fine-tuning and a leaderboard end to end.

## Model parameters (`model_params`)

| Parameter | Default | Description |
|---|---|---|
| `checkpoint` | `mistralai/Timestral-209M` | Any of the four Hub repos, or a local directory with `config.json` and `model.safetensors`. |
| `batch_size` | `64` | Items per forward pass. Shorter histories are left-padded with NaN, which the model reads as missing. |
| `symmetric_averaging` | `True` | Average with the mirrored forecast of the negated series (two passes), as upstream. |
| `clip_nonnegative` | `True` | Clamp at zero the series whose observed context is non-negative, as upstream. |
| `per_shot_horizon_patches` | `32` | Patches decoded per block, capped at the checkpoint's block size. |
| `quantile_real_cap_k` | `1e4` | Clamp forecasts to the observed range widened by `k` last-patch scales; `0` disables. |
| `revision`, `local_files_only` | | Hugging Face download options. |

Horizons longer than one decode block are decoded block by block, feeding back the median. Categorical covariates are factorised to codes and target-encoded upstream for univariate targets. Timestral runs in float32.

## Fine-tuning

`finetune` updates every weight and `peft` trains LoRA adapters on `wQKV`, `wO`, `mlp.w12` and `mlp.w3`. Training windows are univariate target windows without covariates. The loss is the pinball loss over the 21 native quantiles, taken in the model's normalised space: the head's quantiles are read before denormalisation, and the targets are moved into the same frame (location and scale of each block's refine step, then `asinh`). A loss in data units overflows on large-valued series and stops at a saturated head; this one stays finite and keeps its gradient. Training horizons are clamped to one decode block. The usual `TimeSeriesPipeline` options apply: `epochs`, `steps_per_epoch`, `learning_rate`, `batch_size`, `validation_split`, `early_stopping` and `peft_config`.

## Embeddings

One vector per target row: the final encoder states averaged over the observed context patches, `d_model` wide.

## License

Code Apache-2.0. Weights under Mistral AI's Modified MIT license in each model repository. The registry records `commercial_use_ok=None` (unverified), so `license_mode="commercial"` warns; read the license before commercial use.
