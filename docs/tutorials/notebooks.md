# Notebooks

Eighteen Colab notebooks, one feature each. They download real weights and run on Colab's
free GPU tier; the ones that fine-tune are slow on CPU. The `examples/` directory in the
repository has the same material as plain scripts, listed on [Features](../features.md).

| # | Notebook | What it runs | |
|---|---|---|---|
| 1 | Unified API | One `TabularPipeline` call across several models. | [Open in Colab](https://colab.research.google.com/drive/1KcaSdYRjZnMlb0MLmQ5IlnbPDiuEr1Ld?usp=sharing) |
| 2 | Model-aware preprocessing | What `DataProcessor` does per model, and how to override it. | [Open in Colab](https://colab.research.google.com/drive/12BQ12VJrxtTDslgjnjm26yi3a0PYXqZT?usp=sharing) |
| 3 | Fine-tuning strategies | `inference`, `finetune` in its modes, and `peft`, compared. | [Open in Colab](https://colab.research.google.com/drive/1QixfiNCjF1IQV9NooMipPpnH4ETcEQwg?usp=sharing) |
| 4 | Model comparison | `TabularLeaderboard` over models and strategies. | [Open in Colab](https://colab.research.google.com/drive/1PZW3iPQOvwh0kroGytMzYTGc6ZVUzuvg?usp=sharing) |
| 5 | Checkpoint management | Save, load, resume. | [Open in Colab](https://colab.research.google.com/drive/1DBTGEPpYLJjU9Aj7lzHoX3JtwaNOC0jn?usp=sharing) |
| 6 | Advanced usage | PEFT configuration and hybrid strategies. | [Open in Colab](https://colab.research.google.com/drive/1V3XGLeKrXSJwavaULMncZiM7uVE8sz0h?usp=sharing) |
| 7 | Resampling | Context and support/query sampling. | [Open in Colab](https://colab.research.google.com/drive/1EHGrrSm7EalVRvzkH1RUHsNSLzmn10lM?usp=sharing) |
| 8 | Regression, inference | Zero-shot regression. | [Open in Colab](https://colab.research.google.com/drive/1lBt0QZWqlwhEg2ul_nVPAeC-w3Are0At) |
| 9 | Regression, fine-tuning | `turn_by_turn` and native regression fine-tuning. | [Open in Colab](https://colab.research.google.com/drive/1FFuaRBDtJZFAQF-JDIxRAjtgOZ1rmHd1?usp=sharing) |
| 10 | Evaluation metrics | What `evaluate()` returns and the calibration and fairness reports. | [Open in Colab](https://colab.research.google.com/drive/18TxyTyBGAGrIVf6zLjURDChG0vM4V02M?usp=sharing) |
| 11 | Benchmarking | Benchmark suites and result handling. | [Open in Colab](https://colab.research.google.com/drive/1lcoVMPz_3X5_5taNdB9doTGoN05krNRw?usp=sharing) |
| 12 | TabPFN v2.6 | Classification and regression with the native fine-tuning API. | [Open in Colab](https://colab.research.google.com/drive/1-5fh2kU9sDidXmm095489f3sxNLssW_M) |
| 13 | TabICL v2 | Classification and quantile regression. | [Open in Colab](https://colab.research.google.com/drive/13lv9Z5QNzaAp_2ArkTXGRKDjDFbKAq3Q) |
| 14 | Ensembling | The six `TabularEnsemble` strategies. | [Open in Colab](https://colab.research.google.com/drive/19TUTBuJ1VNIbp5hLdU4D64c2_RfwFQC8) |
| 15 | Distillation | Single and multi-teacher `TabDistiller`. | [Open in Colab](https://colab.research.google.com/drive/1Fo2zH7jDgYjkYhgI33SyuVgnrhMsdvUH) |
| 16 | Causal inference | Treatment effects with a TFM as the nuisance learner. | [Open in Colab](https://colab.research.google.com/drive/1CWYo3ynOxw0ysV4iDz_8VNCBjMK3WIyd?usp=sharing) |
| 17 | EXAONE Tabular | End to end with EXAONE. | [Open in Colab](https://colab.research.google.com/drive/1RR-mhJkGW0xpAU0si76ru-1NuggPKDNH) |
| 18 | Timestral | Zero-shot, calibrated, covariate and multivariate forecasts, anomalies, gaps, embeddings, fine-tuning and LoRA, a leaderboard against baselines. Runs offline with `TIMESTRAL_CHECKPOINT=random`. | [Open in Colab](https://colab.research.google.com/github/Lexsi-Labs/TabTune/blob/main/examples/notebooks/18_timestral.ipynb) |

## Scripts in the repository

`examples/01_unified_api.py` through `examples/22_new_models_finetune_effect.py` cover the
tabular side, including the registry (`14`), shift-aware evaluation (`15`), uncertainty
(`18`), the Chronos, TimesFM and Toto families (`19` to `21`) and the validated fine-tuning
benchmark for Kumo, LimiX-2 and TabAbax (`22`). `examples/timeseries/` has ten scripts:
quickstart, fine-tuning, covariates and multivariate targets, anomaly detection,
imputation, probabilistic forecasting and calibration, leaderboard and ensemble, benchmark,
tabular forecasting, and embeddings with series features.

```bash
python examples/01_unified_api.py
python examples/timeseries/quickstart.py
```
