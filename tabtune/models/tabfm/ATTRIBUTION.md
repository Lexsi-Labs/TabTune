# TabFM — Third-Party Attribution

The following files in this directory are **vendored** (ported) from Google
Research's TabFM release and are licensed under the **Apache License, Version 2.0**:

- `model/model.py` — the TabFM PyTorch architecture (verbatim, torch-only).
- `model_loading.py` — the Hugging Face weight loader (`TabFM_HF`, `load()`).
  Adapted only by replacing `from absl import logging` with a stdlib-`logging`
  shim and rewriting one intra-package import path.
- `classifier_and_regressor.py` — the scikit-learn `TabFMClassifier` /
  `TabFMRegressor` and preprocessing / ensembling / calibration engine.
  Adapted only by the same `absl.logging` → stdlib-`logging` shim.

Each vendored file retains its original Apache-2.0 copyright header
(`Copyright 2026 Google LLC`).

**Upstream project:** https://github.com/google-research/tabfm
**Pretrained weights:** https://huggingface.co/google/tabfm-1.0.0-pytorch (`google/tabfm-1.0.0-pytorch`)

A full copy of the Apache-2.0 license is available at
https://www.apache.org/licenses/LICENSE-2.0. The remaining files in this package
(`__init__.py`, `backbone.py`, `classifier.py`, and the regression wrapper) are
original TabTune glue code and follow TabTune's own license.
