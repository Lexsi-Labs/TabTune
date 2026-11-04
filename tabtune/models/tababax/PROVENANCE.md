# Code Provenance

## TabICLv2

The inference implementation is derived from:

https://github.com/soda-inria/tabicl

Reference:

Tag `v2.2.0`, commit `0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3`.

https://github.com/soda-inria/tabicl/commit/0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3

This commit is the version used for the comparison in this file. It is not a record of
the commit that was copied when the repository was created. That copy step was not recorded.

### Files that match the reference version

These files match once the TabAbax attribution header is removed:
`_cgroup_memory.py`, `attention.py`, `encoders.py`, `inference_config.py`,
`layers.py`, `learning.py`, `quantile_dist.py`, `ssmax.py`, and `_torch_devices.py`.

`rope.py` and `sklearn_utils.py` match apart from the header and the third-party
license sentences added in their docstrings.

`kv_cache.py` matches apart from the header and the rename of
`TabICLCache` to `TabAbaxCache`. That rename is the only difference in the file.

### Files that are modified derivatives

- `embedding.py`;
- `interaction.py`;
- `inference.py`;
- `model.py` (from `tabicl.py`);
- `_sklearn/base.py`;
- `_sklearn/classifier.py`;
- `_sklearn/preprocessing.py`;
- `_sklearn/regressor.py`.

`embedding.py` adds `FourierFeatureEmbedding` and `ColInteraction`.
`interaction.py` changes the body of `RowInteraction`.

`inference.py` is a modified derivative, not a file that only differs by its header.
On top of the reference version it adds a block marked
`PATCH BEGIN: auto downgrade helpers` in the file. That block estimates the forward
peak of the batch that will actually run, moves the stored output off the GPU when
the batch no longer fits, and promotes a real out-of-memory failure to disk once
before retrying. The reference version already contains `OffloadConfig`,
`MemoryEstimator`, `AsyncCopyManager`, `DiskTensor`, and the `_cgroup_memory` helpers.
TabAbax changes when and how that machinery is triggered, so the file must not be
described as unchanged.

### Modules that are not in this repository

The reference version also ships `train/`, `prior/`, `_finetune/`, `forecast/`,
`shap/`, and `_unsupervised/`. This repository does not include them.

Derived areas:

- src/tababax/_model/
- src/tababax/_sklearn/base.py
- src/tababax/_sklearn/classifier.py
- src/tababax/_sklearn/regressor.py
- src/tababax/_sklearn/preprocessing.py
- src/tababax/_sklearn/sklearn_utils.py
- src/tababax/_torch_devices.py

The TabICL-derived portions retain the BSD 3-Clause License.

## TabAbax modifications

TabAbax changes include:

- renamed public classes and package namespace;
- modified architecture components;
- changed training and prior configuration, which are not in this repository;
- newly trained checkpoints, distributed separately under CC-BY-NC-4.0;
- inference integration and memory-management changes where applicable.

This repository contains inference code only.

It does not contain:

- pre-training code;
- synthetic prior generation code;
- training datasets;
- optimizer configuration;
- full checkpoint-generation scripts.

The released checkpoints were trained from a TabICLv2-derived implementation
with additional architectural and prior changes.

## Other third-party sources

- rotary-embedding-torch: MIT, Copyright (c) 2021 Phil Wang
- scikit-learn: BSD-3-Clause, commit 1eb422d6c5 of sklearn/utils/validation.py
  (full license and copyright line in LICENSES/scikit-learn-BSD-3-Clause.txt)
- TabR: MIT, commit 75105013189c76bc4f247633c2fb856bc948e579
- PyTabKit: Apache-2.0, commit 949bf81e3964f65a33dd2c252c3713c239c17b2d

See NOTICE for the applicable notices and license texts.

## Methods referenced but not copied

Two places in the model code name a technique after the project it came from. In both cases the
idea is credited in the comment and the implementation was written here.

- `_model/embedding.py`, `FourierFeatureEmbedding`: "Causilo-style" learnable Fourier feature
  embedding. Idea from Causilo (https://github.com/nums-ai/causilo).
- `_model/embedding.py`, `ColInteraction`: the output projection and normalization after a
  column-interaction stage is marked "Follow TabFM". Idea from TabFM
  (https://github.com/google-research/tabfm).
