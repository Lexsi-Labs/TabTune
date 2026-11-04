# TabICL v2

**TabICLv2** is the second generation of `soda-inria`'s scalable tabular in-context learner.
It keeps the two-stage column-then-row attention that made TabICL work and adds **QASSMax**
normalisation plus a **native quantile regression head** — so unlike TabICL v1, it does
regression natively rather than by transfer.

```python
from tabtune import TabularPipeline

pipeline = TabularPipeline(
    model_name="TabICLv2",
    task_type="classification",
    tuning_strategy="inference",
)
pipeline.fit(X_train, y_train)
print(pipeline.evaluate(X_test, y_test))
```

Aliases: `TabICL-v2`, `TabICL2`.

---

## 1. What changed from v1

| | TabICL | TabICLv2 |
|---|---|---|
| Attention | Column-then-row | Column-then-row + **QASSMax** normalisation |
| Regression | ❌ | ✅ native **quantile** head |
| Row ceiling | undeclared | 500,000 (via CPU/disk KV-cache offloading) |
| Feature ceiling | undeclared | 2,000 |
| PEFT | ✅ | ❌ — use `finetune` |
| Licence | BSD-3-Clause | BSD-3-Clause |

!!! success "Commercially deployable"
    BSD-3-Clause weights. TabICLv2 is one of the registry's default
    `commercial_alternatives` when another model's licence blocks you.

---

## 2. Envelope

| Constraint | Value | Severity |
|---|---:|---|
| `max_features` | `2,000` | warn |
| `max_rows` | `500,000` | warn |
| `native_nan` | yes | — |

Upstream supports 500k rows via **CPU/disk offloading of the KV cache** — see
`offload_mode` below.

---

## 3. Model parameters

### 3.1 Classification

```python
model_params = {
    'n_estimators': 8,
    'norm_methods': None,
    'feat_shuffle_method': 'latin',
    'class_shuffle_method': 'shift',
    'outlier_threshold': 4.0,
    'softmax_temperature': 0.9,
    'average_logits': True,
    'support_many_classes': True,
    'batch_size': 8,
    'kv_cache': False,
    'offload_mode': 'auto',
    'disk_offload_dir': None,
    'use_amp': 'auto',
    'use_fa3': 'auto',
    'device': 'cuda',
    'random_state': 42,
    'n_jobs': None,
    'verbose': False,
}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_estimators` | int | `8` | Ensemble members |
| `norm_methods` | str / list | `None` | Feature normalisation scheme(s) |
| `feat_shuffle_method` | str | `'latin'` | Feature permutation across ensemble members |
| `class_shuffle_method` | str | `'shift'` | Class permutation across ensemble members |
| `outlier_threshold` | float | `4.0` | Clipping threshold in standard deviations |
| `softmax_temperature` | float | `0.9` | Lower = sharper predictions |
| `average_logits` | bool | `True` | Average logits vs probabilities across the ensemble |
| `support_many_classes` | bool | `True` | Enable the many-class path |
| `batch_size` | int | `8` | Inference batch size |
| `kv_cache` | bool / str | `False` | Cache keys/values across queries |
| `offload_mode` | str / bool | `'auto'` | `'auto'`, CPU or disk offloading of the KV cache |
| `disk_offload_dir` | str | `None` | Where to spill when offloading to disk |
| `use_amp` | bool / str | `'auto'` | Mixed precision |
| `use_fa3` | bool / str | `'auto'` | FlashAttention-3 when available |
| `model_path` / `checkpoint_version` | str | released ckpt | Pin a specific checkpoint |
| `allow_auto_download` | bool | `True` | Fetch weights from the Hub |
| `n_jobs` | int | `None` | Worker processes |

### 3.2 Regression

The regressor takes the same parameters minus `class_shuffle_method`, `softmax_temperature`,
`average_logits` and `support_many_classes`, and defaults to the regression checkpoint.

---

## 4. Scaling to 500k rows

```python
pipeline = TabularPipeline(
    model_name="TabICLv2",
    task_type="classification",
    model_params={
        "kv_cache": True,
        "offload_mode": "auto",              # spills to CPU, then disk
        "disk_offload_dir": "./.tabicl-kv",
        "batch_size": 4,
        "use_amp": True,
    },
)
```

!!! tip "If you hit OOM"
    In order: lower `batch_size`, lower `n_estimators`, set `offload_mode` explicitly, then
    fall back to context sampling via
    `processor_params={'context_sampling_strategy': 'stratified', 'context_size': 100_000}`.

---

## 5. Supported strategies

| Task | Strategy | Modes |
|---|---|---|
| Classification | `inference` | — |
| Classification | `finetune` | `meta-learning` (default), `sft`, `turn_by_turn` |
| Regression | `inference` | — |
| Regression | `finetune` | `turn_by_turn` (default) — episodic MSE training |

!!! warning "No PEFT"
    `tuning_strategy='peft'` is not supported for TabICLv2. Use `'finetune'`.
    `check_envelope` / `validate_request` will tell you this before weights load.

```python
pipeline = TabularPipeline(
    model_name="TabICLv2",
    task_type="regression",
    tuning_strategy="finetune",
    tuning_params={"epochs": 5, "learning_rate": 1e-5},
)
```

---

## 6. Quantiles and intervals

TabICLv2's native quantile head is exposed through the pipeline:

```python
pipeline.predict_quantiles(X_test, quantiles=[0.1, 0.5, 0.9])
pipeline.predict_intervals(X_test, confidence=0.95)
```

That also makes it a candidate for **CQR** conformal regression, which adapts interval
widths per row rather than using a constant band:

```python
from tabtune.uncertainty import ConformalRegressor
cr = ConformalRegressor(pipeline, method="cqr", alpha=0.1).calibrate(X_cal, y_cal)
lo, hi = cr.predict_interval(X_test)
```

See [Uncertainty Quantification](../user-guide/uncertainty.md).

---

## 7. When to use TabICLv2

**Good fit**

- **Commercial deployment** — BSD-3-Clause
- Large tables: 10k–500k rows, up to 2,000 features
- Regression where you want native quantiles
- A strong general-purpose default across both tasks

**Poor fit**

- PEFT workflows (not supported)
- Beyond 2,000 features
- Very small datasets where TabPFN's prior is stronger

---

## See Also

- [TabICL](tabicl.md) — the v1 model
- [Model Overview](overview.md)
- Runnable example: notebook 13 in the [examples table](../index.md)
