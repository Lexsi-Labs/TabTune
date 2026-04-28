# TabTune Distillation Module

## 1. Overview

The TabTune Distillation module compresses any tabular foundation model (TFM) into a lightweight, production-deployable student using **model-agnostic soft-label knowledge distillation**. It requires only `predict_proba()` — no access to model internals, gradients, or architecture details.

The core algorithm uses **k-fold cross-prediction** to generate out-of-fold soft labels, which prevents *ICL identity leakage* — the failure mode where in-context learning models label their own training data, producing artificially overconfident soft targets that mislead the student.

The module integrates seamlessly with TabTune's `TabularPipeline` and follows TabTune's conventions for configuration, fitting, prediction, and evaluation.

### Key Features

- **Model-agnostic** — works with any TFM that exposes `predict_proba()`: TabPFNv2.6, TabICLv2, TabDPT, Mitra, and others.
- **Four student backends** — `mlp`, `lgbm`, `xgb`/`xgboost`, `catboost`/`cb`.
- **K-fold cross-prediction** — eliminates ICL identity leakage by ensuring teachers never label their own training samples.
- **Adaptive temperature** — per-sample temperature scaling based on teacher entropy, rather than a single global `T`.
- **Confidence weighting** — bell-shaped sample weighting that focuses student training on the clearest teacher signals.
- **Multi-teacher distillation** — averages soft labels across multiple TFMs before training the student.
- **Deployment-ready** — `save()` strips the teacher entirely; the serialised student has no TFM dependency.
- **Advanced strategies** — progressive/curriculum distillation, post-hoc calibration, and self-distillation available via `tabtune.distillation.strategies`.

---

## 2. Installation & Setup

The distillation module lives under `tabtune/distillation/` and is imported as:

```python
from tabtune.distillation import TabDistiller
```

### Dependencies

| Package | Required | Notes |
|---------|----------|-------|
| `numpy` | Yes | Core array operations |
| `pandas` | Yes | DataFrame handling |
| `scikit-learn` | Yes | Metrics, CV splitting, preprocessing |
| `scipy` | Yes | Temperature optimisation (L-BFGS-B), NLL minimisation |
| `joblib` | Yes | Student serialisation (`save`/`load`) |
| `lightgbm` | Optional | `student="lgbm"` backend |
| `xgboost` | Optional | `student="xgb"` / `"xgboost"` backend |
| `catboost` | Optional | `student="catboost"` / `"cb"` backend |
| `torch` | Optional | `student="mlp"` backend |
| `tabtune` | Yes (for integration) | Strategy-only tests run without it |

### File Structure

```
tabtune/distillation/
├── __init__.py              # Public API exports
├── distiller.py             # TabDistiller orchestrator class
├── losses.py                # Hinton KD loss, adaptive temperature, confidence weights
├── strategies.py            # Advanced strategies: CV distillation, progressive,
│                            #   calibration, self-distillation
├── students/
│   ├── mlp_student.py       # PyTorch MLP student
│   └── gbdt_student.py      # GBDTStudent: lgbm / xgb / catboost backends
tests/
└── test_distillation.py     # Unit + integration tests
```

---

## 3. Quick Start

### Classification — TabICLv2 → LightGBM (Example) 

```python
from tabtune.distillation import TabDistiller
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Build distiller
distiller = TabDistiller(
    teachers="TabICLv2",
    student="lgbm",
    task_type="classification",
    temperature=3.0,
    alpha=0.7,
    n_folds=5,
    adaptive_temperature=True,
    confidence_weighting=True,
    device="cuda",
)

# Fit & compare
distiller.fit(X_train, y_train)
results = distiller.compare(X_test, y_test)

```


---

## 4. Architecture

```
TabDistiller (orchestrator)
│
├── fit(X_train, y_train)
│   ├── Step 1: Fit teacher pipeline(s)           [TabularPipeline per teacher]
│   ├── Step 2: Extract label encoder             [from teacher processor]
│   ├── Step 3: Collect soft labels               [k-fold cross-prediction]
│   │           └── Normalize soft labels         [fixes TabDPT float bug]
│   ├── Step 4: Compute adaptive temperatures     [per-sample, from entropy]
│   ├── Step 4: Compute confidence weights        [bell-shaped weighting]
│   ├── Step 5: (Optional) Augment data           [Gaussian noise]
│   ├── Step 6: Preprocess features               [numeric, fillna, float32]
│   └── Step 7: Train student                     [MLP or GBDTStudent]
│
├── predict(X_test)
│   └── student_.predict(preprocess(X_test))
│
├── predict_proba(X_test)
│   └── student_.predict_proba(preprocess(X_test))
│
├── evaluate(X_test, y_test)   → Dict with student metrics
├── compare(X_test, y_test)    → Dict with student + teacher metrics + retention
├── save(path)                 → joblib dump (teacher stripped)
└── TabDistiller.load(path)    → restored student, no TFM dependency
```

Internally, `_collect_soft_labels` routes each teacher through either:
- **K-fold cross-prediction** (ICL inference-mode teachers) — teacher never labels its own training samples
- **Direct prediction** (fine-tuned or pre-fitted teachers) — teacher predicts on full training set

---

## 5. Strategy Reference

### 5.1 Soft Labels (Default)

**Strategy:** Standard Hinton knowledge distillation via soft label targets.

The teacher's probability distribution over classes contains richer information than the hard one-hot label. A temperature parameter `T` smooths the distribution: high `T` flattens it (more information transfer), low `T` sharpens it (closer to hard labels).

#### Loss Function

$$\mathcal{L} = \alpha \cdot T^2 \cdot \text{KL}(p_\text{teacher}^{(T)} \| p_\text{student}^{(T)}) + (1 - \alpha) \cdot \text{CE}(y_\text{hard}, p_\text{student})$$

- `alpha` controls the balance between soft-label loss and hard-label loss.
- The `T²` scaling factor corrects for gradient magnitude (Hinton et al., 2015).

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | `3.0` | Base temperature `T` for soft label smoothing |
| `alpha` | float | `0.7` | Soft-label loss weight (1-alpha = hard-label loss weight) |

#### When To Use

Default for all classification distillation tasks. Works with any GBDT or MLP student.

---

### 5.2 Adaptive Temperature

**Strategy:** Per-sample temperature scaling based on teacher entropy.

Instead of a single global `T`, each training sample gets a temperature proportional to the teacher's local uncertainty (entropy). High-entropy samples (teacher unsure) get a higher temperature, preventing oversmoothing on already-uncertain samples.

#### How It Works

```
entropy(x) = -Σ p_i log(p_i)

T_adaptive(x) = T_base × (1 + β × (entropy(x) - mean_entropy) / std_entropy)
```

The resulting per-sample temperatures are passed to the loss function, replacing the global `T`.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adaptive_temperature` | bool | `True` | Enable adaptive per-sample temperature |
| `temperature` | float | `3.0` | Base temperature used as anchor |

#### Attributes After Fit

- `distiller.soft_labels_` — the collected soft label matrix `(n, n_classes)`.

#### When To Use

Recommended when the teacher has uneven confidence across samples — common with small datasets or near-boundary regions. Enabled by default alongside `confidence_weighting`.

---

### 5.3 Confidence Weighting

**Strategy:** Bell-shaped sample weighting that upweights high-confidence teacher predictions.

Each training sample receives a scalar weight based on the teacher's peak probability (max soft label value). Samples where the teacher is most confident receive higher weight; ambiguous near-boundary samples receive lower weight. This prevents noisy, uncertain teacher predictions from dominating the student's training signal.

#### How It Works

```
confidence(x) = max(soft_labels(x))
weight(x) = bell_curve(confidence(x))  # peaks at high confidence
```

Weights are passed as `sample_weight` to the student's `fit()` call.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence_weighting` | bool | `True` | Enable bell-shaped confidence weighting |

#### When To Use

Recommended for small datasets where teacher errors have an outsized effect. Enabled by default. Can be disabled if teacher quality is high and confidence is uniformly distributed.

---

### 5.4 K-Fold Cross-Prediction

**Strategy:** Out-of-fold soft label generation to prevent ICL identity leakage.

In-context learning models (TabPFN, TabICLv2) condition their predictions on the training set passed at inference time. If the full training set is passed and the model labels *its own* training examples, the soft labels are artificially sharp and overconfident — inflated by memorization rather than generalization. K-fold cross-prediction eliminates this by ensuring each sample's soft label is generated by a model that was **not** trained on it.

#### Algorithm

1. Split training data into `n_folds` stratified folds.
2. For each fold: fit a fresh teacher on the other `n_folds - 1` folds, predict on the held-out fold.
3. Assemble out-of-fold predictions into a full soft-label array `(n, n_classes)`.
4. Student trains on these OOF soft labels.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_folds` | int | `5` | Number of CV folds for cross-prediction |

#### When To Use

**Always enabled for inference-mode ICL teachers.** Automatically skipped for fine-tuned teachers (where the model doesn't memorize training data via in-context conditioning). Increasing `n_folds` reduces variance in soft labels at the cost of `n_folds` teacher fits.

---

### 5.5 Multi-Teacher Distillation

**Strategy:** Average soft labels across multiple TFM teachers before training the student.

When multiple teachers are provided, each teacher runs its own k-fold cross-prediction independently. The resulting soft label matrices are averaged element-wise before the student trains. This produces better-calibrated, lower-variance soft targets than any single teacher alone.

#### Example

```python
distiller = TabDistiller(
    teachers=["TabDPT", "TabICLv2"],
    student="lgbm",
    task_type="classification",
    n_folds=5,
)
distiller.fit(X_train, y_train)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `teachers` | str or list[str] | `"TabDPT"` | Single teacher name or list for multi-teacher |

#### When To Use

When two or more TFMs are available and their predictions are complementary. Compute cost scales linearly with the number of teachers × `n_folds`.

---

### 5.6 Data Augmentation

**Strategy:** Gaussian noise augmentation to expand the soft-label training set.

If `augment_factor` is set, the distiller creates `augment_factor × n` synthetic samples by adding small Gaussian noise (scaled to each feature's standard deviation) to randomly sampled training points. Soft labels and sample weights/temperatures for augmented samples are copied from their source samples.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `augment_factor` | float or None | `None` | Fraction of original dataset to augment (e.g. `0.5` adds 50% more samples) |

#### When To Use

Primarily useful for very small datasets (<200 samples). Disabled by default. Note: augmentation occurs *after* soft label collection, so teacher predictions are not re-run on synthetic samples.

---

## 6. Advanced Strategies

Available via `tabtune.distillation.strategies`:

### 6.1 Cross-Validation Distillation

Standalone function for generating OOF soft labels without instantiating a full `TabDistiller`. Useful when you want to experiment with different student architectures on pre-computed soft labels.

```python
from tabtune.distillation.strategies import cross_validate_soft_labels

soft_labels = cross_validate_soft_labels(
    teacher_name="TabDPT",
    X=X_train,
    y=y_train,
    task_type="classification",
    n_folds=5,
    random_state=42,
)
# soft_labels: ndarray (n, n_classes) — OOF predictions for every training sample
```

---

### 6.2 Progressive / Curriculum Distillation

Curriculum-based training: student first learns on the teacher's most confident predictions, then progressively expands to harder, more ambiguous samples.

```python
from tabtune.distillation.strategies import progressive_distillation

# After collecting soft labels:
student = progressive_distillation(
    distiller=distiller,
    X_train=X_train,
    y_train=y_train,
    soft_labels=soft_labels,
    n_stages=3,
    confidence_percentiles=[33.0, 66.0, 100.0],
)
```

**Stage breakdown (3 stages, default):**
1. Top 33% most confident samples only
2. Top 66% — includes medium-confidence samples
3. Full dataset

#### When To Use

Useful for MLP students on datasets with significant class imbalance or ambiguous boundary regions. GBDT students are less sensitive to curriculum ordering due to boosting's built-in focus on hard samples.

---

### 6.3 Probability Calibration

Post-hoc calibration of the distilled student's output probabilities via temperature scaling (Platt) or isotonic regression.

```python
from tabtune.distillation.strategies import calibrate_student

# Requires a held-out calibration split
T_star = calibrate_student(
    distiller=distiller,
    X_cal=X_cal,
    y_cal=y_cal,
    method="temperature",   # or "isotonic"
)
# T_star: optimal temperature (stored on student for subsequent predict_proba calls)
```

#### Methods

| Method | Description | When To Use |
|--------|-------------|-------------|
| `"temperature"` | Learn scalar `T*` minimising NLL on calibration set via L-BFGS-B | General purpose; low data |
| `"isotonic"` | Per-class isotonic regression calibration | Larger calibration sets (>500 samples) |

---

### 6.4 Self-Distillation

Train a model on its own soft predictions for additional generalization improvement — no larger teacher required.

```python
from tabtune.distillation.strategies import self_distill

distiller = self_distill(
    teacher_name="TabDPT",
    X_train=X_train,
    y_train=y_train,
    task_type="classification",
    n_rounds=2,
    student_type="lgbm",
)
```

**Round 1:** Standard distillation from TFM teacher.  
**Round 2:** Previous student's predictions used as soft targets for a fresh student.

---

## 7. TabDistiller API Reference

### Constructor

```python
TabDistiller(
    teachers: Union[str, list] = "TabDPT",
    student: str = "lgbm",
    task_type: str = "classification",
    temperature: float = 3.0,
    alpha: float = 0.7,
    n_folds: int = 5,
    adaptive_temperature: bool = True,
    confidence_weighting: bool = True,
    augment_factor: Optional[float] = None,
    student_params: Optional[dict] = None,
    teacher_params: Optional[dict] = None,
    device: str = "cpu",
    random_state: int = 42,
)
```

### Teacher Configuration

`teachers` accepts a string, list of strings, or list of pre-fitted `TabularPipeline` objects:

```python
# Single teacher
TabDistiller(teachers="TabDPT", ...)

# Multi-teacher (soft labels averaged)
TabDistiller(teachers=["TabDPT", "TabICLv2"], ...)

# Pre-fitted pipelines (skips teacher fit step)
pipe = TabularPipeline(model_name="TabPFN", ...)
pipe.fit(X_train, y_train)
TabDistiller(teachers=[pipe], ...)
```

### Student Backends

| `student` value | Backend | Notes |
|-----------------|---------|-------|
| `"lgbm"` | LightGBM | Recommended default: fast, robust, near-teacher accuracy |
| `"xgb"` / `"xgboost"` | XGBoost | Strong alternative; slightly slower than LightGBM |
| `"catboost"` / `"cb"` | CatBoost | Best on datasets with categorical features |
| `"mlp"` | PyTorch MLP | Use only when neural student is required; collapses on small/hard datasets |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(X_train, y_train)` | `self` | Fit teacher(s), collect soft labels, train student |
| `predict(X_test)` | `ndarray (n,)` | Student class predictions or regression values |
| `predict_proba(X_test)` | `ndarray (n, k)` | Student probability matrix (classification only) |
| `evaluate(X_test, y_test)` | `dict` | Student metrics only |
| `compare(X_test, y_test)` | `dict` | Student + teacher metrics + retention percentage |
| `save(path)` | — | Serialize student to disk (teacher stripped) |
| `TabDistiller.load(path)` | `TabDistiller` | Restore fitted distiller from disk |



### Fitted Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `teacher_pipelines_` | list | Fitted teacher `TabularPipeline` objects |
| `student_` | object | Fitted student (`MLPStudent` or `GBDTStudent`) |
| `soft_labels_` | ndarray | Collected soft labels `(n, n_classes)` |
| `n_classes_` | int | Number of classes (classification only) |
| `label_encoder_` | LabelEncoder | Extracted from teacher pipeline |
| `_fit_time` | float | Wall-clock time of `fit()` in seconds |

---

## 8. Strategy Selection Guide

| Strategy | Best For | Compute | Typical Retention |
|----------|----------|---------|-------------------|
| **Soft Labels + Adaptive T** | General purpose; recommended default | Medium (n_folds × teacher fits) | 97–99% |
| **Soft Labels (fixed T)** | When adaptive T adds no benefit; fast iteration | Medium | 96–98% |
| **Multi-Teacher** | Two+ available TFMs; noisy datasets | High (n_teachers × n_folds) | 98–99% |
| **Confidence Weighting** | Small datasets; high teacher variance | +0 overhead | +1–2% on hard samples |
| **Progressive/Curriculum** | Large datasets; MLP students; class imbalance | Medium + staging overhead | +1–3% on boundary samples |
| **Self-Distillation** | No larger teacher available | Low (student-only) | +1–2% generalisation |

### Student Selection

```
Need fast inference + near-teacher accuracy?
├── Tabular, mixed numeric/categorical → lgbm (default)
├── Pure numeric, regularisation needed → xgb
├── Significant categorical features → catboost
└── Must be neural / downstream fine-tuning → mlp
    └── Note: MLP collapses on small (<300 samples) or hard datasets.
        Use GBDT students for robustness.
```

---

## 9. Save & Deployment

After distillation, `save()` strips the teacher and serialises only the student. The loaded student runs inference with no TFM dependency — suitable for production environments without GPU.

```python
# Save (teacher stripped)
distiller.save("student_lgbm_v1.joblib")

# Reload & predict — no TFM dependency at runtime
distiller_prod = TabDistiller.load("student_lgbm_v1.joblib")
predictions = distiller_prod.predict(X_new)
probabilities = distiller_prod.predict_proba(X_new)
```

**Typical file sizes:**

| Student | Serialised Size |
|---------|----------------|
| LightGBM | ~50–500 KB |
| XGBoost | ~100–800 KB |
| CatBoost | ~200 KB – 2 MB |
| MLP | ~1–5 MB |

---

## 10. Error Handling & Logging

### Logging

The module uses Python's `logging` module with logger name `tabtune.distillation`. Set log level for detailed output:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Error Handling Design

- **Invalid `student` string** raises `ValueError` with supported values listed.
- **`predict()` / `predict_proba()` before `fit()`** raises `RuntimeError`.
- **`predict_proba()` on regression** raises `ValueError`.
- **`StratifiedKFold` failure** (too few samples per class) falls back automatically to `KFold` with a logged warning.
- **Teacher `predict_proba()` failure** during `compare()` is caught per-teacher — one failed teacher does not abort the comparison; error is logged and `{"error": str(e)}` returned for that teacher.
- **Soft label normalisation** — TabDPT floating-point bug produces rows that don't sum to 1. The distiller detects and renormalises automatically before student training.
- **Label encoder extraction** — supports TabPFN, TabICL, TabICLv2, TabDPT, Mitra, OrionMSP, OrionBiX, ContextTab, Limix. Unsupported models log a warning and continue.

### Verbose Logging Output

When `logging.INFO` is set, `TabDistiller` logs:
- Teacher fit start/complete per teacher
- K-fold progress (`K-fold 1/5 complete`, etc.)
- Soft labels shape after collection
- Adaptive temperature statistics (`mean`, `std`)
- Confidence weight statistics
- Augmentation summary (if enabled)
- Distillation total time

---

## 11. Testing

### Running Tests

```bash
# Strategy unit tests (no GPU needed):
pytest tests/test_distillation.py -k "TestDistillerUnit or TestSoftLabels or TestGBDTStudent" -v

# Full integration tests (GPU + TabTune installed):
pytest tests/test_distillation.py::TestFullDistillation -v

# Quick smoke test:
python tests/test_distillation.py
```

### Test Coverage

| Test Class | What It Tests | GPU Required |
|------------|--------------|-------------|
| `TestDistillerUnit` | Constructor, param validation, `_check_fitted` guard, student routing | No |
| `TestSoftLabels` | K-fold OOF shape, normalisation fix, direct prediction path | No |
| `TestGBDTStudent` | lgbm / xgb / catboost backends, fit/predict, classification + regression | No |
| `TestMLPStudent` | Fit/predict, early stopping, soft-label loss | No |
| `TestAdvancedStrategies` | `cross_validate_soft_labels`, `calibrate_student` (T + isotonic), `progressive_distillation` | No |
| `TestSaveLoad` | `save()` / `load()` roundtrip, student-only serialisation | No |
| `TestFullDistillation` | End-to-end with real TabTune pipelines: TabPFN → lgbm/xgb, multi-teacher | Yes |

---

## 12. References

| # | Paper | Relevance |
|---|-------|-----------|
| 1 | Hinton, G. et al. (2015). *Distilling the Knowledge in a Neural Network.* NeurIPS Workshop. | Soft-label KD loss, temperature scaling, `T²` gradient correction |
| 2 | Hollmann, N. et al. (2023). *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second.* ICLR. | Primary teacher model |
| 3 | Ye, H. et al. (2024). *TabICL: In-Context Learning for Tabular Data.* arXiv. | TabICLv2 teacher; ICL identity leakage motivation |
| 4 | Tanna, A. et al. (2025). *TabTune: A Unified Library for Inference and Fine-Tuning Tabular Foundation Models.* arXiv:2511.02802. | TabTune framework |
| 5 | Guo, C. et al. (2017). *On Calibration of Modern Neural Networks.* ICML. | Temperature calibration (`calibrate_student`) |
| 6 | Xu, Z. et al. (2020). *Data-Free Knowledge Distillation for Object Detection.* CVPR. | Self-distillation motivation |
| 7 | Bengio, Y. et al. (2009). *Curriculum Learning.* ICML. | Progressive distillation curriculum design |
