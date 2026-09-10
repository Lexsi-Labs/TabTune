# Model Registry: Envelopes and Licensing

*New in TabTune 0.2.0.*

The registry is a **torch-free** description of every model TabTune ships. It exists to answer
two questions that previously required downloading a multi-gigabyte checkpoint first:

1. **Will this model accept my data?** — capability envelopes encode the architectural limits
   of each pretrained checkpoint (TabFM's ten-class output head, TabPFN-v3's cell budget,
   Mitra's practical row ceiling), so a mismatch raises a clear message in milliseconds
   instead of an out-of-memory error after a long download.
2. **Can I actually deploy this?** — weight licenses vary sharply, from MIT through CC-BY to
   research-only. The registry records them and can fail fast when the intended use is
   commercial.

Because it pulls in nothing beyond the standard library, `import tabtune` and
`from tabtune.registry import list_models` cost milliseconds and never load torch.

---

## 1. Discovery

```python
from tabtune.registry import list_models, list_model_names, models_dataframe

list_model_names()
# ['TabPFN', 'TabPFNv26', 'TabPFNv3', 'TabICL', 'TabICLv2', 'OrionMSP',
#  'OrionMSPv1.5', 'OrionBix', 'Mitra', 'ContextTab', 'TabDPT', 'Limix',
#  'TabFM', 'XRFM', 'ILTM', 'EXAONETabular']

# Filter by task, strategy, family or licence
for spec in list_models(task="regression", commercial_ok=True):
    print(spec.name, spec.license.name)

# A tidy pandas table for reports and model cards
models_dataframe()[["Model", "Tasks", "Max classes", "License", "Commercial"]]
```

`list_models()` accepts:

| Argument | Type | Meaning |
|---|---|---|
| `task` | `'classification'` / `'regression'` | Model has a head for this task |
| `strategy` | `'inference'` / `'finetune'` / `'peft'` | Model implements this strategy |
| `commercial_ok` | `bool` | Filter on the **weight** licence |
| `family` | `str` | e.g. `'pfn'`, `'icl'`, `'hypernetwork'` |
| `include_unverified_licenses` | `bool` | Include tri-state `None` licences when filtering on `commercial_ok=True` |

---

## 2. Name resolution is forgiving

Lookup ignores case, hyphens, underscores, dots and whitespace, because people copy model
names out of papers, READMEs and Slack messages.

```python
from tabtune.registry import resolve_model_name

resolve_model_name("tabpfn-v2.6")    # 'TabPFNv26'
resolve_model_name("TABPFN_V26")     # 'TabPFNv26'
resolve_model_name("SAP-RPT-1-OSS")  # 'ContextTab'   (upstream rename)
resolve_model_name("Orion-MSP")      # 'OrionMSP'
resolve_model_name("Tab2D")          # 'Mitra'
```

An unknown name raises `ModelNotFoundError` with a suggested match rather than a bare
`KeyError`.

---

## 3. Capability envelopes

A `CapabilityEnvelope` records the limits of a **pretrained checkpoint**. `None` means "no
declared limit" and is deliberately distinct from a large number — TabTune does not invent
limits it has not verified.

```python
from tabtune.registry import check_envelope, get_model_spec

spec = get_model_spec("TabFM")
spec.envelope.describe()
# '<=10 classes; <=500 features'
```

### 3.1 Fields

| Field | Meaning | Severity when exceeded |
|---|---|---|
| `max_classes` | Output-head capacity | **error** (architectural) |
| `min_rows` | Minimum training rows | **error** (architectural) |
| `max_features` | Documented feature ceiling | warn |
| `max_rows` | Documented row ceiling | warn |
| `max_cells` | `rows × features` budget | warn |
| `native_nan` / `native_text` / `native_categorical` | First-class handling, no encoding needed | — |
| `notes` | Free-text caveats | — |

`max_cells` is checked *in addition to* `max_rows` and `max_features`, not instead of them:
several recent checkpoints (TabPFN-v3 in particular) advertise a budget that trades rows
against features, so a single scalar row cap would misrepresent them.

### 3.2 Checking a dataset

```python
# Hard limit — raises EnvelopeError in microseconds, before any weights are fetched
check_envelope("TabFM", n_rows=1_000, n_features=20, n_classes=14)
# EnvelopeError: TabFM supports at most 10 classes (found 14)

# Soft limit — warns, run still meaningful
for v in check_envelope("Mitra", n_rows=50_000, n_features=10, mode="warn"):
    print(f"[{v.severity}] {v.message}")
# [warn] is documented up to 10,000 rows (found 50,000); consider context sampling ...
```

Each violation is an `EnvelopeViolation` with `constraint`, `limit`, `actual`, `severity`
and a human-readable `message`. Hard violations are sorted first.

!!! warning "Hard constraints raise even under `mode='warn'`"
    `max_classes` and `min_rows` are architectural: no amount of extra memory fixes them.
    They raise unless `envelope_mode='ignore'`.

---

## 4. Weight licensing

`LicenseSpec.commercial_use_ok` is **tri-state** on purpose:

| Value | Meaning | `license_mode='commercial'` behaviour |
|---|---|---|
| `True` | Upstream explicitly permits commercial use | passes |
| `False` | Upstream forbids it, or requires a separate grant | raises `LicenseError` |
| `None` | TabTune has not verified it | warns, does not block |

Inventing a restriction is as wrong as ignoring one, so unverified terms warn and point you
upstream rather than blocking your work.

```python
from tabtune.registry import check_license, get_model_spec

get_model_spec("Mitra").license.badge      # 'yes (attribution)'
get_model_spec("TabPFNv3").license.badge   # 'no'
get_model_spec("TabPFN").license.badge     # 'unverified'

check_license("TabPFNv3", "commercial")    # raises LicenseError, suggests alternatives
```

### 4.1 Licence summary

| Model | Weight licence | Commercial |
|---|---|---|
| TabPFN, TabPFNv2.6 | Prior Labs License | unverified |
| TabPFNv3 | TABPFN-3.0 License v1.0 | **no** (research / internal evaluation only) |
| TabICL, TabICLv2 | BSD-3-Clause | yes |
| OrionMSP, OrionMSPv1.5, OrionBix | MIT | yes |
| Mitra | CC-BY-4.0 | yes (attribution required) |
| ContextTab (SAP-RPT-1-OSS) | SAP-RPT-1-OSS | **no** (research use) |
| TabDPT | see upstream | unverified |
| LimiX | LimiX (academic use free) | **no** without authorization |
| TabFM | TabFM Non-Commercial License v1.0 | **no** (code is Apache-2.0, weights are not) |
| xRFM | MIT | yes (no weights to license) |
| iLTM | Apache-2.0 | yes (attribution) |
| EXAONE Tabular | EXAONE AI Model License 1.1 - NC | **no** (code is BSD-3-Clause-LG, weights are not) |

When a check fails, `commercial_alternatives` on the spec names models you can ship instead
— by default Mitra, TabICLv2, OrionMSP, OrionMSPv1.5 and OrionBix.

---

## 5. Enforcement inside the pipeline

`TabularPipeline` runs all three checks — task/strategy validity, envelope, licence — before
loading any weights.

```python
from tabtune import TabularPipeline

pipeline = TabularPipeline(
    model_name="TabFM",
    task_type="classification",
    envelope_mode="error",        # 'error' | 'warn' (default) | 'ignore'
    license_mode="commercial",    # 'research' (default) | 'commercial' | 'ignore'
    validate=True,                # default; set False for an unregistered model
)
```

| Parameter | Values | Effect |
|---|---|---|
| `envelope_mode` | `'error'`, `'warn'`, `'ignore'` | How soft limit violations are reported. Hard limits raise unless `'ignore'`. |
| `license_mode` | `'research'`, `'commercial'`, `'ignore'` | `'commercial'` fails fast on weights that forbid commercial use. |
| `validate` | `bool` | Check model / task / strategy against the registry before loading weights. |

---

## 6. Exceptions

All registry errors derive from `TabTuneError`:

```python
from tabtune.registry import (
    TabTuneError,
    ConfigError,
    ModelNotFoundError,
    UnsupportedTaskError,
    UnsupportedStrategyError,
    EnvelopeError,
    LicenseError,
)
```

| Exception | Raised when |
|---|---|
| `ModelNotFoundError` | Name does not resolve to any registered model |
| `UnsupportedTaskError` | Model has no head for `task_type` |
| `UnsupportedStrategyError` | Model does not implement `tuning_strategy` |
| `EnvelopeError` | Data violates a hard architectural limit |
| `LicenseError` | Weight licence forbids the intended use under `license_mode` |
| `ConfigError` | Invalid configuration |

---

## 7. Registering your own model

Adding a model to TabTune starts with a `ModelSpec` — the registry, validation, error
messages, model cards and generated documentation tables all pick it up automatically.

```python
from tabtune.registry import register_model, CapabilityEnvelope, LicenseSpec, ModelSpec

register_model(ModelSpec(
    name="MyModel",
    family="icl",
    aliases=("my-model",),
    classification_strategies=frozenset({"inference", "finetune"}),
    finetune_modes=frozenset({"sft"}),
    preprocessor_key="tabicl_special",
    envelope=CapabilityEnvelope(max_classes=20, native_nan=True),
    license=LicenseSpec(name="MIT", commercial_use_ok=True),
))
```

Or skip the registry entirely for a one-off:

```python
TabularPipeline("MyModel", validate=False)
```

---

## See Also

- [Typed Configuration](configuration.md) — `validate_against_registry()` in CI
- [Model Overview](../models/overview.md) — all 16 models compared
- Runnable example: `examples/14_registry_and_licensing.py`
