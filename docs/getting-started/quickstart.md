# Quick Start

This fits one model on a public dataset, scores it, saves the pipeline and reloads it. The
example runs on CPU; it is faster on a GPU and nothing in the code changes.

## Load a dataset

The Telco customer churn dataset from OpenML: 7,043 rows, 19 features, a yes/no target.

```python
import openml
from sklearn.model_selection import train_test_split

dataset = openml.datasets.get_dataset(42178)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

Pass the raw frame. Strings, categories and missing values are the `DataProcessor`'s
problem, and it handles them differently per model.

## Zero-shot inference

```python
from tabtune import TabularPipeline

pipe = TabularPipeline(
    model_name="TabICL",
    task_type="classification",
    tuning_strategy="inference",
)
pipe.fit(X_train, y_train)          # preprocessing only; the training set becomes the context
metrics = pipe.evaluate(X_test, y_test)
print(metrics)                      # accuracy, weighted F1, ROC AUC, log loss ...
```

`fit()` does no gradient step here. An in-context model reads the training rows as context
at prediction time, which is why zero-shot is a real baseline and not a placeholder.

## Fine-tune

```python
pipe = TabularPipeline(
    model_name="TabICL",
    task_type="classification",
    tuning_strategy="finetune",
    tuning_params={"epochs": 3, "learning_rate": 1e-5, "device": "cuda"},
)
pipe.fit(X_train, y_train)
print(pipe.evaluate(X_test, y_test))
```

`finetune` updates every parameter. The default `finetune_mode` is `meta-learning`:
episodes of support and query rows drawn from your training set, the same regime the model
was pretrained in. See [Tuning Strategies](../user-guide/tuning-strategies.md) for `sft`,
`turn_by_turn` and the vendors' `native` loops.

## LoRA instead

```python
pipe = TabularPipeline(
    model_name="TabICL",
    task_type="classification",
    tuning_strategy="peft",
    tuning_params={
        "epochs": 3,
        "learning_rate": 1e-4,
        "peft_config": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05},
    },
)
pipe.fit(X_train, y_train)
```

Same data, same API, a fraction of the trainable parameters. TabTune knows which layers to
wrap for each model; see [PEFT & LoRA](../advanced/peft-lora.md).

## Save and reload

```python
pipe.save("churn.joblib")
loaded = TabularPipeline.load("churn.joblib")
loaded.predict(X_test[:5])
```

The file carries the fitted preprocessor, the model weights (or the LoRA adapters) and the
configuration, so a loaded pipeline predicts without refitting.

## Check before you download

Every model has a registry entry with its limits and its weight licence. Ask first:

```python
from tabtune.registry import check_envelope, get_model_spec

spec = get_model_spec("TabPFN")
print(spec.envelope.describe())                # class, feature and row limits in one line
print(spec.license.commercial_use_ok)          # True

for v in check_envelope("TabPFN", n_rows=50_000, n_features=30, n_classes=2):
    print(v.severity, v.message)
```

Or let the pipeline refuse for you:

```python
TabularPipeline(model_name="TabFM", task_type="classification", license_mode="commercial")
# raises LicenseError: the TabFM weights are non-commercial
```

## Next steps

- **[Basic Concepts](basic-concepts.md)**: what the pipeline is made of.
- **[Models](../models/overview.md)**: which of the 24 fits your rows, columns, classes and
  licence.
- **[Leaderboard](../user-guide/leaderboard.md)**: run the comparison above over many
  models at once.
