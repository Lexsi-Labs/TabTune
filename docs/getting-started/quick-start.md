# Quick Start

This quick start guide demonstrates how to run a complete end-to-end workflow with TabTune in just a few steps.

---

## 1. Prepare Your Environment

Ensure you have installed TabTune and its dependencies as per the [Installation Guide](installation.md).

Activate your virtual environment:
```bash
# If using venv
source tabtune-env/bin/activate

# If using conda
conda activate tabtune
``` 

---

## 2. Load a Dataset

We use the **Telco Customer Churn** dataset from OpenML for this example.

```python
import openml
from sklearn.model_selection import train_test_split

# Load dataset
dataset = openml.datasets.get_dataset(42178)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
```

---

## 3. Initialize and Configure the Pipeline

Use the `TabularPipeline` class to define your model, task type, and tuning strategy.

```python
from tabtune import TabularPipeline

# Base fine-tuning example
pipeline = TabularPipeline(
    model_name="TabPFN",
    task_type="classification",
    tuning_strategy="base-ft",
    tuning_params={
        "device": "cuda",         # or "cpu" if GPU unavailable
        "epochs": 3,
        "learning_rate": 2e-5,
        "show_progress": True
    }
)
```

---

## 4. Fit the Model

Train your pipeline on the training data.

```python
pipeline.fit(X_train, y_train)
```

During training, TabTune will automatically handle data preprocessing and apply the chosen tuning strategy.

---

## 5. Evaluate and Predict

After training, evaluate performance and generate predictions:

```python
# Evaluate on test set
metrics = pipeline.evaluate(X_test, y_test)
print("Evaluation metrics:", metrics)

# Make predictions
predictions = pipeline.predict(X_test)
```

Supported metrics include: **Accuracy**, **Weighted F1 Score**, and **ROC AUC Score**.

---

## 6. Save and Load the Pipeline

Persist your trained pipeline for later use:

```python
# Save to disk
pipeline.save("tabtune_pipeline.joblib")

# Load from disk
loaded_pipeline = TabularPipeline.load("tabtune_pipeline.joblib")
results = loaded_pipeline.predict(X_test)
```

---

## 7. Try PEFT (LoRA) Strategy

Switch to parameter-efficient fine-tuning with minimal code changes:

```python
# PEFT fine-tuning
peft_pipeline = TabularPipeline(
    model_name="TabPFN",
    task_type="classification",
    tuning_strategy="peft",
    tuning_params={
        "device": "cuda",
        "epochs": 3,
        "learning_rate": 2e-4,
        "peft_config": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05}
    }
)
peft_pipeline.fit(X_train, y_train)
metrics_peft = peft_pipeline.evaluate(X_test, y_test)
print("PEFT metrics:", metrics_peft)
```

---

### Next Steps
- Explore advanced configurations in the [User Guide](../user-guide/pipeline-overview.md)
- Compare multiple models with the [TabularLeaderboard](../user-guide/leaderboard.md)
- Dive into PEFT internals in [PEFT & LoRA](../advanced/peft-lora.md)
