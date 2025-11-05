# Orion BIX

Orion BIX (Biaxial Interaction eXpert) enhances scalable ICL with biaxial (row × column) interaction modules to better capture complex feature dependencies.

## Key ideas
- Biaxial interaction layers improve feature–feature and sample–sample coupling
- Robust ensembling across transformed feature orders
- Designed for accuracy on complex datasets while remaining scalable

## When to use
- Medium to large datasets (≥50K rows) with strong cross‑feature interactions
- Accuracy‑critical workloads where TabICL underfits

## Inference parameters
- `n_estimators` (int): ensemble size across transformed views
- `softmax_temperature` (float): post‑logit scaling
- `average_logits` (bool): average logits vs. probabilities
- `feat_shuffle_method` (str): feature permutation policy (e.g., `latin`)
- `norm_methods` (list[str]): normalization strategies per view
- `batch_size` (int): ensemble batch size

## Fine‑tuning parameters (via TuningManager)
- `epochs` (int): 3–8 typical
- `learning_rate` (float): 1e‑5 – 3e‑4 (AdamW)
- `support_size` / `query_size`: episodic context/query sizes
- `n_episodes` (int): total adaptation episodes
- `batch_size` (int): episode batch size

## Usage
```python
from tabtune import TabularPipeline

# Inference
pipe = TabularPipeline(
    model_name="OrionBix",
    task_type="classification",
    tuning_strategy="inference",
    model_params={
        "n_estimators": 24,
        "softmax_temperature": 0.9,
        "average_logits": True,
        "feat_shuffle_method": "latin",
        "norm_methods": ["none", "power"],
    },
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# PEFT fine‑tuning
peft_pipe = TabularPipeline(
    model_name="OrionBix",
    task_type="classification",
    tuning_strategy="peft",
    tuning_params={
        "epochs": 5,
        "learning_rate": 2e-5,
        "support_size": 1024,
        "query_size": 256,
        "batch_size": 8,
        "peft_config": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05},
    },
)
peft_pipe.fit(X_train, y_train)
```

## Notes & caveats
- Slightly higher memory than OrionMSP due to biaxial layers
- Increase `n_estimators` for more stable calibration; reduce for speed
- Ensure preprocessing/encodings are consistent between splits
