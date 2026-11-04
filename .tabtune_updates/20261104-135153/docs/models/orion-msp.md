# OrionMSP

OrionMSP is a scalable in-context learning model that leverages multi‑scale synthetic priors and episodic adaptation for robust tabular classification.

## Key ideas
- Multi‑scale synthetic priors to initialize inductive biases
- Column‑then‑row attention for efficient context reasoning
- Ensemble over transformed views for stability
- Works with standard tabular dtypes: numeric, categorical (after encoding)

## When to use
- Datasets from 50K to multi‑million rows where ICL scales better than PFN
- Strong generalization with modest fine‑tuning budget

## Inference parameters
- `n_estimators` (int): number of transformed views to ensemble
- `softmax_temperature` (float): post‑logit scaling (0.5–1.5)
- `average_logits` (bool): average logits vs. probabilities
- `batch_size` (int): batch size for ensembling

## Fine‑tuning parameters (via TuningManager)
- `epochs` (int): number of epochs (typ. 3–10)
- `learning_rate` (float): AdamW LR (1e‑5 – 5e‑4)
- `support_size` (int): context samples per episode (e.g., 512–4096)
- `query_size` (int): query samples per episode (e.g., 128–512)
- `n_episodes` (int): total adaptation episodes
- `batch_size` (int): episode batch size

## Usage
```python
from tabtune import TabularPipeline

# Inference / zero-shot
pipe = TabularPipeline(
    model_name="OrionMSP",
    task_type="classification",
    tuning_strategy="inference",
    model_params={
        "n_estimators": 16,
        "softmax_temperature": 0.9,
        "average_logits": True,
    },
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Base fine-tuning
ft_pipe = TabularPipeline(
    model_name="OrionMSP",
    task_type="classification",
    tuning_strategy="base-ft",
    tuning_params={
        "epochs": 5,
        "learning_rate": 2e-5,
        "support_size": 1024,
        "query_size": 256,
        "batch_size": 8,
    },
)
ft_pipe.fit(X_train, y_train)
```

## PEFT (LoRA)
- Supported; target linear projections in attention and MLPs are typical.
- Start with `r=8, lora_alpha=16, lora_dropout=0.05`.

## Notes & caveats
- Ensure categorical encoding is consistent between train/test splits.
- For very small datasets (<20K), consider TabPFN or TabICL instead.
- Increase `n_estimators` for stability on noisy datasets.