# TabLDM

TabLDM is Xiaomi's tabular foundation model: dual-stream feature grouping, a lightweight
attention residual and a sparse mixture of experts, pretrained only on tables generated
from structural causal models. TabTune vendors `Xiaomi-TabLDM` 0.1.0 with its ten absolute
imports rewritten and adds fine-tuning, LoRA and the constructor conventions in
`tabtune_support.py`.

| | |
|---|---|
| `model_name` | `"TabLDM"` (aliases `Xiaomi-TabLDM`, `XiaomiTabLDM`, `TabLDM-v1`) |
| Classification | `inference`, `finetune`, `peft` |
| Regression | `inference`, `finetune`, `peft` |
| `finetune_mode` | `meta-learning` only |
| Limits | none declared |
| Reads natively | NaN, categorical |
| Weights | [`occams/Xiaomi-TabLDM`](https://huggingface.co/occams/Xiaomi-TabLDM) |
| Paper | [arXiv:2609.03880](https://arxiv.org/abs/2609.03880) |

## Usage

```python
from tabtune import TabularPipeline

pipe = TabularPipeline("TabLDM", task_type="classification", tuning_strategy="inference")
pipe.fit(X_train, y_train)

pipe = TabularPipeline("TabLDM", task_type="classification", tuning_strategy="finetune",
                       tuning_params={"epochs": 5, "learning_rate": 1e-5})
pipe.fit(X_train, y_train)
```

## Fine-tuning

Unlike most inference-only releases, TabLDM kept its supervised entry point:
`TabLDM.forward` branches on `self.training`, and the training branch maps
`X (B, T+Q, H)` plus `y_train (B, T)` to `(B, Q, out_dim)` with gradients intact. Only the
call sites in the sklearn wrapper are under `torch.no_grad()`. So TabTune's fine-tuning
drives the vendor's own training path rather than a reconstruction of it.

Episodes are drawn from the fitted `EnsembleGenerator`, which is the same preprocessing the
model sees at prediction time. Each episode picks one (normalisation method, ensemble
member) pair, so the weights adapt across the preprocessing distribution they are used
with and not just one member's.

Parameters: `epochs`, `steps_per_epoch` (alias `episodes_per_epoch`), `learning_rate`,
`support_size`, `query_size`, `grad_clip` (alias `grad_clip_value`), `seed` (alias
`random_state`), `peft_config` for `peft`. LoRA targets `linear1`, `linear2`,
`attn_res_projs`, `in_linear`, `ssmax_layer`, `decoder` and `out_proj`; the `y_encoder`
and the MoE `router` are excluded.

## Licence

The code is Apache-2.0. The weight licence on the Hub conflicts with the repository's
stated terms, so the registry records `commercial_use_ok=None` (unverified) and
`license_mode="commercial"` warns rather than refuses. Check the terms yourself before
shipping.
