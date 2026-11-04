# Mitra v2

Mitra v2 is AWS's second set of Mitra weights on the same Tab2D architecture: 2D attention
over rows and columns, pretrained on mixed synthetic priors. Because the architecture is
shared, a v2 model *is* a `Tab2D`, and everything TabTune built for [Mitra](mitra.md)
(preprocessing, the episodic loops, LoRA targets) applies unchanged. Only the checkpoint
repositories differ.

| | |
|---|---|
| `model_name` | `"MitraV2"` (aliases `Mitra-2`, `Mitra2`, `mitra-classifier-2`, `mitra-regressor-2`) |
| Classification | `inference`, `finetune`, `peft` |
| Regression | `inference`, `finetune` |
| `finetune_mode` | `meta-learning`, `sft`, `turn_by_turn` |
| Limits | none declared (v1 declares 10,000 rows) |
| Reads natively | nothing; TabTune imputes and encodes |
| Weights | [`autogluon/mitra-classifier-2`](https://huggingface.co/autogluon/mitra-classifier-2), [`autogluon/mitra-regressor-2`](https://huggingface.co/autogluon/mitra-regressor-2), Apache-2.0 |
| Paper | [arXiv:2510.21204](https://arxiv.org/abs/2510.21204) |

## Usage

```python
from tabtune import TabularPipeline

pipe = TabularPipeline("MitraV2", task_type="classification", tuning_strategy="inference")
pipe.fit(X_train, y_train)

reg = TabularPipeline("MitraV2", task_type="regression", tuning_strategy="finetune",
                      tuning_params={"epochs": 5})
reg.fit(X_train, y_train)
```

Every parameter on the [Mitra](mitra.md) page applies. The checkpoint repositories can be
overridden with the `TABTUNE_MITRA_V2_CLS_REPO` and `TABTUNE_MITRA_V2_REG_REPO`
environment variables, which exists so a mirror can be used without a code change.

## Licence

Apache-2.0 for code and weights. `commercial_use_ok=True`.
