# Installation

TabTune needs **Python 3.10 or newer** and PyTorch 2.7 or newer. A GPU is optional for
inference and strongly recommended for fine-tuning.

```bash
git clone https://github.com/Lexsi-Labs/TabTune.git
cd TabTune
pip install -r requirements.txt
pip install -e .
```

That is the whole install. The model code for all 24 tabular models and all eleven time series
models is vendored under `tabtune/models/`, so there is no second package to install per
model. Checkpoints download from the Hugging Face Hub the first time you call `fit()` and
land in the usual Hugging Face cache.

## Optional extras

```bash
pip install "tabtune[distillation]"   # LightGBM students for tabtune.distillation
pip install "tabtune[serving]"        # ONNX export of distilled students
pip install "tabtune[interactive]"    # rich notebook display helpers
pip install "tabtune[colab]"          # pin core packages to the versions Colab ships
pip install "tabtune[docs]"           # build this site
pip install "tabtune[dev]"            # pytest, ruff, black, pre-commit
pip install "tabtune[toto2]"          # the Toto 2 time series model (Python 3.12+)
```

`toto2` is deliberately not part of the default install: its dependencies declare
`requires-python >= 3.12`, and installing them on 3.11 needs `--ignore-requires-python`.

## GPU

Install the PyTorch build that matches your CUDA version before TabTune, following the
[PyTorch instructions](https://pytorch.org/get-started/locally/). TabTune resolves the
device itself: `model_params={"device": "cuda"}` or `tuning_params={"device": "cuda"}`
when you want to be explicit, otherwise CUDA is used when it is available.

```python
import torch
print(torch.cuda.is_available())
```

## Gated and licensed weights

Some checkpoints need a Hugging Face token (ConTextTab is gated; TabPFN v2.6 and later
require accepting Prior Labs' terms). Log in once:

```bash
huggingface-cli login      # or: export HF_TOKEN=hf_...
```

Three models have particular setup needs, documented on their pages:

- [TabFM](../models/tabfm.md) is non-commercial and downloads from Google's release.
- [EXAONE Tabular](../models/exaone.md) ships classification weights only; regression needs
  a locally supplied checkpoint.
- [xRFM](../models/xrfm.md) has no pretrained weights and trains from scratch, which makes it
  the one model that works with no network at all.

Weight licences differ from code licences. The registry records each checkpoint's licence,
and `license_mode="commercial"` refuses the non-commercial ones before anything is
downloaded. See [Model Registry](../user-guide/registry.md).

## Verify it worked

```bash
python -c "import tabtune; print(tabtune.__version__)"
```

`import tabtune` is lazy and pulls in neither torch nor transformers, so this takes
milliseconds. A real smoke test:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tabtune import TabularPipeline

data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)

pipe = TabularPipeline(model_name="TabICL", task_type="classification", tuning_strategy="inference")
pipe.fit(X_train, y_train)
print(pipe.evaluate(X_test, y_test))
```

The first run downloads the TabICL checkpoint (about 100 MB). Everything after that is
cached.

## Google Colab

TabTune detects Colab and skips the optional IPython-heavy integrations so the install
succeeds with Colab's preinstalled packages. Use `pip install "tabtune[colab]"` to keep
NumPy, pandas and scikit-learn on the versions Colab expects.

## Environment variables

| Variable | Effect |
|---|---|
| `HF_TOKEN` | Hugging Face token for gated checkpoints. |
| `HF_HOME` | Where checkpoints are cached. |
| `TOKENIZERS_PARALLELISM=false` | Silences the tokenizer fork warning from ConTextTab's text encoder. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Reduces fragmentation on long fine-tuning runs. |
| `TABTUNE_MITRA_V2_CLS_REPO`, `TABTUNE_MITRA_V2_REG_REPO` | Override the Mitra v2 checkpoint repositories. |

## Next steps

- **[Quick Start](quickstart.md)**: fit, evaluate, save and reload a pipeline on a real
  dataset.
- **[Basic Concepts](basic-concepts.md)**: `TabularPipeline`, `DataProcessor`,
  `TuningManager` and the registry, before the User Guide starts assuming them.
