<p align="center">
  <img src="assets/vuugolelzvmgay7nhq.svg" alt="TabTune Logo" width="1000px"/>
</p>

# TabTune - A Unified Library for Inference and Fine-Tuning Tabular Foundation Models

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![Documentation](https://img.shields.io/badge/docs-available-green.svg)](https://github.com/Lexsi-Labs/TabTune)
[![arXiv](https://img.shields.io/badge/arXiv-2511.02802-b31b1b.svg)](https://arxiv.org/abs/2511.02802)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white)](https://discord.gg/dSB62Q7A)

A powerful and flexible Python library designed to simplify the **training and fine-tuning** of modern foundation models on tabular data.

Provides a **high-level, scikit-learn-compatible API** that abstracts away the complexities of data preprocessing and model-specific training loops, allowing you to focus on results.

---

## 🚀 Core Features

The library is built on **four main components** that work together seamlessly:

- **`DataProcessor`** -- A smart, model-aware data preparation engine.  
  Automatically handles imputation, scaling, and categorical encoding based on the requirements of the selected model (e.g., integer encoding for TabPFN, text embeddings for ContextTab).

- **`TuningManager`** -- The computational core of the library.  
  Manages the model adaptation process, applying the correct training strategy—whether it's _zero-shot inference_, _episodic fine-tuning_ for ICL models, or _full fine-tuning_ with optional PEFT (Parameter-Efficient Fine-Tuning).

- **`TabularPipeline`** -- The main user-facing object.  
  Provides simple yet efficient functionalities - `.fit()`, `.predict()`, `.evaluate()`, `.save()`, and `.load()` API that chains all components into a seamless, end-to-end experience.

- **`TabularLeaderboard`** -- A leaderboard utility for model comparison.  
  Makes it easy to compare multiple models and strategies on the same dataset splits with automatic ranking and metric reporting.

---

## 🤔 Why TabTune?

Using diverse tabular foundation models often requires writing model-specific boilerplate for data preparation, training, and inference. TabTune solves this by providing:

- **Unified API**: A single, consistent interface (`.fit()`, `.predict()`, `.evaluate()`) for multiple models like TabPFN, TabICL, Mitra, ContextTab, TabDPT, OrionMSP, and OrionBix.

- **Automated Preprocessing**: The DataProcessor is model-aware, automatically applying the correct transformations without manual configuration.

- **Flexible Fine-Tuning Strategies**: 
  - **Inference mode** for zero-shot predictions
  - **Meta-learning mode** for episodic fine-tuning (recommended for ICL models)
  - **Supervised Fine-Tuning (SFT)** for task-optimized learning
  - **PEFT mode** for parameter-efficient adaptation using LoRA adapters

- **Easy Model Comparison**: The TabularLeaderboard allows you to benchmark multiple models and strategies to quickly find the best performer.

- **Checkpoint Management**: Automatic saving and loading of fine-tuned model weights with support for resuming training.

---

## 📊 Supported Models

TabTune has built-in support for a growing list of powerful tabular models, each with its own specialized preprocessing and tuning pipeline handled automatically.

| Model       | Family / Paradigm | Key Innovation | Supported Strategies |
|-------------|------------------|----------------|----------------------|
| **TabPFN-v2**  | PFN / ICL        | Approximates Bayesian inference on synthetic data | Inference, Meta-Learning FT, SFT, PEFT* |
| **TabICL**  | Scalable ICL     | Two-stage column-then-row attention | Inference, Meta-Learning FT, SFT, PEFT |
| **OrionMSP** | Scalable ICL | Multi-Scale Sparse Attention for Tabular In-Context Learning | Inference, Meta-Learning FT, SFT, PEFT |
| **OrionBix** | Scalable ICL | Tabular BiAxial In-Context Learning with biaxial attention mechanism | Inference, Meta-Learning FT, SFT, PEFT |
| **Mitra**   | Scalable ICL        | 2D attention (row & column); mixed synthetic priors | Inference, Meta-Learning FT, SFT, PEFT |
| **ContextTab** | Semantics-Aware ICL | Modality-specific semantic embeddings | Inference, Full Fine-Tuning, PEFT* |
| **TabDPT**     | Denoising Transformer      | Denoising pre-training for feature representation | Inference, Meta-Learning FT, SFT, PEFT |

*Note: PEFT for ContextTab and TabPFN is experimental; 'base-ft' strategy is fully supported.*

---

## ⚙️ Installation

```bash
git clone https://github.com/Lexsi-Labs/TabTune.git
cd TabTune
pip install -r requirements.txt
pip install -e .
```

---

## ⚡ Quick Start: End-to-End Workflow

Here is a complete example of loading a dataset, fine-tuning a TabPFN model, saving the pipeline, and making predictions.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import openml
from tabtune.TabularPipeline.pipeline import TabularPipeline

# 1. Load a dataset from OpenML
dataset = openml.datasets.get_dataset(42178)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 2. Configure and Initialize the Pipeline
pipeline = TabularPipeline(
    model_name="TabPFN",
    task_type="classification",
    tuning_strategy="inference",  # or 'finetune', 'base-ft', 'peft'
    tuning_params={"device": "cpu"}
)

# 3. Fit the pipeline on the raw training data
pipeline.fit(X_train, y_train)

# 4. Save the fine-tuned pipeline
pipeline.save("fitted_pipeline.joblib")

# 5. Load the pipeline and make predictions on new data
loaded_pipeline = TabularPipeline.load("fitted_pipeline.joblib")
predictions = loaded_pipeline.predict(X_test)

# 6. Evaluate the pipeline
metrics = pipeline.evaluate(X_test, y_test)
print(metrics)
```

---

## 🎯 Tuning Strategies

TabTune provides multiple fine-tuning strategies to suit different use cases:

### Inference Mode
Zero-shot predictions without any training. The model uses its pre-trained weights directly on your data.

```python
pipeline = TabularPipeline(
    model_name="TabPFN",
    tuning_strategy="inference"
)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Base Fine-Tuning (`base-ft`)
Full parameter fine-tuning. Updates all model weights using task data.

- **Meta-Learning (default for ICL models)**: Episodic training that mimics the in-context learning paradigm
- **SFT (Supervised Fine-Tuning)**: Standard supervised training on batches

```python
pipeline = TabularPipeline(
    model_name="TabICL",
    tuning_strategy="finetune",  # Defaults to 'base-ft'
    tuning_params={
        "epochs": 5,
        "learning_rate": 1e-5,
        "finetune_mode": "meta-learning"  # or "sft"
    }
)
pipeline.fit(X_train, y_train)
```

### PEFT Mode (Parameter-Efficient Fine-Tuning)
Applies LoRA (Low-Rank Adaptation) adapters to only a subset of parameters, reducing memory and computation.

```python
pipeline = TabularPipeline(
    model_name="TabICL",
    tuning_strategy="peft",
    tuning_params={
        "epochs": 10,
        "learning_rate": 5e-5,
        "peft_config": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05
        }
    }
)
pipeline.fit(X_train, y_train)
```

**PEFT Support by Model**:
- ✅ **Full Support**: TabICL, OrionMSP, OrionBix, TabDPT, Mitra
- ⚠️ **Experimental**: ContextTab and TabPFN (may cause prediction issues; use 'base-ft' instead)

---

## 📊 Evaluation Metrics

When calling `.evaluate()`, TabTune computes the following metrics:

- **Accuracy** -- Fraction of correct predictions
- **Weighted F1 Score** -- Harmonic mean of precision and recall, weighted by class support
- **ROC AUC Score** -- Area under the Receiver Operating Characteristic curve (binary and multi-class supported)
- **Matthews Correlation Coefficient (MCC)** -- Correlation between predicted and actual values
- **Precision & Recall** -- Per-class performance metrics
- **Brier Score** -- Mean squared error of probabilistic predictions

```python
metrics = pipeline.evaluate(X_test, y_test)
print(metrics)
# Output: {'accuracy': 0.92, 'f1_score': 0.89, 'roc_auc_score': 0.95, ...}
```

---

## 🏆 Model Comparison with TabularLeaderboard

The `TabularLeaderboard` makes it easy to compare multiple models and strategies on the same dataset.

```python
from tabtune.TabularLeaderboard.leaderboard import TabularLeaderboard

# 1. Initialize the leaderboard with your data splits
leaderboard = TabularLeaderboard(X_train, X_test, y_train, y_test)

# 2. Add model configurations to compare
leaderboard.add_model(
    model_name='TabICL',
    tuning_strategy='inference',
    model_params={'n_estimators': 16}
)

leaderboard.add_model(
    model_name='TabICL',
    tuning_strategy='finetune',
    model_params={'n_estimators': 16},
    tuning_params={'epochs': 5, 'learning_rate': 1e-5, 'finetune_mode': 'meta-learning'}
)

leaderboard.add_model(
    model_name='TabPFN',
    tuning_strategy='inference'
)

# 3. Run the benchmark and display ranked results
leaderboard.run()
```

---

## 🛠️ API Reference

### TabularPipeline Constructor

```python
TabularPipeline(
    model_name: str,
    task_type: str = 'classification',
    tuning_strategy: str = 'inference',
    tuning_params: dict | None = None,
    processor_params: dict | None = None,
    model_params: dict | None = None,
    model_checkpoint_path: str | None = None,
    finetune_mode: str = 'meta-learning'
)
```

#### Key Parameters:

- **`model_name`** (str): The name of the model to use (e.g., `'TabPFN'`, `'TabICL'`, `'ContextTab'`, `'Mitra'`, `'TabDPT'`, `'OrionMSP'`, `'OrionBix'`).

- **`task_type`** (str): The type of task, either `'classification'` or `'regression'` (currently only classification is fully supported).

- **`tuning_strategy`** (str): The strategy for model adaptation (`'inference'`, `'finetune'`, `'base-ft'`, or `'peft'`).

- **`tuning_params`** (dict, optional): Parameters for the `TuningManager`:
  - `epochs` (int): Number of training epochs
  - `learning_rate` (float): Learning rate for optimization
  - `batch_size` (int): Batch size for fine-tuning
  - `device` (str): 'cuda' or 'cpu'
  - `save_checkpoint_path` (str): Path to save fine-tuned weights
  - `checkpoint_dir` (str): Directory for automatic checkpoint saving
  - `finetune_mode` (str): 'meta-learning' or 'sft' (episodic vs. supervised)
  - `peft_config` (dict): Configuration for LoRA adapters
  - `show_progress` (bool): Whether to show progress bars

- **`processor_params`** (dict, optional): Parameters for the `DataProcessor`:
  - `imputation_strategy` (str): 'mean', 'median', 'iterative', 'knn'
  - `categorical_encoding` (str): 'onehot', 'ordinal', 'target', 'hashing', 'binary'
  - `scaling_strategy` (str): 'standard', 'minmax', 'robust', 'power_transform'
  - `resampling_strategy` (str): 'smote', 'random_over', 'random_under', 'tomek', 'kmeans', 'knn'
  - `feature_selection_strategy` (str): 'variance', 'select_k_best_anova', 'select_k_best_chi2'

- **`model_params`** (dict, optional): Model-specific parameters.

- **`model_checkpoint_path`** (str, optional): Path to a `.pt` file containing pre-trained model weights.

- **`finetune_mode`** (str, optional): Default fine-tuning mode. Can be overridden in `tuning_params`.

---

## 💾 Checkpoint Management

### Automatic Checkpoint Saving

Fine-tuned models are automatically saved during training:

```python
tuning_params = {
    'save_checkpoint_path': './checkpoints/my_model.pt',
    'checkpoint_dir': './checkpoints'  # Used if save_checkpoint_path is None
}
```

### Manual Checkpoint Loading

```python
# Load pre-trained weights when initializing
pipeline = TabularPipeline(
    model_name="TabPFN",
    model_checkpoint_path="./checkpoints/pretrained.pt"
)
```

### Pipeline Serialization

```python
# Save entire pipeline
pipeline.save("my_pipeline.joblib")

# Load and use
loaded_pipeline = TabularPipeline.load("my_pipeline.joblib")
predictions = loaded_pipeline.predict(X_test)
```

---

## 🔧 PEFT/LoRA Configuration

LoRA (Low-Rank Adaptation) adapters can significantly reduce memory usage during fine-tuning.

```python
peft_config = {
    'r': 8,                   # LoRA rank (lower = fewer parameters)
    'lora_alpha': 16,         # Scaling factor for LoRA updates
    'lora_dropout': 0.05,     # Dropout in LoRA modules
    'target_modules': None    # Auto-detect by model (optional override)
}

pipeline = TabularPipeline(
    model_name="TabICL",
    tuning_strategy="peft",
    tuning_params={
        'epochs': 10,
        'learning_rate': 5e-5,
        'peft_config': peft_config
    }
)
```

**Memory Savings**: PEFT typically reduces memory usage by 60-80% compared to full fine-tuning.

---

## 📚 Example Notebooks

Below are 9 Example Notebooks showcasing all the features of the Library in-depth!

| Serial No. | Name | Task Performed | Link To Notebook |
|---|------|------|------|
| 1 | Unified API | Showcasing A Unified API Across Multiple Models |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1KcaSdYRjZnMlb0MLmQ5IlnbPDiuEr1Ld?usp=sharing) |
| 2 |  Automated Model-Aware Preprocessing | The Automated preprocessing system explained |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/12BQ12VJrxtTDslgjnjm26yi3a0PYXqZT?usp=sharing) |
| 3 | Fine-Tuning Strategies | TabTune's four fine-tuning strategies |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1QixfiNCjF1IQV9NooMipPpnH4ETcEQwg?usp=sharing) |
| 4 | Model Comparison | Model Comparison with TabularLeaderboard |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1PZW3iPQOvwh0kroGytMzYTGc6ZVUzuvg?usp=sharing) |
| 5 | Checkpoint Management | Checkpoint Management - Save/Load Pipelines |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1DBTGEPpYLJjU9Aj7lzHoX3JtwaNOC0jn?usp=sharing) |
| 6 | Advanced Usage | PEFT Configuration and Hybrid Strategies |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1V3XGLeKrXSJwavaULMncZiM7uVE8sz0h?usp=sharing) |
| 7 | Data Sampling |  Data Sampling and Resampling Strategies for Inference |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1TUwxsfk6E0LDepc3bONeDZLslYAMesbZ?usp=sharing) |
| 8 | Evaluation Metrics | Evaluation Metrics involved |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/18TxyTyBGAGrIVf6zLjURDChG0vM4V02M?usp=sharing) |
| 9 | Benchmarking | Standard Benchmarking Techniques |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1lcoVMPz_3X5_5taNdB9doTGoN05krNRw?usp=sharing) |



---

## 🚀 Advanced Usage

### Custom Preprocessing

Override default preprocessing for specific needs:

```python
processor_params = {
    'imputation_strategy': 'iterative',
    'categorical_encoding': 'target',
    'scaling_strategy': 'robust',
    'resampling_strategy': 'smote'
}

pipeline = TabularPipeline(
    model_name="TabICL",
    processor_params=processor_params
)
```

### Hybrid Fine-Tuning

Combine meta-learning with PEFT for optimal results:

```python
pipeline = TabularPipeline(
    model_name="TabICL",
    tuning_strategy="peft",
    tuning_params={
        'epochs': 20,
        'learning_rate': 1e-5,
        'finetune_mode': 'meta-learning',
        'peft_config': {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1
        }
    }
)
```

---

## 📖 Documentation

For detailed documentation, API reference, model configurations, and usage examples, please visit: **[Documentation](https://tabtune.lexsi.ai/)**

---

## 🙏 Acknowledgments

TabTune is built upon the excellent work of the following projects and research teams:


- **[OrionMSP](https://github.com/Lexsi-Labs/OrionMSP)** - Multi-Scale Sparse Attention for Tabular In-Context Learning
- **[OrionBix](https://github.com/Lexsi-Labs/OrionBix)** - Tabular BiAxial In-Context Learnin
- **[TabPFN](https://github.com/PriorLabs/TabPFN)** - Prior-data Fitted Networks for tabular data
- **[TabICL](https://github.com/soda-inria/tabicl)** - Tabular In-Context Learning with scalable attention
- **[Mitra (Tab2D)](https://github.com/autogluon/autogluon)** - 2D Attention mechanism (Tab2D) for tabular data, included within AutoGluon
- **[ContextTab](https://github.com/SAP-samples/contexttab)** - Semantics-Aware In-Context Learning for Tabular Data
- **[TabDPT](https://github.com/layer6ai-labs/TabDPT-inference)** - Denoising Pre-training Transformer for Tabular Data
- **[AutoGluon](https://github.com/autogluon/autogluon)** - AutoML framework that inspired our unified API design

---

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `batch_size` in `tuning_params`
- Use `tuning_strategy='peft'` for PEFT mode
- Decrease `n_ensembles` or `context_size` for inference

### PEFT Compatibility Issues
- Some models have experimental PEFT support; use 'base-ft' strategy instead
- Check logs for model-specific warnings

### Device Mismatch
- Ensure `device` parameter matches your hardware (cuda/cpu)
- Use `torch.cuda.is_available()` to check GPU availability

---

## 🗃️ License

This project is released under the MIT License.  
Please cite appropriately if used in academic or production projects.

**Citation:**

```bibtex
@misc{tanna2025tabtuneunifiedlibraryinference,
      title={TabTune: A Unified Library for Inference and Fine-Tuning Tabular Foundation Models}, 
      author={Aditya Tanna and Pratinav Seth and Mohamed Bouadi and Utsav Avaiya and Vinay Kumar Sankarapu},
      year={2025},
      eprint={2511.02802},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.02802}, 
}
```

---

## 📫 Join Community / Contribute

- Issues and discussions are welcomed on the [GitHub issue tracker](https://github.com/Lexsi-Labs/TabTune/issues) and [Discord](https://discord.gg/dSB62Q7A) .
- Please see the **Contributing** section for contribution standards, code reviews, and documentation tips.

---
## Contact

<p align="center">
  <img src="assets/lexsi-labs-logo-white.svg" alt="Lexsi Labs Logo" width="200px"/>
</p>

Website: **https://lexsi.ai**
