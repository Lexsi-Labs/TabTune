<div style="text-align: center;">
  <img src="assets/tabtune.svg" alt="TabTune Logo" style="width: 600px; height: auto;" />
</div>


---

**TabTune** is a powerful and flexible Python library designed to simplify the training and fine-tuning of modern foundation models on tabular data.  
It provides a high-level, scikit-learn-compatible API that abstracts away the complexities of data preprocessing, model-specific training loops, and benchmarking, letting you focus on delivering results.

Whether you are a practitioner aiming for production-grade pipelines or a researcher exploring advanced architectures, TabTune streamlines your workflow for tabular deep learning.

---

## 🚀 TabTune v1.0 - First Release

**Welcome to TabTune! This initial release provides a complete, production-ready framework for tabular foundation models.**

**Core Components:**

- **Unified API (`TabularPipeline`):** Single, scikit-learn-compatible interface for all models with `.fit()`, `.predict()`, `.save()`, and `.load()` methods.
- **Smart Data Processing (`DataProcessor`):** Model-aware preprocessing that automatically handles imputation, scaling, categorical encoding, and feature transformations for each model.
- **Flexible Tuning (`TuningManager`):** Three tuning strategies—zero-shot `inference`, supervised fine-tuning (`base-ft`) with full parameter updates, and memory-efficient `peft` (LoRA) adapters. Supports episodic meta-learning for ICL models.
- **Model Comparison (`TabularLeaderboard`):** Systematic benchmarking tool for comparing multiple models and strategies on your datasets.

**Supported Models (7 Models):**

- **TabPFN-v2:** Fast Bayesian inference for small datasets (<10K rows) with experimental PEFT support.
- **TabICL:** Scalable ICL with two-stage attention, full PEFT support, ideal for 10K-1M row datasets.
- **OrionMSP:** Multi-scale prior ICL for balanced generalization on 50K-2M+ row datasets, full PEFT support.
- **OrionBix:** Biaxial interaction expert for high-accuracy scenarios (50K-2M+ rows), full PEFT support.
- **TabDPT:** Denoising transformer for very large datasets (100K-5M rows), full PEFT support.
- **Mitra:** 2D cross-attention model for complex patterns and mixed data types, full PEFT support.
- **ContextTab:** Semantics-aware ICL with text embedding integration, experimental PEFT support.

**Key Capabilities:**

- ✅ **Multiple Training Paradigms:** Supports supervised fine-tuning (SFT) with full parameter updates, episodic meta-learning for in-context learning models, and parameter-efficient PEFT strategies.
- ✅ **PEFT (LoRA) Support:** Parameter-efficient fine-tuning for 5 out of 7 models (TabICL, OrionMSP, OrionBix, TabDPT, Mitra) with full support.
- ✅ **Meta-Learning Integration:** Episodic training with support/query sets for ICL models (TabICL, OrionMSP, OrionBix, Mitra) enabling fast task adaptation.
- ✅ **Comprehensive Documentation:** Extensive guides, API references, troubleshooting, and model-specific documentation.
- ✅ **Production Ready:** Model serialization, reproducible training, and deployment-ready pipelines.
- ✅ **Extensible Architecture:** Modular design for easy integration of custom processors and models.

---

## ⭐ Core Features

- **Unified API:** Single interface for model training, inference, and evaluation across multiple tabular model families.
- **Automated Preprocessing:** Model-aware data processing for feature scaling, encoding, imputation, and transformation.
- **Flexible Fine-Tuning:** Choose between zero-shot inference, full fine-tuning, or memory-efficient PEFT strategies.
- **Model Comparison:** Built-in leaderboard for systematic benchmarking and strategy evaluation.
- **Extensible Design:** Modular codebase for easy integration of custom data processors and models.

---

## 📦 Supported Models

| Model          | Family / Paradigm         | Key Innovation                       | PEFT Support    |
|----------------|--------------------------|--------------------------------------|-----------------|
| **TabPFN-v2**  | PFN / ICL                | Bayesian approximation on synthetic  | ⚠️ Experimental |
| **TabICL**     | Scalable ICL             | Two-stage column-row attention       | ✅ Full Support |
| **OrionMSP**  | Scalable ICL             | Multi-Scale Prior (MSP) ICL          | ✅ Full Support |
| **Orion BIX**  | Scalable ICL             | Biaxial Interaction eXpert           | ✅ Full Support |
| **Mitra**      | Scalable ICL             | 2D attention, synthetic priors       | ✅ Full Support |
| **ContextTab** | Semantics-Aware ICL      | Modality-specific embeddings         | ⚠️ Experimental |
| **TabDPT**     | Denoising Transformer    | Denoising pre-trained transformer    | ✅ Full Support |

<sup>✅ Full Support: Reliable LoRA integration<br>⚠️ Experimental: Known issues may occur; use base-ft if unstable</sup>

---

## ⚡ Quick Start

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import openml
from tabtune import TabularPipeline

# Load dataset
dataset = openml.datasets.get_dataset(42178)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Init and fit pipeline
pipeline = TabularPipeline(
    model_name="TabPFN",
    task_type="classification",
    tuning_strategy="base-ft",
    tuning_params={"device": "cpu"}
)
pipeline.fit(X_train, y_train)

# Save and load pipeline for prediction
pipeline.save("churn_pipeline.joblib")
loaded_pipeline = TabularPipeline.load("churn_pipeline.joblib")
predictions = loaded_pipeline.predict(X_test)
metrics = pipeline.evaluate(X_test, y_test)
print(metrics)
```

---

## 📝 Why TabTune?

- **No Boilerplate:** Avoids repetitive code for model-specific data loading, training, and inference.
- **Consistent Results:** Automates best practices for tabular DL research and model selection.
- **Fast Iteration:** Easily compare new models with your data, using the same consistent API.
- **Production Ready:** Model and config serialization for robust deployment and reproducibility.
- **Community-Driven:** Extensible design and open contribution policy.

---

## 📑 Explore the Documentation

- **[Getting Started](getting-started/installation.md):** Installation, setup, and basic usage.
- **[User Guide](user-guide/pipeline-overview.md):** In-depth tutorials for each component.
- **[Supported Models](models/overview.md):** Model details and design notes.
- **[Advanced Topics](advanced/peft-lora.md):** PEFT/LoRA, custom preprocessing, and more.
- **[API Reference](api/pipeline.md):** Complete Python API and class/method details.
- **[Examples & Benchmarks](examples/classification.md):** End-to-end code notebooks.

---

## 🏆 Example Notebooks

| Name                    | Task Type               | Colab Link                                                                 |
|-------------------------|-------------------------|----------------------------------------------------------------------------|
| TabPFN Inference        | Inference & Fine-Tune   | [Open In Colab](https://colab.research.google.com/drive/1aLwPgWZFtf20nEBnjkPxTbcjsW9hwANW)             |
| Mitra Inference/FineTune | Inference & Fine-Tune   | [Open In Colab](https://colab.research.google.com/drive/1Vnnfshuy1PZs5SpLSgKWdFg-4JVudxMY)             |
| OrionMSP/BIX Examples  | Inference & Fine-Tune   | *Coming soon*                                                              |
| TabDPT Large Dataset    | Large-Scale Training    | *Coming soon*                                                              |
| PEFT Fine-Tuning Guide  | Parameter-Efficient FT  | *Coming soon*                                                              |
| Leaderboard Benchmarking| Model Comparison        | *Coming soon*                                                              |

---

## 📂 Project Structure

```
tabtune/
├── Dataprocess/
├── models/
├── TabularPipeline/
├── TuningManager/
├── TabularLeaderboard/
├── benchmarking/
├── data/
├── logger.py
└── run.py
```

See [User Guide](user-guide/pipeline-overview.md) for a full file/module breakdown.

---

## 🏢 Developed by Lexsi Labs

<div style="text-align: center; margin: 30px 0;">
  <img src="assets/lexsi-labs-logo.svg" alt="Lexsi Labs Logo" style="height: 390px; width: auto; margin-bottom: 20px; position: relative;" />
  <p style="font-size: 1.1em; color: #666; margin-top: 10px;">
    Created by the team at <strong>Lexsi Labs</strong>, TabTune extends frontier AI research into the tabular domain.
  </p>
</div>

---

## 🗃️ License

This project is released under the MIT License.  
Please cite appropriately if used in academic or production projects.

**BibTeX Citation:**

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

- Issues and discussions are welcomed on the [GitHub issue tracker](https://github.com/Lexsi-Labs/TabTune/issues).
- Please see the **Contributing** section for contribution standards, code reviews, and documentation tips.

---

**Get started with TabTune and accelerate your tabular deep learning workflows today!**
