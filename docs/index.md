<div style="text-align: center;">
  <img src="assets/tabtune.svg" alt="TabTune Logo" style="width: 600px; height: auto;" />
</div>
<div style="text-align: center;">
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg"/></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.6+-red.svg"/></a>
<a href="https://github.com/Lexsi-Labs/TabTune"><img src="https://img.shields.io/badge/docs-available-green.svg"/></a>
<a href="https://arxiv.org/abs/2511.02802"><img src="https://img.shields.io/badge/arXiv-2511.02802-b31b1b.svg"/></a>
<a href="https://discord.gg/dSB62Q7A"><img src="https://img.shields.io/badge/Discord-%235865F2.svg?&amp;logo=discord&amp;logoColor=white"/></a>
</div>

---

**TabTune** is a powerful and flexible Python library designed to simplify the training and fine-tuning of modern foundation models on tabular data.  
It provides a high-level, scikit-learn-compatible API that abstracts away the complexities of data preprocessing, model-specific training loops, and benchmarking, letting you focus on delivering results.

Whether you are a practitioner aiming for production-grade pipelines or a researcher exploring advanced architectures, TabTune streamlines your workflow for tabular deep learning.

---

## 🚀 Welcome to TabTune

**This documentation provides a complete, production-ready framework for tabular foundation models.**

**Core Components:**

- **Unified API (`TabularPipeline`):** Single, scikit-learn-compatible interface for all models with `.fit()`, `.predict()`, `.save()`, and `.load()` methods.
- **Smart Data Processing (`DataProcessor`):** Model-aware preprocessing that automatically handles imputation, scaling, categorical encoding, and feature transformations for each model.
- **Flexible Tuning (`TuningManager`):** Three tuning strategies—zero-shot `inference`, supervised fine-tuning (`base-ft`) with full parameter updates, and memory-efficient `peft` (LoRA) adapters. Supports episodic meta-learning for ICL models.
- **Model Comparison (`TabularLeaderboard`):** Systematic benchmarking tool for comparing multiple models and strategies on your datasets.
- **🆕 Ensembling Module (`TabularEnsemble`):** Unified framework to combine multiple tabular foundation models using six strategies, including weighted averaging, stacking, and deep ensembles for improved accuracy and uncertainty estimation. 


**Key Capabilities:**

- ✅ **Multiple Training Paradigms:** Supports supervised fine-tuning (SFT) with full parameter updates, episodic meta-learning for in-context learning models, and parameter-efficient PEFT strategies.
- ✅ **PEFT (LoRA) Support:** Parameter-efficient fine-tuning for 5 out of 9 models (TabICL, OrionMSP, OrionBix, TabDPT, Mitra) with full support.
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
## 🚀 What's New in this release

-   ✅ **TabPFN v3 Integration** -- Full support for the latest PriorLabs Model : `TabPFNv3`, with end-to-end inference and fine-tuning (native, meta-learning, SFT, PEFT/LoRA) for both classification and regression. Added as a new model entry alongside the existing TabPFNv2.6 integration.

-   ✅ **Causal Inference Module Integration** -- Full support for treatment effect estimation using tabular foundation models through a unified `CausalAnalysis` API, enabling identification, estimation, and refutation workflows.

-   ✅ **Six Causal Estimators** -- Includes Double Machine Learning (DML), S-Learner, T-Learner, X-Learner, R-Learner, and Causal Forests for robust average and heterogeneous treatment effect estimation.

-   ✅ **Built-in Causal Validation** -- Supports formal causal identification, placebo tests, random common cause checks, subset stability analysis, and sensitivity analysis through an integrated refutation framework.

-   ✅ **Fairness & Compliance Audits** -- Includes proxy attribute detection and counterfactual fairness evaluation with automated reporting for fairness-critical deployments.

-   ✅ **Counterfactual & Heterogeneous Effect Analysis** -- Supports per-row Conditional Average Treatment Effects (CATE), counterfactual prediction, and treatment effect exploration at the individual level.

-   ✅ **CausalLeaderboard Benchmarking** -- Compare multiple `(Estimator × TFM)` combinations using treatment effect stability, confidence intervals, and refutation pass rates.

---

## 📊 Supported Models

| Model | Family / Paradigm | Key Innovation | Supported Strategies |
|-------|------------------|----------------|----------------------|
| **TabPFN-v2** | PFN / ICL | Approximates Bayesian inference on synthetic data | Inference, Meta-Learning FT, SFT, PEFT*, Regression, Regression FT |
| **TabICL** | Scalable ICL | Two-stage column-then-row attention | Inference, Meta-Learning FT, SFT, PEFT |
| **OrionMSP v1.0** | Scalable ICL | Multi-Scale Sparse Attention | Inference, Meta-Learning FT, SFT, PEFT |
| **OrionMSP v1.5** | Scalable ICL | Stabilized prototype refinement | Inference, Meta-Learning FT, SFT, PEFT |
| **OrionBix** | Scalable ICL | Tabular Bi-Axial In-Context Learning | Inference, Meta-Learning FT, SFT, PEFT |
| **Mitra** | Scalable ICL | 2D attention (row & column) | Inference, Meta-Learning FT, SFT, PEFT, Regression, Regression-FT |
| **ContextTab** | Semantics-Aware ICL | Modality-specific semantic embeddings | Inference, Full Fine-Tuning, PEFT*, Regression, Regression-FT |
| **TabDPT** | Denoising Transformer | Denoising pre-training | Inference, Meta-Learning FT, SFT, Regression, Regression-FT |
| **LimiX** | Probabilistic / ICL | Likelihood-based mixture modeling; uncertainty-aware | Inference, Regression, Regression-FT |
| **TabPFN-v2.6** | PFN / ICL | Latest PriorLabs release with native finetuning API | Inference, Meta-Learning FT, SFT, Native FT, Regression, Regression FT |
| **TabPFN-v3** | PFN / ICL | Newest PriorLabs Prior-Fitted Network; updated architecture and checkpoints | Inference, Meta-Learning FT, SFT, Native FT, PEFT, Regression, Regression FT |
| **TabICLv2** | Scalable ICL | Improved column-then-row attention | Inference, FT, Regression, Regression FT |
 
*Note: PEFT for ContextTab and TabPFN is experimental; `inference` strategy is fully supported.*

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
    tuning_strategy="finetune",
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

### Google Colab Support

- TabTune auto-detects Colab sessions, skipping optional IPython-heavy integrations (e.g., rich leaderboard display) so installs succeed with Colab’s preinstalled packages.
- In Colab, install with `pip install "tabtune[colab]"` to keep core runtime packages (NumPy, pandas, scikit-learn, IPython) on the versions Colab expects.
- For richer notebook display helpers outside Colab, opt into `pip install "tabtune[interactive]"`.

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

|Below are 16 Example Notebooks showcasing all the features of the Library in-depth!

| Serial No. | Name | Task Performed | Link To Notebook |
|---|------|------|------|
| 1 | Unified API | Showcasing A Unified API Across Multiple Models |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1KcaSdYRjZnMlb0MLmQ5IlnbPDiuEr1Ld?usp=sharing) |
| 2 |  Automated Model-Aware Preprocessing | The Automated preprocessing system explained |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/12BQ12VJrxtTDslgjnjm26yi3a0PYXqZT?usp=sharing) |
| 3 | Fine-Tuning Strategies | TabTune's four fine-tuning strategies |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1QixfiNCjF1IQV9NooMipPpnH4ETcEQwg?usp=sharing) |
| 4 | Model Comparison | Model Comparison with TabularLeaderboard |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1PZW3iPQOvwh0kroGytMzYTGc6ZVUzuvg?usp=sharing) |
| 5 | Checkpoint Management | Checkpoint Management - Save/Load Pipelines |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1DBTGEPpYLJjU9Aj7lzHoX3JtwaNOC0jn?usp=sharing) |
| 6 | Advanced Usage | PEFT Configuration and Hybrid Strategies |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1V3XGLeKrXSJwavaULMncZiM7uVE8sz0h?usp=sharing) |
| 7 |  Resampling |  Resampling Strategies |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1EHGrrSm7EalVRvzkH1RUHsNSLzmn10lM?usp=sharing) |
| 8 | Regression - 1| Introduction to Regression - Inference |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1lBt0QZWqlwhEg2ul_nVPAeC-w3Are0At) |
| 9 | Regression - 2| Introduction to Regression - Finetune |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1FFuaRBDtJZFAQF-JDIxRAjtgOZ1rmHd1?usp=sharing) |
| 10 | Evaluation Metrics | Evaluation Metrics involved |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/18TxyTyBGAGrIVf6zLjURDChG0vM4V02M?usp=sharing) |
| 11 | Benchmarking | Standard Benchmarking Techniques |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1lcoVMPz_3X5_5taNdB9doTGoN05krNRw?usp=sharing) |
| 12 | TabPFNv2.6 | TabPFNv2.6 — Classification and Regression |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1-5fh2kU9sDidXmm095489f3sxNLssW_M) |
| 13 | TabICLv2 | TabICLv2 — Classification and Regression |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/13lv9Z5QNzaAp_2ArkTXGRKDjDFbKAq3Q) |
| 14 | Ensembling Strategies| TabTune's 6 Ensembling Strategies  |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/19TUTBuJ1VNIbp5hLdU4D64c2_RfwFQC8) |
| 15 | Distillation | With Single and Multi Teachers |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1Fo2zH7jDgYjkYhgI33SyuVgnrhMsdvUH)| 
| 16 | Causal Inference | Estimate Treatment Effect using TFMs |[![Open In Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1CWYo3ynOxw0ysV4iDz_8VNCBjMK3WIyd?usp=sharing)| 

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
├── ensemble/
├── distillation/
├── causal/
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
