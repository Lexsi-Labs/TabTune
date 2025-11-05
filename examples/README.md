# TabTune Examples

This directory contains comprehensive examples demonstrating TabTune's main contributions and features. Each example is self-contained and uses diverse datasets from different industries to showcase real-world applications.

## Example Overview

The examples are organized in a logical progression from basic concepts to advanced usage:

### Core Contributions (Examples 1-5)

| Example | File | Description | Dataset | Industry |
|---------|------|-------------|---------|----------|
| **01** | `01_unified_api.py` | Unified API across multiple models | OpenML 31 (credit-g) | Finance |
| **02** | `02_automated_preprocessing.py` | Model-aware automatic preprocessing | OpenML 37 (diabetes) | Healthcare |
| **03** | `03_finetuning_strategies.py` | Four fine-tuning strategies comparison | OpenML 42178 (Telco Churn) | Telecom |
| **04** | `04_model_comparison.py` | TabularLeaderboard for model benchmarking | OpenML 1461 (bank-marketing) | Banking |
| **05** | `05_checkpoint_management.py` | Save/load pipelines and checkpoint management | OpenML 45547 (Cardiovascular) | Healthcare |

### Advanced Features (Examples 6-9)

| Example | File | Description | Dataset | Industry |
|---------|------|-------------|---------|----------|
| **06** | `06_advanced_usage.py` | PEFT configuration and hybrid strategies | OpenML 41027/1480 | HR/Business |
| **07** | `07_data_sampling.py` | Resampling and feature selection strategies | OpenML 1489 (phoneme) | Technology |
| **08** | `08_evaluation_metrics.py` | Fairness and calibration evaluation | OpenML 31 (credit-g) | Finance |
| **09** | `09_benchmarking.py` | Running comprehensive benchmarks | Multiple datasets | Various |

## Quick Start

### Running an Example

```bash
cd TabTune_Internal
python examples/01_unified_api.py
```

### Prerequisites

- Python 3.11+
- PyTorch 2.0+
- All TabTune dependencies (see main README.md)
- OpenML account (for downloading datasets)

## Detailed Example Descriptions

### Example 1: Unified API (`01_unified_api.py`)

**Main Contribution**: Demonstrates TabTune's unified API that works identically across different models.

**Key Learning Points**:
- Same `.fit()`, `.predict()`, `.evaluate()` interface for all models
- No model-specific APIs to learn
- Consistent behavior regardless of underlying model complexity

**Dataset**: German Credit (Finance) - ~1000 samples, binary classification

---

### Example 2: Automated Preprocessing (`02_automated_preprocessing.py`)

**Main Contribution**: Shows how DataProcessor automatically handles model-specific preprocessing requirements.

**Key Learning Points**:
- Automatic preprocessing adaptation for different models
- TabPFN vs ContextTab vs ICL models - all handled automatically
- Custom preprocessing when needed via `processor_params`

**Dataset**: Pima Indians Diabetes (Healthcare) - ~768 samples, mixed features, some missing values

---

### Example 3: Fine-Tuning Strategies (`03_finetuning_strategies.py`)

**Main Contribution**: Compares all four fine-tuning strategies side-by-side.

**Fine-Tuning Strategies**:
1. **Inference**: Zero-shot predictions (no training)
2. **Meta-Learning**: Episodic fine-tuning (recommended for ICL models)
3. **SFT**: Supervised Fine-Tuning (standard batch training)
4. **PEFT**: Parameter-Efficient Fine-Tuning with LoRA adapters

**Key Learning Points**:
- When to use each strategy
- How to configure each strategy
- Performance comparison

**Dataset**: Telco Customer Churn (Telecom) - ~7043 samples, imbalanced dataset

---

### Example 4: Model Comparison (`04_model_comparison.py`)

**Main Contribution**: Demonstrates TabularLeaderboard for easy model benchmarking.

**Key Learning Points**:
- Compare multiple models/configurations simultaneously
- Automatic ranking by metrics
- Clean, organized results
  
This example includes comparisons for TabPFN, TabICL (inference and fine-tuned), OrionMSP (inference and fine-tuned), and OrionBix (inference and fine-tuned).

**Dataset**: Bank Marketing (Banking) - ~45211 samples, large dataset

---

### Example 5: Checkpoint Management (`05_checkpoint_management.py`)

**Main Contribution**: Shows checkpoint and pipeline serialization capabilities.

**Key Learning Points**:
- Save/load complete pipelines (model + preprocessing)
- Automatic checkpoint saving during training
- Resume training from checkpoints

**Dataset**: Cardiovascular Disease (Healthcare) - ~70000 samples, large dataset

---

### Example 6: Advanced Usage (`06_advanced_usage.py`)

**Main Contribution**: Demonstrates advanced PEFT configuration and hybrid strategies.

**Key Learning Points**:
- Detailed PEFT/LoRA configuration tuning
- Hybrid strategies (combining PEFT with meta-learning)
- Custom preprocessing pipelines
- Advanced tuning parameters

**Dataset**: Employee Satisfaction or Sick dataset (HR/Business or Healthcare)

---

### Example 7: Data Sampling (`07_data_sampling.py`)

**Main Contribution**: Showcases data preprocessing and balancing techniques.

**Key Learning Points**:
- Resampling strategies: SMOTE, random oversampling, KNN resampling
- Feature selection: variance threshold, ANOVA-based selection
- Impact of resampling on class balance and performance

**Dataset**: Phoneme (Technology) - ~5404 samples, highly imbalanced (perfect for demo)

---

### Example 8: Evaluation Metrics (`08_evaluation_metrics.py`)

**Main Contribution**: Demonstrates comprehensive evaluation including fairness and calibration.

**Key Learning Points**:
- Calibration metrics (Brier score, calibration curves)
- Fairness evaluation with sensitive features
- Demographic parity, equalized odds, equal opportunity
- Comparing standard vs. fairness-aware metrics

**Dataset**: German Credit (Finance) - Contains demographic features for fairness analysis

---

### Example 9: Benchmarking (`09_benchmarking.py`)

**Main Contribution**: Shows how to run comprehensive benchmarks across multiple datasets.

**Key Learning Points**:
- Use BenchmarkPipeline for large-scale evaluations
- Configure benchmark suites (talent, openml-cc18)
- Systematic evaluation workflow

**Dataset**: Multiple datasets from benchmark suites

---

## Industry Coverage

The examples use datasets from diverse industries to showcase TabTune's applicability:

- **Finance**: Credit scoring, banking (Examples 1, 4, 8)
- **Healthcare**: Disease diagnosis, patient care (Examples 2, 5)
- **Telecom**: Customer churn prediction (Example 3)
- **Technology**: Signal processing (Example 7)
- **HR/Business**: Employee satisfaction (Example 6)

## Common Patterns

All examples follow consistent patterns:

1. **Setup**: Reproducibility (seeds) and logging configuration
2. **Data Loading**: OpenML dataset loading with fallback to sklearn datasets
3. **Demonstration**: Clear, step-by-step code with comments
4. **Summary**: Key takeaways and best practices

## Running Multiple Examples

To run all examples sequentially:

```bash
for i in {1..9}; do
    python example/0${i}_*.py
done
```

## Troubleshooting

### Dataset Loading Issues

If OpenML dataset loading fails:
- Examples automatically fall back to sklearn datasets
- Check your internet connection
- Verify OpenML API access

### Memory Issues

For large datasets or models:
- Use PEFT strategy (Example 6)
- Reduce batch sizes in tuning_params
- Use CPU instead of GPU for smaller models

### Model-Specific Issues

- Some models require Hugging Face authentication (ContextTab)
- Check model documentation for specific requirements
- Use `tuning_strategy='inference'` for quick testing

## Next Steps

After running these examples:

1. **Experiment**: Modify parameters in examples to see effects
2. **Your Data**: Replace datasets with your own data
3. **Production**: Use patterns from Example 5 for deployment
4. **Research**: Use Example 9 for systematic evaluations

## Additional Resources

- **Main Documentation**: See `/docs` directory
- **API Reference**: See `/docs/api` directory
- **Model Guides**: See `/docs/models` directory
- **Contributing**: See `/docs/contributing` directory

## Support

For questions or issues:
- Check the main README.md
- Review example comments for explanations
- Open an issue on GitHub
- Contact: contact@lexsi.ai
