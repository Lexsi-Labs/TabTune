# FAQ

Frequently asked questions about TabTune, covering installation, usage, model selection, and troubleshooting.

---

## Installation & Setup

### Which Python versions are supported?
**Python 3.10+** is required. Python 3.11+ is recommended for best performance.

### Do I need a GPU?
No, TabTune works on CPU for many models. However, a GPU is **strongly recommended** for:
- Training/fine-tuning (base-ft and peft strategies)
- Large datasets (>100K rows)
- Faster inference

Models like TabPFN and TabICL can run on CPU for inference, but training will be significantly slower.

### How do I install TabTune with GPU support?
Install PyTorch with CUDA support first, then install TabTune:

```bash
# Install PyTorch with CUDA (check your CUDA version first with nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install TabTune
pip install -r requirements.txt
pip install -e .
```

---

## Tasks & Models

### Does TabTune support regression?
Not yet; regression support is planned for future releases. Currently, TabTune focuses on classification tasks (binary and multi-class).

### Which models support PEFT (LoRA)?
**Full PEFT Support:**
- TabICL
- OrionMSP
- OrionBix
- TabDPT
- Mitra

**Experimental PEFT Support:**
- TabPFN (may have stability issues)
- ContextTab (may have stability issues)

If you encounter issues with experimental models, use `base-ft` strategy instead.

### How do I choose the right model for my dataset?
See the [Model Selection Guide](../user-guide/model-selection.md) for detailed guidance. Quick reference:

- **<10K rows**: TabPFN (inference) or TabICL
- **10K-100K rows**: TabICL or Mitra
- **100K-1M rows**: OrionBix, OrionMSP, or TabDPT
- **>1M rows**: TabDPT
- **Text-heavy features**: ContextTab
- **High accuracy needed**: OrionBix or TabDPT with base-ft

---

## Usage & Workflow

### What's the difference between inference, base-ft, and peft strategies?

- **`inference`**: Zero-shot predictions using pre-trained weights. No training occurs. Fastest, lowest accuracy.
- **`base-ft`**: Full fine-tuning of all model parameters. Slowest, highest accuracy, requires most memory.
- **`peft`**: Parameter-efficient fine-tuning using LoRA adapters. Faster than base-ft, uses less memory, high accuracy.

See [Tuning Strategies](../user-guide/tuning-strategies.md) for detailed comparisons.

### How do I save and load models?

```python
# Save pipeline (includes preprocessing and model state)
pipeline.save("my_pipeline.joblib")

# Load pipeline
loaded_pipeline = TabularPipeline.load("my_pipeline.joblib")
predictions = loaded_pipeline.predict(X_test)
```

**Note**: Saved pipelines include the DataProcessor state, so preprocessing is automatically applied.

### What file formats are supported for input data?
TabTune accepts **pandas DataFrames and Series**. You can load data from:
- CSV files: `pd.read_csv()`
- Excel files: `pd.read_excel()`
- Parquet files: `pd.read_parquet()`
- Any format that pandas supports

### How do I handle missing values?
The `DataProcessor` automatically handles missing values based on your configuration:

```python
pipeline = TabularPipeline(
    model_name="TabICL",
    processor_params={
        "imputation_strategy": "mean"  # Options: 'mean', 'median', 'mode', 'knn'
    }
)
```

Most models have sensible defaults, so you often don't need to specify this.

---

## Data & Preprocessing

### Can I use my own custom preprocessing?
Yes! See [Custom Preprocessing](../advanced/custom-preprocessing.md) for details on:
- Creating custom preprocessors
- Extending the data pipeline
- Integrating domain-specific transformations

### What are the memory requirements?
Memory usage varies by model and dataset size:

- **TabPFN**: ~2-4 GB (small datasets)
- **TabICL**: ~4-8 GB (medium datasets)
- **OrionBix/OrionMSP**: ~8-16 GB (large datasets)
- **TabDPT**: ~12-24 GB (very large datasets)
- **Mitra**: ~16-32 GB (complex datasets)

**PEFT strategy** reduces memory by 40-60% compared to base-ft.

---

## Training & Performance

### How long does training take?
Training time depends on:
- Dataset size (rows and features)
- Model choice
- Strategy (inference: 0s, peft: fast, base-ft: slower)
- Hardware (GPU vs CPU)

Rough estimates:
- **Inference**: Instant (no training)
- **PEFT**: 5-30 minutes for medium datasets
- **Base-ft**: 30 minutes to several hours for large datasets

### How do I debug training issues?
1. **Check logs**: TabTune uses structured logging - enable verbose mode
2. **Reduce dataset size**: Test with a smaller subset first
3. **Use CPU**: Test on CPU to rule out GPU-specific issues
4. **Lower batch size**: Reduce memory pressure
5. **Check data quality**: Ensure no invalid values or type mismatches

See [Troubleshooting](../user-guide/troubleshooting.md) for detailed solutions.

### Why is my model overfitting?
Common causes and solutions:

- **Too many epochs**: Reduce `epochs` in `tuning_params`
- **Too high learning rate**: Lower `learning_rate` (try 1e-5 to 2e-5)
- **Dataset too small**: Use more data or a simpler model
- **Try PEFT**: LoRA adapters often generalize better

---

## Technical Issues

### I get "CUDA out of memory" errors. How do I fix this?
**Solutions:**
1. Use `peft` strategy instead of `base-ft` (40-60% less memory)
2. Reduce `batch_size` in `tuning_params`
3. Use a smaller model (TabICL instead of TabDPT)
4. Process data in chunks
5. Use CPU instead of GPU (slower but no memory limits)

### ModuleNotFoundError: No module named 'tabtune'
**Solution**: Install TabTune in development mode:
```bash
cd TabTune_Internal
pip install -e .
```

### Import errors or version conflicts
**Solution**: Use a virtual environment:
```bash
python -m venv tabtune-env
source tabtune-env/bin/activate  # Linux/macOS
# tabtune-env\Scripts\activate   # Windows
pip install -r requirements.txt
pip install -e .
```

### Model predictions are all the same class
**Possible causes:**
- Model not trained (using inference with poor pre-trained weights)
- Data preprocessing issue (check DataProcessor summary)
- Severe class imbalance (use resampling strategies)
- Wrong model for dataset size

**Solution**: Try fine-tuning with `base-ft` or `peft` strategy.

---

## Comparison & Evaluation

### How do I compare multiple models?
Use `TabularLeaderboard`:
```python
from tabtune import TabularLeaderboard

leaderboard = TabularLeaderboard(X_train, X_test, y_train, y_test)
leaderboard.add_model("TabICL", "base-ft")
leaderboard.add_model("OrionBix", "peft")
results = leaderboard.run(rank_by="roc_auc_score")
```

See [Model Comparison](../user-guide/leaderboard.md) for detailed examples.

### What evaluation metrics are available?
Default metrics in `.evaluate()`:
- **Accuracy**: Overall correctness
- **Weighted F1 Score**: Class-balanced F1
- **ROC AUC Score**: Binary and multi-class supported
- **Precision**: Weighted average
- **Recall**: Weighted average
- **MCC**: Matthews Correlation Coefficient

### How do I interpret the evaluation metrics?
- **Accuracy**: Simple but can be misleading with imbalanced classes
- **F1 Score**: Better for imbalanced datasets (weighted average)
- **ROC AUC**: Best for ranking/model comparison, works with imbalanced data
- **MCC**: Comprehensive metric that accounts for all confusion matrix values

---

## Advanced Topics

### Can I use TabTune for production deployment?
Yes! TabTune pipelines are production-ready:
- Save complete pipelines with `.save()`
- Includes all preprocessing transformations
- Reproducible results
- Handles new data automatically

**Best practices:**
- Use `base-ft` or `peft` for best accuracy
- Save checkpoints during training
- Log hyperparameters and preprocessing config
- Test on validation sets before deployment

### How do I fine-tune hyperparameters?
See [Hyperparameter Tuning](../advanced/hyperparameter-tuning.md) for:
- Search strategies (grid, random, Bayesian)
- Hyperparameter spaces for each model
- Integration with Optuna and other tools
- Best practices and validation strategies

### Can I use multiple GPUs?
Yes, for supported models. See [Multi-GPU Training](../advanced/multi-gpu.md) for configuration details.

---

## Support & Community

### Where can I get help?
- **GitHub Issues**: [TabTune_Internal Issues](https://github.com/Lexsi-Labs/TabTune_Internal/issues)
- **Documentation**: Browse the [User Guide](../user-guide/pipeline-overview.md)
- **FAQ**: This page!

### How do I report a bug?
Open an issue on GitHub with:
- TabTune version
- Python version
- Error message and traceback
- Minimal reproducible example
- System information (OS, GPU if applicable)

### Can I contribute to TabTune?
Yes! See the [Contributing Guide](../contributing/setup.md) for:
- Development setup
- Code standards
- How to add new models
- Documentation guidelines