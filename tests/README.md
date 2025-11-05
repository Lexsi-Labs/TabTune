# TabTune Testing Guide

## Quick Start

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run fast tests only (skip slow tests)
pytest -m "not slow"

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_tabular_pipeline.py

# Run fine-tuning tests only
pytest tests/test_finetuning.py -v

# Run fine-tuning tests for specific model
pytest tests/test_finetuning.py -k "TabICL" -v
```

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── pytest.ini               # Pytest configuration
├── test_imports.py          # Package import tests
├── test_tabular_pipeline.py # TabularPipeline core functionality tests
├── test_data_processor.py   # DataProcessor and preprocessing tests
├── test_leaderboard.py      # TabularLeaderboard tests
├── test_finetuning.py       # Fine-tuning tests (meta-learning & SFT)
├── test_peft.py             # Parameter-Efficient Fine-Tuning (PEFT/LoRA) tests
├── test_model_integration.py # Full model integration tests (fit/predict/evaluate)
├── test_multiclass.py       # Multiclass classification tests
├── test_evaluation_advanced.py # Advanced evaluation metrics (calibration, fairness)
└── test_edge_cases.py       # Edge cases and error handling tests
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=tabtune --cov-report=html

# Run specific test class
pytest tests/test_tabular_pipeline.py::TestTabularPipelineFit

# Run specific test
pytest tests/test_tabular_pipeline.py::TestTabularPipelineFit::test_fit_tabpfn_inference

# Run only slow tests
pytest -m slow

# Run only fast tests
pytest -m "not slow"
```

### Debugging

```bash
# Show print statements
pytest -s

# Drop into debugger on failure
pytest --pdb

# Show all discovered tests without running
pytest --collect-only
```

## Writing Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*` (optional, for grouping)
- Test methods: `test_*`

### Using Fixtures

Fixtures are defined in `conftest.py` and used as function parameters:

```python
def test_example(minimal_data, random_seed):
    X_train, X_test, y_train, y_test = minimal_data
    # Use the data
    pass
```

**Available fixtures:**
- `minimal_data`: Small dataset (~35 training samples) for quick tests
- `small_classification_data`: Binary classification dataset (200 samples)
- `multiclass_data`: Multiclass dataset (3 classes, 300 samples)
- `openml_data`: OpenML iris dataset (falls back to synthetic if unavailable)
- `missing_data_dataframe`: Dataset with missing values
- `random_seed`: Random seed for reproducibility
- `supported_models`: List of supported model names
- `supported_strategies`: List of tuning strategies
- `fast_finetune_params`: Fast fine-tuning parameters optimized for small datasets
  - Includes meta-learning params (support_size, query_size, n_episodes)
  - Includes TabDPT-specific params (steps_per_epoch to prevent hanging)
  - Designed to work with `minimal_data` fixture
- `peft_config`: PEFT/LoRA configuration for testing

### Test Categories

**Unit Tests (Fast)**
- Test individual components in isolation
- Should run quickly (< 1 second each)
- Don't require model training

```python
def test_pipeline_initialization():
    pipeline = TabularPipeline(model_name='TabPFN', tuning_strategy='inference')
    assert pipeline.model_name == 'TabPFN'
```

**Integration Tests (Slow)**
- Test complete workflows (fit → predict → evaluate)
- May take longer (> 1 second each)
- Require actual model training

Mark slow tests with `@pytest.mark.slow`:

```python
@pytest.mark.slow
def test_full_pipeline(minimal_data):
    X_train, X_test, y_train, y_test = minimal_data
    pipeline = TabularPipeline(model_name='TabPFN', tuning_strategy='inference')
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    assert len(predictions) == len(X_test)
```

### Test Template

```python
"""
Tests for MyNewFeature.
"""
import pytest
from tabtune import TabularPipeline


class TestMyNewFeature:
    """Test suite for MyNewFeature."""
    
    def test_feature_initialization(self):
        """Test that feature can be initialized."""
        # Arrange
        feature = MyNewFeature(param='value')
        # Act & Assert
        assert feature.param == 'value'
    
    @pytest.mark.slow
    def test_feature_integration(self, minimal_data):
        """Test feature works in full pipeline."""
        X_train, X_test, y_train, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        assert predictions is not None
        assert len(predictions) == len(X_test)
```

## Writing Good Tests

1. **Test One Thing**: Each test should verify one behavior
2. **Use Descriptive Names**: Test names should clearly describe what they test
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Use Fixtures**: Reuse test data through fixtures
5. **Test Edge Cases**: Test with missing values, empty data, invalid inputs
6. **Test Error Cases**: Verify that errors are raised appropriately

## When Adding New Code

### Must Add Tests For:
- ✅ New models → `test_tabular_pipeline.py` and `test_model_integration.py`
- ✅ New preprocessing → `test_data_processor.py`
- ✅ New tuning strategies → `test_finetuning.py`
- ✅ New evaluation metrics → `test_evaluation_advanced.py`
- ✅ PEFT/LoRA features → `test_peft.py`
- ✅ Edge cases → `test_edge_cases.py`
- ✅ Bug fixes → Add regression test to relevant test file

## Before Submitting

1. **Run All Tests**: Ensure all existing tests pass
   ```bash
   pytest
   ```

2. **Check Coverage**: Aim for >80% coverage on new code
   ```bash
   pytest --cov=tabtune --cov-report=term-missing
   ```

3. **Update Existing Tests**: If you change behavior, update relevant tests

## Fine-Tuning Tests

Fine-tuning tests cover both meta-learning and supervised fine-tuning (SFT) strategies across all supported models:

- **Meta-Learning Tests**: Episodic fine-tuning with support/query splits
- **SFT Tests**: Standard batch-wise supervised fine-tuning
- **Models Tested**: TabPFN, TabICL, OrionMSP, OrionBix, Mitra, ContextTab, TabDPT

**Running Fine-Tuning Tests:**
```bash
# Run all fine-tuning tests
pytest tests/test_finetuning.py -v

# Run only meta-learning tests
pytest tests/test_finetuning.py -k "meta_learning" -v

# Run only SFT tests
pytest tests/test_finetuning.py -k "sft" -v

# Run tests for specific model
pytest tests/test_finetuning.py -k "TabICL" -v
```

**Note**: Fine-tuning tests use the `fast_finetune_params` fixture which is optimized for the `minimal_data` dataset (~35 samples). For custom fine-tuning tests, ensure your parameters are compatible with the dataset size.

## Common Issues

### Import Errors
Ensure you've installed the package:
```bash
pip install -e .
```

Also ensure test dependencies are installed:
```bash
pip install -e ".[dev]"
```

### Fixture Not Found
Ensure fixtures are defined in `conftest.py` or in the same file.

### Slow Test Execution
- Use `minimal_data` fixture instead of creating large datasets
- Skip slow tests during development: `pytest -m "not slow"`
- Use smaller models for testing (TabPFN is often faster)
- Fine-tuning tests can be slow - run specific subsets during development

### Memory Issues
- Use `minimal_data` or `small_classification_data` fixtures
- Clean up large objects in tests
- Consider using `@pytest.fixture(scope="session")` for expensive setup

### Fine-Tuning Test Failures
- **TabDPT hanging**: Ensure `steps_per_epoch` is small enough for your dataset size
- **Meta-learning failures**: Check that `support_size + query_size <= training_samples`
- **Dimension mismatch errors**: Ensure tensor shapes are consistent (especially for TabICL/Orion* SFT)
- **ContextTab dtype errors**: ContextTab requires labels to be `Long` type for loss calculation

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
