"""
Pytest configuration and shared fixtures for TabTune tests.

This file provides reusable fixtures that can be used across all test modules.
"""
import pytest
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def small_classification_data(random_seed):
    """
    Small synthetic classification dataset for fast tests.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) pandas DataFrames/Series
    """
    np.random.seed(random_seed)
    n_samples = 200
    
    # Create a simple dataset with mixed types
    X = pd.DataFrame({
        'numerical_1': np.random.randn(n_samples),
        'numerical_2': np.random.randn(n_samples),
        'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'categorical_2': np.random.choice(['X', 'Y'], n_samples),
    })
    
    # Binary classification target
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def multiclass_data(random_seed):
    """
    Multiclass classification dataset.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) with 3 classes
    """
    np.random.seed(random_seed)
    n_samples = 300
    
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
    })
    
    y = pd.Series(np.random.randint(0, 3, n_samples))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def openml_data(random_seed):
    """
    Load OpenML dataset for testing (uses iris dataset - OpenML ID 61).
    Falls back to synthetic data if OpenML is unavailable.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) pandas DataFrames/Series
    """
    try:
        import openml
        
        # Use iris dataset (OpenML ID 61) - multiclass, ~150 samples
        # Good for testing: small, multiclass, no missing values
        dataset = openml.datasets.get_dataset(61, download_data=True, download_qualities=False)
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        
        # Convert to pandas DataFrame/Series
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name='target')
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_seed, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        # Fallback to synthetic multiclass data if OpenML fails
        np.random.seed(random_seed)
        n_samples = 300
        
        X = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randn(n_samples),
        })
        
        y = pd.Series(np.random.randint(0, 3, n_samples))
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_seed, stratify=y
        )
        
        return X_train, X_test, y_train, y_test


@pytest.fixture
def minimal_data(random_seed):
    """
    Minimal dataset for quick smoke tests.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) with 50 samples
    """
    np.random.seed(random_seed)
    n_samples = 50
    
    X = pd.DataFrame({
        'feat1': np.random.randn(n_samples),
        'feat2': np.random.randn(n_samples),
    })
    
    y = pd.Series(np.random.randint(0, 2, n_samples))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_seed, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def missing_data_dataframe():
    """
    Dataset with missing values for testing imputation.
    
    Returns:
        pd.DataFrame: DataFrame with NaN values
    """
    np.random.seed(42)
    data = {
        'col1': [1, 2, np.nan, 4, 5, 6, np.nan, 8],
        'col2': [1.1, 2.2, 3.3, np.nan, 5.5, 6.6, 7.7, np.nan],
        'col3': ['A', 'B', 'C', 'A', np.nan, 'B', 'C', 'A'],
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def supported_models():
    """List of supported model names."""
    return [
        'TabPFN',
        'TabICL',
        'OrionMSP',
        'OrionBix',
        'Mitra',
        'ContextTab',
        'TabDPT',
    ]


@pytest.fixture(scope="module")
def supported_strategies():
    """List of supported tuning strategies."""
    return [
        'inference',
        'finetune',
        'base-ft',
        'peft',
    ]


@pytest.fixture
def fast_finetune_params():
    """Fast fine-tuning parameters for testing (1 epoch, small batch)."""
    return {
        'epochs': 1,
        'batch_size': 8,
        'learning_rate': 1e-5,
        'show_progress': False,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
        # Meta-learning parameters (small values for minimal_data: ~35 train samples)
        'support_size': 15,  # Must be <= available training samples
        'query_size': 10,    # Must be <= available training samples (support + query <= train_size)
        'n_episodes': 10,    # Small number for fast testing
        # TabDPT-specific parameters (prevents hanging on small datasets)
        'steps_per_epoch': 2,  # Small number to prevent TabDPT from hanging/stuck
    }


@pytest.fixture
def peft_config():
    """PEFT configuration for testing."""
    return {
        'r': 4,
        'lora_alpha': 8,
        'lora_dropout': 0.05
    }


@pytest.fixture
def sensitive_features(minimal_data):
    """Synthetic sensitive features for fairness testing."""
    _, _, _, y_test = minimal_data
    # Create binary sensitive feature
    np.random.seed(42)
    return pd.Series(np.random.choice(['GroupA', 'GroupB'], len(y_test)))


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""
    import logging
    logging.getLogger().handlers.clear()
    yield
    logging.getLogger().handlers.clear()

