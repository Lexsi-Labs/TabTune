# Custom Preprocessing: Extending TabTune's Data Pipeline

This document explains how to create custom preprocessors, extend the data pipeline, and integrate domain-specific transformations with TabTune.

---

## 1. Introduction

While TabTune provides comprehensive automatic preprocessing, you may need custom transformations for:

- Domain-specific feature engineering
- Specialized encoding for your data
- Integration with existing pipelines
- Research and experimentation
- Non-standard data types

This guide shows how to extend TabTune's preprocessing architecture.

---

## 2. Preprocessing Architecture

### 2.1 Class Hierarchy

```
BasePreprocessor (Abstract)
    ├── StandardPreprocessor (Default)
    ├── TabPFNPreprocessor
    ├── TabICLPreprocessor
    ├── MitraPreprocessor
    ├── ContextTabPreprocessor
    ├── TabDPTPreprocessor
    ├── OrionBixPreprocessor
    ├── OrionMSPPreprocessor
    └── YourCustomPreprocessor
```

### 2.2 Base Class Interface

```python
from abc import ABC, abstractmethod
import pandas as pd

class BasePreprocessor(ABC):
    """Abstract base for all preprocessors."""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Learn preprocessing parameters from data."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to data."""
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit and transform in one call."""
        self.fit(X, y)
        return self.transform(X)
    
    def get_config(self) -> dict:
        """Return preprocessing configuration."""
        return self.config
```

---

## 3. Creating Custom Preprocessors

### 3.1 Simple Custom Preprocessor

Create a basic preprocessor for specialized transformations:

```python
import pandas as pd
import numpy as np
from tabtune.preprocessing.base import BasePreprocessor
from sklearn.preprocessing import StandardScaler

class CustomFeatureEngineeringPreprocessor(BasePreprocessor):
    """Custom preprocessor with feature engineering."""
    
    def __init__(self, scale_numericals=True, create_interactions=True):
        super().__init__(
            scale_numericals=scale_numericals,
            create_interactions=create_interactions
        )
        self.scaler = StandardScaler()
        self.numerical_cols = None
        self.categorical_cols = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Learn column types and scaling."""
        # Identify numerical and categorical columns
        self.numerical_cols = X.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.categorical_cols = X.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        # Fit scaler on numerical columns
        if self.config['scale_numericals']:
            self.scaler.fit(X[self.numerical_cols])
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations."""
        X_transformed = X.copy()
        
        # Scale numerical features
        if self.config['scale_numericals']:
            X_transformed[self.numerical_cols] = self.scaler.transform(
                X[self.numerical_cols]
            )
        
        # Create interaction features
        if self.config['create_interactions']:
            X_transformed = self._create_interactions(X_transformed)
        
        return X_transformed
    
    def _create_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial interaction features."""
        if len(self.numerical_cols) >= 2:
            # Create pairwise interactions
            for i, col1 in enumerate(self.numerical_cols):
                for col2 in self.numerical_cols[i+1:]:
                    X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
        
        return X
```

### 3.2 Domain-Specific Preprocessor (Finance Example)

```python
import pandas as pd
import numpy as np
from tabtune.preprocessing.base import BasePreprocessor

class FinancialDataPreprocessor(BasePreprocessor):
    """Specialized preprocessor for financial data."""
    
    def __init__(self, 
                 handle_outliers=True,
                 create_ratios=True,
                 normalize_by_scale=True):
        super().__init__(
            handle_outliers=handle_outliers,
            create_ratios=create_ratios,
            normalize_by_scale=normalize_by_scale
        )
        self.outlier_bounds = {}
        self.scaling_factors = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Learn financial data patterns."""
        # Detect outliers (IQR method)
        if self.config['handle_outliers']:
            for col in X.select_dtypes(include=[np.number]):
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.outlier_bounds[col] = {
                    'lower': Q1 - 1.5 * IQR,
                    'upper': Q3 + 1.5 * IQR
                }
        
        # Learn scaling factors by sector
        if self.config['normalize_by_scale']:
            if 'sector' in X.columns:
                for sector in X['sector'].unique():
                    sector_data = X[X['sector'] == sector]
                    self.scaling_factors[sector] = sector_data.mean()
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply financial transformations."""
        X_transformed = X.copy()
        
        # Handle outliers with clipping
        if self.config['handle_outliers']:
            for col, bounds in self.outlier_bounds.items():
                X_transformed[col] = X_transformed[col].clip(
                    lower=bounds['lower'],
                    upper=bounds['upper']
                )
        
        # Create financial ratios
        if self.config['create_ratios']:
            if 'revenue' in X_transformed.columns and 'costs' in X_transformed.columns:
                X_transformed['profit_margin'] = (
                    X_transformed['revenue'] - X_transformed['costs']
                ) / X_transformed['revenue']
        
        # Normalize by sector
        if self.config['normalize_by_scale']:
            if 'sector' in X_transformed.columns:
                for sector, factors in self.scaling_factors.items():
                    sector_mask = X_transformed['sector'] == sector
                    # Normalize numeric columns by sector mean
                    for col in X_transformed.select_dtypes(include=[np.number]):
                        X_transformed.loc[sector_mask, col] /= factors[col]
        
        return X_transformed
```

---

## 4. Integrating Custom Preprocessors

### 4.1 Register Custom Preprocessor

```python
# In your codebase or configuration file
from tabtune.data_processor import DataProcessor
from your_module import CustomFeatureEngineeringPreprocessor

# Register custom preprocessor
DataProcessor.register_preprocessor(
    'custom_features',
    CustomFeatureEngineeringPreprocessor
)
```

### 4.2 Use Custom Preprocessor with Pipeline

```python
from tabtune import TabularPipeline

# Use custom preprocessor
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='base-ft',
    processor_params={
        'preprocessor_type': 'custom_features',
        'scale_numericals': True,
        'create_interactions': True
    }
)

pipeline.fit(X_train, y_train)
```

### 4.3 Chaining Preprocessors

Combine multiple preprocessors:

```python
class ChainedPreprocessor(BasePreprocessor):
    """Sequentially apply multiple preprocessors."""
    
    def __init__(self, preprocessors: list):
        super().__init__(preprocessors=preprocessors)
        self.preprocessor_chain = preprocessors
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit each preprocessor in chain."""
        for preprocessor in self.preprocessor_chain:
            preprocessor.fit(X, y)
            X = preprocessor.transform(X)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessors in sequence."""
        for preprocessor in self.preprocessor_chain:
            X = preprocessor.transform(X)
        return X

# Usage
preprocessors = [
    CustomFeatureEngineeringPreprocessor(create_interactions=True),
    FinancialDataPreprocessor(handle_outliers=True),
    YourSpecializedPreprocessor()
]

chained = ChainedPreprocessor(preprocessors)
X_transformed = chained.fit_transform(X_train, y_train)
```

---

## 5. Feature Engineering Examples

### 5.1 Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

class PolynomialPreprocessor(BasePreprocessor):
    """Add polynomial features."""
    
    def __init__(self, degree=2, include_bias=False):
        super().__init__(degree=degree, include_bias=include_bias)
        self.poly = PolynomialFeatures(
            degree=degree,
            include_bias=include_bias
        )
        self.feature_names = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit polynomial transformer."""
        numerical = X.select_dtypes(include=[np.number])
        self.poly.fit(numerical)
        self.feature_names = self.poly.get_feature_names_out(
            numerical.columns
        )
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform to polynomial features."""
        numerical = X.select_dtypes(include=[np.number])
        poly_features = self.poly.transform(numerical)
        
        return pd.DataFrame(
            poly_features,
            columns=self.feature_names,
            index=X.index
        )
```

### 5.2 Statistical Features

```python
class StatisticalFeaturePreprocessor(BasePreprocessor):
    """Extract statistical features from groups."""
    
    def __init__(self, groupby_col=None, agg_functions=None):
        super().__init__(
            groupby_col=groupby_col,
            agg_functions=agg_functions or ['mean', 'std', 'min', 'max']
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """No fitting needed for statistical features."""
        pass
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create statistical aggregates."""
        X_transformed = X.copy()
        
        if self.config['groupby_col'] and self.config['groupby_col'] in X:
            groupby_col = self.config['groupby_col']
            
            # Aggregate statistics
            for col in X.select_dtypes(include=[np.number]):
                for func in self.config['agg_functions']:
                    agg_values = X.groupby(groupby_col)[col].agg(func)
                    X_transformed[f'{col}_{func}_by_{groupby_col}'] = (
                        X[groupby_col].map(agg_values)
                    )
        
        return X_transformed
```

### 5.3 Text Feature Extraction

```python
from sklearn.feature_extraction.text import TfidfVectorizer

class TextFeaturePreprocessor(BasePreprocessor):
    """Extract features from text columns."""
    
    def __init__(self, text_columns=None, max_features=100):
        super().__init__(
            text_columns=text_columns,
            max_features=max_features
        )
        self.vectorizers = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit TF-IDF vectorizers."""
        text_cols = self.config['text_columns'] or [
            col for col in X.columns
            if X[col].dtype == 'object'
        ]
        
        for col in text_cols:
            vectorizer = TfidfVectorizer(
                max_features=self.config['max_features'],
                lowercase=True
            )
            vectorizer.fit(X[col].astype(str))
            self.vectorizers[col] = vectorizer
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform text to features."""
        X_transformed = X.copy()
        
        for col, vectorizer in self.vectorizers.items():
            tfidf_matrix = vectorizer.transform(X[col].astype(str))
            feature_names = vectorizer.get_feature_names_out()
            
            for i, fname in enumerate(feature_names):
                X_transformed[f'{col}_tfidf_{fname}'] = (
                    tfidf_matrix[:, i].toarray().flatten()
                )
        
        return X_transformed
```

---

## 6. Integration with scikit-learn

### 6.1 Use scikit-learn Transformers

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

class SklearnPreprocessor(BasePreprocessor):
    """Wrap scikit-learn preprocessing pipeline."""
    
    def __init__(self, steps=None):
        super().__init__(steps=steps or [])
        self.pipeline = Pipeline(steps or [])
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit sklearn pipeline."""
        self.pipeline.fit(X, y)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using sklearn pipeline."""
        result = self.pipeline.transform(X)
        return pd.DataFrame(result, index=X.index)

# Usage with sklearn pipeline
sklearn_steps = [
    ('scaler', RobustScaler()),
    ('pca', PCA(n_components=50))
]

preprocessor = SklearnPreprocessor(steps=sklearn_steps)
pipeline = TabularPipeline(
    model_name='TabICL',
    processor_params={
        'custom_preprocessor': preprocessor
    }
)
```

---

## 7. Validation and Monitoring

### 7.1 Validation Framework

```python
class ValidatingPreprocessor(BasePreprocessor):
    """Preprocessor with validation."""
    
    def __init__(self, validators=None):
        super().__init__(validators=validators or [])
        self.validators = validators
    
    def validate(self, X: pd.DataFrame) -> dict:
        """Run validators and return results."""
        results = {}
        
        for validator in self.validators:
            results[validator.name] = validator(X)
        
        return results
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit with validation."""
        validation_results = self.validate(X)
        
        for name, passed in validation_results.items():
            if not passed:
                print(f"⚠️ Validation failed: {name}")
```

### 7.2 Data Quality Checks

```python
class DataQualityValidator:
    """Validate data quality before preprocessing."""
    
    def __init__(self, name, check_func):
        self.name = name
        self.check_func = check_func
    
    def __call__(self, X: pd.DataFrame) -> bool:
        """Run validation check."""
        return self.check_func(X)

# Define checks
no_all_nulls = DataQualityValidator(
    'no_all_nulls',
    lambda X: not X.isnull().all().any()
)

sufficient_samples = DataQualityValidator(
    'sufficient_samples',
    lambda X: len(X) >= 100
)

numeric_columns_exist = DataQualityValidator(
    'numeric_columns',
    lambda X: X.select_dtypes(include=[np.number]).shape[1] > 0
)
```

---

## 8. Performance Optimization

### 8.1 Caching Transformed Data

```python
from functools import lru_cache
import hashlib

class CachedPreprocessor(BasePreprocessor):
    """Cache preprocessor outputs."""
    
    def __init__(self, cache_size=128):
        super().__init__(cache_size=cache_size)
        self.cache = {}
        self.cache_size = cache_size
    
    def _get_cache_key(self, X: pd.DataFrame) -> str:
        """Generate cache key from data hash."""
        data_hash = hashlib.md5(
            pd.util.hash_pandas_object(X).values
        ).hexdigest()
        return data_hash
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform with caching."""
        cache_key = self._get_cache_key(X)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Perform transformation
        result = self._transform_impl(X)
        
        # Cache result if space available
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = result
        
        return result
    
    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        """Override this for your transformation."""
        return X
```

### 8.2 Parallel Processing

```python
from joblib import Parallel, delayed

class ParallelPreprocessor(BasePreprocessor):
    """Apply preprocessing in parallel."""
    
    def __init__(self, n_jobs=-1, batch_size=1000):
        super().__init__(n_jobs=n_jobs, batch_size=batch_size)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform in parallel batches."""
        n_jobs = self.config['n_jobs']
        batch_size = self.config['batch_size']
        
        # Split into batches
        batches = [
            X.iloc[i:i+batch_size]
            for i in range(0, len(X), batch_size)
        ]
        
        # Process in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._transform_batch)(batch)
            for batch in batches
        )
        
        return pd.concat(results, ignore_index=True)
    
    def _transform_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Override for batch transformation."""
        return batch
```

---

## 9. Testing Custom Preprocessors

### 9.1 Unit Tests

```python
import unittest
import pandas as pd
import numpy as np

class TestCustomPreprocessor(unittest.TestCase):
    """Test custom preprocessor."""
    
    def setUp(self):
        """Create test data."""
        self.X = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['x', 'y', 'x', 'y', 'z']
        })
        self.y = pd.Series([0, 1, 0, 1, 0])
    
    def test_fit_transform(self):
        """Test fit_transform."""
        preprocessor = CustomFeatureEngineeringPreprocessor()
        result = preprocessor.fit_transform(self.X, self.y)
        
        self.assertEqual(len(result), len(self.X))
        self.assertGreater(result.shape[1], self.X.shape[1])
    
    def test_transform_consistency(self):
        """Test consistency across calls."""
        preprocessor = CustomFeatureEngineeringPreprocessor()
        preprocessor.fit(self.X, self.y)
        
        result1 = preprocessor.transform(self.X)
        result2 = preprocessor.transform(self.X)
        
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_no_data_leakage(self):
        """Test train/test independence."""
        X_train = self.X.iloc[:3]
        X_test = self.X.iloc[3:]
        
        preprocessor = CustomFeatureEngineeringPreprocessor()
        preprocessor.fit(X_train)
        
        result_test = preprocessor.transform(X_test)
        self.assertEqual(len(result_test), len(X_test))

if __name__ == '__main__':
    unittest.main()
```

---

## 10. Best Practices

### ✅ Do's

- ✅ Inherit from `BasePreprocessor`
- ✅ Implement `fit()` and `transform()`
- ✅ Prevent data leakage (fit on train only)
- ✅ Return DataFrames with proper indices
- ✅ Document configuration parameters
- ✅ Handle edge cases (empty data, NaNs)
- ✅ Test thoroughly
- ✅ Cache expensive computations

### ❌ Don'ts

- ❌ Don't modify input data in-place
- ❌ Don't fit on test data
- ❌ Don't hardcode column names
- ❌ Don't ignore NaN values silently
- ❌ Don't create state in `transform()`
- ❌ Don't forget to preserve index
- ❌ Don't skip error handling

---

## 11. Real-World Example: Complete Custom Pipeline

```python
class ComprehensivePreprocessor(BasePreprocessor):
    """Complete preprocessing pipeline."""
    
    def __init__(self):
        super().__init__()
        self.numerical_cols = None
        self.categorical_cols = None
        self.transformers = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit all transformers."""
        self.numerical_cols = X.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.categorical_cols = X.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        # Fit numerical transformer
        from sklearn.preprocessing import StandardScaler
        self.transformers['scaler'] = StandardScaler()
        self.transformers['scaler'].fit(X[self.numerical_cols])
        
        # Fit categorical transformer
        from sklearn.preprocessing import LabelEncoder
        self.transformers['encoders'] = {}
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.transformers['encoders'][col] = le
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations."""
        X_transformed = pd.DataFrame(index=X.index)
        
        # Transform numerical
        scaled = self.transformers['scaler'].transform(
            X[self.numerical_cols]
        )
        X_transformed[self.numerical_cols] = scaled
        
        # Transform categorical
        for col in self.categorical_cols:
            encoded = self.transformers['encoders'][col].transform(
                X[col].astype(str)
            )
            X_transformed[col] = encoded
        
        return X_transformed

# Usage
preprocessor = ComprehensivePreprocessor()
preprocessor.fit(X_train, y_train)
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

pipeline = TabularPipeline(
    model_name='TabICL',
    processor_params={'custom_preprocessor': preprocessor}
)
```

---

## 12. Next Steps

- [Data Processing](../user-guide/data-processing.md) - Standard preprocessing
- [API Reference](../api/data-processor.md) - DataProcessor API
- [Examples](../examples/classification.md) - Full examples with custom preprocessing

---

Extend TabTune's preprocessing with custom transformers tailored to your domain!