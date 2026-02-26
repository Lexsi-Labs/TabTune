# Regression Examples: End-to-End Workflows with TabTune

This document provides practical, complete examples for regression tasks
using TabTune across various scenarios and complexity levels.

------------------------------------------------------------------------

## 1. Quick Start Regression

### 1.1 5-Minute Example

``` python
from tabtune import TabularPipeline
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train
pipeline = TabularPipeline(
    model_name='TabPFN',
    task_type='regression',
    tuning_strategy='inference'
)

pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)
metrics = pipeline.evaluate(X_test, y_test)

```

------------------------------------------------------------------------

## 2. Fine-Tuned Regression

### 2.1 Housing Price Prediction

``` python
import pandas as pd
from sklearn.model_selection import train_test_split
from tabtune import TabularPipeline

df = pd.read_csv('housing_prices.csv')

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = TabularPipeline(
    model_name='TabICL',
    task_type='regression',
    tuning_strategy='base-ft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-5
    }
)

pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)

print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"R2: {metrics['r2_score']:.4f}")
```



## 3. Cross-Validation for Regression

``` python
from sklearn.model_selection import KFold
import numpy as np
from tabtune import TabularPipeline

def cross_validate_regression(X, y, model_name, params, k=5):

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmse_scores = []
    r2_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]

        pipeline = TabularPipeline(
            model_name=model_name,
            task_type='regression',
            tuning_strategy='base-ft',
            tuning_params=params
        )

        pipeline.fit(X_train_fold, y_train_fold)
        metrics = pipeline.evaluate(X_val_fold, y_val_fold)

        rmse_scores.append(metrics['rmse'])
        r2_scores.append(metrics['r2_score'])

        print(f"Fold {fold_idx+1}/{k}: RMSE={metrics['rmse']:.4f}, R2={metrics['r2_score']:.4f}")

    print(f"Mean RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"Mean R2: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

    return rmse_scores, r2_scores
```

------------------------------------------------------------------------

## 4. Real-World Dataset Example

### 4.1 Energy Efficiency Prediction

``` python
import openml
from sklearn.model_selection import train_test_split
from tabtune import TabularPipeline

dataset = openml.datasets.get_dataset(1471)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = TabularPipeline(
    model_name='OrionMSP1.5',
    task_type='regression',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'peft_config': {'r': 8}
    }
)

pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)

print("Energy Efficiency Results")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
print(f"R2 Score: {metrics['r2_score']:.4f}")
```

------------------------------------------------------------------------

## 6. Next Steps

-   Model Selection Guide
-   Hyperparameter Tuning
-   Regression Framework
-   Saving & Loading

------------------------------------------------------------------------

These examples cover the full spectrum of regression tasks with TabTune!
