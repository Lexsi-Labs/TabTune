# Classification Examples: End-to-End Workflows with TabTune

This document provides practical, complete examples for classification tasks using TabTune across various scenarios and complexity levels.

---

## 1. Quick Start Classification

### 1.1 5-Minute Example

Minimal code to get predictions:

```python
from tabtune import TabularPipeline
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train
pipeline = TabularPipeline(model_name='TabPFN', tuning_strategy='inference')
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)
metrics = pipeline.evaluate(X_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### 1.2 10-Minute Example with Fine-Tuning

```python
from tabtune import TabularPipeline
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create pipeline with fine-tuning
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='base-ft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-5
    }
)

# Train
pipeline.fit(X_train, y_train)

# Evaluate
metrics = pipeline.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
print(f"ROC AUC: {metrics['roc_auc_score']:.4f}")
```

---

## 2. Binary Classification

### 2.1 Credit Card Fraud Detection

Real-world binary classification example:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tabtune import TabularPipeline

# Load fraud detection dataset
df = pd.read_csv('creditcard.csv')

# Separate features and target
X = df.drop('Class', axis=1)  # Class: 0=normal, 1=fraud
y = df['Class']

print(f"Class distribution: {y.value_counts().to_dict()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train pipeline
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'learning_rate': 2e-4,
        'peft_config': {'r': 8, 'lora_alpha': 16}
    }
)

print("Training...")
pipeline.fit(X_train, y_train)

# Evaluate
metrics = pipeline.evaluate(X_test, y_test)

print("\n=== Results ===")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
print(f"ROC AUC: {metrics['roc_auc_score']:.4f}")

# Save for deployment
pipeline.save('fraud_detection_model.joblib')

# In production
loaded = TabularPipeline.load('fraud_detection_model.joblib')
new_transactions = pd.read_csv('new_transactions.csv')
fraud_predictions = loaded.predict(new_transactions)
```

### 2.2 Customer Churn Prediction

```python
import pandas as pd
from tabtune import TabularPipeline
from sklearn.model_selection import train_test_split

# Load customer data
df = pd.read_csv('customer_churn.csv')

# Preprocessing
X = df.drop(['CustomerID', 'Churn'], axis=1)
y = (df['Churn'] == 'Yes').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Pipeline
pipeline = TabularPipeline(
    model_name='OrionMSP',
    tuning_strategy='base-ft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'support_size': 48,
        'query_size': 32,
        'learning_rate': 2e-5
    }
)

pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)

print(f"Churn Prediction Accuracy: {metrics['accuracy']:.4f}")

# Identify high-risk customers
churn_probs = pipeline.predict_proba(X_test)[:, 1]
high_risk = np.where(churn_probs > 0.7)[0]
print(f"High-risk customers: {len(high_risk)}")
```

---

## 3. Multi-Class Classification

### 3.1 Iris Dataset (3-Class)

Classic machine learning example:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from tabtune import TabularPipeline

# Load iris dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Simple inference baseline
print("=== Inference Baseline ===")
pipeline_inf = TabularPipeline(
    model_name='TabPFN',
    tuning_strategy='inference'
)
pipeline_inf.fit(X_train, y_train)
inf_metrics = pipeline_inf.evaluate(X_test, y_test)
print(f"Inference Accuracy: {inf_metrics['accuracy']:.4f}")

# Fine-tuned model
print("\n=== Fine-Tuned Model ===")
pipeline_ft = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='base-ft',
    tuning_params={
        'device': 'cuda',
        'epochs': 5,
        'support_size': 48,
        'query_size': 32
    }
)
pipeline_ft.fit(X_train, y_train)
ft_metrics = pipeline_ft.evaluate(X_test, y_test)
print(f"Fine-Tuned Accuracy: {ft_metrics['accuracy']:.4f}")

# Compare
improvement = (ft_metrics['accuracy'] - inf_metrics['accuracy']) * 100
print(f"\nImprovement: +{improvement:.2f}%")
```

### 3.2 Multi-Class Document Classification

```python
import pandas as pd
from tabtune import TabularPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load documents
df = pd.read_csv('documents.csv')  # Columns: text, category

# Feature extraction from text
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['text']).toarray()
y = pd.factorize(df['category'])[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train classifier
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='base-ft',
    tuning_params={'epochs': 5}
)

pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)

print(f"Classification Accuracy: {metrics['accuracy']:.4f}")
print(f"Weighted F1: {metrics['f1_score']:.4f}")
```

---

<!-- ## 4. Imbalanced Classification

### 4.1 Handling Class Imbalance

```python
import pandas as pd
import numpy as np
from tabtune import TabularPipeline
from sklearn.model_selection import train_test_split

# Load imbalanced dataset
df = pd.read_csv('imbalanced_data.csv')
X = df.drop('target', axis=1)
y = df['target']

print(f"Class distribution: {y.value_counts().to_dict()}")
print(f"Imbalance ratio: {y.value_counts().min() / y.value_counts().max():.2%}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# Method 1: Using SMOTE in preprocessing
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='base-ft',
    tuning_params={'epochs': 5}
)

pipeline.fit(X_train_balanced, y_train_balanced)
metrics = pipeline.evaluate(X_test, y_test)

print(f"\nWith SMOTE:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")

# Method 2: Class weights (if supported)
# Adjust loss to penalize minority class more
```

### 4.2 Threshold Optimization for Imbalanced Data

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

# Get prediction probabilities
probs = pipeline.predict_proba(X_test)[:, 1]

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(y_test, probs)

# F1 score optimization
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[best_threshold_idx]

print(f"Default threshold: 0.5")
print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"F1 improvement: {f1_scores[best_threshold_idx]:.4f}")

# Use optimal threshold
predictions_optimized = (probs > optimal_threshold).astype(int)
```
 -->


---

## 5. Cross-Validation

### 5.1 k-Fold Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tabtune import TabularPipeline

def cross_validate(X, y, model_name, params, k=5):
    """Perform k-fold cross-validation."""
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    f1_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='base-ft',
            tuning_params=params
        )
        
        pipeline.fit(X_train_fold, y_train_fold)
        metrics = pipeline.evaluate(X_val_fold, y_val_fold)
        
        scores.append(metrics['accuracy'])
        f1_scores.append(metrics['f1_score'])
        
        print(f"Fold {fold_idx+1}/{k}: Acc={metrics['accuracy']:.4f}, "
              f"F1={metrics['f1_score']:.4f}")
    
    print(f"\nMean Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"Mean F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    
    return scores, f1_scores

# Usage
import pandas as pd
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

scores, f1s = cross_validate(
    X, y,
    model_name='TabICL',
    params={'epochs': 5},
    k=5
)
```


---
<!-- 
## 7. Hyperparameter Optimization

### 7.1 Bayesian Optimization

```python
import optuna
from tabtune import TabularPipeline
from sklearn.model_selection import train_test_split

def objective(trial):
    """Optuna objective for hyperparameter tuning."""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    epochs = trial.suggest_int('epochs', 1, 10)
    support_size = trial.suggest_int('support_size', 24, 96, step=8)
    
    # Train
    pipeline = TabularPipeline(
        model_name='TabICL',
        tuning_strategy='base-ft',
        tuning_params={
            'device': 'cuda',
            'learning_rate': learning_rate,
            'epochs': epochs,
            'support_size': support_size,
            'query_size': support_size // 2
        }
    )
    
    pipeline.fit(X_train, y_train)
    metrics = pipeline.evaluate(X_val, y_val)
    
    return metrics['accuracy']

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, n_jobs=1)

# Get best parameters
print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")

# Train final model with best parameters
best_pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='base-ft',
    tuning_params={
        'device': 'cuda',
        **study.best_params
    }
)

best_pipeline.fit(X_train, y_train)
final_metrics = best_pipeline.evaluate(X_test, y_test)
print(f"Final Test Accuracy: {final_metrics['accuracy']:.4f}")
```
 -->
---

## 9. Real-World Datasets

### 9.1 Adult Income Dataset

Predicting income level (binary classification):

```python
import pandas as pd
from tabtune import TabularPipeline
from sklearn.model_selection import train_test_split
import openml

# Download from OpenML
dataset = openml.datasets.get_dataset(1590)  # Adult dataset
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
pipeline = TabularPipeline(
    model_name='TabDPT',
    tuning_strategy='base-ft',
    tuning_params={
        'device': 'cuda',
        'epochs': 3,
        'support_size': 1024
    }
)

pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)

print(f"Adult Dataset Results:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
```

### 9.2 MNIST-Like Tabular Classification

```python
from sklearn.datasets import load_digits
from tabtune import TabularPipeline
from sklearn.model_selection import train_test_split

# Load digits (multi-class: 0-9)
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Classify
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={
        'device': 'cuda',
        'epochs': 3,
        'peft_config': {'r': 8}
    }
)

pipeline.fit(X_train, y_train)
metrics = pipeline.evaluate(X_test, y_test)

print(f"Digit Recognition (0-9):")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Classes: {len(set(y))}")
```

---

## 10. Troubleshooting

### 10.1 Common Issues

```python
# Issue: Low accuracy
# Solution: Try different model and strategy
from tabtune import TabularLeaderboard

lb = TabularLeaderboard(X_train, X_val, y_train, y_val)
for model in ['TabPFN', 'TabICL', 'TabDPT']:
    lb.add_model(model, 'inference')
results = lb.run()  # Compare

# Issue: Out of memory
# Solution: Use PEFT
pipeline = TabularPipeline(
    model_name='TabICL',
    tuning_strategy='peft',
    tuning_params={'peft_config': {'r': 4}}
)

# Issue: Slow training
# Solution: Reduce batch size or use fewer epochs
pipeline = TabularPipeline(
    tuning_params={'batch_size': 8, 'epochs': 3}
)
```

---

## 11. Next Steps

- [Model Selection Guide](../user-guide/model-selection.md) - Choose right model
- [Hyperparameter Tuning](../advanced/hyperparameter-tuning.md) - Optimize performance
- [TabularLeaderboard](../user-guide/leaderboard.md) - Compare models systematically
- [Saving & Loading](../user-guide/saving-loading.md) - Deploy models

---

These examples cover the full spectrum of classification tasks with TabTune!