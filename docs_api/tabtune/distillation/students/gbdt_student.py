"""
GBDT Student — Knowledge Distillation via Gradient Boosting
================================================================
Supports three backends: LightGBM, XGBoost, CatBoost.


"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GBDTStudent:
    """GBDT student for knowledge distillation.

    Parameters
    ----------
    task_type : str
    backend : str, 'lgbm', 'xgb', or 'catboost'
    n_classes : int or None
    n_estimators : int, default=500
    max_depth : int, default=6
    learning_rate : float, default=0.05
    early_stopping_rounds : int, default=50
    cat_features : list or None
        Column indices of categorical features (CatBoost only).
    random_state : int, default=42
    """

    def __init__(
        self,
        task_type="classification",
        backend="lgbm",
        n_classes=None,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        cat_features=None,
        random_state=42,
        **kwargs,
    ):
        self.task_type = task_type
        self.backend = backend
        self.n_classes = n_classes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.cat_features = cat_features
        self.random_state = random_state

        self.models_ = []
        self._label_map = None
        self._label_inv_map = None
        self._mode = None

    def fit(self, X, y, soft_labels, sample_weights=None):
        """Train GBDT on soft labels."""
        X_np = self._to_numpy(X)
        y_np = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

        if sample_weights is not None:
            sw = np.asarray(sample_weights, dtype=np.float64)
        else:
            sw = None

        if self.task_type == "classification":
            unique_labels = np.unique(y_np)
            self._label_map = {lab: i for i, lab in enumerate(unique_labels)}
            self._label_inv_map = {i: lab for lab, i in self._label_map.items()}
            n_classes = soft_labels.shape[1] if soft_labels.ndim == 2 else len(unique_labels)
            self.n_classes = n_classes

            self._mode = "multi_regressor"
            self.models_ = []

            # Split data for early stopping (10% held out)
            from sklearn.model_selection import train_test_split
            try:
                idx_train, idx_val = train_test_split(
                    np.arange(len(X_np)), test_size=0.1,
                    random_state=self.random_state
                )
            except ValueError:
                idx_train = np.arange(len(X_np))
                idx_val = idx_train[-max(1, len(X_np)//10):]

            for c in range(n_classes):
                target = soft_labels[:, c].astype(np.float64)
                model = self._create_regressor()
                self._fit_one_regressor(
                    model, X_np, target, idx_train, idx_val, sw
                )
                self.models_.append(model)

            logger.info(f"[GBDTStudent] Trained {n_classes} class regressors ({self.backend})")

        else:
            # Regression
            soft_np = np.asarray(soft_labels, dtype=np.float64).ravel()

            from sklearn.model_selection import train_test_split
            try:
                idx_train, idx_val = train_test_split(
                    np.arange(len(X_np)), test_size=0.1,
                    random_state=self.random_state
                )
            except ValueError:
                idx_train = np.arange(len(X_np))
                idx_val = idx_train[-max(1, len(X_np)//10):]

            model = self._create_regressor()
            self._fit_one_regressor(
                model, X_np, soft_np, idx_train, idx_val, sw
            )
            self.models_ = [model]
            logger.info(f"[GBDTStudent] Trained regression model ({self.backend})")

        return self

    def _fit_one_regressor(self, model, X_np, target, idx_train, idx_val, sw):
        """Fit a single regressor with backend-appropriate early stopping."""
        fit_kwargs = {}
        if sw is not None:
            fit_kwargs["sample_weight"] = sw[idx_train]

        if self.backend == "lgbm":
            model.fit(
                X_np[idx_train], target[idx_train],
                eval_set=[(X_np[idx_val], target[idx_val])],
                callbacks=[self._lgbm_early_stopping()],
                **fit_kwargs,
            )
        elif self.backend == "xgb":
            model.fit(
                X_np[idx_train], target[idx_train],
                eval_set=[(X_np[idx_val], target[idx_val])],
                verbose=False,
                **fit_kwargs,
            )
        elif self.backend == "catboost":
            # CatBoost uses eval_set as a Pool or (X, y) tuple
            # and has its own early stopping via early_stopping_rounds in constructor
            cb_kwargs = {}
            if sw is not None:
                cb_kwargs["sample_weight"] = sw[idx_train]

            model.fit(
                X_np[idx_train], target[idx_train],
                eval_set=(X_np[idx_val], target[idx_val]),
                **cb_kwargs,
            )
        else:
            model.fit(X_np[idx_train], target[idx_train], **fit_kwargs)

    def predict(self, X):
        if self.task_type == "classification":
            probas = self.predict_proba(X)
            preds = probas.argmax(axis=1)
            if self._label_inv_map:
                preds = np.array([self._label_inv_map.get(p, p) for p in preds])
            return preds
        else:
            X_np = self._to_numpy(X)
            return self.models_[0].predict(X_np)

    def predict_proba(self, X):
        if self.task_type != "classification":
            raise ValueError("predict_proba only for classification.")
        X_np = self._to_numpy(X)

        raw = np.column_stack([m.predict(X_np) for m in self.models_])
        raw = np.clip(raw, 0, None)
        sums = raw.sum(axis=1, keepdims=True)
        sums = np.where(sums < 1e-10, 1.0, sums)
        return raw / sums

    def _create_regressor(self):
        if self.backend == "lgbm":
            try:
                import lightgbm as lgb
            except ImportError:
                raise ImportError("LightGBM not installed. pip install lightgbm")
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                verbose=-1, n_jobs=-1,
            )
        elif self.backend == "xgb":
            try:
                import xgboost as xgb
            except ImportError:
                raise ImportError("XGBoost not installed. pip install xgboost")
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                verbosity=0, n_jobs=-1,
                early_stopping_rounds=self.early_stopping_rounds,
            )
        elif self.backend == "catboost":
            try:
                from catboost import CatBoostRegressor
            except ImportError:
                raise ImportError(
                    "CatBoost not installed. pip install catboost"
                )
            return CatBoostRegressor(
                iterations=self.n_estimators,
                depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                l2_leaf_reg=3.0,
                random_seed=self.random_state,
                verbose=0,
                allow_writing_files=False,
                bootstrap_type="Bernoulli",
                early_stopping_rounds=self.early_stopping_rounds,
                cat_features=self.cat_features,
            )
        raise ValueError(f"Unknown backend: {self.backend}")

    def _lgbm_early_stopping(self):
        """Create LightGBM early stopping callback."""
        try:
            import lightgbm as lgb
            return lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds, verbose=False
            )
        except (ImportError, AttributeError):
            return None

    @staticmethod
    def _to_numpy(X):
        X_np = (
            X.to_numpy().astype(np.float32)
            if hasattr(X, "to_numpy")
            else np.asarray(X, dtype=np.float32)
        )
        return np.nan_to_num(X_np, nan=0.0, posinf=1e6, neginf=-1e6)