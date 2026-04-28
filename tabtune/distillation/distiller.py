"""
TabDistiller — Model-Agnostic Distillation for Tabular Foundation Models
=========================================================================
Compresses any TabTune TFM into a lightweight student (MLP or GBDT)
using only predict_proba(). No model internals required.

Core algorithm:
  1. Fit teacher pipeline(s)
  2. Collect soft labels via k-fold cross-prediction (prevents ICL identity leakage)
  3. Normalize soft labels (fixes TabDPT floating-point bug)
  4. Optionally compute adaptive temperatures and confidence weights
  5. Train student on soft labels with Hinton loss
  6. Wrap student for sklearn-compatible deployment
"""

from __future__ import annotations

import copy
import logging
import time
import pickle
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    mean_squared_error, r2_score,
)

logger = logging.getLogger(__name__)


class TabDistiller:
    """Model-agnostic knowledge distillation for TabTune pipelines.

    Parameters
    ----------
    teachers : str, list[str], or list[TabularPipeline]
        Teacher model name(s) or pre-fitted pipelines.
    student : str
        Student type: 'mlp' or 'lgbm'.
    task_type : str
        'classification' or 'regression'.
    temperature : float
        Base temperature for KD loss.
    alpha : float
        Weight for soft-label loss (1-alpha for hard-label loss).
    n_folds : int
        Number of folds for cross-prediction soft labels.
    adaptive_temperature : bool
        Use per-sample adaptive temperature based on teacher entropy.
    confidence_weighting : bool
        Use bell-shaped confidence weighting.
    augment_factor : float or None
        If set, augment training data with Gaussian noise (fraction of original).
    student_params : dict or None
        Extra params passed to student constructor.
    teacher_params : dict or None
        Extra params passed to teacher pipeline constructor.
    device : str
        'cpu' or 'cuda'.
    random_state : int
    """

    def __init__(
        self,
        teachers: Union[str, list] = "TabPFN",
        student: str = "mlp",
        task_type: str = "classification",
        temperature: float = 3.0,
        alpha: float = 0.7,
        n_folds: int = 5,
        adaptive_temperature: bool = True,
        confidence_weighting: bool = True,
        augment_factor: Optional[float] = None,
        student_params: Optional[dict] = None,
        teacher_params: Optional[dict] = None,
        device: str = "cpu",
        random_state: int = 42,
        **kwargs,
    ):
        # Normalize teachers to list
        if isinstance(teachers, str):
            self.teacher_names = [teachers]
            self._teacher_objects = None
        elif isinstance(teachers, list):
            # Could be list of strings or list of pipelines
            if all(isinstance(t, str) for t in teachers):
                self.teacher_names = teachers
                self._teacher_objects = None
            else:
                # Pre-fitted pipelines
                self.teacher_names = [
                    getattr(t, 'model_name', f'teacher_{i}')
                    for i, t in enumerate(teachers)
                ]
                self._teacher_objects = teachers
        else:
            self.teacher_names = [str(teachers)]
            self._teacher_objects = None

        self.student_type = student
        self.task_type = task_type
        self.temperature = temperature
        self.alpha = alpha
        self.n_folds = n_folds
        self.adaptive_temperature = adaptive_temperature
        self.confidence_weighting = confidence_weighting
        self.augment_factor = augment_factor
        self.student_params = student_params or {}
        self.teacher_params = teacher_params or {}
        self.device = device
        self.random_state = random_state

        # Fitted state
        self.teacher_pipelines_: Optional[list] = None
        self.student_: Optional[object] = None
        self.soft_labels_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
        self.label_encoder_: Optional[object] = None
        self.processor_: Optional[object] = None
        self._is_fitted = False
        self._fit_time = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "TabDistiller":
        """Fit teacher(s) and train student via soft-label distillation.

        Parameters
        ----------
        X_train : DataFrame
        y_train : Series

        Returns
        -------
        self
        """
        from ..TabularPipeline.pipeline import TabularPipeline

        t0 = time.time()
        X_train = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train.reset_index(drop=True)
        y_train = pd.Series(y_train).reset_index(drop=True) if not isinstance(y_train, pd.Series) else y_train.reset_index(drop=True)

        logger.info(f"[TabDistiller] Starting distillation: "
                    f"teachers={self.teacher_names}, student={self.student_type}, "
                    f"task={self.task_type}, n={len(X_train)}, f={X_train.shape[1]}")

        # ── Step 1: Fit or adopt teachers ──────────────────────────────
        if self._teacher_objects is not None:
            self.teacher_pipelines_ = list(self._teacher_objects)
            logger.info(f"[TabDistiller] Using {len(self.teacher_pipelines_)} pre-fitted teacher(s)")
        else:
            self.teacher_pipelines_ = []
            for name in self.teacher_names:
                logger.info(f"[TabDistiller] Fitting teacher: {name}")
                pipe = TabularPipeline(
                    model_name=name,
                    task_type=self.task_type,
                    tuning_strategy="inference",
                    model_params=self.teacher_params,
                )
                pipe.fit(X_train, y_train)
                self.teacher_pipelines_.append(pipe)

        # ── Step 2: Get label encoder ──────────────────────────────────
        self.label_encoder_ = self._get_label_encoder(self.teacher_pipelines_[0])
        self.processor_ = getattr(self.teacher_pipelines_[0], 'processor', None)

        if self.task_type == "classification":
            self.n_classes_ = len(np.unique(y_train))

        # ── Step 3: Collect soft labels ────────────────────────────────
        soft_labels = self._collect_soft_labels(X_train, y_train)

        # BUG 1 FIX: Normalize soft labels (TabDPT floating-point issue)
        if self.task_type == "classification" and soft_labels.ndim == 2:
            row_sums = soft_labels.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
            soft_labels = soft_labels / row_sums
            # Clamp for numerical stability
            soft_labels = np.clip(soft_labels, 1e-8, 1.0)
            soft_labels = soft_labels / soft_labels.sum(axis=1, keepdims=True)

        self.soft_labels_ = soft_labels
        logger.info(f"[TabDistiller] Soft labels collected: shape={soft_labels.shape}")

        # ── Step 4: Compute adaptive T and confidence weights ──────────
        sample_temperatures = None
        sample_weights = None

        if self.task_type == "classification" and soft_labels.ndim == 2:
            if self.adaptive_temperature:
                from .losses import compute_adaptive_temperatures
                sample_temperatures = compute_adaptive_temperatures(
                    soft_labels, T_base=self.temperature
                )
                logger.info(f"[TabDistiller] Adaptive T: mean={sample_temperatures.mean():.2f}, "
                            f"std={sample_temperatures.std():.2f}")

            if self.confidence_weighting:
                from .losses import compute_confidence_weights
                sample_weights = compute_confidence_weights(soft_labels)
                logger.info(f"[TabDistiller] Confidence weights: mean={sample_weights.mean():.2f}")

        # ── Step 5: Data augmentation (optional) ───────────────────────
        if self.augment_factor is not None and self.augment_factor > 0:
            X_train, y_train, soft_labels, sample_weights, sample_temperatures = \
                self._augment_data(X_train, y_train, soft_labels,
                                   sample_weights, sample_temperatures)

        # ── Step 6: Preprocess features for student ────────────────────
        X_processed = self._preprocess_features(X_train)

        # ── Step 7: Train student ──────────────────────────────────────
        self.student_ = self._train_student(
            X_processed, y_train, soft_labels,
            sample_weights, sample_temperatures,
        )

        self._is_fitted = True
        self._fit_time = time.time() - t0
        logger.info(f"[TabDistiller] Distillation complete in {self._fit_time:.1f}s")
        return self

    def predict(self, X) -> np.ndarray:
        """Predict using the distilled student."""
        self._check_fitted()
        X_processed = self._preprocess_features(pd.DataFrame(X))
        return self.student_.predict(X_processed)

    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities using the distilled student."""
        self._check_fitted()
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification.")
        X_processed = self._preprocess_features(pd.DataFrame(X))
        return self.student_.predict_proba(X_processed)

    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate student on test data.

        Returns rounded metrics dict. For classification: accuracy, f1_weighted,
        f1_macro, roc_auc. For regression: mse, rmse, mae, r2.
        """
        self._check_fitted()
        X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
        y_test = np.asarray(y_test)
        preds = self.predict(X_test)

        if self.task_type == "classification":
            metrics = {
                "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                "f1_weighted": round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4),
                "f1_macro": round(float(f1_score(y_test, preds, average="macro", zero_division=0)), 4),
            }
            try:
                probas = self.predict_proba(X_test)
                if self.n_classes_ == 2:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_test, probas[:, 1])), 4)
                else:
                    metrics["roc_auc"] = round(float(roc_auc_score(
                        y_test, probas, multi_class="ovr", average="macro"
                    )), 4)
            except Exception as e:
                logger.warning(f"[TabDistiller] AUC computation failed: {e}")
                metrics["roc_auc"] = None
            return metrics
        else:
            from sklearn.metrics import mean_absolute_error as mae_fn
            preds_f = np.asarray(preds, dtype=float)
            y_f = np.asarray(y_test, dtype=float)
            return {
                "mse": round(float(mean_squared_error(y_f, preds_f)), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_f, preds_f))), 4),
                "mae": round(float(mae_fn(y_f, preds_f)), 4),
                "r2": round(float(r2_score(y_f, preds_f)), 4),
            }

    def compare(self, X_test, y_test) -> dict:
        """Compare teacher(s), student, and report metrics."""
        self._check_fitted()
        X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
        y_test_np = np.asarray(y_test)

        results = {}

        # Student metrics
        student_metrics = self.evaluate(X_test, y_test)
        results["student"] = student_metrics
        logger.info(f"[Compare] Student: {student_metrics}")

        # Teacher metrics
        for pipe, name in zip(self.teacher_pipelines_, self.teacher_names):
            try:
                t_preds = pipe.predict(X_test)
                if self.task_type == "classification":
                    t_metrics = {"accuracy": accuracy_score(y_test_np, t_preds)}
                    try:
                        t_probas = pipe.predict_proba(X_test)
                        if self.n_classes_ == 2:
                            t_metrics["roc_auc"] = roc_auc_score(y_test_np, t_probas[:, 1])
                        else:
                            t_metrics["roc_auc"] = roc_auc_score(
                                y_test_np, t_probas, multi_class="ovr", average="macro"
                            )
                    except Exception:
                        t_metrics["roc_auc"] = None
                else:
                    t_metrics = {
                        "mse": mean_squared_error(y_test_np, t_preds),
                        "r2": r2_score(y_test_np, t_preds),
                    }
                results[f"teacher_{name}"] = t_metrics
                logger.info(f"[Compare] Teacher {name}: {t_metrics}")
            except Exception as e:
                logger.warning(f"[Compare] Teacher {name} failed: {e}")
                results[f"teacher_{name}"] = {"error": str(e)}

        # Retention
        if self.task_type == "classification":
            s_key = "roc_auc" if student_metrics.get("roc_auc") else "accuracy"
            s_val = student_metrics.get(s_key, 0)
            for name in self.teacher_names:
                t_data = results.get(f"teacher_{name}", {})
                t_val = t_data.get(s_key, 0)
                if t_val and t_val > 0:
                    retention = s_val / t_val * 100
                    results[f"retention_{name}"] = round(retention, 1)
                    logger.info(f"[Compare] Retention vs {name}: {retention:.1f}%")

        return results

    def save(self, path: str):
        """Save distiller to disk."""
        self._check_fitted()
        # Strip teacher pipelines to save space (they're large)
        save_obj = {
            "student": self.student_,
            "student_type": self.student_type,
            "task_type": self.task_type,
            "n_classes": self.n_classes_,
            "teacher_names": self.teacher_names,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "device": self.device,
            "random_state": self.random_state,
            "adaptive_temperature": self.adaptive_temperature,
            "confidence_weighting": self.confidence_weighting,
            "student_params": self.student_params,
        }
        joblib.dump(save_obj, path)
        logger.info(f"[TabDistiller] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TabDistiller":
        """Load a saved distiller."""
        save_obj = joblib.load(path)
        d = cls(
            teachers=save_obj["teacher_names"],
            student=save_obj["student_type"],
            task_type=save_obj["task_type"],
            temperature=save_obj["temperature"],
            alpha=save_obj["alpha"],
            device=save_obj.get("device", "cpu"),
            random_state=save_obj.get("random_state", 42),
        )
        d.student_ = save_obj["student"]
        d.n_classes_ = save_obj["n_classes"]
        d.teacher_names = save_obj["teacher_names"]
        d._is_fitted = True
        logger.info(f"[TabDistiller] Loaded from {path}")
        return d

    # ═══════════════════════════════════════════════════════════════
    # Internal methods
    # ═══════════════════════════════════════════════════════════════

    def _collect_soft_labels(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Collect soft labels from teacher(s), using k-fold for ICL models."""
        from ..TabularPipeline.pipeline import TabularPipeline

        all_soft = []

        for pipe, name in zip(self.teacher_pipelines_, self.teacher_names):
            strategy = getattr(pipe, 'tuning_strategy', 'inference')
            is_icl_inference = strategy == 'inference'

            if is_icl_inference and self.n_folds > 1:
                # K-fold cross-prediction to prevent identity leakage
                logger.info(f"[TabDistiller] K-fold cross-prediction for {name} "
                            f"(n_folds={self.n_folds})")
                soft = self._kfold_soft_labels(name, X, y)
            else:
                # Direct prediction (fine-tuned or pre-fitted)
                logger.info(f"[TabDistiller] Direct soft labels for {name}")
                if self.task_type == "classification":
                    soft = pipe.predict_proba(X)
                else:
                    soft = np.asarray(pipe.predict(X)).ravel()

            all_soft.append(soft)

        # Average across teachers if multi-teacher
        if len(all_soft) == 1:
            return all_soft[0]
        else:
            return np.mean(all_soft, axis=0)

    def _kfold_soft_labels(self, teacher_name: str, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Generate out-of-fold soft labels via stratified k-fold."""
        from ..TabularPipeline.pipeline import TabularPipeline

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # Try stratified, fall back to regular KFold
        if self.task_type == "classification":
            try:
                kf = StratifiedKFold(
                    n_splits=self.n_folds, shuffle=True,
                    random_state=self.random_state
                )
                splits = list(kf.split(X, y))
            except ValueError:
                logger.warning("[TabDistiller] StratifiedKFold failed, falling back to KFold")
                kf = KFold(
                    n_splits=self.n_folds, shuffle=True,
                    random_state=self.random_state
                )
                splits = list(kf.split(X))
        else:
            kf = KFold(
                n_splits=self.n_folds, shuffle=True,
                random_state=self.random_state
            )
            splits = list(kf.split(X))

        # Initialize output array from first fold
        first_train_idx, first_val_idx = splits[0]
        pipe = TabularPipeline(
            model_name=teacher_name,
            task_type=self.task_type,
            tuning_strategy="inference",
            model_params=self.teacher_params,
        )
        pipe.fit(X.iloc[first_train_idx], y.iloc[first_train_idx])

        if self.task_type == "classification":
            first_preds = pipe.predict_proba(X.iloc[first_val_idx])
            n_classes = first_preds.shape[1]
            soft_labels = np.zeros((len(X), n_classes), dtype=np.float64)
            soft_labels[first_val_idx] = first_preds
        else:
            first_preds = np.asarray(pipe.predict(X.iloc[first_val_idx])).ravel()
            soft_labels = np.zeros(len(X), dtype=np.float64)
            soft_labels[first_val_idx] = first_preds

        logger.info(f"[TabDistiller] K-fold 1/{self.n_folds} complete")

        # Remaining folds
        for fold_idx, (train_idx, val_idx) in enumerate(splits[1:], 2):
            pipe = TabularPipeline(
                model_name=teacher_name,
                task_type=self.task_type,
                tuning_strategy="inference",
                model_params=self.teacher_params,
            )
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])

            if self.task_type == "classification":
                preds = pipe.predict_proba(X.iloc[val_idx])
                soft_labels[val_idx] = preds
            else:
                preds = np.asarray(pipe.predict(X.iloc[val_idx])).ravel()
                soft_labels[val_idx] = preds

            logger.info(f"[TabDistiller] K-fold {fold_idx}/{self.n_folds} complete")

        return soft_labels

    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for the student.

        Converts to numeric, fills NaN, ensures float32.
        This is the same preprocessing used for both distilled student
        and hard-label baseline, ensuring fair comparison.
        """
        X = pd.DataFrame(X).copy()
        # Select numeric columns only
        X_num = X.select_dtypes(include=[np.number])
        if X_num.shape[1] == 0:
            # Fallback: try to convert all columns
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            X_num = X
        X_num = X_num.fillna(0).astype(np.float32)
        return X_num

    def _train_student(self, X, y, soft_labels, sample_weights, sample_temperatures):
        """Train the student model on soft labels."""
        X_np = X.to_numpy().astype(np.float32) if hasattr(X, 'to_numpy') else np.asarray(X, dtype=np.float32)
        n_features = X_np.shape[1]

        # Split for validation
        try:
            if self.task_type == "classification":
                X_tr, X_val, y_tr, y_val, sl_tr, sl_val = train_test_split(
                    X, y, soft_labels, test_size=0.15,
                    random_state=self.random_state, stratify=y,
                )
            else:
                X_tr, X_val, y_tr, y_val, sl_tr, sl_val = train_test_split(
                    X, y, soft_labels, test_size=0.15,
                    random_state=self.random_state,
                )
        except ValueError:
            X_tr, X_val, y_tr, y_val, sl_tr, sl_val = train_test_split(
                X, y, soft_labels, test_size=0.15,
                random_state=self.random_state,
            )

        # Split sample weights/temperatures to match
        if sample_weights is not None:
            sw_tr = sample_weights[X_tr.index] if hasattr(X_tr, 'index') else sample_weights[:len(X_tr)]
        else:
            sw_tr = None

        if sample_temperatures is not None:
            st_tr = sample_temperatures[X_tr.index] if hasattr(X_tr, 'index') else sample_temperatures[:len(X_tr)]
        else:
            st_tr = None

        if self.student_type == "mlp":
            return self._train_mlp_student(
                X_tr, y_tr, sl_tr, X_val, y_val, sw_tr, st_tr, n_features
            )
        elif self.student_type == "lgbm":
            return self._train_gbdt_student(X_tr, y_tr, sl_tr, sw_tr)
        else:
            raise ValueError(f"Unknown student type: {self.student_type}")

    def _train_mlp_student(self, X_tr, y_tr, soft_labels, X_val, y_val,
                           sample_weights, sample_temperatures, n_features=None):
        """Train MLP student."""
        from .students.mlp_student import MLPStudent

        if n_features is None:
            n_features = X_tr.shape[1] if hasattr(X_tr, 'shape') else len(X_tr.columns)

        params = {
            "task_type": self.task_type,
            "n_features": n_features,
            "n_classes": self.n_classes_,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "device": self.device,
            "random_state": self.random_state,
        }
        params.update(self.student_params)

        student = MLPStudent(**params)
        student.fit(
            X_tr, y_tr, soft_labels,
            X_val=X_val, y_val=y_val,
            sample_weights=sample_weights,
            sample_temperatures=sample_temperatures,
        )
        return student

    def _train_gbdt_student(self, X_tr, y_tr, soft_labels, sample_weights):
        """Train GBDT student."""
        from .students.gbdt_student import GBDTStudent

        params = {
            "task_type": self.task_type,
            "n_classes": self.n_classes_,
        }
        params.update(self.student_params)

        student = GBDTStudent(**params)
        student.fit(X_tr, y_tr, soft_labels, sample_weights=sample_weights)
        return student

    def _get_label_encoder(self, pipeline):
        """Extract label encoder from a fitted pipeline.

        BUG 3 FIX: Handles all known models with clear error for unsupported ones.
        """
        # Try common paths
        if hasattr(pipeline, 'processor'):
            proc = pipeline.processor
            # Path 1: TabPFN, TabICL, TabDPT, etc.
            if hasattr(proc, 'custom_preprocessor_'):
                cp = proc.custom_preprocessor_
                if hasattr(cp, 'label_encoder_'):
                    return cp.label_encoder_
            # Path 2: regression processor
            if hasattr(proc, 'label_encoder_'):
                return proc.label_encoder_

        # Path 3: model-level
        if hasattr(pipeline, 'model'):
            m = pipeline.model
            if hasattr(m, 'classes_'):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                le.classes_ = np.array(m.classes_)
                return le

        # Fallback: create from training data if available
        if hasattr(pipeline, 'y_train_processed_') and pipeline.y_train_processed_ is not None:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.fit(pipeline.y_train_processed_)
            return le

        logger.warning(
            f"[TabDistiller] Could not extract label encoder from "
            f"{getattr(pipeline, 'model_name', 'unknown')}. "
            f"This may cause issues with label mapping. "
            f"Supported models: TabPFN, TabICL, TabICLv2, TabDPT, Mitra, "
            f"OrionMSP, OrionBiX, ContextTab, Limix."
        )
        return None

    def _augment_data(self, X, y, soft_labels, sample_weights, sample_temperatures):
        """Augment training data with Gaussian noise, re-querying teacher for soft labels.

        Unlike simply copying old soft labels, this re-predicts on the perturbed
        features so the student learns the teacher's decision boundary shape.
        Falls back to copying soft labels if teacher re-query fails.
        """
        n_aug = int(len(X) * self.augment_factor)
        if n_aug == 0:
            return X, y, soft_labels, sample_weights, sample_temperatures

        rng = np.random.RandomState(self.random_state + 1)
        idx = rng.choice(len(X), size=n_aug, replace=True)

        X_np = X.to_numpy() if hasattr(X, 'to_numpy') else np.asarray(X)
        feature_std = np.std(X_np, axis=0, keepdims=True)
        feature_std = np.where(feature_std == 0, 1.0, feature_std)
        noise = rng.normal(0, 0.1, size=(n_aug, X_np.shape[1]))  # 10% of feature std
        X_aug = pd.DataFrame(
            X_np[idx] + noise * feature_std,
            columns=X.columns if hasattr(X, 'columns') else None,
        )

        # Re-query teacher for augmented soft labels (more accurate than copying)
        aug_soft = None
        try:
            if self.teacher_pipelines_:
                teacher_preds = []
                for pipe in self.teacher_pipelines_:
                    if self.task_type == "classification":
                        teacher_preds.append(pipe.predict_proba(X_aug))
                    else:
                        teacher_preds.append(np.asarray(pipe.predict(X_aug)).ravel())
                if len(teacher_preds) == 1:
                    aug_soft = teacher_preds[0]
                else:
                    aug_soft = np.mean(teacher_preds, axis=0)
                logger.info(f"[TabDistiller] Re-queried teacher for {n_aug} augmented samples")
        except Exception as e:
            logger.warning(f"[TabDistiller] Teacher re-query failed ({e}), copying original soft labels")
            aug_soft = None

        if aug_soft is None:
            # Fallback: copy soft labels from original indices
            aug_soft = soft_labels[idx]

        # Hard labels from teacher (not ground truth)
        if self.task_type == "classification" and aug_soft.ndim == 2:
            y_aug = pd.Series(np.argmax(aug_soft, axis=1))
        else:
            y_aug = y.iloc[idx].reset_index(drop=True) if hasattr(y, 'iloc') else pd.Series(np.asarray(y)[idx])

        X_out = pd.concat([X, X_aug], ignore_index=True)
        y_out = pd.concat([y, y_aug], ignore_index=True)

        if soft_labels.ndim == 2:
            sl_out = np.concatenate([soft_labels, aug_soft], axis=0)
        else:
            sl_out = np.concatenate([soft_labels, aug_soft])

        # Augmented samples get slightly lower weight (0.8x)
        if sample_weights is not None:
            aug_weights = np.full(n_aug, 0.8, dtype=np.float32)
            sw_out = np.concatenate([sample_weights, aug_weights])
        else:
            sw_out = None

        if sample_temperatures is not None:
            if aug_soft is not None and aug_soft.ndim == 2:
                from .losses import compute_adaptive_temperatures
                aug_temps = compute_adaptive_temperatures(aug_soft, T_base=self.temperature)
                st_out = np.concatenate([sample_temperatures, aug_temps])
            else:
                st_out = np.concatenate([sample_temperatures, sample_temperatures[idx]])
        else:
            st_out = None

        logger.info(f"[TabDistiller] Augmented {n_aug} samples (total: {len(X_out)})")
        return X_out, y_out, sl_out, sw_out, st_out

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("TabDistiller is not fitted. Call fit() first.")


def preprocess_for_baseline(X: pd.DataFrame) -> pd.DataFrame:
    """Preprocess features identically to distiller for fair baseline comparison.

    Use this to train a hard-label baseline MLP on the same processed features.
    """
    X = pd.DataFrame(X).copy()
    X_num = X.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X_num = X
    return X_num.fillna(0).astype(np.float32)
