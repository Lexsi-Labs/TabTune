"""
Advanced Distillation Strategies
==================================
Industry-grade techniques for robust knowledge distillation:

1. Cross-Validation Distillation:
   Uses K-fold CV to generate out-of-fold soft labels from teachers,
   preventing train-data leakage (teacher never labels its own train data).

2. Progressive / Curriculum Distillation:
   Trains student in stages — easy samples first (high teacher confidence),
   then progressively includes harder samples.

3. Probability Calibration:
   Post-hoc calibration of student probabilities via temperature scaling
   on held-out data (Platt scaling / isotonic regression).

4. Self-Distillation:
   Student teaches itself — train once, use predictions as soft targets
   for a fresh student. Proven to improve generalization.
"""

from __future__ import annotations

import logging
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

logger = logging.getLogger(__name__)


def cross_validate_soft_labels(
    teacher_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "classification",
    n_folds: int = 5,
    teacher_params: Optional[dict] = None,
    random_state: int = 42,
) -> np.ndarray:
    """Generate out-of-fold soft labels via cross-validation.

    Each fold's teacher only labels data it was NOT trained on,
    preventing optimistic data leakage in the soft targets.

    This is critical for small datasets where teacher overfitting
    can produce artificially sharp soft labels on training data.

    Parameters
    ----------
    teacher_name : str
    X : DataFrame
    y : Series
    task_type : str
    n_folds : int
    teacher_params : dict or None
    random_state : int

    Returns
    -------
    soft_labels : ndarray
        Out-of-fold soft predictions for every sample.
    """
    from ..TabularPipeline.pipeline import TabularPipeline

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    if task_type == "classification":
        try:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            splits = list(kf.split(X, y))
        except ValueError:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            splits = list(kf.split(X))
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(kf.split(X))

    # Determine output shape from first fold
    first_train_idx, first_val_idx = splits[0]
    pipe = TabularPipeline(
        model_name=teacher_name, task_type=task_type,
        tuning_strategy="inference", model_params=teacher_params or {},
    )
    pipe.fit(X.iloc[first_train_idx], y.iloc[first_train_idx])

    if task_type == "classification":
        first_preds = pipe.predict_proba(X.iloc[first_val_idx])
        n_classes = first_preds.shape[1]
        soft_labels = np.zeros((len(X), n_classes), dtype=np.float64)
        soft_labels[first_val_idx] = first_preds
    else:
        first_preds = pipe.predict(X.iloc[first_val_idx])
        soft_labels = np.zeros(len(X), dtype=np.float64)
        soft_labels[first_val_idx] = np.asarray(first_preds).ravel()

    # Remaining folds
    for fold_idx, (train_idx, val_idx) in enumerate(splits[1:], 2):
        pipe = TabularPipeline(
            model_name=teacher_name, task_type=task_type,
            tuning_strategy="inference", model_params=teacher_params or {},
        )
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])

        if task_type == "classification":
            preds = pipe.predict_proba(X.iloc[val_idx])
            soft_labels[val_idx] = preds
        else:
            preds = pipe.predict(X.iloc[val_idx])
            soft_labels[val_idx] = np.asarray(preds).ravel()

        logger.info(f"[CV-Distill] Fold {fold_idx}/{n_folds} complete")

    logger.info(f"[CV-Distill] Generated OOF soft labels: shape={soft_labels.shape}")
    return soft_labels


def progressive_distillation(
    distiller,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    soft_labels: np.ndarray,
    n_stages: int = 3,
    confidence_percentiles: Optional[List[float]] = None,
):
    """Curriculum-based progressive distillation.

    Stage 1: Train on most confident teacher predictions only (top N%)
    Stage 2: Include medium-confidence samples
    Stage 3: Full dataset

    The student first learns clear patterns, then adapts to ambiguous ones.

    Parameters
    ----------
    distiller : TabDistiller
    X_train, y_train : training data
    soft_labels : teacher soft predictions
    n_stages : int
    confidence_percentiles : list of float
        Percentile thresholds for each stage (ascending).
        Default: [33, 66, 100] for 3 stages.

    Returns
    -------
    Fitted student
    """
    if confidence_percentiles is None:
        confidence_percentiles = np.linspace(100 / n_stages, 100, n_stages).tolist()

    # Compute confidence scores
    if soft_labels.ndim == 2:
        confidence = np.max(soft_labels, axis=1)
    else:
        # Regression: inverse of residual magnitude
        mean_pred = np.mean(soft_labels)
        confidence = 1.0 / (1.0 + np.abs(soft_labels - mean_pred))

    logger.info(f"[Progressive] {n_stages} stages, percentiles: {confidence_percentiles}")

    for stage, pct in enumerate(confidence_percentiles, 1):
        threshold = np.percentile(confidence, 100 - pct)
        mask = confidence >= threshold
        n_samples = mask.sum()

        logger.info(f"[Progressive] Stage {stage}/{n_stages}: "
                    f"{n_samples} samples (confidence >= {threshold:.3f})")

        X_stage = X_train[mask].reset_index(drop=True)
        y_stage = y_train[mask].reset_index(drop=True)
        soft_stage = soft_labels[mask]

        # Reduced epochs per stage (except last)
        original_epochs = distiller.student_params.get("epochs", 200)
        if stage < n_stages:
            stage_epochs = max(20, original_epochs // n_stages)
        else:
            stage_epochs = original_epochs

        distiller.student_params["epochs"] = stage_epochs

        if distiller.student_ is None:
            # First stage: init student
            distiller.student_ = distiller._train_mlp_student(
                X_stage, y_stage, soft_stage, None, None, None, None
            )
        else:
            # Subsequent stages: continue training existing student
            distiller.student_.fit(
                X_stage, y_stage, soft_stage, None, None
            )

    distiller.student_params["epochs"] = original_epochs
    return distiller.student_


def calibrate_student(distiller, X_cal, y_cal, method: str = "temperature"):
    """Post-hoc probability calibration of the distilled student.

    Learns an optimal temperature T* on calibration data that
    minimizes negative log-likelihood.

    Parameters
    ----------
    distiller : TabDistiller (fitted)
    X_cal : DataFrame
        Calibration data (held-out, not used for training).
    y_cal : Series
        Calibration labels.
    method : str
        'temperature' (Platt-style temperature scaling) or
        'isotonic' (isotonic regression per class).

    Returns
    -------
    float : optimal temperature (for temperature method)
    """
    if distiller.task_type != "classification":
        logger.warning("[Calibrate] Calibration only for classification. Skipping.")
        return None

    X_cal = pd.DataFrame(X_cal) if not isinstance(X_cal, pd.DataFrame) else X_cal
    y_cal = np.asarray(y_cal)

    if method == "temperature":
        return _temperature_calibration(distiller, X_cal, y_cal)
    elif method == "isotonic":
        return _isotonic_calibration(distiller, X_cal, y_cal)
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def _temperature_calibration(distiller, X_cal, y_cal):
    """Find optimal temperature T* that minimizes NLL on calibration set."""
    from scipy.optimize import minimize_scalar
    from sklearn.preprocessing import LabelEncoder

    probas = distiller.predict_proba(X_cal)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_cal)

    def nll(T):
        scaled = np.exp(np.log(np.clip(probas, 1e-10, 1.0)) / T)
        scaled = scaled / scaled.sum(axis=1, keepdims=True)
        ll = np.mean(np.log(np.clip(scaled[np.arange(len(y_enc)), y_enc], 1e-10, 1.0)))
        return -ll

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T_star = result.x

    # Store calibration temperature on student
    if hasattr(distiller.student_, "_calibration_temperature"):
        distiller.student_._calibration_temperature = T_star
    else:
        distiller.student_._calibration_temperature = T_star

    logger.info(f"[Calibrate] Optimal temperature T*={T_star:.3f} (NLL={result.fun:.4f})")
    return T_star


def _isotonic_calibration(distiller, X_cal, y_cal):
    """Per-class isotonic regression calibration."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.preprocessing import LabelEncoder

    probas = distiller.predict_proba(X_cal)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_cal)
    n_classes = probas.shape[1]

    calibrators = []
    for c in range(n_classes):
        binary = (y_enc == c).astype(float)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(probas[:, c], binary)
        calibrators.append(ir)

    distiller.student_._isotonic_calibrators = calibrators
    logger.info(f"[Calibrate] Isotonic calibration fitted for {n_classes} classes")
    return calibrators


class _StudentAsTeacherWrapper:
    """Wraps a fitted TabDistiller to behave like a TabularPipeline for multi-round self-distillation."""

    def __init__(self, distiller, name: str = "prev_student"):
        self.distiller = distiller
        self.model_name = name
        self.tuning_strategy = "finetuned"  # prevents k-fold re-fitting

    def fit(self, X, y):
        pass  # already fitted

    def predict(self, X):
        return self.distiller.predict(X)

    def predict_proba(self, X):
        return self.distiller.predict_proba(X)


def self_distill(
    teacher_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str = "classification",
    n_rounds: int = 2,
    student_type: str = "mlp",
    blend_alpha: float = 0.5,
    **kwargs,
):
    """Self-distillation: train model, use its predictions to train a fresh copy.

    Proven to improve generalization even without a larger teacher.
    Round 1: standard TFM -> student distillation.
    Round 2+: blend TFM soft labels with previous student's soft labels,
    then train a fresh student on the blended targets.

    Parameters
    ----------
    teacher_name : str
    X_train, y_train : data
    task_type : str
    n_rounds : int
        Number of self-distillation rounds.
    student_type : str
    blend_alpha : float
        Weight for original TFM teacher in rounds 2+.
        0.5 = equal blend of TFM and previous student.

    Returns
    -------
    TabDistiller (fitted after n_rounds)
    """
    from .distiller import TabDistiller

    current_distiller = None

    for round_num in range(1, n_rounds + 1):
        logger.info(f"[Self-Distill] Round {round_num}/{n_rounds}")

        if round_num == 1:
            # First round: standard distillation from TFM
            current_distiller = TabDistiller(
                teachers=teacher_name,
                student=student_type,
                task_type=task_type,
                **kwargs,
            )
            current_distiller.fit(X_train, y_train)
        else:
            # Wrap previous student as a "teacher"
            prev_wrapper = _StudentAsTeacherWrapper(
                current_distiller, name=f"student_r{round_num - 1}"
            )

            # Create new distiller with both original TFM + previous student
            new_distiller = TabDistiller(
                teachers=[teacher_name],  # will be fitted normally
                student=student_type,
                task_type=task_type,
                **kwargs,
            )

            # Fit teachers (the TFM)
            new_distiller.fit(X_train, y_train)

            # Now blend: get soft labels from TFM and previous student
            try:
                if task_type == "classification":
                    tfm_soft = new_distiller.soft_labels_
                    prev_soft = current_distiller.predict_proba(X_train)

                    # Blend: alpha * TFM + (1-alpha) * previous student
                    blended = blend_alpha * tfm_soft + (1 - blend_alpha) * prev_soft

                    # Normalize
                    row_sums = blended.sum(axis=1, keepdims=True)
                    row_sums = np.where(row_sums < 1e-10, 1.0, row_sums)
                    blended = blended / row_sums

                    new_distiller.soft_labels_ = blended
                    logger.info(f"[Self-Distill] Blended soft labels: "
                                f"{blend_alpha:.1f}*TFM + {1 - blend_alpha:.1f}*prev_student")

                    # Re-train student on blended soft labels
                    X_processed = new_distiller._preprocess_features(X_train)
                    new_distiller.student_ = new_distiller._train_student(
                        X_processed, y_train, blended, None, None,
                    )

                else:
                    tfm_soft = new_distiller.soft_labels_
                    prev_soft = np.asarray(current_distiller.predict(X_train)).ravel()
                    blended = blend_alpha * tfm_soft + (1 - blend_alpha) * prev_soft
                    new_distiller.soft_labels_ = blended

                    X_processed = new_distiller._preprocess_features(X_train)
                    new_distiller.student_ = new_distiller._train_student(
                        X_processed, y_train, blended, None, None,
                    )

            except Exception as e:
                logger.warning(f"[Self-Distill] Blending failed ({e}), "
                               f"keeping round {round_num} as standard distillation")

            current_distiller = new_distiller

    return current_distiller