"""
TabTune Tabular Ensemble
=========================
User-facing API for ensembling multiple tabular foundation models.

This module provides :class:`TabularEnsemble`, the single entry-point for
creating, fitting, evaluating, and predicting with ensembles of TabTune
pipelines.  It supports all six strategies defined in
:mod:`tabtune.ensemble.strategies`.

Usage
-----
::

    from tabtune.ensemble import TabularEnsemble

    ensemble = TabularEnsemble(
        models=[
            {"model_name": "TabPFN",   "tuning_strategy": "inference"},
            {"model_name": "TabICLv2", "tuning_strategy": "inference"},
            {"model_name": "OrionMSP", "tuning_strategy": "inference"},
        ],
        ensemble_strategy="greedy_selection",
        task_type="classification",
    )
    ensemble.fit(X_train, y_train)
    predictions = ensemble.predict(X_test)
    metrics     = ensemble.evaluate(X_test, y_test)
    leaderboard = ensemble.get_leaderboard()
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder

from .strategies import (
    CascadeStackingEnsemble,
    GreedyEnsembleSelection,
    RandomInitEnsemble,
    StackingEnsemble,
    TemperatureScaledBlending,
    WeightedAveraging,
    get_strategy,
    _probas_to_preds,
    _to_numpy,
)

logger = logging.getLogger(__name__)


class TabularEnsemble:
    """Ensemble of tabular foundation models via TabTune pipelines.

    Fits multiple TFMs on the training data, collects their predictions
    (via cross-validation for stacking / cascade, or holdout for other
    strategies), and learns an optimal combination strategy.

    Parameters
    ----------
    models : list of dict
        Each dict specifies a model config::

            {
                "model_name": "TabPFN",
                "tuning_strategy": "inference",
                "tuning_params": {},   # optional
                "model_params": {},    # optional
                "finetune_mode": "",   # optional
            }

    ensemble_strategy : str
        How to combine model predictions.  One of:

        * ``"weighted_averaging"`` -- performance-proportional weights
        * ``"greedy_selection"``  -- Caruana et al. ensemble selection
          (**recommended default**)
        * ``"stacking"`` -- meta-learner on cross-validated predictions
        * ``"temperature_scaled"`` -- calibrate then combine
        * ``"cascade_stacking"`` -- multi-level stacking
        * ``"random_init"`` -- deep ensembles via multiple random seeds

    task_type : str
        ``"classification"`` or ``"regression"``.

    metric : str
        Metric for weight optimisation.

        * Classification: ``"accuracy"``, ``"log_loss"``, ``"f1_score"``
        * Regression: ``"mse"``, ``"rmse"``, ``"r2"``, ``"mae"``

    cv_folds : int
        Number of CV folds for stacking / cascade OOF predictions.

    holdout_fraction : float
        Fraction of training data held out for strategy fitting (used
        with non-stacking strategies when no explicit validation set is
        provided).

    meta_learner : str
        For ``"stacking"``: ``"lr"`` (logistic / ridge), ``"gbdt"``,
        or ``"mlp"``.

    greedy_ensemble_size : int
        For ``"greedy_selection"`` / final GES in ``"cascade_stacking"``:
        number of iterations in Caruana selection.

    include_gbdt_baselines : bool
        Whether to include XGBoost / LightGBM / CatBoost as base models
        alongside TFMs.  Enables TFM + GBDT hybrid ensembles.

    n_cascade_levels : int
        For ``"cascade_stacking"``: number of stacking levels (default 2).

    n_seeds : int
        For ``"random_init"``: number of random seeds per model (default 5).

    base_seed : int
        For ``"random_init"``: starting seed value.

    weight_scheme : str
        For ``"weighted_averaging"``: one of ``"uniform"``,
        ``"performance"`` (default), or ``"inverse_error"``.

    verbose : bool
        Print progress and model-level metrics.

    Attributes
    ----------
    pipelines_ : dict[str, object]
        Fitted TabTune pipelines and GBDT baselines keyed by pipeline ID.
    strategy_ : object
        The fitted ensemble strategy instance.
    individual_scores_ : dict[str, float]
        Validation score for each individual model.
    ensemble_score_ : float or None
        Ensemble validation score.
    fit_times_ : dict[str, float]
        Wall-clock fitting time per model (seconds).
    """

    # ==================================================================
    # Construction
    # ==================================================================

    def __init__(
        self,
        models: List[Dict[str, Any]],
        ensemble_strategy: str = "greedy_selection",
        task_type: str = "classification",
        metric: str = "accuracy",
        cv_folds: int = 5,
        holdout_fraction: float = 0.2,
        meta_learner: str = "lr",
        greedy_ensemble_size: int = 50,
        include_gbdt_baselines: bool = False,
        n_cascade_levels: int = 2,
        n_seeds: int = 5,
        base_seed: int = 42,
        weight_scheme: str = "performance",
        verbose: bool = True,
    ) -> None:
        # --- Validate inputs ---
        valid_strategies = (
            "weighted_averaging", "greedy_selection", "stacking",
            "temperature_scaled", "cascade_stacking", "random_init",
        )
        if ensemble_strategy not in valid_strategies:
            raise ValueError(
                f"ensemble_strategy must be one of {valid_strategies}, "
                f"got '{ensemble_strategy}'."
            )
        if task_type not in ("classification", "regression"):
            raise ValueError(
                f"task_type must be 'classification' or 'regression', "
                f"got '{task_type}'."
            )
        if not models:
            logger.warning(
                "[TabularEnsemble] Initialised with empty models list. "
                "fit() will require at least one model."
            )
        if holdout_fraction <= 0 or holdout_fraction >= 1:
            raise ValueError("holdout_fraction must be in (0, 1).")
        if cv_folds < 2:
            raise ValueError("cv_folds must be >= 2.")

        self.models = models
        self.ensemble_strategy = ensemble_strategy
        self.task_type = task_type
        self.metric = metric
        self.cv_folds = cv_folds
        self.holdout_fraction = holdout_fraction
        self.meta_learner = meta_learner
        self.greedy_ensemble_size = greedy_ensemble_size
        self.include_gbdt_baselines = include_gbdt_baselines
        self.n_cascade_levels = n_cascade_levels
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        self.weight_scheme = weight_scheme
        self.verbose = verbose

        # State (populated after fit)
        self.pipelines_: Dict[str, Any] = {}
        self.strategy_: Any = None
        self.individual_scores_: Dict[str, float] = {}
        self.ensemble_score_: Optional[float] = None
        self.fit_times_: Dict[str, float] = {}
        self._is_fitted: bool = False
        self._n_classes: Optional[int] = None
        self._y_le_: Optional[Any] = None

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _get_pipeline_id(self, config: Dict[str, Any]) -> str:
        """Generate a unique, human-readable ID for a model configuration."""
        name = config["model_name"]
        strategy = config.get("tuning_strategy", "inference")
        mode = config.get("finetune_mode", "")
        suffix = f"_{mode}" if mode else ""
        return f"{name}_{strategy}{suffix}"

    def _build_pipeline(self, config: Dict[str, Any]) -> Any:
        """Build a TabTune :class:`TabularPipeline` from a config dict.

        Returns
        -------
        TabularPipeline
            An unfitted pipeline ready for ``.fit()``.

        Raises
        ------
        ImportError
            If ``tabtune`` is not installed.
        """
        try:
            from tabtune import TabularPipeline
        except ImportError as exc:
            raise ImportError(
                "TabTune is required. Install via: pip install tabtune"
            ) from exc

        return TabularPipeline(
            model_name=config["model_name"],
            task_type=self.task_type,
            tuning_strategy=config.get("tuning_strategy", "inference"),
            tuning_params=config.get("tuning_params", {}),
            model_params=config.get("model_params", {}),
            finetune_mode=config.get("finetune_mode", "meta-learning"),
        )

    def _build_gbdt_baselines(self) -> Dict[str, Any]:
        """Instantiate GBDT baseline models (XGBoost / LightGBM / sklearn).

        Returns
        -------
        dict[str, sklearn-compatible estimator]
        """
        baselines: Dict[str, Any] = {}
        if self.task_type == "classification":
            try:
                from sklearn.ensemble import GradientBoostingClassifier
                baselines["GBDT_sklearn"] = GradientBoostingClassifier(
                    n_estimators=100, max_depth=5
                )
            except ImportError:
                pass
            try:
                from lightgbm import LGBMClassifier
                baselines["LightGBM"] = LGBMClassifier(
                    n_estimators=200, max_depth=6, verbose=-1
                )
            except ImportError:
                pass
            try:
                from xgboost import XGBClassifier
                baselines["XGBoost"] = XGBClassifier(
                    n_estimators=200, max_depth=6, verbosity=0,
                    eval_metric="logloss",
                )
            except ImportError:
                pass
        else:  # regression
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                baselines["GBDT_sklearn"] = GradientBoostingRegressor(
                    n_estimators=100, max_depth=5
                )
            except ImportError:
                pass
            try:
                from lightgbm import LGBMRegressor
                baselines["LightGBM"] = LGBMRegressor(
                    n_estimators=200, max_depth=6, verbose=-1
                )
            except ImportError:
                pass
            try:
                from xgboost import XGBRegressor
                baselines["XGBoost"] = XGBRegressor(
                    n_estimators=200, max_depth=6, verbosity=0,
                )
            except ImportError:
                pass
        return baselines

    def _get_model_output(
        self, pipeline: Any, X: Any, *, is_tabtune: bool = True,
    ) -> np.ndarray:
        """Get predictions / probabilities from a fitted model.

        Parameters
        ----------
        pipeline : fitted estimator
        X : features
        is_tabtune : bool
            ``True`` for TabTune pipelines, ``False`` for sklearn models.

        Returns
        -------
        ndarray
            ``(n, k)`` probabilities for classification, ``(n,)`` for
            regression.
        """
        if is_tabtune:
            if self.task_type == "classification":
                try:
                    return np.asarray(pipeline.predict_proba(X))
                except Exception:
                    preds = np.asarray(pipeline.predict(X))
                    # One-hot encode as fallback
                    n_cls = self._n_classes or int(preds.max() + 1)
                    probas = np.zeros((len(preds), n_cls))
                    for i, c in enumerate(preds.astype(int)):
                        if 0 <= c < n_cls:
                            probas[i, c] = 1.0
                    return probas
            else:
                return np.asarray(pipeline.predict(X))
        else:
            # Sklearn model
            X_arr = X.values if hasattr(X, "values") else X
            if self.task_type == "classification" and hasattr(pipeline, "predict_proba"):
                return np.asarray(pipeline.predict_proba(X_arr))
            else:
                return np.asarray(pipeline.predict(X_arr))

    def _log(self, msg: str) -> None:
        """Print a message if verbose mode is on."""
        if self.verbose:
            print(msg)

    @staticmethod
    def _encode_for_skip(
        *dfs: pd.DataFrame,
    ) -> Tuple[np.ndarray, ...]:
        """Encode DataFrames into float numpy arrays for skip connections.

        Uses :class:`OrdinalEncoder` fitted on the first DataFrame only
        (no leakage).

        Parameters
        ----------
        *dfs : DataFrames (train, val, test, ...)

        Returns
        -------
        tuple of ndarrays
        """
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
        results = []
        for i, df in enumerate(dfs):
            if i == 0:
                results.append(enc.fit_transform(df).astype(float))
            else:
                results.append(enc.transform(df).astype(float))
        return tuple(results)

    # ==================================================================
    # Scoring helpers
    # ==================================================================

    def _score_individual(
        self,
        output: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Compute validation score for a single model output."""
        if self.task_type == "classification" and output.ndim == 2:
            preds = _probas_to_preds(output)
            return float(accuracy_score(y, preds))
        elif self.task_type == "regression":
            return float(r2_score(y, output))
        return 0.0

    # ==================================================================
    # Fit
    # ==================================================================

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
    ) -> "TabularEnsemble":
        """Fit all base models and learn the ensemble combination strategy.

        Parameters
        ----------
        X_train : DataFrame or array-like
            Training features.
        y_train : Series or array-like
            Training target.
        X_val : DataFrame or array-like, optional
            Validation features.  If ``None``, a holdout is carved from
            *X_train*.
        y_val : Series or array-like, optional
            Validation target.

        Returns
        -------
        self

        Raises
        ------
        RuntimeError
            If fewer than 1 model fits successfully.
        """
        # --- Validate ---
        if not self.models:
            raise ValueError(
                "[TabularEnsemble] models list is empty. "
                "Provide at least one model configuration."
            )

        # --- Coerce inputs ---
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train)

        self._n_classes = int(y_train.nunique()) if self.task_type == "classification" else None
        if self.task_type == "classification":
            from sklearn.preprocessing import LabelEncoder as _LabelEncoder
            self._y_le_ = _LabelEncoder().fit(y_train)

        # --- Dispatch to strategy-specific fitting ---
        if self.ensemble_strategy == "cascade_stacking":
            return self._fit_cascade(X_train, y_train, X_val, y_val)
        elif self.ensemble_strategy == "random_init":
            return self._fit_random_init(X_train, y_train, X_val, y_val)
        else:
            return self._fit_standard(X_train, y_train, X_val, y_val)

    # ------------------------------------------------------------------
    # Standard fit (weighted_averaging, greedy_selection, stacking, temp_scaled)
    # ------------------------------------------------------------------

    def _fit_standard(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[Any],
        y_val: Optional[Any],
    ) -> "TabularEnsemble":
        """Fit workflow for non-cascade, non-deep-ensemble strategies."""
        # --- Split data ---
        if X_val is None:
            stratify = y_train if self.task_type == "classification" else None
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train, y_train,
                test_size=self.holdout_fraction,
                random_state=42,
                stratify=stratify,
            )
        else:
            X_fit, y_fit = X_train, y_train
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)
            if not isinstance(y_val, pd.Series):
                y_val = pd.Series(y_val)

        y_val_np = _to_numpy(y_val)
        
        
        
        if self.task_type == "classification" and self._y_le_ is not None:
            y_val_np = self._y_le_.transform(y_val_np)

        # --- Step 1: Fit TFM base models ---
        model_val_outputs: Dict[str, np.ndarray] = {}
        self._log(f"\n{'='*60}")
        self._log(f"  Fitting {len(self.models)} base model(s)")
        self._log(f"{'='*60}")

        for config in self.models:
            pid = self._get_pipeline_id(config)
            self._log(f"\n  > Fitting: {pid}")
            try:
                t0 = time.time()
                pipeline = self._build_pipeline(config)
                pipeline.fit(X_fit, y_fit)
                self.fit_times_[pid] = time.time() - t0
                self.pipelines_[pid] = pipeline

                output = self._get_model_output(pipeline, X_val, is_tabtune=True)
                model_val_outputs[pid] = output
                self.individual_scores_[pid] = self._score_individual(output, y_val_np)

                self._log(
                    f"    score={self.individual_scores_[pid]:.4f}  "
                    f"time={self.fit_times_[pid]:.1f}s"
                )
            except Exception as exc:
                logger.error("[Ensemble] Failed to fit %s: %s", pid, exc)
                self._log(f"    FAILED: {exc}")

        # --- Step 2: GBDT baselines (optional) ---
        if self.include_gbdt_baselines:
            self._log(f"\n  Adding GBDT baselines ...")
            baselines = self._build_gbdt_baselines()
            for name, model in baselines.items():
                try:
                    t0 = time.time()
                    X_arr = X_fit.values if hasattr(X_fit, "values") else X_fit
                    y_arr = y_fit.values if hasattr(y_fit, "values") else y_fit
                    model.fit(X_arr, y_arr)
                    self.fit_times_[name] = time.time() - t0
                    self.pipelines_[name] = model

                    output = self._get_model_output(
                        model, X_val, is_tabtune=False
                    )
                    model_val_outputs[name] = output
                    self.individual_scores_[name] = self._score_individual(
                        output, y_val_np
                    )
                    self._log(
                        f"    {name}: score={self.individual_scores_[name]:.4f}  "
                        f"time={self.fit_times_[name]:.1f}s"
                    )
                except Exception as exc:
                    logger.error("[Ensemble] GBDT baseline %s failed: %s", name, exc)

        if len(model_val_outputs) < 1:
            raise RuntimeError(
                "[Ensemble] No models fitted successfully. "
                "Cannot build an ensemble."
            )
        if len(model_val_outputs) < 2:
            logger.warning(
                "[Ensemble] Only 1 model fitted. Ensemble may not be effective."
            )

        # --- Step 3: Fit the ensemble strategy ---
        self._log(f"\n{'='*60}")
        self._log(f"  Fitting strategy: {self.ensemble_strategy}")
        self._log(f"{'='*60}")

        strategy_kwargs: Dict[str, Any] = {"task_type": self.task_type}

        if self.ensemble_strategy == "weighted_averaging":
            strategy_kwargs["weight_scheme"] = self.weight_scheme
        elif self.ensemble_strategy == "greedy_selection":
            strategy_kwargs["ensemble_size"] = self.greedy_ensemble_size
            strategy_kwargs["metric"] = self.metric
        elif self.ensemble_strategy == "stacking":
            strategy_kwargs["meta_learner"] = self.meta_learner
        # temperature_scaled needs no extra kwargs

        self.strategy_ = get_strategy(self.ensemble_strategy, **strategy_kwargs)

        if self.ensemble_strategy == "stacking":
            X_orig = X_val.values if hasattr(X_val, "values") else X_val
            self.strategy_.fit(model_val_outputs, y_val_np, X_original=X_orig)
        elif self.ensemble_strategy == "weighted_averaging":
            self.strategy_.fit(model_val_outputs, y_val_np, metric=self.metric)
        else:
            self.strategy_.fit(model_val_outputs, y_val_np)

        # Ensemble validation score
        try:
            ensemble_preds = self.strategy_.predict(model_val_outputs)
            if self.task_type == "classification":
                self.ensemble_score_ = float(accuracy_score(y_val_np, ensemble_preds))
            else:
                self.ensemble_score_ = float(r2_score(y_val_np, ensemble_preds))
        except Exception as exc:
            logger.warning("[Ensemble] Could not score ensemble on val: %s", exc)
            self.ensemble_score_ = None

        self._log_fit_summary()

        # --- Step 4: Refit on full training data (if holdout was used) ---
        if X_val is not None and id(X_val) != id(X_train):
            self._refit_on_full_data(X_train, y_train)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Cascade stacking fit
    # ------------------------------------------------------------------

    def _fit_cascade(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[Any],
        y_val: Optional[Any],
    ) -> "TabularEnsemble":
        """Fit workflow for multi-level cascade stacking."""
        if X_val is None:
            stratify = y_train if self.task_type == "classification" else None
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train, y_train,
                test_size=self.holdout_fraction,
                random_state=42,
                stratify=stratify,
            )
        else:
            X_fit, y_fit = X_train, y_train
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)
            if not isinstance(y_val, pd.Series):
                y_val = pd.Series(y_val)

        y_fit_np = _to_numpy(y_fit)
        y_val_np = _to_numpy(y_val)
        
        if self.task_type == "classification" and self._y_le_ is not None:
            y_val_np = self._y_le_.transform(y_val_np)

        # --- FIX: For regression, n_classes=1 (one output column per model).
        # For classification, n_classes = number of unique classes.
        if self.task_type == "classification":
            n_classes = int(np.unique(y_fit_np).shape[0])
        else:
            n_classes = 1

        n_models = len(self.models)

        self._log(f"\n{'='*60}")
        self._log(f"  Cascade Stacking: {self.n_cascade_levels} levels x {n_models} models")
        self._log(f"{'='*60}")

        # Encode original features for skip connections.
        # --- FIX: Store the encoder so _predict_cascade can reuse it.
        self._cascade_encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
        X_fit_enc = self._cascade_encoder_.fit_transform(X_fit).astype(float)
        X_val_enc = self._cascade_encoder_.transform(X_val).astype(float)

        cur_X_fit_df = X_fit.copy()
        cur_X_val_df = X_val.copy()

        all_level_val_outputs: List[Dict[str, np.ndarray]] = []

        # --- FIX: Store per-level pipelines (keyed "L{level}_{pid}") so
        # _predict_cascade can replay the cascade at test time.
        self._cascade_level_pipelines_: Dict[str, Any] = {}
        self._cascade_n_classes_ = n_classes

        for level in range(self.n_cascade_levels):
            self._log(f"\n  -- Level {level + 1}/{self.n_cascade_levels} "
                       f"(input dim: {cur_X_fit_df.shape[1]}) --")

            # OOF generation
            oof_mat, fold_count = self._generate_oof(
                cur_X_fit_df, y_fit_np, n_classes, level_label=f"L{level + 1}"
            )

            # Full-train val predictions
            val_outputs: Dict[str, np.ndarray] = {}
            self._log(f"    Full-train fit for val predictions ...")
            y_fit_ser = pd.Series(y_fit_np)
            for config in self.models:
                pid = self._get_pipeline_id(config)
                try:
                    t0 = time.time()
                    pipe = self._build_pipeline(config)
                    pipe.fit(cur_X_fit_df, y_fit_ser)
                    self.fit_times_[f"L{level+1}_{pid}"] = time.time() - t0

                    output = self._get_model_output(pipe, cur_X_val_df, is_tabtune=True)
                    val_outputs[pid] = output

                    # --- FIX: Store EVERY level's pipeline (not just last).
                    # Key format: "L{level+1}_{pid}"
                    self._cascade_level_pipelines_[f"L{level+1}_{pid}"] = pipe
                    # Also keep last-level in self.pipelines_ for backward compat
                    if level == self.n_cascade_levels - 1:
                        self.pipelines_[pid] = pipe

                    score = self._score_individual(output, y_val_np)
                    self.individual_scores_[f"L{level+1}_{pid}"] = score
                    self._log(f"      {pid}: score={score:.4f}")
                except Exception as exc:
                    logger.error(
                        "[Cascade] Level %d, model %s failed: %s",
                        level + 1, pid, exc,
                    )

            all_level_val_outputs.append(val_outputs)

            # Skip connection: concat outputs + original encoded features.
            # For classification outputs are (n, n_classes); for regression (n,).
            # np.column_stack handles both 1D and 2D arrays correctly.
            if val_outputs:
                val_pred_parts = [val_outputs[self._get_pipeline_id(c)]
                                  for c in self.models
                                  if self._get_pipeline_id(c) in val_outputs]
                val_preds_stacked = np.column_stack(val_pred_parts)
            else:
                val_preds_stacked = np.zeros((len(X_val), 0))

            cur_X_fit_arr = np.hstack([oof_mat, X_fit_enc])
            cur_X_val_arr = np.hstack([val_preds_stacked, X_val_enc])
            cur_X_fit_df = pd.DataFrame(cur_X_fit_arr)
            cur_X_val_df = pd.DataFrame(cur_X_val_arr)

        # Final GES across all level outputs
        self._log(f"\n  Final GES over {sum(len(d) for d in all_level_val_outputs)} candidates")

        self.strategy_ = CascadeStackingEnsemble(
            n_levels=self.n_cascade_levels,
            n_folds=self.cv_folds,
            ges_size=self.greedy_ensemble_size,
            task_type=self.task_type,
        )
        self.strategy_.fit(
            all_level_val_outputs, y_val_np,
            model_names=[self._get_pipeline_id(c) for c in self.models],
        )

        # Ensemble validation score
        try:
            flat_outputs = {}
            for level_idx, level_out in enumerate(all_level_val_outputs):
                for name, arr in level_out.items():
                    flat_outputs[f"L{level_idx+1}_{name}"] = arr
            ensemble_preds = self.strategy_.predict(flat_outputs)
            # --- FIX: Use correct metric for scoring
            if self.task_type == "classification":
                self.ensemble_score_ = float(accuracy_score(y_val_np, ensemble_preds))
            else:
                self.ensemble_score_ = float(r2_score(y_val_np, ensemble_preds))
        except Exception as exc:
            logger.warning("[Cascade] Could not score ensemble: %s", exc)
            self.ensemble_score_ = None

        self._log_fit_summary()
        self._is_fitted = True
        return self

    def _generate_oof(
        self,
        X_fit: pd.DataFrame,
        y_fit: np.ndarray,
        n_classes: int,
        level_label: str = "L0",
    ) -> Tuple[np.ndarray, int]:
        """Generate OOF predictions for all models via K-fold CV.

        For classification, each model produces ``n_classes`` probability
        columns.  For regression (``n_classes=1``), each model produces
        a single prediction column.

        Returns
        -------
        oof_mat : ndarray of shape (n_train, n_models * n_classes)
        fold_count : number of folds used
        """
        n_train = len(X_fit)
        n_models = len(self.models)
        oof_mat = np.zeros((n_train, n_models * n_classes), dtype=float)

        is_reg = self.task_type == "regression"

        if self.task_type == "classification":
            kf = StratifiedKFold(
                n_splits=self.cv_folds, shuffle=True,
                random_state=self.base_seed,
            )
            split_iter = kf.split(X_fit, y_fit)
        else:
            kf = KFold(
                n_splits=self.cv_folds, shuffle=True,
                random_state=self.base_seed,
            )
            split_iter = kf.split(X_fit)

        for fold_idx, (tr_idx, oof_idx) in enumerate(split_iter):
            self._log(f"    [{level_label}] Fold {fold_idx + 1}/{self.cv_folds}")
            X_fold_tr = X_fit.iloc[tr_idx]
            X_fold_oof = X_fit.iloc[oof_idx]
            y_fold_tr = y_fit[tr_idx]

            for k, config in enumerate(self.models):
                try:
                    pipe = self._build_pipeline(config)
                    pipe.fit(X_fold_tr, pd.Series(y_fold_tr))
                    c0, c1 = k * n_classes, (k + 1) * n_classes

                    if is_reg:
                        # --- FIX: For regression, use predict() → 1D array.
                        # Assign into single-column slot per model.
                        preds = np.asarray(pipe.predict(X_fold_oof)).ravel()
                        oof_mat[oof_idx, c0] = preds
                    else:
                        # Classification: predict_proba → (n, n_classes)
                        oof_mat[oof_idx, c0:c1] = pipe.predict_proba(X_fold_oof)
                except Exception as exc:
                    logger.error(
                        "[OOF] Fold %d, model %s failed: %s",
                        fold_idx + 1, config.get("model_name", "?"), exc,
                    )

        return oof_mat, self.cv_folds

    # ------------------------------------------------------------------
    # Random-init (deep ensembles) fit
    # ------------------------------------------------------------------

    def _fit_random_init(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[Any],
        y_val: Optional[Any],
    ) -> "TabularEnsemble":
        """Fit workflow for deep ensembles via random initialisation."""
        if X_val is None:
            stratify = y_train if self.task_type == "classification" else None
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train, y_train,
                test_size=self.holdout_fraction,
                random_state=42,
                stratify=stratify,
            )
        else:
            X_fit, y_fit = X_train, y_train
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)
            if not isinstance(y_val, pd.Series):
                y_val = pd.Series(y_val)

        y_val_np = _to_numpy(y_val)
        # Encode for scoring 
        if self.task_type == "classification" and self._y_le_ is not None:
            y_val_np = self._y_le_.transform(y_val_np)
        seeds = [self.base_seed + m for m in range(self.n_seeds)]

        self._log(f"\n{'='*60}")
        self._log(f"  Deep Ensembles: {len(self.models)} models x {self.n_seeds} seeds")
        self._log(f"  Seeds: {seeds}")
        self._log(f"{'='*60}")

        val_per_seed: Dict[str, List[np.ndarray]] = {
            self._get_pipeline_id(c): [] for c in self.models
        }

        for config in self.models:
            pid = self._get_pipeline_id(config)
            self._log(f"\n  Model: {pid}")
            for seed in seeds:
                self._log(f"    seed={seed} ...", )
                try:
                    t0 = time.time()
                    # Inject seed into tuning_params
                    cfg_copy = dict(config)
                    tp = dict(cfg_copy.get("tuning_params", {}))
                    tp["seed"] = int(seed)
                    cfg_copy["tuning_params"] = tp

                    pipe = self._build_pipeline(cfg_copy)
                    pipe.fit(X_fit, y_fit)
                    elapsed = time.time() - t0

                    output = self._get_model_output(pipe, X_val, is_tabtune=True)
                    val_per_seed[pid].append(output)

                    # Store last-seed pipeline for predict
                    self.pipelines_[f"{pid}_seed{seed}"] = pipe
                    self.fit_times_[f"{pid}_seed{seed}"] = elapsed

                    score = self._score_individual(output, y_val_np)
                    self._log(f"      score={score:.4f}  time={elapsed:.1f}s")
                except Exception as exc:
                    logger.error(
                        "[DeepEnsemble] %s seed=%d failed: %s", pid, seed, exc
                    )

        # Score individual model averages
        for pid, outputs in val_per_seed.items():
            if outputs:
                avg = np.stack(outputs, axis=0).mean(axis=0)
                self.individual_scores_[pid] = self._score_individual(avg, y_val_np)

        # Fit the RandomInitEnsemble strategy
        self.strategy_ = RandomInitEnsemble(
            n_seeds=self.n_seeds,
            base_seed=self.base_seed,
            task_type=self.task_type,
        )
        self.strategy_.fit(val_per_seed, y_val_np)

        # Ensemble validation score
        try:
            ensemble_preds = self.strategy_.predict(val_per_seed)
            if self.task_type == "classification":
                self.ensemble_score_ = float(accuracy_score(y_val_np, ensemble_preds))
            else:
                self.ensemble_score_ = float(r2_score(y_val_np, ensemble_preds))
        except Exception as exc:
            logger.warning("[DeepEnsemble] Could not score ensemble: %s", exc)

        self._log_fit_summary()
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Refit & logging
    # ------------------------------------------------------------------

    def _refit_on_full_data(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> None:
        """Refit all base models on the full training data for final use."""
        self._log(f"\n  Refitting all models on full training data ...")
        for config in self.models:
            pid = self._get_pipeline_id(config)
            if pid in self.pipelines_:
                try:
                    pipeline = self._build_pipeline(config)
                    pipeline.fit(X_train, y_train)
                    self.pipelines_[pid] = pipeline
                except Exception as exc:
                    logger.error("[Ensemble] Refit failed for %s: %s", pid, exc)

    def _log_fit_summary(self) -> None:
        """Print a summary after fitting."""
        if not self.verbose:
            return
        print(f"\n{'='*60}")
        print(f"  Ensemble Fit Summary")
        print(f"{'='*60}")
        if self.individual_scores_:
            best_name = max(self.individual_scores_, key=self.individual_scores_.get)
            best_score = self.individual_scores_[best_name]
            print(f"  Best individual: {best_name} = {best_score:.4f}")
        if self.ensemble_score_ is not None:
            print(f"  Ensemble score:  {self.ensemble_score_:.4f}")
            if self.individual_scores_:
                improvement = self.ensemble_score_ - best_score
                print(f"  Improvement:     {improvement:+.4f}")
        if hasattr(self.strategy_, "weights_") and self.strategy_.weights_:
            # Show top-5 weights
            sorted_w = sorted(
                self.strategy_.weights_.items(), key=lambda x: x[1], reverse=True
            )[:5]
            print(f"  Top weights: {dict(sorted_w)}")
        print(f"{'='*60}\n")

    # ==================================================================
    # Predict
    # ==================================================================

    def predict(self, X_test: Any) -> np.ndarray:
        """Generate predictions from the ensemble.

        Parameters
        ----------
        X_test : DataFrame or array-like

        Returns
        -------
        ndarray of shape ``(n,)``

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        if self.ensemble_strategy == "random_init":
            preds = self._predict_random_init(X_test)
        elif self.ensemble_strategy == "cascade_stacking":
            preds = self._predict_cascade(X_test)
        else:
            model_outputs = self._collect_predictions(X_test)
            preds = self.strategy_.predict(model_outputs)

        
        if self.task_type == "classification" and self._y_le_ is not None:
            try:
                preds = self._y_le_.inverse_transform(np.asarray(preds).astype(int))
            except Exception:
                pass  # Fall back to returning raw indices if decoding fails

        return preds

    def predict_proba(self, X_test: Any) -> np.ndarray:
        """Generate probability predictions (classification only).

        Parameters
        ----------
        X_test : DataFrame or array-like

        Returns
        -------
        ndarray of shape ``(n, k)``

        Raises
        ------
        ValueError
            If task_type is not classification.
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification.")
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)

        if self.ensemble_strategy == "random_init":
            return self._predict_proba_random_init(X_test)
        elif self.ensemble_strategy == "cascade_stacking":
            return self._predict_proba_cascade(X_test)

        model_outputs = self._collect_predictions(X_test)
        return self.strategy_.predict_proba(model_outputs)

    def _collect_predictions(self, X_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Gather predictions from all fitted pipelines."""
        outputs: Dict[str, np.ndarray] = {}
        for pid, pipeline in self.pipelines_.items():
            try:
                is_tabtune = hasattr(pipeline, "predict_proba") and hasattr(pipeline, "evaluate")
                output = self._get_model_output(pipeline, X_test, is_tabtune=is_tabtune)
                outputs[pid] = output
            except Exception as exc:
                logger.error("[Ensemble] Predict failed for %s: %s", pid, exc)
        return outputs

    def _predict_random_init(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict for random_init strategy (collect per-seed outputs)."""
        per_seed: Dict[str, List[np.ndarray]] = {}
        for pid_key, pipeline in self.pipelines_.items():
            # pid_key format: "{model}_seed{N}"
            base_pid = "_".join(pid_key.split("_")[:-1])  # remove _seed{N}
            if base_pid not in per_seed:
                per_seed[base_pid] = []
            try:
                output = self._get_model_output(pipeline, X_test, is_tabtune=True)
                per_seed[base_pid].append(output)
            except Exception as exc:
                logger.error("[DeepEnsemble] Predict failed for %s: %s", pid_key, exc)
        return self.strategy_.predict(per_seed)

    def _predict_proba_random_init(self, X_test: pd.DataFrame) -> np.ndarray:
        """predict_proba for random_init strategy."""
        per_seed: Dict[str, List[np.ndarray]] = {}
        for pid_key, pipeline in self.pipelines_.items():
            base_pid = "_".join(pid_key.split("_")[:-1])
            if base_pid not in per_seed:
                per_seed[base_pid] = []
            try:
                output = self._get_model_output(pipeline, X_test, is_tabtune=True)
                per_seed[base_pid].append(output)
            except Exception as exc:
                logger.error("[DeepEnsemble] Predict failed for %s: %s", pid_key, exc)
        return self.strategy_.predict_proba(per_seed)

    def _replay_cascade(self, X_test: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Replay the cascade enrichment at test time.

        Runs each level's stored pipelines on progressively enriched
        features (mirroring what _fit_cascade did during training) and
        returns a flat dict of ``"L{level}_{pid}" -> predictions`` that
        matches the keys expected by ``self.strategy_``.
        """
        # Encode raw features with the same encoder used during fit
        X_test_enc = self._cascade_encoder_.transform(X_test).astype(float)
        n_classes = self._cascade_n_classes_

        cur_X_test_df = X_test.copy()
        flat_outputs: Dict[str, np.ndarray] = {}

        for level in range(self.n_cascade_levels):
            level_outputs: Dict[str, np.ndarray] = {}

            for config in self.models:
                pid = self._get_pipeline_id(config)
                level_key = f"L{level+1}_{pid}"
                pipe = self._cascade_level_pipelines_.get(level_key)
                if pipe is None:
                    logger.warning("[Cascade] No pipeline stored for %s", level_key)
                    continue
                try:
                    output = self._get_model_output(pipe, cur_X_test_df, is_tabtune=True)
                    level_outputs[pid] = output
                    flat_outputs[level_key] = output
                except Exception as exc:
                    logger.error("[Cascade] Predict failed for %s: %s", level_key, exc)

            # Build enriched features for next level:
            # [level outputs stacked | original encoded features]
            if level < self.n_cascade_levels - 1:
                if level_outputs:
                    test_pred_parts = [level_outputs[self._get_pipeline_id(c)]
                                       for c in self.models
                                       if self._get_pipeline_id(c) in level_outputs]
                    test_preds_stacked = np.column_stack(test_pred_parts)
                else:
                    test_preds_stacked = np.zeros((len(X_test), 0))
                cur_X_test_arr = np.hstack([test_preds_stacked, X_test_enc])
                cur_X_test_df = pd.DataFrame(cur_X_test_arr)

        return flat_outputs

    def _predict_cascade(self, X_test: pd.DataFrame) -> np.ndarray:
        """Predict for cascade_stacking strategy."""
        flat = self._replay_cascade(X_test)
        return self.strategy_.predict(flat)

    def _predict_proba_cascade(self, X_test: pd.DataFrame) -> np.ndarray:
        """predict_proba for cascade_stacking strategy."""
        flat = self._replay_cascade(X_test)
        return self.strategy_.predict_proba(flat)

    # ==================================================================
    # Evaluate
    # ==================================================================

    def evaluate(self, X_test: Any, y_test: Any) -> Dict[str, Any]:
        """Evaluate the ensemble and all individual models.

        Parameters
        ----------
        X_test : DataFrame or array-like
        y_test : Series or array-like

        Returns
        -------
        dict
            Contains ``"ensemble"`` metrics, ``"individual"`` per-model
            metrics, ``"weights"`` from the strategy, and ``"fit_times"``.
        """
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        y_test = _to_numpy(y_test)

        
        if self.task_type == "classification" and self._y_le_ is not None:
            y_test_enc = self._y_le_.transform(y_test)
        else:
            y_test_enc = y_test

        preds = self.predict(X_test)
        result: Dict[str, Any] = {
            "ensemble": {},
            "individual": {},
            "weights": {},
            "fit_times": dict(self.fit_times_),
        }

        # Ensemble metrics
        if self.task_type == "classification":
            # preds and y_test are both in original label space → direct comparison
            result["ensemble"]["accuracy"] = float(accuracy_score(y_test, preds))
            result["ensemble"]["f1_score"] = float(f1_score(y_test, preds, average="weighted"))
            try:
                probas = self.predict_proba(X_test)
                # Use integer-encoded y_test for probability-based metrics
                if probas.shape[1] == 2:
                    result["ensemble"]["roc_auc"] = float(
                        roc_auc_score(y_test_enc, probas[:, 1])
                    )
                else:
                    result["ensemble"]["roc_auc"] = float(
                        roc_auc_score(y_test_enc, probas, multi_class="ovr")
                    )
                result["ensemble"]["log_loss"] = float(log_loss(y_test_enc, probas))
                result["ensemble"]["brier_score"] = float(np.mean([
                    brier_score_loss((y_test_enc == c).astype(int), probas[:, c])
                    for c in range(probas.shape[1])
                ]))
            except Exception:
                pass
        else:
            result["ensemble"]["r2"] = float(r2_score(y_test, preds))
            result["ensemble"]["mse"] = float(mean_squared_error(y_test, preds))
            result["ensemble"]["rmse"] = float(np.sqrt(mean_squared_error(y_test, preds)))
            result["ensemble"]["mae"] = float(mean_absolute_error(y_test, preds))

        # Individual model scores
        for pid, pipeline in self.pipelines_.items():
            try:
                is_tabtune = hasattr(pipeline, "evaluate")
                if is_tabtune:
                    ind_preds = np.asarray(pipeline.predict(X_test))
                else:
                    X_arr = X_test.values if hasattr(X_test, "values") else X_test
                    ind_preds = np.asarray(pipeline.predict(X_arr))

                if self.task_type == "classification":
                    result["individual"][pid] = {
                        "accuracy": float(accuracy_score(y_test, ind_preds)),
                        "f1_score": float(f1_score(y_test, ind_preds, average="weighted")),
                    }
                else:
                    result["individual"][pid] = {
                        "r2": float(r2_score(y_test, ind_preds)),
                        "rmse": float(np.sqrt(mean_squared_error(y_test, ind_preds))),
                    }
            except Exception as exc:
                result["individual"][pid] = {"error": str(exc)}

        # Weights
        if hasattr(self.strategy_, "weights_") and self.strategy_.weights_:
            result["weights"] = dict(self.strategy_.weights_)

        return result

    # ==================================================================
    # Leaderboard
    # ==================================================================

    def get_leaderboard(self) -> pd.DataFrame:
        """Return a DataFrame ranking all models plus the ensemble.

        Returns
        -------
        DataFrame
            Columns: ``model``, ``val_score``, ``fit_time``, ``weight``.
            Sorted by ``val_score`` descending.
        """
        rows: List[Dict[str, Any]] = []
        for pid, score in self.individual_scores_.items():
            weight = None
            if hasattr(self.strategy_, "weights_") and self.strategy_.weights_:
                weight = self.strategy_.weights_.get(pid)
            rows.append({
                "model": pid,
                "val_score": score,
                "fit_time": self.fit_times_.get(pid, 0.0),
                "weight": weight,
            })
        rows.append({
            "model": f"ENSEMBLE ({self.ensemble_strategy})",
            "val_score": self.ensemble_score_,
            "fit_time": sum(self.fit_times_.values()),
            "weight": None,
        })
        df = pd.DataFrame(rows)
        ascending = self.metric in ("mse", "rmse", "mae", "log_loss")
        df = df.sort_values("val_score", ascending=ascending).reset_index(drop=True)
        return df

    # ==================================================================
    # Uncertainty (deep ensembles only)
    # ==================================================================

    def get_uncertainty(self) -> Optional[np.ndarray]:
        """Return per-sample epistemic uncertainty (deep ensembles only).

        Returns
        -------
        ndarray or None
            ``(n,)`` array of uncertainty values, or ``None`` if not
            applicable.
        """
        if isinstance(self.strategy_, RandomInitEnsemble):
            return self.strategy_.uncertainty_
        return None

    # ==================================================================
    # Serialisation helpers
    # ==================================================================

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary of the ensemble.

        Returns
        -------
        dict
        """
        return {
            "ensemble_strategy": self.ensemble_strategy,
            "task_type": self.task_type,
            "metric": self.metric,
            "n_models": len(self.models),
            "n_fitted_pipelines": len(self.pipelines_),
            "ensemble_score": self.ensemble_score_,
            "individual_scores": dict(self.individual_scores_),
            "fit_times": dict(self.fit_times_),
            "is_fitted": self._is_fitted,
        }