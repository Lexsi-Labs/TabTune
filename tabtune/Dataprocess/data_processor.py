from __future__ import annotations

import pandas as pd
import numpy as np
import logging
from .._internal.deprecation import warn_once
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, chi2
from category_encoders import TargetEncoder, HashingEncoder, BinaryEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids, NeighbourhoodCleaningRule

logger = logging.getLogger(__name__)


from .tabpfn_preprocessor import TabPFNPreprocessor
from .tabicl_preprocessor import TabICLPreprocessor
from .orion_msp_preprocessor import OrionMSPPreprocessor
from .contexttab_preprocessor import ContextTabPreprocessor
from .mitra_preprocessor import MitraPreprocessor 
from .orion_bix_preprocessor import OrionBixPreprocessor
from .tabdpt_preprocessor import TabDPTPreprocessor
from .limix_preprocessor import LimixPreprocessor
from .tabiclv2_preprocessor import TabICLv2Preprocessor
from .tabfm_preprocessor import TabFMPreprocessor
from .xrfm_preprocessor import XRFMPreprocessor
from .iltm_preprocessor import ILTMPreprocessor
from .exaone_preprocessor import EXAONEPreprocessor
from .regression.base_processor import RegressionDataProcessor
from .regression.tabpfn_processor import TabPFNRegressionProcessor
from .regression.contexttab_processor import ContextTabRegressionProcessor
from .regression.tabdpt_processor import TabDPTRegressionProcessor
from .regression.mitra_processor import MitraRegressionProcessor
from .regression.limix_processor import LimixRegressionProcessor
from .regression.tabfm_processor import TabFMRegressionProcessor
from .regression.xrfm_processor import XRFMRegressionProcessor
from .regression.iltm_processor import ILTMRegressionProcessor
from .regression.exaone_processor import EXAONERegressionProcessor

class DataProcessor(BaseEstimator, TransformerMixin):
    """
    The complete Data Preparation Engine for the TabTune library. Integrates
    a full suite of standard preprocessing tools with custom, model-specific
    logic.
    """
    #: Strategy applied when the user gives no explicit value and the model has
    #: no model-aware default of its own.
    _FALLBACK_STRATEGIES = {
        'imputation_strategy': 'mean',
        'categorical_encoding': 'onehot',
        'scaling_strategy': 'standard',
    }

    #: model_name -> preprocessing defaults. ``categorical_encoding`` doubles as
    #: the key selecting the model-specific preprocessor in
    #: :meth:`_get_custom_preprocessor`.
    MODEL_AWARE_DEFAULTS = {
        'TabPFN': {'categorical_encoding': 'tabpfn_special'},
        'TabICL': {'categorical_encoding': 'tabicl_special'},
        'OrionMSP': {'categorical_encoding': 'orion_msp_special'},
        'OrionMSPv1.5': {'categorical_encoding': 'orion_msp_special'},
        'ContextTab': {'categorical_encoding': 'contexttab_special'},
        'Mitra': {'categorical_encoding': 'mitra_special'},
        'OrionBix': {'categorical_encoding': 'orion_bix_special'},
        'TabDPT': {'categorical_encoding': 'tabdpt_special'},
        'Limix': {'categorical_encoding': 'limix_special'},
        'TabICLv2': {'categorical_encoding': 'tabiclv2_special'},
        'TabPFNv26': {'categorical_encoding': 'tabpfn_special'},
        'TabPFNv3': {'categorical_encoding': 'tabpfn_special'},
        'TabFM': {'categorical_encoding': 'tabfm_special'},
        'XRFM': {'categorical_encoding': 'xrfm_special'},
        'ILTM': {'categorical_encoding': 'iltm_special'},
        'EXAONETabular': {'categorical_encoding': 'exaone_special'},
    }

    def __init__(
        self,
        model_name: str | None = None,
        override_types: dict[str, str] | None = None,
        imputation_strategy: str | None = None,
        categorical_encoding: str | None = None,
        scaling_strategy: str | None = None,
        resampling_strategy: str | None = None,
        feature_selection_strategy: str | None = None,
        feature_selection_k: int = 10,
        correlation_threshold: float | None = None,
        model_params: dict | None = None,
        task_type: str = 'classification',
    ) -> None:
        """Model-aware preprocessing engine.

        Args:
            model_name: Name of the target model. Selects a model-specific
                preprocessor and supplies defaults for unset strategies.
            override_types: Optional explicit column-type overrides.
            imputation_strategy: ``'mean'``, ``'median'``, ``'iterative'``,
                ``'knn'`` or ``'none'``. ``None`` means "let the model decide".
            categorical_encoding: ``'onehot'``, ``'ordinal'``, ``'target'``,
                ``'hashing'``, ``'binary'`` or ``'none'``. ``None`` selects the
                model-aware preprocessor.
            scaling_strategy: ``'standard'``, ``'minmax'``, ``'robust'``,
                ``'power_transform'`` or ``'none'``.
            resampling_strategy: ``'smote'``, ``'random_over'``,
                ``'random_under'``, ``'tomek'``, ``'kmeans'`` or ``'knn'``.
                Classification only; applied by :meth:`fit_resample`.
            feature_selection_strategy: ``'variance'``, ``'select_k_best_anova'``,
                ``'select_k_best_chi2'`` or ``'correlation'``.
            feature_selection_k: ``k`` for the ``select_k_best_*`` strategies.
            model_params: Model-specific parameters forwarded to the
                model-aware preprocessor.
            task_type: ``'classification'`` or ``'regression'``.

        .. versionchanged:: 0.2.0
           Strategy arguments now default to ``None`` ("auto") instead of
           concrete values, and explicit user choices are honoured rather than
           being silently overwritten by the model-aware defaults. Imputation,
           scaling and feature selection are applied as a pre-stage when a
           model-specific preprocessor is also active, and they now take effect
           for classification as well as regression.
        """
        self.model_name = model_name
        self.task_type = task_type
        self.override_types = override_types
        self.imputation_strategy = imputation_strategy
        self.categorical_encoding = categorical_encoding
        self.scaling_strategy = scaling_strategy
        self.resampling_strategy = resampling_strategy
        self.feature_selection_strategy = feature_selection_strategy
        self.feature_selection_k = feature_selection_k
        self.correlation_threshold = correlation_threshold
        self.model_params = model_params or {}

        # Record what the caller asked for *before* defaults are filled in, so
        # an explicit choice can be distinguished from an unset one. Prior to
        # 0.2.0 this distinction did not exist and user values were discarded.
        self._user_specified = {
            name
            for name in (
                'imputation_strategy',
                'categorical_encoding',
                'scaling_strategy',
                'feature_selection_strategy',
                'resampling_strategy',
            )
            if getattr(self, name) is not None
        }

        self._set_model_aware_defaults()

        # --- Internal State Attributes ---
        self.column_types_ = {}
        self.numerical_cols_ = []
        self.categorical_cols_ = []
        self._is_fitted = False
        self.custom_preprocessor_ = None
        self.imputer_ = None
        self.scaler_ = None
        self.encoder_ = None
        self.resampler_ = None
        self.selector_ = None
        self.label_encoder_ = None
        self.regression_processor_ = None
        self.feature_names_ = None
        self.feature_names_in_ = None
        self._correlation_cols_to_drop = []
        self.original_cols_ = None
        self.processing_summary_ = {}
        # Standard steps that run *before* a model-specific preprocessor.
        self._pre_stage_active = False

    # ------------------------------------------------------------------ setup

    def user_specified(self, name: str) -> bool:
        """Return whether the caller explicitly set the strategy ``name``.

        Used to decide whether a model-aware default may overwrite a value and
        whether the standard pre-stage should run at all.
        """
        return name in getattr(self, '_user_specified', set())

    def _set_model_aware_defaults(self) -> None:
        """Fill unset strategies from the model's defaults.

        Explicit user choices always win. Overriding ``categorical_encoding``
        for a model that ships a dedicated preprocessor disables that
        preprocessor, which is occasionally what the user wants and is usually a
        mistake, so it warns.
        """
        config = self.MODEL_AWARE_DEFAULTS.get(self.model_name) if self.model_name else None

        if config and self.user_specified('categorical_encoding'):
            expected = config.get('categorical_encoding')
            if expected and self.categorical_encoding != expected:
                warn_once(
                    f"categorical_encoding={self.categorical_encoding!r} overrides the "
                    f"model-aware preprocessor for {self.model_name!r}. TabTune will use "
                    f"generic encoding instead of {expected!r}, which frequently degrades "
                    f"accuracy for this model. Omit categorical_encoding to keep the "
                    f"model-aware pipeline.",
                    UserWarning,
                    key=f"cat-encoding-override:{self.model_name}",
                )

        for name, value in (config or {}).items():
            if not self.user_specified(name):
                setattr(self, name, value)

        # Anything still unset and not model-supplied falls back to the generic
        # default. ``categorical_encoding`` is deliberately left as None for
        # unknown models so that _get_custom_preprocessor returns None and the
        # standard path takes over with the documented default.
        for name, fallback in self._FALLBACK_STRATEGIES.items():
            if getattr(self, name) is None and not config:
                setattr(self, name, fallback)

    def _get_custom_preprocessor(self):
        """Factory method to return the correct custom preprocessor instance."""
        special_encoders = {
            'tabpfn_special': TabPFNPreprocessor,
            'tabicl_special': TabICLPreprocessor,
            'orion_msp_special': OrionMSPPreprocessor,
            'contexttab_special': ContextTabPreprocessor,
            'mitra_special': MitraPreprocessor,
            'orion_bix_special': OrionBixPreprocessor,
            'tabdpt_special': TabDPTPreprocessor,
            'limix_special': LimixPreprocessor,
            'tabiclv2_special': TabICLv2Preprocessor,
            'tabfm_special': TabFMPreprocessor,
            'xrfm_special': XRFMPreprocessor,
            'iltm_special': ILTMPreprocessor,
            'exaone_special': EXAONEPreprocessor,
        }
        if self.categorical_encoding in special_encoders:
            logger.info(f"[DataProcessor] Using special preprocessor for: {self.model_name}")
            PreprocessorClass = special_encoders[self.categorical_encoding]
            if self.categorical_encoding == 'tabfm_special':
                # TabFM preprocessor needs task_type (label-encode target for classification).
                return PreprocessorClass(task_type=self.task_type)
            if self.categorical_encoding == 'xrfm_special':
                # XRFM preprocessor needs task_type (label-encode target for classification).
                return PreprocessorClass(task_type=self.task_type)
            if self.categorical_encoding == 'iltm_special':
                # ILTM preprocessor needs task_type (label-encode target for classification).
                return PreprocessorClass(task_type=self.task_type)
            if self.categorical_encoding == 'exaone_special':
                # EXAONE preprocessor needs task_type (label-encode target for
                # classification). Features pass through untouched: the wrappers
                # run EXAONEFeatureEncoder themselves, so encoding here would
                # double-encode the frame.
                return PreprocessorClass(task_type=self.task_type)
            if self.categorical_encoding == 'contexttab_special':
                # Extract regression parameters from model_params
                regression_type = self.model_params.get('regression_type', 'l2')
                num_regression_bins = self.model_params.get('num_regression_bins', 16)
                # Pass task_type to ContextTab preprocessor
                return PreprocessorClass(regression_type=regression_type, num_regression_bins=num_regression_bins, task_type=self.task_type)
            elif self.categorical_encoding == 'tabdpt_special':
                # Pass task_type to TabDPT preprocessor
                return PreprocessorClass(task_type=self.task_type)
            elif self.categorical_encoding == 'limix_special':
                # Pass task_type to Limix preprocessor
                return PreprocessorClass(task_type=self.task_type)
            elif self.categorical_encoding == 'mitra_special':
                # Pass task_type to Mitra preprocessor
                return PreprocessorClass(task_type=self.task_type)
            return PreprocessorClass()
        return None

    def _get_regression_processor(self):
        """Factory method to return the correct regression processor instance.
        
        Note: Most models handle target normalization internally, so default is 'none'.
        Only override if explicitly specified in model_params.
        """
        # Get target_scaling from model_params, but use model-specific defaults if not specified
        target_scaling = self.model_params.get('target_scaling', None)
        
        if self.model_name == 'TabPFN':
            # TabPFN handles normalization internally, default to 'none'
            if target_scaling is None:
                target_scaling = 'none'
            return TabPFNRegressionProcessor(target_scaling_strategy=target_scaling)
        elif self.model_name == 'ContextTab':
            # ContextTab handles normalization internally, default to 'none'
            if target_scaling is None:
                target_scaling = 'none'
            regression_type = self.model_params.get('regression_type', 'l2')
            num_regression_bins = self.model_params.get('num_regression_bins', 16)
            return ContextTabRegressionProcessor(
                target_scaling_strategy=target_scaling,
                regression_type=regression_type,
                num_regression_bins=num_regression_bins
            )
        elif self.model_name == 'TabDPT':
            # TabDPT handles normalization internally, default to 'none'
            if target_scaling is None:
                target_scaling = 'none'
            return TabDPTRegressionProcessor(target_scaling_strategy=target_scaling)
        elif self.model_name == 'Mitra':
            # Mitra handles normalization internally, default to 'none'
            if target_scaling is None:
                target_scaling = 'none'
            return MitraRegressionProcessor(target_scaling_strategy=target_scaling)
        elif self.model_name == 'Limix':
            # LimiX handles normalization internally, default to 'none'
            if target_scaling is None:
                target_scaling = 'none'
            return LimixRegressionProcessor(target_scaling_strategy=target_scaling)
        elif self.model_name in ('TabPFNv26', 'TabPFNv3'):
            # TabPFN v2.6 / v3 handle target normalization internally -> default 'none'.
            if target_scaling is None:
                target_scaling = 'none'
            return TabPFNRegressionProcessor(target_scaling_strategy=target_scaling)
        elif self.model_name == 'TabFM':
            # TabFM handles feature encoding + target internally, default to 'none'.
            if target_scaling is None:
                target_scaling = 'none'
            return TabFMRegressionProcessor(target_scaling_strategy=target_scaling)
        elif self.model_name == 'XRFM':
            # The xRFM wrapper standardises the target internally, default to 'none'.
            if target_scaling is None:
                target_scaling = 'none'
            return XRFMRegressionProcessor(target_scaling_strategy=target_scaling)
        elif self.model_name == 'ILTM':
            # iLTM normalises the regression target internally, default to 'none'.
            if target_scaling is None:
                target_scaling = 'none'
            return ILTMRegressionProcessor(target_scaling_strategy=target_scaling)
        elif self.model_name == 'EXAONETabular':
            # The vendored EXAONE regressor centres/scales the target internally
            # and predict() already returns the ORIGINAL space, so any
            # pipeline-level scaling would never be inverted -> force 'none'.
            if target_scaling is None:
                target_scaling = 'none'
            return EXAONERegressionProcessor(target_scaling_strategy=target_scaling)

        # Fallback to generic processor (use 'standard' for unknown models)
        if target_scaling is None:
            target_scaling = 'standard'
        return RegressionDataProcessor(target_scaling_strategy=target_scaling)


    # -------------------------------------------------------------- fit/transform

    @property
    def has_pre_stage(self) -> bool:
        """Whether standard steps run before the model-specific preprocessor.

        True when the user explicitly asked for imputation, scaling or feature
        selection *and* a model-specific preprocessor is also in play. Before
        0.2.0 those requests were silently discarded in this situation.
        """
        return bool(self._pre_stage_active)

    def _wants_pre_stage(self) -> bool:
        return any(
            self.user_specified(name)
            for name in ('imputation_strategy', 'scaling_strategy', 'feature_selection_strategy')
        )

    def fit(self, X: "pd.DataFrame", y: "pd.Series | None" = None) -> "DataProcessor":
        """Fit every preprocessing component.

        Args:
            X: Training features, a :class:`pandas.DataFrame`.
            y: Training target. Required for target encoding, supervised
                feature selection and regression target scaling.

        Returns:
            ``self``.
        """
        X_fit = X.copy()
        self.original_cols_ = X_fit.columns.tolist()
        self.feature_names_in_ = list(self.original_cols_)
        y_fit = y.copy() if y is not None else None

        # Column types are needed by both paths: the standard pipeline uses them
        # directly, and the pre-stage needs them to know what to impute/scale.
        self._infer_column_types(X_fit)

        self.custom_preprocessor_ = self._get_custom_preprocessor()

        if self.custom_preprocessor_:
            self.processing_summary_['strategy'] = 'custom'
            self.processing_summary_['steps'] = {}

            # Pre-stage: honour explicit imputation/scaling/feature-selection
            # requests before the model-aware preprocessor sees the data.
            # ``categorical_encoding`` is excluded by design - for these models
            # it is the switch that *selects* the preprocessor, so running a
            # second encoder here would fight with it.
            if self._wants_pre_stage():
                logger.info(
                    "[DataProcessor] Applying standard pre-stage (%s) before the "
                    "%s preprocessor.",
                    ", ".join(
                        f"{n.replace('_strategy', '')}={getattr(self, n)!r}"
                        for n in ('imputation_strategy', 'scaling_strategy', 'feature_selection_strategy')
                        if self.user_specified(n)
                    ),
                    self.model_name,
                )
                self._fit_standard_components(X_fit, y_fit, include_encoding=False)
                self._pre_stage_active = True
                X_fit = self._apply_standard_transforms(X_fit)

            self.custom_preprocessor_.fit(X_fit, y)

            if hasattr(self.custom_preprocessor_, 'get_summary'):
                summary = self.custom_preprocessor_.get_summary()
                if isinstance(summary, dict):
                    self.processing_summary_['steps'].update(summary)
        else:
            self.processing_summary_['strategy'] = 'standard'
            self.processing_summary_.setdefault('steps', {})
            self._fit_standard_components(X_fit, y_fit)

        # Target handling applies to both paths.
        if self.task_type == 'classification':
            if y_fit is not None and not self.custom_preprocessor_:
                # Model-specific preprocessors own their own label encoding.
                self.label_encoder_ = LabelEncoder().fit(y_fit)
                self.processing_summary_['target_encoding'] = 'LabelEncoder'
        elif self.task_type == 'regression':
            if y_fit is not None:
                self.regression_processor_ = self._get_regression_processor()
                self.regression_processor_.fit(y_fit)
                self.processing_summary_['target_encoding'] = type(
                    self.regression_processor_
                ).__name__

        self._is_fitted = True
        self.feature_names_ = self._resolve_feature_names(X_fit)
        logger.info("[DataProcessor] All components for pipeline have been fitted.")
        return self

    def transform(
        self, X: "pd.DataFrame", y: "pd.Series | None" = None
    ) -> "pd.DataFrame | tuple":
        """Apply the fitted preprocessing pipeline.

        Args:
            X: Features to transform.
            y: Optional target. When given, the transformed target is returned
                alongside the features.

        Returns:
            The transformed features, or a ``(features, target)`` tuple when
            ``y`` is provided and a target transformer was fitted.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before calling transform().")

        X_transformed = X.copy()

        if self.custom_preprocessor_:
            if self._pre_stage_active:
                X_transformed = self._apply_standard_transforms(X_transformed)
            if self.model_name == 'TabPFN':
                return self.custom_preprocessor_.transform(X_transformed)
            return self.custom_preprocessor_.transform(X_transformed, y)

        X_transformed = self._apply_standard_transforms(X_transformed)

        if y is not None:
            if self.task_type == 'classification' and self.label_encoder_ is not None:
                return X_transformed, self.label_encoder_.transform(y)
            if self.task_type == 'regression' and self.regression_processor_ is not None:
                return X_transformed, self.regression_processor_.transform(y)

        return X_transformed

    def fit_transform(
        self, X: "pd.DataFrame", y: "pd.Series | None" = None, **fit_params
    ) -> "pd.DataFrame | tuple":
        """Fit, transform, and apply resampling when configured.

        Returns:
            The same shape as :meth:`transform`, after resampling is applied.

        .. versionchanged:: 0.2.0
           Resampling now also runs for models with a model-specific
           preprocessor. Previously ``fit_transform`` returned early for those
           models and ``resampling_strategy`` had no effect.
        """
        self.fit(X, y)
        transformed = self.transform(X, y)

        if not self.resampling_strategy or y is None:
            return transformed

        if isinstance(transformed, tuple):
            X_transformed, y_transformed = transformed
        else:
            X_transformed, y_transformed = transformed, y

        X_resampled, y_resampled = self.fit_resample(X_transformed, y_transformed)
        if isinstance(transformed, tuple):
            return X_resampled, y_resampled
        return X_resampled

    def fit_resample(self, X: "pd.DataFrame", y: "pd.Series") -> tuple:
        """Fit and apply the configured resampler to already-transformed data.

        Exposed publicly so :class:`~tabtune.TabularPipeline` can resample
        without going through :meth:`fit_transform`. Resampling is a
        classification-only operation and is skipped with a warning otherwise.

        Args:
            X: Transformed features.
            y: Transformed target.

        Returns:
            A ``(features, target)`` tuple. The inputs are returned unchanged
            when no strategy is configured or resampling is not applicable.
        """
        if not self.resampling_strategy:
            return X, y

        if self.task_type != 'classification':
            warn_once(
                f"resampling_strategy={self.resampling_strategy!r} is only supported for "
                f"classification; ignoring it for task_type={self.task_type!r}.",
                UserWarning,
                key=f"resample-task:{self.resampling_strategy}",
            )
            return X, y

        self._fit_resampler(y)
        if self.resampler_ is None:
            warn_once(
                f"Unknown resampling_strategy={self.resampling_strategy!r}; no resampling "
                f"applied. Valid values: 'smote', 'random_over', 'random_under', "
                f"'tomek', 'kmeans', 'knn'.",
                UserWarning,
                key=f"resample-unknown:{self.resampling_strategy}",
            )
            return X, y

        n_before = len(y)
        try:
            X_resampled, y_resampled = self.resampler_.fit_resample(X, y)
        except Exception as exc:
            # Resampling is a convenience, not a correctness requirement:
            # SMOTE in particular fails on tiny minority classes. Degrading to
            # the original data is better than losing the run.
            warn_once(
                f"Resampling with {self.resampling_strategy!r} failed ({exc}); "
                f"continuing with the original data.",
                UserWarning,
                key=f"resample-failed:{self.resampling_strategy}",
            )
            return X, y

        logger.info(
            "[DataProcessor] Resampled with '%s': %d -> %d rows.",
            self.resampling_strategy,
            n_before,
            len(y_resampled),
        )
        self.processing_summary_['resampling'] = {
            'strategy': self.resampling_strategy,
            'rows_before': int(n_before),
            'rows_after': int(len(y_resampled)),
        }
        return X_resampled, y_resampled

    def _resolve_feature_names(self, X_fit: "pd.DataFrame") -> list[str]:
        """Best-effort output feature names, for downstream label recovery."""
        try:
            if self.custom_preprocessor_ is not None:
                for attr in ('feature_names_out_', 'feature_names_', 'columns_'):
                    names = getattr(self.custom_preprocessor_, attr, None)
                    if names is not None:
                        return list(names)
                return list(self.original_cols_)
            transformed = self._apply_standard_transforms(X_fit.head(min(len(X_fit), 5)))
            return list(getattr(transformed, 'columns', self.original_cols_))
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.debug("[DataProcessor] Could not resolve feature names: %s", exc)
            return list(self.original_cols_ or [])

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return output feature names (scikit-learn transformer protocol)."""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before get_feature_names_out().")
        return list(self.feature_names_ or self.original_cols_ or [])

    def get_processing_summary(self) -> str:
        """
        Returns a formatted string summarizing the data processing steps.
        """
        if not self._is_fitted:
            logger.warning("[DataProcessor] DataProcessor has not been fitted yet.")
            raise RuntimeError("DataProcessor has not been fitted yet.")

        summary_lines = ["--- Data Processing Summary ---"]

        if self.processing_summary_.get('strategy') == 'custom':
            summary_lines.append(f"\n[Custom Preprocessing for '{self.model_name}']")
            
            steps = self.processing_summary_.get('steps', {})
            if not steps:
                summary_lines.append("  - No detailed summary available for this preprocessor.")
            else:
                summary_lines.append("\n  Applied Steps:")
                # --- NEW: Detailed loop for rich summary ---
                for i, (step_name, step_info) in enumerate(steps.items()):
                    summary_lines.append(f"    {i+1}. {step_name}:")
                    # Some entries (e.g. a per-column breakdown) are structured
                    # data rather than a description/details pair, so read the
                    # optional keys defensively instead of indexing directly.
                    description = step_info.get('description') if isinstance(step_info, dict) else None
                    if description:
                        summary_lines.append(f"       - {description}")
                    details = step_info.get('details', []) if isinstance(step_info, dict) else []
                    for detail_line in details:
                        summary_lines.append(f"       - {detail_line}")
                
        elif self.processing_summary_.get('strategy') == 'standard':
            summary_lines.append("\n[Standard Preprocessing Pipeline]")
            
            steps = self.processing_summary_.get('steps', {})
            processed_cols = set()

            if 'imputation' in steps:
                step_info = steps['imputation']
                summary_lines.append(f"\n1. Imputation (Strategy: '{step_info['strategy']}')")
                summary_lines.append(f"   - Applied to {len(step_info['columns'])} numerical features: {', '.join(f'`{c}`' for c in step_info['columns'])}")
                processed_cols.update(step_info['columns'])

            if 'categorical_encoding' in steps:
                step_info = steps['categorical_encoding']
                summary_lines.append(f"\n2. Categorical Encoding (Strategy: '{step_info['strategy']}')")
                summary_lines.append(f"   - Applied to {len(step_info['columns'])} categorical features: {', '.join(f'`{c}`' for c in step_info['columns'])}")
                processed_cols.update(step_info['columns'])
            
            if 'scaling' in steps:
                step_info = steps['scaling']
                summary_lines.append(f"\n3. Scaling (Strategy: '{step_info['strategy']}')")
                summary_lines.append(f"   - Applied to {len(step_info['columns'])} features (original numerical + encoded categorical).")
                processed_cols.update(step_info['columns'])

            if 'feature_selection' in steps:
                step_info = steps['feature_selection']
                summary_lines.append(f"\n4. Feature Selection (Strategy: '{step_info['strategy']}')")
                if 'dropped_columns' in step_info and step_info['dropped_columns']:
                    summary_lines.append(f"   - Removed {len(step_info['dropped_columns'])} features: {', '.join(f'`{c}`' for c in step_info['dropped_columns'])}")
                else:
                    summary_lines.append("   - No features were removed by this step.")
            
            untouched_features = [col for col in self.original_cols_ if col not in processed_cols and col not in (steps.get('feature_selection', {}).get('dropped_columns', []))]
            summary_lines.append(f"\n[Untouched Features]")
            if untouched_features:
                summary_lines.append(f"  - {len(untouched_features)} features were not modified: {', '.join(f'`{c}`' for c in untouched_features)}")
            else:
                summary_lines.append("  - All features were processed by at least one step.")
        else:
            summary_lines.append("No processing steps were recorded.")

        resampling = self.processing_summary_.get('resampling')
        if resampling:
            summary_lines.append("\n[Resampling]")
            if isinstance(resampling, dict):
                summary_lines.append(
                    f"  - Strategy: '{resampling['strategy']}' "
                    f"({resampling['rows_before']} -> {resampling['rows_after']} rows)."
                )
            else:  # pre-0.2.0 summaries stored a bare strategy string
                summary_lines.append(f"  - Strategy: '{resampling}' applied to the training data.")

        if self.has_pre_stage:
            summary_lines.append("\n[Standard Pre-Stage]")
            summary_lines.append(
                "  - Explicit imputation/scaling/feature-selection settings were applied "
                f"before the '{self.model_name}' model-aware preprocessor."
            )

        return "\n".join(summary_lines)

    def _infer_column_types(self, X: "pd.DataFrame") -> None:
        """Split columns into numerical and categorical, honouring overrides."""
        self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(exclude=np.number).columns.tolist()

        for column, kind in (self.override_types or {}).items():
            if column not in X.columns:
                warn_once(
                    f"override_types references unknown column {column!r}; ignoring it.",
                    UserWarning,
                    key=f"override-unknown:{column}",
                )
                continue
            if kind == 'numerical':
                self.categorical_cols_ = [c for c in self.categorical_cols_ if c != column]
                if column not in self.numerical_cols_:
                    self.numerical_cols_.append(column)
            elif kind == 'categorical':
                self.numerical_cols_ = [c for c in self.numerical_cols_ if c != column]
                if column not in self.categorical_cols_:
                    self.categorical_cols_.append(column)
            else:
                warn_once(
                    f"override_types[{column!r}]={kind!r} is not recognised; expected "
                    f"'numerical' or 'categorical'.",
                    UserWarning,
                    key=f"override-kind:{column}",
                )

        self.column_types_ = {
            'numerical': list(self.numerical_cols_),
            'categorical': list(self.categorical_cols_),
        }

    def _fit_standard_components(self, X: "pd.DataFrame", y=None, *, include_encoding: bool = True) -> None:
        """Fit imputation, encoding, scaling and feature selection.

        Args:
            X: Training features. Mutated in place for imputation, matching the
                previous behaviour.
            y: Training target, used by supervised encoders and selectors.
            include_encoding: Fit the categorical encoder. Disabled for the
                pre-stage, where the model-specific preprocessor owns encoding.
        """
        steps = self.processing_summary_.setdefault('steps', {})

        if self.imputation_strategy not in (None, 'none') and self.numerical_cols_:
            imputer_map = {
                'mean': lambda: SimpleImputer(strategy='mean'),
                'median': lambda: SimpleImputer(strategy='median'),
                'most_frequent': lambda: SimpleImputer(strategy='most_frequent'),
                'iterative': lambda: IterativeImputer(random_state=42),
                'knn': lambda: KNNImputer(),
            }
            factory = imputer_map.get(self.imputation_strategy)
            if factory is None:
                warn_once(
                    f"Unknown imputation_strategy={self.imputation_strategy!r}; using 'mean'. "
                    f"Valid values: {sorted(imputer_map)}.",
                    UserWarning,
                    key=f"impute-unknown:{self.imputation_strategy}",
                )
                factory = imputer_map['mean']
            self.imputer_ = factory()
            X[self.numerical_cols_] = self.imputer_.fit_transform(X[self.numerical_cols_])
            steps['imputation'] = {
                'strategy': self.imputation_strategy,
                'columns': list(self.numerical_cols_),
            }

        if include_encoding and self.categorical_encoding not in (None, 'none') and self.categorical_cols_:
            encoder_map = {
                'onehot': lambda: OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                'ordinal': lambda: OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
                'target': lambda: TargetEncoder(),
                'hashing': lambda: HashingEncoder(),
                'binary': lambda: BinaryEncoder(),
            }
            factory = encoder_map.get(self.categorical_encoding)
            if factory is None:
                warn_once(
                    f"Unknown categorical_encoding={self.categorical_encoding!r}; using "
                    f"'onehot'. Valid values: {sorted(encoder_map)}.",
                    UserWarning,
                    key=f"encoding-unknown:{self.categorical_encoding}",
                )
                factory = encoder_map['onehot']
            self.encoder_ = factory()
            self.encoder_.fit(X[self.categorical_cols_], y)
            steps['categorical_encoding'] = {
                'strategy': self.categorical_encoding,
                'columns': list(self.categorical_cols_),
            }

        X_encoded = self._apply_encoding(X)
        numeric_for_scaling = X_encoded.select_dtypes(include=np.number).columns.tolist()

        if self.scaling_strategy not in (None, 'none') and numeric_for_scaling:
            scaler_map = {
                'standard': lambda: StandardScaler(),
                'minmax': lambda: MinMaxScaler(),
                'robust': lambda: RobustScaler(),
                'power_transform': lambda: PowerTransformer(),
            }
            factory = scaler_map.get(self.scaling_strategy)
            if factory is None:
                warn_once(
                    f"Unknown scaling_strategy={self.scaling_strategy!r}; using 'standard'. "
                    f"Valid values: {sorted(scaler_map)}.",
                    UserWarning,
                    key=f"scaling-unknown:{self.scaling_strategy}",
                )
                factory = scaler_map['standard']
            self.scaler_ = factory()
            self.scaler_.fit(X_encoded[numeric_for_scaling])
            steps['scaling'] = {
                'strategy': self.scaling_strategy,
                'columns': list(numeric_for_scaling),
            }

        X_scaled = self._apply_scaling(X_encoded, numeric_for_scaling)

        if self.feature_selection_strategy not in (None, 'none'):
            self._fit_feature_selector(X_scaled, y)

    def _apply_standard_transforms(self, X):
        X_transformed = X.copy()
        if self.imputer_ and self.numerical_cols_:
            X_transformed[self.numerical_cols_] = self.imputer_.transform(X_transformed[self.numerical_cols_])
        
        X_transformed = self._apply_encoding(X_transformed)
        numeric_for_scaling = X_transformed.select_dtypes(include=np.number).columns.tolist()
        X_transformed = self._apply_scaling(X_transformed, numeric_for_scaling)

        if self.selector_ or self._correlation_cols_to_drop:
            X_transformed = self._apply_feature_selection(X_transformed)
            
        return X_transformed
        
    def _apply_encoding(self, X):
        if self.encoder_ is None or not self.categorical_cols_:
            return X
        encoded_data = self.encoder_.transform(X[self.categorical_cols_])
        try:
            encoded_cols = self.encoder_.get_feature_names_out(self.categorical_cols_)
        except Exception as exc:
            # category_encoders transformers do not all implement the
            # get_feature_names_out protocol; fall back to positional names.
            logger.debug("[DataProcessor] Encoder has no feature names: %s", exc)
            encoded_cols = [f"cat_{i}" for i in range(np.asarray(encoded_data).shape[1])]
        encoded_df = pd.DataFrame(
            np.asarray(encoded_data), index=X.index, columns=list(encoded_cols)
        )
        X_transformed = X.drop(columns=self.categorical_cols_)
        return pd.concat([X_transformed, encoded_df], axis=1)

    def _apply_scaling(self, X, cols_to_scale):
        if not self.scaler_ or not cols_to_scale: return X
        X_scaled = X.copy()
        X_scaled[cols_to_scale] = self.scaler_.transform(X_scaled[cols_to_scale])
        return X_scaled
        
    def _fit_feature_selector(self, X: "pd.DataFrame", y) -> None:
        """Fit the configured feature selector.

        ``select_k_best_*`` are supervised and need ``y``; when it is absent the
        step is skipped with a warning rather than raising, so an unsupervised
        ``transform``-only flow still works.
        """
        strategy = self.feature_selection_strategy
        logger.debug("[DataProcessor] Fitting feature selector: %r", strategy)

        if strategy == 'correlation':
            self._fit_correlation_selector(X, threshold=self.correlation_threshold)
            return

        k = min(self.feature_selection_k, X.shape[1]) if X.shape[1] else self.feature_selection_k
        selector_map = {
            'variance': lambda: VarianceThreshold(threshold=0.0),
            'select_k_best_anova': lambda: SelectKBest(f_classif, k=k),
            'select_k_best_chi2': lambda: SelectKBest(chi2, k=k),
        }
        factory = selector_map.get(strategy)
        if factory is None:
            warn_once(
                f"Unknown feature_selection_strategy={strategy!r}; skipping feature "
                f"selection. Valid values: {sorted(selector_map)} or 'correlation'.",
                UserWarning,
                key=f"fs-unknown:{strategy}",
            )
            return

        if strategy.startswith('select_k_best') and y is None:
            warn_once(
                f"feature_selection_strategy={strategy!r} is supervised but no target was "
                f"provided; skipping feature selection.",
                UserWarning,
                key=f"fs-needs-y:{strategy}",
            )
            return

        self.selector_ = factory()
        X_to_fit = X.copy()
        if strategy == 'select_k_best_chi2':
            # chi2 requires non-negative inputs.
            X_to_fit = MinMaxScaler().fit_transform(X_to_fit)
        try:
            self.selector_.fit(X_to_fit, y)
        except Exception as exc:
            warn_once(
                f"Feature selection with {strategy!r} failed ({exc}); continuing without it.",
                UserWarning,
                key=f"fs-failed:{strategy}",
            )
            self.selector_ = None
            return

        dropped_cols = X.columns[~self.selector_.get_support()].tolist()
        self.processing_summary_.setdefault('steps', {})['feature_selection'] = {
            'strategy': strategy,
            'k': k,
            'dropped_columns': dropped_cols,
        }
            
    def _apply_feature_selection(self, X):
        X_selected = X
        if self.selector_ and self.feature_selection_strategy != 'correlation':
             selected_cols = X.columns[self.selector_.get_support()]
             X_selected = pd.DataFrame(self.selector_.transform(X), index=X.index, columns=selected_cols)
        
        if self._correlation_cols_to_drop:
             X_selected = X_selected.drop(columns=self._correlation_cols_to_drop, errors='ignore')
             
        return X_selected
        
    def _fit_correlation_selector(self, X: "pd.DataFrame", threshold: float | None = None) -> None:
        """Drop one of every pair of features correlated above ``threshold``."""
        threshold = 0.9 if threshold is None else float(threshold)
        numeric = X.select_dtypes(include=np.number)
        if numeric.shape[1] < 2:
            self._correlation_cols_to_drop = []
            return
        corr_matrix = numeric.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self._correlation_cols_to_drop = [
            column for column in upper.columns if (upper[column] > threshold).any()
        ]
        logger.debug(
            "[DataProcessor] Correlation filter identified %d columns to drop.",
            len(self._correlation_cols_to_drop),
        )
        self.processing_summary_.setdefault('steps', {})['feature_selection'] = {
            'strategy': 'correlation',
            'threshold': threshold,
            'dropped_columns': list(self._correlation_cols_to_drop),
        }

    def _fit_resampler(self, y):
        if self.resampling_strategy:
            k_neighbors = 5
            if self.resampling_strategy == 'smote':
                min_class_count = pd.Series(y).value_counts().min()
                k_neighbors = max(1, min_class_count - 1)
            
            resampler_map = {
                'smote': SMOTE(random_state=42, k_neighbors=k_neighbors),
                'random_over': RandomOverSampler(random_state=42),
                'random_under': RandomUnderSampler(random_state=42),
                'tomek': TomekLinks(),
                'kmeans': ClusterCentroids(random_state=42),
                'knn': NeighbourhoodCleaningRule()
            }
            self.resampler_ = resampler_map.get(self.resampling_strategy)
