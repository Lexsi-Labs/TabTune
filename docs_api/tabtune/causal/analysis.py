"""
tabtune.causal.analysis
=======================

User-facing entry point for the causal module.

:class:`CausalAnalysis` mirrors :class:`tabtune.TabularPipeline` in shape:

* Same constructor argument names (``model_name``, ``task_type``,
  ``tuning_strategy``, ``tuning_params``, ``processor_params``,
  ``model_params``) plus the causal-specific ``treatment``, ``outcome``,
  ``confounders``, ``sensitive``, ``estimator``, ``treatment_model``,
  and ``estimator_params``.
* Same ``.fit(X, y)`` / ``.evaluate(X, y, output_format='rich')``
  lifecycle.
* ``evaluate(..., output_format='rich')`` returns the same dict it logs
  -- callers always get a programmable result back.

The class orchestrates the three-step causal discipline:

1. **Identify** via :class:`GraphBuilder` + :class:`Identifier`.
2. **Estimate** via one of the estimators in
   :mod:`tabtune.causal.estimators`, each consuming TabTune TFMs as
   nuisance learners through :class:`_TabTuneSklearnAdapter`.
3. **Refute** via :class:`Refuter`, augmented by the proxy and
   counterfactual-fairness audits when a sensitive attribute is supplied.

The composed report is rendered via :class:`Reporter`.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .adapters import as_sklearn
from .audit.counterfactual import CounterfactualFairness
from .audit.proxy import ProxyAuditor
from .estimators import ESTIMATOR_REGISTRY, BaseCausalEstimator
from .graph import GraphBuilder, Identifier
from .refute import Refuter
from .reporter import Reporter

logger = logging.getLogger(__name__)


# Estimators that expect a regression outcome model by default.
# (Multi-level discrete treatments still build a *classifier* for the
# treatment model regardless of the outcome task.)
_REGRESSION_OUTCOME_ESTIMATORS = {"dml", "causal_forest", "r_learner"}


# ---------------------------------------------------------------------------
# CausalAnalysis
# ---------------------------------------------------------------------------
class CausalAnalysis:
    """
    User-facing causal-inference pipeline.

    Parameters
    ----------
    model_name : str
        Identifier of the TabTune TFM to use as the outcome nuisance learner.
        Examples: ``'TabPFNv26'``, ``'TabICLv2'``, ``'Mitra'``,
        ``'OrionMSP'``, ``'TabDPT'``.
    task_type : str
        ``'classification'`` or ``'regression'`` of the *outcome*. Determines
        the task type of the outcome model. The treatment model's task type
        is inferred from the treatment column (typically classification).
    tuning_strategy : str, default ``'inference'``
        One of ``'inference' | 'finetune' | 'peft'``. Forwarded to the
        underlying :class:`TabularPipeline`.
    treatment : str
        Column name of the treatment variable T.
    outcome : str
        Column name of the outcome variable Y.
    confounders : list of str
        Column names of the confounders X.
    sensitive : list of str, optional
        Column names of sensitive attributes S. Used for proxy and
        counterfactual-fairness audits; never used to estimate the effect.
    estimator : str, default ``'dml'``
        Key into :data:`ESTIMATOR_REGISTRY`. One of ``'dml'``,
        ``'s_learner'``, ``'t_learner'``, ``'x_learner'``, ``'r_learner'``,
        ``'causal_forest'``.
    treatment_model : dict, optional
        Override the treatment nuisance learner. Accepts a sub-dict with
        keys mirroring :class:`TabularPipeline` (``model_name``,
        ``tuning_strategy``, ``tuning_params``, ...). When omitted, the
        treatment model uses the same TFM as the outcome model.
    estimator_params : dict, optional
        Estimator-specific knobs (``n_folds``, ``confidence_level``, ...).
    tuning_params : dict, optional
        Forwarded to the TabTune pipeline used by both nuisance learners
        (commonly ``{'device': 'cuda'}``).
    processor_params, model_params : dict, optional
        Forwarded to the TabTune pipeline.
    graph : networkx.DiGraph, optional
        User-supplied causal DAG. When omitted, the canonical confounder
        graph is used.
    infer_graph : bool, default False
        If True and ``graph`` is None, try to infer a DAG via the PC
        algorithm (requires ``causal-learn``).
    verbose : bool, default True

    Examples
    --------
    >>> from tabtune.causal import CausalAnalysis
    >>> ca = CausalAnalysis(
    ...     model_name='TabPFNv26',
    ...     task_type='regression',
    ...     tuning_strategy='inference',
    ...     treatment='loan_approved',
    ...     outcome='default_12m',
    ...     confounders=['age', 'income', 'credit_score'],
    ...     sensitive=['race', 'gender'],
    ...     estimator='dml',
    ...     tuning_params={'device': 'cuda'},
    ... )
    >>> ca.fit(X_train, y_train)
    >>> result = ca.evaluate(X_test, y_test, output_format='rich')
    """

    def __init__(
        self,
        model_name: str,
        task_type: str,
        treatment: str,
        outcome: str,
        confounders: Iterable[str],
        tuning_strategy: str = "inference",
        sensitive: Iterable[str] | None = None,
        estimator: str = "dml",
        treatment_model: dict | None = None,
        estimator_params: dict | None = None,
        tuning_params: dict | None = None,
        processor_params: dict | None = None,
        model_params: dict | None = None,
        graph: Any | None = None,
        infer_graph: bool = False,
        verbose: bool = True,
    ):
        # User-facing arguments
        self.model_name = model_name
        self.task_type = task_type
        self.tuning_strategy = tuning_strategy
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = list(confounders)
        self.sensitive = list(sensitive or [])
        self.estimator = estimator
        self.treatment_model = treatment_model
        self.estimator_params = estimator_params or {}
        self.tuning_params = tuning_params or {}
        self.processor_params = processor_params or {}
        self.model_params = model_params or {}
        self.graph = graph
        self.infer_graph = bool(infer_graph)
        self.verbose = bool(verbose)

        # Validate estimator choice early
        if self.estimator not in ESTIMATOR_REGISTRY:
            raise ValueError(
                f"Unknown estimator '{self.estimator}'. "
                f"Available: {sorted(ESTIMATOR_REGISTRY.keys())}."
            )

        # Internal state set during fit()
        self._is_fitted = False
        self._estimator_obj: BaseCausalEstimator | None = None
        self._builder: GraphBuilder | None = None
        self._identification: dict | None = None
        self._train_df: pd.DataFrame | None = None

        if self.verbose:
            logger.info(
                "[Causal] Initialised CausalAnalysis: estimator=%s, "
                "outcome_model=%s/%s, treatment=%s, outcome=%s, |X|=%d, |S|=%d",
                self.estimator,
                self.model_name,
                self.tuning_strategy,
                self.treatment,
                self.outcome,
                len(self.confounders),
                len(self.sensitive),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_treatment_task_type(self, df: pd.DataFrame) -> str:
        """Decide whether the treatment model is regression or classification."""
        t = df[self.treatment]
        n_unique = t.nunique()
        
        if not pd.api.types.is_numeric_dtype(t):
            return "classification"
        if n_unique <= 10:
            return "classification"
        return "regression"

    def _build_outcome_model(self):
        """Build the sklearn-wrapped TabTune pipeline for the outcome."""
        # DML / Causal Forest / R-Learner expect a regressor for the outcome
        # model (since residualisation uses E[Y|X] as a real number).
        if self.estimator in _REGRESSION_OUTCOME_ESTIMATORS:
            outcome_task = "regression"
        else:
            outcome_task = self.task_type
        return as_sklearn(
            model_name=self.model_name,
            task_type=outcome_task,
            tuning_strategy=self.tuning_strategy,
            tuning_params=self.tuning_params,
            processor_params=self.processor_params,
            model_params=self.model_params,
        )

    def _build_treatment_model(self, df: pd.DataFrame):
        """Build the sklearn-wrapped TabTune pipeline for the treatment."""
        treat_task = self._resolve_treatment_task_type(df)
        spec = self.treatment_model or {}
        return as_sklearn(
            model_name=spec.get("model_name", self.model_name),
            task_type=treat_task,
            tuning_strategy=spec.get("tuning_strategy", self.tuning_strategy),
            tuning_params=spec.get("tuning_params", self.tuning_params),
            processor_params=spec.get("processor_params", self.processor_params),
            model_params=spec.get("model_params", self.model_params),
        )

    def _join_xy(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
        """Build the dataframe the estimator and graph builder consume."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        df = X.copy()
        # Outcome must live as a column for DoubleML / EconML.
        df[self.outcome] = np.asarray(y).ravel()
        # Sanity-check column presence.
        for col in (self.treatment, *self.confounders):
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' missing from X.")
        return df

    def _estimator_factory(self):
        """Return a fresh-estimator factory used by the refuter."""
        outcome_model_proto = self._build_outcome_model()
        treatment_model_proto = self._build_treatment_model(self._train_df)

        EstimatorCls = ESTIMATOR_REGISTRY[self.estimator]
        params = deepcopy(self.estimator_params)

        def _factory():
            # Use fresh adapter instances so cross-fitting state isn't shared.
            return EstimatorCls(
                treatment=self.treatment,
                outcome=self.outcome,
                confounders=list(self.confounders),
                outcome_model=as_sklearn(
                    model_name=outcome_model_proto.model_name,
                    task_type=outcome_model_proto.task_type,
                    tuning_strategy=outcome_model_proto.tuning_strategy,
                    tuning_params=outcome_model_proto.tuning_params,
                    processor_params=outcome_model_proto.processor_params,
                    model_params=outcome_model_proto.model_params,
                ),
                treatment_model=as_sklearn(
                    model_name=treatment_model_proto.model_name,
                    task_type=treatment_model_proto.task_type,
                    tuning_strategy=treatment_model_proto.tuning_strategy,
                    tuning_params=treatment_model_proto.tuning_params,
                    processor_params=treatment_model_proto.processor_params,
                    model_params=treatment_model_proto.model_params,
                ),
                estimator_params=params,
            )

        return _factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "CausalAnalysis":
        """
        Identify the effect, then fit the nuisance learners and the chosen
        causal estimator.

        Parameters
        ----------
        X : pandas.DataFrame
            Must include the treatment column and every confounder.
            Sensitive columns are optional but recommended; they enable
            the audit blocks downstream.
        y : pandas.Series or numpy.ndarray
            Outcome values. Joined into ``X`` as a column named
            ``self.outcome``.

        Returns
        -------
        self : CausalAnalysis
        """
        df = self._join_xy(X, y)
        self._train_df = df

        # Step 1 -- Identify
        self._builder = GraphBuilder(
            treatment=self.treatment,
            outcome=self.outcome,
            confounders=self.confounders,
        )
        if self.graph is not None:
            self._builder.from_user_dag(self.graph)
        elif self.infer_graph:
            try:
                self._builder.infer(df)
            except ImportError as exc:
                logger.warning(
                    "[Causal] Could not run PC algorithm (%s); "
                    "falling back to default graph.",
                    exc,
                )
                self._builder.default()
        else:
            self._builder.default()
        identifier = Identifier(self._builder)
        self._identification = identifier.identify(df=df, verbose=self.verbose)

        if not self._identification.get("identifiable", False):
            logger.warning(
                "[Causal] Effect not identifiable from the supplied graph. "
                "Proceeding with estimation anyway; treat results with caution."
            )

        # Step 2 -- Estimate
        factory = self._estimator_factory()
        self._estimator_obj = factory()
        if self.verbose:
            logger.info("[Causal] Fitting estimator '%s'...", self.estimator)
        self._estimator_obj.fit(df)

        self._is_fitted = True
        if self.verbose:
            logger.info("[Causal] Fit complete.")
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        X_test: pd.DataFrame | None = None,
        y_test: pd.Series | np.ndarray | None = None,
        include: Iterable[str] = ("effect",),
        output_format: str = "rich",
        output_path: str | None = None,
    ) -> dict:
        """
        Compose the requested blocks of the causal report and return them.

        Parameters
        ----------
        X_test, y_test : optional
            When provided, refutation and audits run on the concatenated
            train+test data, mirroring :meth:`TabularPipeline.evaluate`.
            When omitted, refutation and audits use the training data.
        include : iterable of str
            Subset of ``{'effect', 'identification', 'refutation', 'proxy',
            'counterfactual_fairness'}``. ``'effect'`` is always included.
        output_format : ``'rich' | 'json' | 'silent'``, default ``'rich'``
            ``'rich'`` pretty-prints via the TabTune logger. ``'json'``
            prints a JSON dump. ``'silent'`` returns the dict without
            side effects.
        output_path : str, optional
            If supplied, also write an HTML model card to this path.

        Returns
        -------
        report : dict
            The composed report.
        """
        self._require_fit()
        include = set(include)
        include.add("effect")  # always include the headline

        # Build the audit dataframe (train + test if provided).
        if X_test is not None and y_test is not None:
            df_audit = pd.concat(
                [self._train_df, self._join_xy(X_test, y_test)],
                axis=0,
                ignore_index=True,
            )
        else:
            df_audit = self._train_df

        report: dict[str, Any] = {
            "meta": {
                "estimator": self.estimator,
                "model_name": self.model_name,
                "task_type": self.task_type,
                "tuning_strategy": self.tuning_strategy,
                "treatment": self.treatment,
                "outcome": self.outcome,
                "n_confounders": len(self.confounders),
                "sensitive": list(self.sensitive),
            }
        }

        if "identification" in include and self._identification is not None:
            report["identification"] = self._identification

        # Always run effect
        confidence_level = float(self.estimator_params.get("confidence_level", 0.95))
        report["effect"] = self._estimator_obj.ate(confidence_level=confidence_level)

        if "refutation" in include:
            refuter = Refuter(
                self._estimator_obj,
                df_audit,
                estimator_factory=self._estimator_factory(),
            )
            report["refutation"] = refuter.run()

        if "proxy" in include and self.sensitive:
            auditor = ProxyAuditor(
                df=df_audit,
                confounders=self.confounders,
                sensitive=self.sensitive,
                outcome=self.outcome,
            )
            report["proxy"] = auditor.run()

        if "counterfactual_fairness" in include and self.sensitive:
           
            cf = CounterfactualFairness(
                estimator=self._estimator_obj,
                df=df_audit,
                sensitive=self.sensitive,
                feature_columns=self.confounders + [self.treatment],
            )
            report["counterfactual_fairness"] = cf.run()

        # Output side effects
        reporter = Reporter(report)
        if output_format == "rich":
            reporter.to_rich()
        elif output_format == "json":
            print(reporter.to_json())
        elif output_format == "silent":
            pass
        else:
            logger.warning(
                "[Causal] Unknown output_format='%s'. Returning dict silently.",
                output_format,
            )

        if output_path is not None:
            reporter.to_html(output_path)

        return report

   
    def evaluate_refutation(self, output_format: str = "rich") -> dict:
        """Run only the refutation block and return its dict."""
        self._require_fit()
        refuter = Refuter(
            self._estimator_obj,
            self._train_df,
            estimator_factory=self._estimator_factory(),
        )
        result = refuter.run()
        if output_format == "rich":
            Reporter({"refutation": result}).to_rich()
        elif output_format == "json":
            print(json.dumps(result, indent=2, default=str))
        return result

    def evaluate_proxy(self, output_format: str = "rich") -> dict:
        """Run only the proxy audit."""
        self._require_fit()
        if not self.sensitive:
            raise ValueError("evaluate_proxy() requires sensitive=[...] at construction.")
        auditor = ProxyAuditor(
            df=self._train_df,
            confounders=self.confounders,
            sensitive=self.sensitive,
            outcome=self.outcome,
        )
        result = auditor.run()
        if output_format == "rich":
            Reporter({"proxy": result}).to_rich()
        elif output_format == "json":
            print(json.dumps(result, indent=2, default=str))
        return result

    def evaluate_counterfactual_fairness(self, output_format: str = "rich") -> dict:
        """Run only the counterfactual-fairness audit."""
        self._require_fit()
        if not self.sensitive:
            raise ValueError(
                "evaluate_counterfactual_fairness() requires sensitive=[...]."
            )
        cf = CounterfactualFairness(
            estimator=self._estimator_obj,
            df=self._train_df,
            sensitive=self.sensitive,
            feature_columns=self.confounders + [self.treatment],
        )
        result = cf.run()
        if output_format == "rich":
            Reporter({"counterfactual_fairness": result}).to_rich()
        elif output_format == "json":
            print(json.dumps(result, indent=2, default=str))
        return result

    def evaluate_identification(self, output_format: str = "rich") -> dict:
        """Return the identification result computed at fit time."""
        self._require_fit()
        if output_format == "rich":
            Reporter({"identification": self._identification or {}}).to_rich()
        elif output_format == "json":
            print(json.dumps(self._identification, indent=2, default=str))
        return self._identification or {}

    def predict_cate(self, X_query: pd.DataFrame, return_ci: bool = False):
        """Return per-row Conditional Average Treatment Effects."""
        self._require_fit()
        return self._estimator_obj.cate(X_query, return_ci=return_ci)

    def predict_counterfactual(self, row: pd.DataFrame, intervention: dict) -> dict:
        """Predict the counterfactual outcome for a single row."""
        self._require_fit()
        return self._estimator_obj.counterfactual(row, intervention)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True) -> dict:
        """Return constructor-equivalent parameters (sklearn-style)."""
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "tuning_strategy": self.tuning_strategy,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "confounders": list(self.confounders),
            "sensitive": list(self.sensitive),
            "estimator": self.estimator,
            "treatment_model": deepcopy(self.treatment_model),
            "estimator_params": deepcopy(self.estimator_params),
            "tuning_params": deepcopy(self.tuning_params),
            "processor_params": deepcopy(self.processor_params),
            "model_params": deepcopy(self.model_params),
        }

    def _require_fit(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("You must call .fit() before this method.")
