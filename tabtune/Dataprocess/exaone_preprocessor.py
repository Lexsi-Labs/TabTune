"""
EXAONE Tabular (LG AI Research) model-aware preprocessor for TabTune.

EXAONE is a **raw-frame self-preprocessing** model, exactly like TabFM and iLTM:
both wrappers (``EXAONETabularClassifier`` /
``EXAONETabularRegressorWrapper``) hold their own fitted
:class:`~tabtune.models.exaone.episode_features.EXAONEFeatureEncoder` and call it
inside ``fit`` / ``predict`` / ``predict_proba``, and the vendored engine then
runs its own standardisation, per-member categorical permutation, quantile
mapping, attention-based feature selection and support-only imputation on top.
The pipeline therefore hands it the raw mixed-type frame and this preprocessor
**passes features through untouched**.

That is the one genuine difference from the TabFM / iLTM preprocessors, which
also emit a numeric matrix for the ``TuningManager`` to build episodes from.
EXAONE does not need that: its fine-tuning episodes come from
``model.prepare_episode_features(X_raw)``, i.e. from the *model's own* fitted
encoder, so producing a second encoding here would either be dead weight or --
worse -- get fed back into a model that has already encoded the same frame, and
the two vocabularies would not agree.

So this class exists for the two remaining reasons every TabTune per-model
preprocessor exists:

1.  it fits a ``LabelEncoder`` on the target so ``TabularPipeline.evaluate``
    and the probability-alignment code can map predictions back to the original
    label space (``evaluate()`` raises without it); and
2.  it keeps the ``DataProcessor`` contract uniform -- ``fit`` / ``transform`` /
    ``get_summary`` -- so nothing downstream needs to special-case EXAONE.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class EXAONEPreprocessor(BaseEstimator, TransformerMixin):
    """Feature passthrough + label bookkeeping for EXAONE Tabular."""

    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type

        self.label_encoder_ = None
        self.columns_ = []
        self.categorical_cols_ = []
        self.numerical_cols_ = []

    def fit(self, X, y=None):
        logger.info("[EXAONE] Fitting EXAONE preprocessor (task=%s)", self.task_type)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.columns_ = X.columns.tolist()
        self.categorical_cols_ = X.select_dtypes(exclude=np.number).columns.tolist()
        self.numerical_cols_ = X.select_dtypes(include=np.number).columns.tolist()

        if self.task_type == "classification" and y is not None:
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(y)
        return self

    def transform(self, X, y=None):
        """Return the frame unchanged; encode the target when one is supplied.

        Deliberately a no-op on ``X``. The vendored EXAONE stack rejects object
        columns, strings and DataFrames at its own boundary, but the TabTune
        wrappers encode with :class:`EXAONEFeatureEncoder` immediately before
        calling it -- so the frame that reaches this method must stay raw.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_ or None)
        X_final = X

        if y is not None:
            if self.task_type == "classification" and self.label_encoder_ is not None:
                y_final = self.label_encoder_.transform(y)
            else:
                y_final = np.asarray(y, dtype=float).ravel()
            return X_final, y_final
        return X_final

    def get_summary(self):
        return {
            "EXAONE Preprocessing": {
                "description": "Passthrough. Raw mixed-type frames go straight to the "
                "vendored EXAONE engine, which ordinal-encodes categoricals, "
                "standardises, quantile-maps, selects features by attention and "
                "imputes from the support set -- all internally. TabTune only fits "
                "a LabelEncoder here so evaluation can recover the original labels.",
                "details": [
                    f"{len(self.categorical_cols_)} categorical, "
                    f"{len(self.numerical_cols_)} numerical columns (unmodified).",
                ],
            }
        }
