"""Regression tests for OrionMSP validation on scikit-learn 1.6 and 1.8+.

scikit-learn 1.8 removed ``force_all_finite`` and ``BaseEstimator._validate_data``.
OrionMSPv1.5's ``_validate_data`` backport then called ``check_X_y`` with the
old keyword and ``fit`` raised ``TypeError``.
"""

import numpy as np
import pytest

from tabtune._internal.sklearn_compat import check_X_y


def test_check_X_y_accepts_legacy_force_all_finite_with_nan():
    X = np.array([[1.0, np.nan], [3.0, 4.0]])
    y = np.array([0, 1])

    X_out, y_out = check_X_y(
        X,
        y,
        accept_sparse=False,
        ensure_2d=True,
        dtype=None,
        force_all_finite="allow-nan",
    )

    assert X_out.shape == (2, 2)
    assert np.isnan(X_out[0, 1])
    assert list(y_out) == [0, 1]


def test_check_X_y_rejects_conflicting_finite_keywords():
    X = np.array([[1.0, 2.0]])
    y = np.array([0])
    with pytest.raises(TypeError, match="conflicting values"):
        check_X_y(
            X,
            y,
            force_all_finite="allow-nan",
            ensure_all_finite=True,
        )


def test_orionmsp_v15_ensemble_fit_allows_nan_on_current_sklearn():
    """The fit path that used to call raw check_X_y(..., force_all_finite=...)."""
    from tabtune.models.orionmsp_v15.sklearn.preprocessing import EnsembleGenerator

    X = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0], [2.0, 3.0, 4.0]])
    y = np.array([0, 1, 0, 1])

    generator = EnsembleGenerator(
        n_estimators=2,
        norm_methods=["none"],
        feat_shuffle_method="random",
        class_shift=False,
        random_state=0,
    )
    generator.fit(X, y)

    assert generator.n_features_in_ == X.shape[1]
    assert generator.n_classes_ == 2
