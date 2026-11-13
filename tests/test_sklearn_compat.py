"""Validation compatibility and the OrionMSPv1.5 preprocessing regression."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tabtune._internal import sklearn_compat as compat
from tabtune.models.orionmsp_v15.sklearn import preprocessing as orion_preprocessing
from tabtune.models.orionmsp_v15.sklearn.preprocessing import EnsembleGenerator

pytestmark = pytest.mark.unit


@pytest.fixture
def data():
    X = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    y = np.array([0, 1, 0, 1])
    return X, y


@pytest.mark.parametrize("validator", ["check_array", "check_X_y"])
@pytest.mark.parametrize("keyword", ["force_all_finite", "ensure_all_finite"])
@pytest.mark.parametrize("policy", [True, False, "allow-nan"])
def test_finite_validation_policy(data, validator, keyword, policy):
    X, y = data
    validate = getattr(compat, validator)
    args = (X, y) if validator == "check_X_y" else (X,)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        validate(*args, **{keyword: policy})

        X[0, 0] = np.nan
        if policy is True:
            with pytest.raises(ValueError, match="NaN"):
                validate(*args, **{keyword: policy})
        else:
            result = validate(*args, **{keyword: policy})
            checked_X = result[0] if validator == "check_X_y" else result
            assert np.isnan(checked_X[0, 0])

        X[0, 0] = np.inf
        if policy is False:
            validate(*args, **{keyword: policy})
        else:
            with pytest.raises(ValueError, match="infinity|inf"):
                validate(*args, **{keyword: policy})


@pytest.mark.parametrize("validator", ["check_array", "check_X_y"])
def test_duplicate_keywords(data, validator):
    X, y = data
    validate = getattr(compat, validator)
    args = (X, y) if validator == "check_X_y" else (X,)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        validate(*args, force_all_finite="allow-nan", ensure_all_finite="allow-nan")
    with pytest.raises(TypeError, match=f"{validator} received conflicting values"):
        validate(*args, force_all_finite=True, ensure_all_finite=False)


@pytest.mark.parametrize("validator", ["check_array", "check_X_y"])
def test_legacy_validator_keyword_and_argument_forwarding(data, monkeypatch, validator):
    """A validator without the new spelling still receives the old spelling."""
    X, y = data

    def legacy_array(array, *, force_all_finite, copy):
        assert array is X
        assert force_all_finite == "allow-nan"
        assert copy is True
        return array.copy()

    def legacy_X_y(features, targets, *, force_all_finite, copy):
        assert targets is y
        return legacy_array(features, force_all_finite=force_all_finite, copy=copy), y

    if validator == "check_array":
        monkeypatch.setattr(compat, "_sklearn_check_array", legacy_array)
        monkeypatch.setattr(compat, "_CHECK_ARRAY_SUPPORTS_ENSURE", False)
        result = compat.check_array(X, ensure_all_finite="allow-nan", copy=True)
    else:
        monkeypatch.setattr(compat, "_sklearn_check_X_y", legacy_X_y)
        monkeypatch.setattr(compat, "_CHECK_X_Y_SUPPORTS_ENSURE", False)
        result, checked_y = compat.check_X_y(X, y, ensure_all_finite="allow-nan", copy=True)
        assert checked_y is y
    np.testing.assert_array_equal(result, X)
    assert result is not X


def test_check_X_y_preserves_length_and_target_validation(data):
    X, y = data
    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        compat.check_X_y(X, y[:-1], force_all_finite="allow-nan")
    with pytest.raises(ValueError, match="NaN"):
        compat.check_X_y(X, [0, np.nan, 0, 1], force_all_finite="allow-nan")


def test_orion_ensemble_fit_and_transform(data):
    """Exercise the fit path from issue #42 without loading model weights."""
    X, y = data
    ensemble = EnsembleGenerator(n_estimators=1, norm_methods=["none"], random_state=42)
    with warnings.catch_warnings():
        # sklearn 1.6 separately deprecates BaseEstimator._validate_data;
        # only the finite-value keyword warning belongs to this regression.
        warnings.filterwarnings("error", message=".*force_all_finite.*", category=FutureWarning)
        assert ensemble.fit(X, y) is ensemble
        transformed = ensemble.transform(X)
        with pytest.raises(ValueError, match="expecting 2 features"):
            ensemble._validate_data(X[:, :1], reset=False)
    assert transformed
    assert ensemble.n_features_in_ == X.shape[1]


@pytest.mark.skipif(
    not hasattr(orion_preprocessing, "_validate_data"),
    reason="The local validation backport is only installed when sklearn removes _validate_data",
)
def test_orion_validation_preserves_nan_copy_and_feature_count(data):
    X, y = data
    X[0, 0] = np.nan
    ensemble = EnsembleGenerator(n_estimators=1)
    checked_X, checked_y = ensemble._validate_data(X, y, copy=True)
    assert np.isnan(checked_X[0, 0])
    assert checked_X is not X
    np.testing.assert_array_equal(checked_y, y)
    assert ensemble.n_features_in_ == X.shape[1]
    with pytest.raises(ValueError, match="expecting 2 features"):
        ensemble._validate_data(X[:, :1], y, reset=False)


@pytest.mark.parametrize("invalid", ["infinity", "length"])
def test_orion_ensemble_rejects_invalid_training_data(data, invalid):
    X, y = data
    if invalid == "infinity":
        X[0, 0] = np.inf
        match = "infinity|inf"
    else:
        y = y[:-1]
        match = "inconsistent numbers of samples"
    ensemble = EnsembleGenerator(n_estimators=1, norm_methods=["none"])
    with pytest.raises(ValueError, match=match):
        ensemble.fit(X, y)
