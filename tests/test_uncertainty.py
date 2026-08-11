"""Tests for conformal prediction, recalibration and the uncertainty report.

The conformal wrappers carry one load-bearing property - marginal coverage at
least ``1 - alpha`` under exchangeability - and the whole point of shipping
them is that the property holds *by construction*, not by luck. It is asserted
statistically here (multiple seeds, mean coverage) alongside the exact
finite-sample quantile arithmetic on a hand-computable case, because an
off-by-one in the quantile index produces coverage that looks fine on one seed
and undercovers systematically in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss

from tabtune.evaluation import expected_calibration_error
from tabtune.uncertainty import (
    ConformalClassifier,
    ConformalRegressor,
    Recalibrator,
    size_stratified_coverage,
    uncertainty_report,
)
from tabtune.uncertainty._scaling import apply_temperature, fit_temperature

pytestmark = pytest.mark.unit

ALPHA = 0.1


def _three_class_split(seed: int, n: int = 3200):
    """Train/calibration/test split of noisy 3-class data (n_cal=800, n_test=1400)."""
    X, y = make_classification(
        n_samples=n,
        n_features=10,
        n_informative=6,
        n_classes=3,
        n_clusters_per_class=1,
        flip_y=0.15,
        random_state=seed,
    )
    return (
        (X[:1000], y[:1000]),
        (X[1000:1800], y[1000:1800]),
        (X[1800:], y[1800:]),
    )


class _FixedProba:
    """Stub classifier returning a pre-set probability matrix, for exact tests."""

    def __init__(self, proba: np.ndarray, classes: np.ndarray) -> None:
        self._proba = np.asarray(proba, dtype=float)
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X) -> np.ndarray:
        return self._proba[: len(X)]


class _Overconfident:
    """Sharpen a fitted classifier's probabilities (p**3, renormalised)."""

    def __init__(self, estimator) -> None:
        self.estimator = estimator
        self.classes_ = estimator.classes_

    def predict_proba(self, X) -> np.ndarray:
        proba = self.estimator.predict_proba(X) ** 3.0
        return proba / proba.sum(axis=1, keepdims=True)


# ------------------------------------------------------- coverage guarantee


@pytest.mark.parametrize("method", ["lac", "aps"])
def test_marginal_coverage_holds_statistically(method):
    """Mean coverage over seeds must sit in [1-a-0.03, 1-a+0.08] and never exceed 1."""
    coverages = []
    for seed in (0, 1, 2):
        (X_train, y_train), (X_cal, y_cal), (X_test, y_test) = _three_class_split(seed)
        model = LogisticRegression(max_iter=500).fit(X_train, y_train)
        conformal = ConformalClassifier(model, method=method, alpha=ALPHA)
        conformal.calibrate(X_cal, y_cal)
        assert conformal.n_cal_ == 800
        measured = conformal.coverage(X_test, y_test)
        assert 0.0 <= measured["coverage"] <= 1.0
        coverages.append(measured["coverage"])
    mean_coverage = float(np.mean(coverages))
    assert mean_coverage >= 1.0 - ALPHA - 0.03
    # A well-specified model must not be absurdly conservative either.
    assert mean_coverage <= 1.0 - ALPHA + 0.08


def test_regression_absolute_coverage_holds_statistically():
    coverages = []
    for seed in (0, 1, 2):
        X, y = make_regression(n_samples=3200, n_features=8, noise=10.0, random_state=seed)
        model = LinearRegression().fit(X[:1000], y[:1000])
        conformal = ConformalRegressor(model, method="absolute", alpha=ALPHA)
        conformal.calibrate(X[1000:1800], y[1000:1800])
        measured = conformal.coverage(X[1800:], y[1800:])
        assert measured["avg_width"] > 0
        coverages.append(measured["coverage"])
    mean_coverage = float(np.mean(coverages))
    assert 1.0 - ALPHA - 0.03 <= mean_coverage <= 1.0 - ALPHA + 0.08


def test_regression_intervals_are_ordered_and_symmetric():
    X, y = make_regression(n_samples=1200, n_features=5, noise=5.0, random_state=0)
    model = LinearRegression().fit(X[:600], y[:600])
    conformal = ConformalRegressor(model).calibrate(X[600:1000], y[600:1000])
    lower, upper = conformal.predict_interval(X[1000:])
    assert np.all(lower <= upper)
    predictions = model.predict(X[1000:])
    np.testing.assert_allclose(upper - predictions, predictions - lower)


# -------------------------------------------------- finite-sample quantile


def test_finite_sample_quantile_on_a_hand_computable_case():
    """n_cal=9, alpha=0.1: k = ceil(10 * 0.9) = 9, so q_hat is the largest score."""
    p_true = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
    proba = np.column_stack([p_true, 1.0 - p_true])
    stub = _FixedProba(proba, classes=np.array([0, 1]))
    y_cal = np.zeros(9, dtype=int)  # LAC scores are 1 - p_true

    conformal = ConformalClassifier(stub, method="lac", alpha=0.1)
    conformal.calibrate(np.zeros((9, 1)), y_cal)
    assert conformal.n_cal_ == 9
    assert conformal.q_hat_ == pytest.approx(0.95)  # max of 1 - p_true


def test_finite_sample_quantile_interior_index():
    """n_cal=9, alpha=0.5: k = ceil(10 * 0.5) = 5, the 5th smallest score."""
    p_true = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
    stub = _FixedProba(np.column_stack([p_true, 1.0 - p_true]), classes=np.array([0, 1]))
    conformal = ConformalClassifier(stub, method="lac", alpha=0.5)
    conformal.calibrate(np.zeros((9, 1)), np.zeros(9, dtype=int))
    # scores sorted: 0.15, 0.25, ..., 0.95 -> 5th smallest is 0.55
    assert conformal.q_hat_ == pytest.approx(0.55)


def test_too_small_calibration_split_yields_maximal_sets():
    """n_cal=5, alpha=0.1 needs k=6 > n: threshold inf, every set is full."""
    p_true = np.full(5, 0.9)
    stub = _FixedProba(np.column_stack([p_true, 1.0 - p_true]), classes=np.array([0, 1]))
    conformal = ConformalClassifier(stub, alpha=0.1)
    conformal.calibrate(np.zeros((5, 1)), np.zeros(5, dtype=int))
    assert conformal.q_hat_ == float("inf")
    assert conformal.predict_set(np.zeros((5, 1))).all()


# ------------------------------------------------------------- lac vs aps


def test_aps_sets_are_at_least_as_large_as_lac_on_average():
    (X_train, y_train), (X_cal, y_cal), (X_test, _) = _three_class_split(0)
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    lac = ConformalClassifier(model, method="lac", alpha=ALPHA).calibrate(X_cal, y_cal)
    aps = ConformalClassifier(model, method="aps", alpha=ALPHA).calibrate(X_cal, y_cal)
    assert aps.set_sizes(X_test).mean() >= lac.set_sizes(X_test).mean()


def test_deterministic_aps_never_returns_an_empty_set():
    (X_train, y_train), (X_cal, y_cal), (X_test, _) = _three_class_split(1)
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    aps = ConformalClassifier(model, method="aps", alpha=0.4).calibrate(X_cal, y_cal)
    assert aps.set_sizes(X_test).min() >= 1


def test_randomized_aps_is_seeded_and_covers():
    (X_train, y_train), (X_cal, y_cal), (X_test, y_test) = _three_class_split(2)
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)

    def build():
        conformal = ConformalClassifier(
            model, method="aps", alpha=ALPHA, randomized=True, random_state=7
        )
        return conformal.calibrate(X_cal, y_cal)

    first, second = build(), build()
    assert np.array_equal(first.predict_set(X_test), second.predict_set(X_test))
    assert first.coverage(X_test, y_test)["coverage"] >= 1.0 - ALPHA - 0.04


# ------------------------------------------------------------ label mapping


def test_string_labels_round_trip_and_columns_follow_classes():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=900), "f2": rng.normal(size=900)})
    y = pd.Series(
        np.where(X["f1"] > 0.3, "cat", np.where(X["f2"] > 0.3, "dog", "fish"))
    )
    model = LogisticRegression(max_iter=500).fit(X[:500], y[:500])
    conformal = ConformalClassifier(model, alpha=ALPHA)
    conformal.calibrate(X[500:700], y[500:700])

    assert np.array_equal(conformal.classes_, model.classes_)
    sets = conformal.predict_set(X[700:])
    assert sets.shape == (200, 3) and sets.dtype == bool

    # Membership of the true label must be read from the classes_-ordered
    # column, which is exactly what coverage() does.
    measured = conformal.coverage(X[700:], y[700:])
    assert measured["coverage"] >= 1.0 - ALPHA - 0.06


def test_calibration_labels_outside_the_class_space_are_rejected():
    stub = _FixedProba(np.full((4, 2), 0.5), classes=np.array(["no", "yes"]))
    conformal = ConformalClassifier(stub)
    with pytest.raises(ValueError, match="does not appear"):
        conformal.calibrate(np.zeros((4, 1)), np.array(["no", "yes", "maybe", "no"]))


# ------------------------------------------------------------------ errors


def test_predict_before_calibrate_raises():
    stub = _FixedProba(np.full((2, 2), 0.5), classes=np.array([0, 1]))
    with pytest.raises(RuntimeError, match="calibrate"):
        ConformalClassifier(stub).predict_set(np.zeros((2, 1)))
    with pytest.raises(RuntimeError, match="calibrate"):
        ConformalRegressor(LinearRegression().fit([[0.0]], [0.0])).predict_interval([[0.0]])


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.2, 1.7])
def test_bad_alpha_is_rejected(alpha):
    stub = _FixedProba(np.full((2, 2), 0.5), classes=np.array([0, 1]))
    with pytest.raises(ValueError, match="alpha"):
        ConformalClassifier(stub, alpha=alpha)
    with pytest.raises(ValueError, match="alpha"):
        ConformalRegressor(LinearRegression(), alpha=alpha)


def test_unknown_methods_name_what_is_supported():
    stub = _FixedProba(np.full((2, 2), 0.5), classes=np.array([0, 1]))
    with pytest.raises(ValueError, match="lac.*aps"):
        ConformalClassifier(stub, method="raps")
    with pytest.raises(ValueError, match="absolute.*cqr"):
        ConformalRegressor(LinearRegression(), method="jackknife")
    with pytest.raises(ValueError, match="temperature.*isotonic"):
        Recalibrator(stub, method="platt")


def test_cqr_on_a_model_without_quantiles_is_actionable():
    model = LinearRegression().fit([[0.0], [1.0]], [0.0, 1.0])
    with pytest.raises(ValueError, match="predict_quantiles.*TabPFN"):
        ConformalRegressor(model, method="cqr")


def test_wrappers_reject_pipelines_missing_the_needed_method():
    with pytest.raises(TypeError, match="predict_proba"):
        ConformalClassifier(LinearRegression())
    with pytest.raises(TypeError, match="predict_proba"):
        Recalibrator(LinearRegression())


# --------------------------------------------------------------------- cqr


class _QuantileStub:
    """Regressor with a TabPFN-shaped predict_quantiles for the CQR path."""

    def __init__(self, spread: float) -> None:
        self.spread = spread

    def fit(self, X, y):
        self._model = LinearRegression().fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_quantiles(self, X, quantiles=None):
        center = self._model.predict(X)
        lo_q, hi_q = quantiles
        return {lo_q: center - self.spread, hi_q: center + self.spread}


def test_cqr_conformalizes_a_quantile_band():
    X, y = make_regression(n_samples=2400, n_features=6, noise=8.0, random_state=0)
    # A deliberately too-narrow band: conformalization must widen it to cover.
    model = _QuantileStub(spread=1.0).fit(X[:800], y[:800])
    conformal = ConformalRegressor(model, method="cqr", alpha=ALPHA)
    conformal.calibrate(X[800:1600], y[800:1600])
    assert conformal.q_hat_ > 0  # the band had to be widened
    measured = conformal.coverage(X[1600:], y[1600:])
    assert measured["coverage"] >= 1.0 - ALPHA - 0.04


# ------------------------------------------------------------- recalibrator


def test_temperature_scaling_improves_nll_and_ece_of_an_overconfident_model():
    (X_train, y_train), (X_cal, y_cal), (X_test, y_test) = _three_class_split(0)
    model = _Overconfident(LogisticRegression(max_iter=500).fit(X_train, y_train))
    recalibrated = Recalibrator(model, method="temperature").fit(X_cal, y_cal)
    assert recalibrated.temperature_ > 1.0  # sharpened probabilities need softening

    raw = model.predict_proba(X_test)
    fixed = recalibrated.predict_proba(X_test)
    assert log_loss(y_test, fixed) < log_loss(y_test, raw)
    ece_raw, _ = expected_calibration_error(y_test, raw, labels=model.classes_)
    ece_fixed, _ = expected_calibration_error(y_test, fixed, labels=model.classes_)
    assert ece_fixed < ece_raw


def test_temperature_scaling_preserves_predictions():
    (X_train, y_train), (X_cal, y_cal), (X_test, _) = _three_class_split(1)
    model = _Overconfident(LogisticRegression(max_iter=500).fit(X_train, y_train))
    recalibrated = Recalibrator(model).fit(X_cal, y_cal)
    raw_argmax = np.argmax(model.predict_proba(X_test), axis=1)
    assert np.array_equal(recalibrated.predict(X_test), model.classes_[raw_argmax])


def test_isotonic_rows_sum_to_one_and_per_class_maps_are_monotone():
    (X_train, y_train), (X_cal, y_cal), (X_test, _) = _three_class_split(2)
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    recalibrated = Recalibrator(model, method="isotonic").fit(X_cal, y_cal)

    proba = recalibrated.predict_proba(X_test)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert proba.min() >= 0.0

    grid = np.linspace(0.0, 1.0, 101)
    for calibrator in recalibrated.calibrators_:
        mapped = calibrator.predict(grid)
        assert np.all(np.diff(mapped) >= -1e-12)


def test_recalibrator_round_trips_string_labels():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"f1": rng.normal(size=600), "f2": rng.normal(size=600)})
    y = pd.Series(np.where(X["f1"] + X["f2"] > 0, "pos", "neg"))
    model = LogisticRegression(max_iter=500).fit(X[:300], y[:300])
    recalibrated = Recalibrator(model).fit(X[300:450], y[300:450])
    assert set(recalibrated.predict(X[450:])) <= {"pos", "neg"}
    assert np.array_equal(recalibrated.classes_, model.classes_)


def test_recalibrator_requires_fit_before_use():
    stub = _FixedProba(np.full((2, 2), 0.5), classes=np.array([0, 1]))
    recalibrated = Recalibrator(stub)
    with pytest.raises(RuntimeError, match="fit"):
        recalibrated.predict_proba(np.zeros((2, 1)))
    assert not hasattr(recalibrated, "classes_")  # sklearn-style: absent until fit


def test_conformal_composes_over_a_recalibrator():
    (X_train, y_train), (X_cal, y_cal), (X_test, y_test) = _three_class_split(0)
    model = _Overconfident(LogisticRegression(max_iter=500).fit(X_train, y_train))
    recalibrated = Recalibrator(model).fit(X_cal[:400], y_cal[:400])
    conformal = ConformalClassifier(recalibrated, alpha=ALPHA)
    conformal.calibrate(X_cal[400:], y_cal[400:])
    measured = conformal.coverage(X_test, y_test)
    assert measured["coverage"] >= 1.0 - ALPHA - 0.05
    assert np.array_equal(conformal.classes_, model.classes_)


# ------------------------------------------------- shared temperature fit


def test_fit_temperature_recovers_a_known_distortion():
    """Probabilities sharpened by T=1/3 need T close to 3 to undo."""
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2000, 3))
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y_index = np.array([rng.choice(3, p=row) for row in proba])
    sharpened = proba**3.0
    sharpened /= sharpened.sum(axis=1, keepdims=True)
    temperature, nll = fit_temperature(sharpened, y_index)
    assert temperature == pytest.approx(3.0, rel=0.15)
    recovered = apply_temperature(sharpened, temperature)
    np.testing.assert_allclose(recovered, proba, atol=0.02)
    assert nll > 0


def test_fit_temperature_validates_input():
    with pytest.raises(ValueError, match="non-empty"):
        fit_temperature(np.empty((0, 2)), np.array([], dtype=int))
    with pytest.raises(ValueError, match="y_index"):
        fit_temperature(np.full((3, 2), 0.5), np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="temperature"):
        apply_temperature(np.full((2, 2), 0.5), 0.0)


def test_distillation_calibrate_student_still_uses_the_same_fit():
    """The refactored distillation path must match fit_temperature exactly."""
    from types import SimpleNamespace

    from tabtune.distillation.strategies import calibrate_student

    rng = np.random.default_rng(1)
    proba = rng.dirichlet(np.ones(3) * 0.5, size=400)
    y_cal = rng.integers(0, 3, size=400)

    distiller = SimpleNamespace(
        task_type="classification",
        predict_proba=lambda X: proba,
        student_=SimpleNamespace(),
    )
    temperature = calibrate_student(distiller, pd.DataFrame(np.zeros((400, 1))), y_cal)
    expected, _ = fit_temperature(proba, y_cal)
    assert temperature == pytest.approx(expected)
    assert distiller.student_._calibration_temperature == pytest.approx(expected)


# ------------------------------------------------------------------ report


def test_uncertainty_report_without_a_calibration_split_is_calibration_only():
    (X_train, y_train), _, (X_test, y_test) = _three_class_split(0)
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    report = uncertainty_report(model, X_test, y_test)
    assert set(report) == {"ece", "mce", "brier"}
    assert 0.0 <= report["ece"] <= 1.0


def test_uncertainty_report_with_a_calibration_split_adds_conformal_block():
    (X_train, y_train), (X_cal, y_cal), (X_test, y_test) = _three_class_split(0)
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    report = uncertainty_report(model, X_test, y_test, X_cal=X_cal, y_cal=y_cal)
    assert set(report) == {"ece", "mce", "brier", "coverage", "avg_set_size", "sscs", "alpha"}
    assert report["alpha"] == ALPHA
    assert report["coverage"] >= 1.0 - ALPHA - 0.04
    assert report["avg_set_size"] >= 1.0
    # SSCS is worst-group coverage, so it can never exceed the best stratum;
    # being well below `coverage` is the expected honest signal.
    assert 0.0 <= report["sscs"] <= 1.0


def test_uncertainty_report_rejects_half_a_calibration_split():
    (X_train, y_train), (X_cal, _), (X_test, y_test) = _three_class_split(0)
    model = LogisticRegression(max_iter=500).fit(X_train, y_train)
    with pytest.raises(ValueError, match="together"):
        uncertainty_report(model, X_test, y_test, X_cal=X_cal)


def test_uncertainty_report_for_regression_requires_and_uses_the_split():
    X, y = make_regression(n_samples=2400, n_features=6, noise=8.0, random_state=0)
    model = LinearRegression().fit(X[:800], y[:800])
    with pytest.raises(ValueError, match="calibration split"):
        uncertainty_report(model, X[1600:], y[1600:])
    report = uncertainty_report(
        model, X[1600:], y[1600:], X_cal=X[800:1600], y_cal=y[800:1600]
    )
    assert set(report) == {"coverage", "avg_width", "alpha"}
    assert report["coverage"] >= 1.0 - ALPHA - 0.04


def test_size_stratified_coverage_finds_the_worst_stratum():
    sizes = np.array([1] * 20 + [2] * 20)
    covered = np.array([True] * 20 + [True] * 10 + [False] * 10)
    assert size_stratified_coverage(sizes, covered) == pytest.approx(0.5)
    # Tiny strata are skipped; all-tiny falls back to marginal coverage.
    assert size_stratified_coverage(np.array([1, 2]), np.array([True, False])) == 0.5
    assert np.isnan(size_stratified_coverage(np.array([]), np.array([])))


# -------------------------------------------------------------- integration


@pytest.mark.integration
@pytest.mark.model_xrfm
def test_pipeline_uncertainty_report_end_to_end_with_xrfm():
    """A real TabularPipeline (xRFM trains from scratch, offline) through the
    pipeline method and the raw ConformalClassifier wrapper."""
    pytest.importorskip("torch")
    try:
        from tabtune import TabularPipeline
    except Exception as exc:  # pragma: no cover - unrelated dependency issues
        pytest.skip(f"full TabTune package not importable: {exc}")

    rng = np.random.RandomState(0)
    n = 260
    X = pd.DataFrame(
        {
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "cat": rng.choice(["red", "green", "blue"], n),
        }
    )
    y = pd.Series(np.where(X["f1"] + 0.5 * X["f2"] + 0.3 * rng.randn(n) > 0, "pos", "neg"))
    X_train, y_train = X[:140], y[:140]
    X_cal, y_cal = X[140:200], y[140:200]
    X_test, y_test = X[200:], y[200:]

    pipe = TabularPipeline(
        model_name="XRFM",
        task_type="classification",
        tuning_strategy="inference",
        model_params={"device": "cpu"},
    )
    pipe.fit(X_train, y_train)

    report = pipe.uncertainty_report(X_test, y_test, X_cal=X_cal, y_cal=y_cal, alpha=0.2)
    assert {"ece", "mce", "brier", "coverage", "avg_set_size", "sscs", "alpha"} <= set(report)
    assert 0.0 <= report["coverage"] <= 1.0

    conformal = ConformalClassifier(pipe, alpha=0.2).calibrate(X_cal, y_cal)
    sets = conformal.predict_set(X_test)
    assert sets.shape == (len(X_test), 2)
    assert set(conformal.classes_.tolist()) == {"neg", "pos"}
    assert conformal.coverage(X_test, y_test)["coverage"] >= 0.5


class TestTrainingFrameGuard:
    """Calibrating on the pipeline's own training frame must raise, not lie."""

    class _PipelineLike:
        """Duck-typed stand-in carrying X_raw_train like TabularPipeline."""

        def __init__(self, X_train, y_train):
            from sklearn.linear_model import LogisticRegression

            self.X_raw_train = X_train
            self.task_type = "classification"
            self._clf = LogisticRegression(max_iter=300).fit(X_train, y_train)
            self.classes_ = self._clf.classes_

        def predict(self, X):
            return self._clf.predict(X)

        def predict_proba(self, X):
            return self._clf.predict_proba(X)

    @pytest.fixture()
    def pipeline_like(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(120, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return self._PipelineLike(X, y), X, y

    def test_classifier_rejects_training_frame(self, pipeline_like):
        model, X, y = pipeline_like
        with pytest.raises(ValueError, match="training frame"):
            ConformalClassifier(model, alpha=0.1).calibrate(X, y)

    def test_recalibrator_rejects_training_frame(self, pipeline_like):
        model, X, y = pipeline_like
        with pytest.raises(ValueError, match="training frame"):
            Recalibrator(model).fit(X, y)

    def test_regressor_rejects_training_frame(self):
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(7)
        X = rng.normal(size=(100, 3))
        y = X @ np.array([1.0, -2.0, 0.5])

        class _RegPipelineLike:
            X_raw_train = X
            task_type = "regression"
            _reg = LinearRegression().fit(X, y)

            def predict(self, Xq):
                return self._reg.predict(Xq)

        with pytest.raises(ValueError, match="training frame"):
            ConformalRegressor(_RegPipelineLike(), alpha=0.1).calibrate(X, y)

    def test_disjoint_split_passes(self, pipeline_like):
        model, X, y = pipeline_like
        rng = np.random.default_rng(8)
        X_cal = rng.normal(size=(60, 4))
        y_cal = (X_cal[:, 0] + X_cal[:, 1] > 0).astype(int)
        conformal = ConformalClassifier(model, alpha=0.1).calibrate(X_cal, y_cal)
        assert conformal.q_hat_ is not None

    def test_subsample_of_training_data_passes(self, pipeline_like):
        # Only the exact frame is rejected: a guard must never false-alarm.
        model, X, y = pipeline_like
        conformal = ConformalClassifier(model, alpha=0.1).calibrate(X[:50], y[:50])
        assert conformal.q_hat_ is not None
