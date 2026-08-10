"""
Tests for XRFM fine-tuning + PEFT mechanics.

xRFM is a kernel/Recursive-Feature-Machine method: 'finetune' = full RFM
(re)training with user hyperparameters (+ warm-started continued refinement
from an already-fitted M), and 'peft' = a frozen-base low-rank update of the
AGOP-learned Mahalanobis matrix M (NO LoRA nn.Linear machinery). These tests
validate, without any downloads or GPU:

  * the rank-r truncated M update has the promised algebraic properties
    (symmetry, max-normalisation, rank of the additive correction <= r);
  * `TuningManager._finetune_xrfm` applies hyperparameters + trains, and
    warm-starts (continued refinement) when the model is already fitted;
  * `TuningManager._peft_xrfm` adapts the leaf M matrices while the model
    keeps producing valid predictions;
  * the regression finetune path works end-to-end.

Run:  pytest tests/test_xrfm_finetune.py -v
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def _require_full_tabtune():
    try:
        import tabtune  # noqa: F401
        from tabtune.TuningManager.tuning import TuningManager  # noqa: F401
    except Exception as e:
        pytest.skip(f"full TabTune package not importable (unrelated to XRFM): {e}")


def _clf_data(n=100, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n)})
    y = pd.Series(np.where(X["f1"] + 0.5 * X["f2"] > 0, "pos", "neg"))
    return X, y


def _reg_data(n=100, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({"f1": rng.randn(n), "f2": rng.randn(n)})
    y = pd.Series(3.0 * X["f1"] - 2.0 * X["f2"] + 0.1 * rng.randn(n))
    return X, y


@pytest.mark.unit
@pytest.mark.model_xrfm
@pytest.mark.finetuning
class TestLowRankMUpdate:
    """Algebraic contract of the rank-r truncated M update (the PEFT core)."""

    @pytest.fixture(autouse=True)
    def _needs_tabtune(self):
        _require_full_tabtune()

    def _update(self):
        from tabtune.TuningManager.tuning import TuningManager
        return TuningManager._xrfm_low_rank_M_update

    def test_rank_of_correction_bounded(self):
        update = self._update()
        torch.manual_seed(0)
        d, r = 12, 3
        A = torch.randn(d, d)
        agop = A @ A.T  # PSD AGOP surrogate
        M_base = torch.eye(d)
        M_new = update(M_base, agop, rank=r, alpha=0.7)
        # M_new = (I + alpha*delta)/c with rank(delta) <= r, so M_new has at
        # most r eigenvalues different from the common 1/c "frozen" level.
        eigvals = torch.linalg.eigvalsh(M_new)
        base_level = eigvals.min()
        n_modified = int((eigvals > base_level + 1e-6).sum())
        assert n_modified <= r
        # and the correction relative to the frozen base has rank <= r
        residual = M_new - base_level * M_base
        assert int(torch.linalg.matrix_rank(residual, atol=1e-5)) <= r

    def test_symmetry_and_normalisation(self):
        update = self._update()
        torch.manual_seed(1)
        d = 8
        A = torch.randn(d, d)
        M_new = update(torch.eye(d), A @ A.T, rank=2, alpha=0.5)
        assert torch.allclose(M_new, M_new.T, atol=1e-6)
        assert float(M_new.max()) == pytest.approx(1.0, abs=1e-5)
        assert torch.all(torch.isfinite(M_new))

    def test_diagonal_M_topk(self):
        update = self._update()
        agop = torch.tensor([5.0, 1.0, 3.0, 0.5])
        M_base = torch.ones(4)
        M_new = update(M_base, agop, rank=2, alpha=1.0)
        # only the two largest AGOP entries get a boost
        boosted = (M_new > M_new.min() + 1e-8)
        assert boosted.tolist() == [True, False, True, False]

    def test_alpha_zero_keeps_base_direction(self):
        update = self._update()
        torch.manual_seed(2)
        d = 6
        M_base = torch.eye(d) * 0.5
        A = torch.randn(d, d)
        M_new = update(M_base, A @ A.T, rank=2, alpha=0.0)
        # alpha=0 -> just the max-normalised frozen base
        assert torch.allclose(M_new, M_base / M_base.max(), atol=1e-6)


@pytest.mark.unit
@pytest.mark.model_xrfm
@pytest.mark.finetuning
class TestFinetuneXRFM:
    @pytest.fixture(autouse=True)
    def _needs_tabtune(self):
        _require_full_tabtune()

    def test_finetune_applies_hyperparams_and_trains(self, tmp_path):
        from tabtune.TuningManager.tuning import TuningManager
        from tabtune.models.xrfm import XRFMClassifier
        X, y = _clf_data()
        model = XRFMClassifier(device="cpu", verbose=False, random_state=0)
        tuned = TuningManager()._finetune_xrfm(
            model, X, y,
            params={"iters": 2, "bandwidth": 6.0, "device": "cpu"},
            save_path=str(tmp_path / "xrfm_ckpt.pt"),
        )
        assert tuned._is_fitted
        assert tuned.iters == 2 and tuned.bandwidth == 6.0
        assert (tuned.predict(X) == y.values).mean() > 0.7
        assert (tmp_path / "xrfm_ckpt.joblib").exists()  # joblib, not torch state_dict

    def test_finetune_warm_start_refines_existing_M(self):
        from tabtune.TuningManager.tuning import TuningManager
        from tabtune.models.xrfm import XRFMClassifier
        X, y = _clf_data()
        model = XRFMClassifier(device="cpu", iters=2, verbose=False, random_state=0)
        model.fit(X, y)
        M_before = [leaf.M.clone() for leaf in model.leaf_models()]
        centers_before = [leaf.centers.clone() for leaf in model.leaf_models()]
        X2, y2 = _clf_data(seed=7)  # "new" adaptation data
        tuned = TuningManager()._finetune_xrfm(model, X2, y2,
                                               params={"refine_iters": 1, "device": "cpu"})
        leaves = tuned.leaf_models()
        assert len(leaves) == len(M_before)
        # continued refinement: M and the kernel predictor were both updated
        assert any(not torch.allclose(leaf.M, M0) for leaf, M0 in zip(leaves, M_before))
        assert any(leaf.centers.shape != c0.shape or not torch.allclose(leaf.centers, c0)
                   for leaf, c0 in zip(leaves, centers_before))
        assert (tuned.predict(X2) == y2.values).mean() > 0.6

    def test_finetune_regression_path(self):
        from tabtune.TuningManager.tuning import TuningManager
        from tabtune.models.regression.xrfm.regressor import XRFMRegressorWrapper
        X, y = _reg_data()
        model = XRFMRegressorWrapper(device="cpu", tuning_strategy="finetune",
                                     verbose=False, random_state=0)
        tuned = TuningManager()._finetune_xrfm_regression(
            model, X, y, params={"iters": 2, "device": "cpu"})
        assert tuned._is_fitted and tuned.iters == 2
        preds = tuned.predict(X)
        assert np.all(np.isfinite(preds))
        assert np.mean((preds - y.values) ** 2) < np.var(y.values)

    def test_tune_dispatch_finetune_and_inference(self):
        from tabtune.TuningManager.tuning import TuningManager
        from tabtune.models.xrfm import XRFMClassifier
        X, y = _clf_data()
        # inference dispatch
        m1 = TuningManager().tune(XRFMClassifier(device="cpu", iters=1, verbose=False),
                                  X, y, strategy="inference")
        assert m1._is_fitted
        # finetune dispatch (checkpoint goes to ./checkpoints/*.joblib)
        m2 = TuningManager().tune(XRFMClassifier(device="cpu", verbose=False),
                                  X, y, strategy="finetune",
                                  params={"iters": 1, "device": "cpu",
                                          "finetune_mode": "meta-learning"})
        assert m2._is_fitted and m2.iters == 1

    def test_tune_dispatch_regression_finetune(self):
        from tabtune.TuningManager.tuning import TuningManager
        from tabtune.models.regression.xrfm.regressor import XRFMRegressorWrapper
        X, y = _reg_data()
        model = XRFMRegressorWrapper(device="cpu", tuning_strategy="finetune", verbose=False)
        tuned = TuningManager().tune(model, X, y, strategy="finetune",
                                     params={"iters": 1, "device": "cpu",
                                             "finetune_mode": "turn_by_turn"})
        assert tuned._is_fitted


@pytest.mark.unit
@pytest.mark.model_xrfm
@pytest.mark.finetuning
class TestPeftXRFM:
    @pytest.fixture(autouse=True)
    def _needs_tabtune(self):
        _require_full_tabtune()

    def test_peft_adapts_leaf_M_and_keeps_predicting(self):
        from tabtune.TuningManager.tuning import TuningManager
        from tabtune.models.xrfm import XRFMClassifier
        X, y = _clf_data()
        model = XRFMClassifier(device="cpu", iters=2, verbose=False, random_state=0)
        model.fit(X, y)
        M_base = [leaf.M.clone() for leaf in model.leaf_models()]
        X2, y2 = _clf_data(seed=7)
        tuned = TuningManager()._peft_xrfm(model, X2, y2,
                                           params={"lora_rank": 2, "peft_alpha": 0.5,
                                                   "device": "cpu"})
        leaves = tuned.leaf_models()
        assert any(not torch.allclose(leaf.M, M0) for leaf, M0 in zip(leaves, M_base))
        for leaf in leaves:  # engine invariant preserved
            assert float(leaf.M.max()) == pytest.approx(1.0, abs=1e-4)
        proba = tuned.predict_proba(X2)
        assert proba.shape == (len(X2), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert (tuned.predict(X2) == y2.values).mean() > 0.6

    def test_peft_on_unfitted_model_fits_base_first(self):
        from tabtune.TuningManager.tuning import TuningManager
        from tabtune.models.xrfm import XRFMClassifier
        X, y = _clf_data()
        model = XRFMClassifier(device="cpu", iters=1, verbose=False, random_state=0)
        tuned = TuningManager()._peft_xrfm(model, X, y, peft_config={"r": 4, "alpha": 0.3})
        assert tuned._is_fitted
        assert tuned.predict(X).shape == (len(X),)

    def test_tune_dispatch_peft_strategy(self):
        from tabtune.TuningManager.tuning import TuningManager
        from tabtune.models.xrfm import XRFMClassifier
        X, y = _clf_data()
        model = XRFMClassifier(device="cpu", iters=1, verbose=False, random_state=0)
        tuned = TuningManager().tune(model, X, y, strategy="peft",
                                     params={"lora_rank": 2, "device": "cpu",
                                             "finetune_mode": "meta-learning"})
        assert tuned._is_fitted

    def test_tune_dispatch_finetune_method_peft(self):
        # strategy='finetune' + finetune_method='peft' routes to the PEFT path
        from tabtune.TuningManager.tuning import TuningManager
        from tabtune.models.xrfm import XRFMClassifier
        X, y = _clf_data()
        model = XRFMClassifier(device="cpu", iters=1, verbose=False, random_state=0)
        tuned = TuningManager().tune(model, X, y, strategy="finetune",
                                     params={"finetune_method": "peft", "device": "cpu",
                                             "finetune_mode": "meta-learning",
                                             "peft_config": {"r": 2}})
        assert tuned._is_fitted
