"""
Fine-tuning / PEFT mechanics tests for ILTM in TabTune.

Everything runs on a TINY randomly-initialised iLTM hypernetwork (no
pretrained weights, no downloads, CPU-only): LoRA injection + freezing
semantics, real gradient loops (full FT and LoRA-only) reducing the episode
loss, the TuningManager episode tensor builder, and the full
``TuningManager.tune`` dispatch (finetune / peft / regression) including the
``.pt`` state-dict checkpoint round-trip.

Run:  pytest tests/test_iltm_finetune.py -v
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tabtune.models.iltm.iltm_model import iLTM
from tabtune.models.iltm.utils import full_main_forward
from tabtune.TuningManager.peft_utils import LoRALinear, apply_tabular_lora

TINY_ARCH = dict(
    n_dims=16, hn_n_layers=2, hn_hidden_size=32, rf_size=64,
    n_classes_limit=6, main_n_layers=3, pca_sampling="zeropad", pca_fit="reduced",
)
FAST_TUNING = dict(device="cpu", epochs=1, steps_per_epoch=10, support_size=32,
                   query_size=16, learning_rate=1e-3, show_progress=False)


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return iLTM(**TINY_ARCH)


def _episode_forward(model, xs, ys, xq, n_classes, training=True):
    rf, pca, main_network, norm = model(xs, ys, n_classes, training=training)
    return full_main_forward(
        xq, n_classes, int(xq.shape[0]), vars(model), rf, pca, norm, main_network,
        torch.device("cpu"), use_amp=False, training_finetuning=training,
    )


def _separable_data(n=120, h=6, k=3, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, k, n)
    X = rng.randn(n, h).astype(np.float32) + 2.5 * np.eye(k, h)[y].astype(np.float32)
    return X, y


def _episode_loss(model, X, y, k, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    s_idx, q_idx = idx[:64], idx[64:96]
    xs = torch.tensor(X[s_idx]); ys = torch.tensor(y[s_idx], dtype=torch.long)
    xq = torch.tensor(X[q_idx]); yq = torch.tensor(y[q_idx], dtype=torch.long)
    logits = _episode_forward(model, xs, ys, xq, k)
    return torch.nn.functional.cross_entropy(logits, yq)


@pytest.mark.unit
@pytest.mark.model_iltm
class TestILTMLoraInjection:
    def test_lora_wraps_and_freezes(self):
        model = _tiny_model()
        n_before = sum(p.numel() for p in model.parameters())
        model = apply_tabular_lora("ILTM", model, {"r": 4, "lora_alpha": 8})
        lora_modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
        assert len(lora_modules) > 0
        # every base weight frozen; only LoRA adapters trainable
        for name, p in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert p.requires_grad
            else:
                assert not p.requires_grad
        n_after = sum(p.numel() for p in model.parameters())
        assert n_after > n_before  # adapters added

    def test_forward_still_works_after_lora(self):
        model = _tiny_model()
        X, y = _separable_data()
        # The initial transformation block draws fresh random features every
        # forward -> seed torch so both episode forwards see the same rf.
        torch.manual_seed(123)
        loss_base = _episode_loss(model, X, y, 3, seed=0)
        model = apply_tabular_lora("ILTM", model, {"r": 4, "lora_dropout": 0.0})
        torch.manual_seed(123)
        loss_lora = _episode_loss(model, X, y, 3, seed=0)
        assert torch.isfinite(loss_lora)
        # B is zero-initialised -> LoRA output starts identical to base
        assert torch.allclose(loss_base, loss_lora, atol=1e-5)

    def test_only_lora_params_change_when_training(self):
        model = apply_tabular_lora("ILTM", _tiny_model(), {"r": 4})
        base_before = {n: p.clone() for n, p in model.named_parameters()
                       if n.startswith("hypernetwork_block.") and "lora_" not in n}
        X, y = _separable_data()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        for step in range(5):
            opt.zero_grad()
            loss = _episode_loss(model, X, y, 3, seed=step)
            loss.backward()
            opt.step()
        for n, p in model.named_parameters():
            if n in base_before:
                assert torch.equal(p, base_before[n]), f"frozen base param {n} changed"
        assert any(p.abs().sum() > 0 for n, p in model.named_parameters() if "lora_B" in n)


@pytest.mark.unit
@pytest.mark.model_iltm
@pytest.mark.finetuning
class TestILTMGradientLoops:
    def _run_loop(self, model, X, y, steps=40, lr=3e-3):
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        losses = []
        for step in range(steps):
            opt.zero_grad()
            loss = _episode_loss(model, X, y, 3, seed=step % 7)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        return losses

    def test_full_finetune_reduces_loss(self):
        model = _tiny_model(seed=3)
        X, y = _separable_data(seed=3)
        losses = self._run_loop(model, X, y)
        assert np.mean(losses[-5:]) < np.mean(losses[:5])

    def test_lora_finetune_reduces_loss(self):
        model = apply_tabular_lora("ILTM", _tiny_model(seed=4), {"r": 8, "lora_alpha": 16})
        X, y = _separable_data(seed=4)
        losses = self._run_loop(model, X, y)
        assert np.mean(losses[-5:]) < np.mean(losses[:5])


@pytest.mark.unit
@pytest.mark.model_iltm
class TestEpisodeTensors:
    def test_shapes_and_dtypes(self):
        from tabtune.TuningManager.tuning import TuningManager
        X = np.random.RandomState(0).randn(50, 5).astype(np.float32)
        y = np.random.RandomState(0).randint(0, 3, 50)
        s_idx, q_idx = np.arange(32), np.arange(32, 48)
        xs, ys, xq, yq = TuningManager._iltm_episode_tensors(X, y, s_idx, q_idx, torch.device("cpu"))
        assert xs.shape == (32, 5) and xq.shape == (16, 5)
        assert ys.dtype == torch.int64 and yq.dtype == torch.int64
        xs, ys, xq, yq = TuningManager._iltm_episode_tensors(
            X, y.astype(np.float32), s_idx, q_idx, torch.device("cpu"), is_reg=True)
        assert ys.dtype == torch.float32 and yq.dtype == torch.float32

    def test_tensors_feed_the_model(self):
        from tabtune.TuningManager.tuning import TuningManager
        model = _tiny_model()
        X, y = _separable_data()
        xs, ys, xq, yq = TuningManager._iltm_episode_tensors(
            X, y, np.arange(40), np.arange(40, 60), torch.device("cpu"))
        logits = _episode_forward(model, xs, ys, xq, 3)
        assert logits.shape == (20, 3)


@pytest.mark.integration
@pytest.mark.model_iltm
@pytest.mark.finetuning
class TestTuningManagerDispatch:
    """Full tune() dispatch on the wrapper with a tiny LOCAL checkpoint."""

    @pytest.fixture()
    def tiny_checkpoint(self, tmp_path):
        path = tmp_path / "tiny.pth"
        torch.save(_tiny_model(seed=1).state_dict(), path)
        return str(path)

    def _clf_wrapper(self, tiny_checkpoint):
        from tabtune.models.iltm.classifier import ILTMClassifier
        return ILTMClassifier(device="cpu", checkpoint=tiny_checkpoint, n_ensemble=2, **TINY_ARCH)

    def _reg_wrapper(self, tiny_checkpoint):
        from tabtune.models.regression.iltm.regressor import ILTMRegressorWrapper
        return ILTMRegressorWrapper(device="cpu", checkpoint=tiny_checkpoint, n_ensemble=2, **TINY_ARCH)

    def _frame(self, seed=0):
        X, y = _separable_data(seed=seed)
        return pd.DataFrame(X), pd.Series(y)

    def test_inference_dispatch(self, tiny_checkpoint):
        from tabtune.TuningManager.tuning import TuningManager
        model = self._clf_wrapper(tiny_checkpoint)
        X, y = self._frame()
        TuningManager().tune(model, X, y, strategy="inference")
        assert model._is_fitted
        assert model.predict_proba(X[:5]).shape == (5, 3)

    def test_finetune_dispatch_changes_weights_and_checkpoints(self, tiny_checkpoint, tmp_path):
        from tabtune.TuningManager.tuning import TuningManager
        model = self._clf_wrapper(tiny_checkpoint)
        X, y = self._frame()
        model._load_model()
        before = {n: p.detach().clone() for n, p in model.model_.named_parameters()
                  if n.startswith("hypernetwork_block.")}
        ckpt = str(tmp_path / "iltm_ft.pt")
        params = dict(FAST_TUNING, save_checkpoint_path=ckpt)
        model = TuningManager().tune(model, X, y, strategy="finetune", params=params)
        changed = any(n in before and not torch.equal(before[n], p)
                      for n, p in model.model_.named_parameters())
        assert changed, "fine-tuning did not update the hypernetwork weights"
        # tune() saved + reloaded the state dict (round-trip)
        import os
        assert os.path.exists(ckpt)
        state = torch.load(ckpt, map_location="cpu")
        assert any(k.startswith("hypernetwork_block.") for k in state)
        assert model.predict(X[:5]).shape == (5,)

    def test_peft_dispatch_attaches_lora_and_only_trains_adapters(self, tiny_checkpoint, tmp_path):
        from tabtune.TuningManager.tuning import TuningManager
        model = self._clf_wrapper(tiny_checkpoint)
        X, y = self._frame()
        model._load_model()
        base_before = {n: p.clone() for n, p in model.model_.named_parameters()
                       if n.startswith("hypernetwork_block.")}
        params = dict(FAST_TUNING, save_checkpoint_path=str(tmp_path / "iltm_peft.pt"),
                      peft_config={"r": 4, "lora_alpha": 8})
        model = TuningManager().tune(model, X, y, strategy="peft", params=params)
        lora = [m for m in model.model_.modules() if isinstance(m, LoRALinear)]
        assert lora, "no LoRA adapters attached under peft"
        for n, p in model.model_.named_parameters():
            if not n.startswith("hypernetwork_block."):
                continue  # data-dependent norm module, recreated per forward
            if "lora_" in n:
                assert p.requires_grad
            else:
                assert not p.requires_grad
                assert torch.equal(base_before[n.replace(".base", "")], p), f"base param {n} changed under peft"
        # the engine still predicts through the adapted module
        assert model.estimator_._model is model.model_
        assert model.predict_proba(X[:5]).shape == (5, 3)

    def test_finetune_method_peft_routes_to_lora(self, tiny_checkpoint):
        from tabtune.TuningManager.tuning import TuningManager
        model = self._clf_wrapper(tiny_checkpoint)
        X, y = self._frame()
        params = dict(FAST_TUNING, finetune_method="peft", peft_config={"r": 4})
        model = TuningManager().tune(model, X, y, strategy="finetune", params=params)
        assert any(isinstance(m, LoRALinear) for m in model.model_.modules())

    def test_sft_mode(self, tiny_checkpoint):
        from tabtune.TuningManager.tuning import TuningManager
        model = self._clf_wrapper(tiny_checkpoint)
        X, y = self._frame()
        params = dict(FAST_TUNING, finetune_mode="sft")
        model = TuningManager().tune(model, X, y, strategy="finetune", params=params)
        assert model._is_fitted

    def test_regression_finetune_dispatch(self, tiny_checkpoint):
        from tabtune.TuningManager.tuning import TuningManager
        model = self._reg_wrapper(tiny_checkpoint)
        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(100, 5).astype(np.float32))
        y = pd.Series(2.0 * X[0] - X[1] + 0.1 * rng.randn(100))
        model._load_model()
        before = {n: p.clone() for n, p in model.model_.named_parameters()
                  if n.startswith("hypernetwork_block.")}
        params = dict(FAST_TUNING, finetune_mode="turn_by_turn")
        model = TuningManager().tune(model, X, y, strategy="finetune", params=params)
        assert any(n in before and not torch.equal(before[n], p)
                   for n, p in model.model_.named_parameters())
        preds = model.predict(X[:5])
        assert preds.shape == (5,) and np.isfinite(preds).all()
