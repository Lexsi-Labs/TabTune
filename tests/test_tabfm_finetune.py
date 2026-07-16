"""
Tests for TabFM fine-tuning + PEFT mechanics on the real architecture.

Validates, without HF weights or a GPU (torch only):
  * `apply_tabular_lora("TabFM", ...)` wraps the real Linear leaves, freezes the
    base weights and keeps the forward working;
  * a real gradient loop over the model's forward reduces the loss (the core of
    `_finetune_tabfm`), proving the training path is wired correctly;
  * `TuningManager._tabfm_episode_tensors` builds correctly-shaped tensors that
    match the model's forward contract (skipped if the full package can't import).

The architecture (`model.py`) and LoRA utilities (`peft_utils.py`) have no
relative imports, so they are loaded **standalone** -- these tests do not depend
on the rest of the TabTune model zoo importing successfully.

Run:  pytest tests/test_tabfm_finetune.py -v
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_standalone(relpath, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register so dataclasses in the module resolve
    spec.loader.exec_module(mod)
    return mod


try:
    TabFM = _load_standalone("tabtune/models/tabfm/model/model.py", "tabfm_model_std").TabFM
    _peft = _load_standalone("tabtune/TuningManager/peft_utils.py", "tabfm_peft_std")
    LoRALinear, apply_tabular_lora = _peft.LoRALinear, _peft.apply_tabular_lora
except Exception as e:  # pragma: no cover
    pytest.skip(f"TabFM model/peft standalone import failed: {e}", allow_module_level=True)


def _tiny_model(is_classifier=True, max_classes=3):
    torch.manual_seed(0)
    m = TabFM(
        embed_dim=8, max_classes=max_classes, col_num_blocks=1, col_nhead=2,
        col_num_inds=2, row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, feature_group_size=3,
        num_freq=8, is_classifier=is_classifier,
    )
    with torch.no_grad():
        m.cell_embedder.fourier_frequencies.normal_(0, 0.5)
        m.cell_embedder.fourier_frequencies_cat.normal_(0, 0.5)
    return m.float()


@pytest.mark.unit
@pytest.mark.model_tabfm
@pytest.mark.finetuning
class TestEpisodeTensors:
    def _builder(self):
        try:
            from tabtune.TuningManager.tuning import TuningManager
        except Exception as e:  # heavy import chain; skip if a dep is missing
            pytest.skip(f"could not import TuningManager: {e}")
        return TuningManager._tabfm_episode_tensors

    def test_classification_shapes(self):
        build = self._builder()
        X = np.random.randn(50, 4).astype(np.float32)
        y = np.random.randint(0, 3, 50)
        s_idx, q_idx = np.arange(0, 30), np.arange(30, 45)
        x_t, y_t, ts, d_t, cm, yq = build(X, y, s_idx, q_idx, None, torch.device("cpu"), False)
        assert x_t.shape == (1, 45, 4)          # support(30)+query(15) along T
        assert y_t.shape == (1, 45)
        assert int(ts.item()) == 30              # train_size == len(support)
        assert int(d_t.item()) == 4              # d == H
        assert cm is None
        assert yq.shape == (15,)
        assert y_t.dtype == torch.long

    def test_regression_shapes_and_dtype(self):
        build = self._builder()
        X = np.random.randn(40, 3).astype(np.float32)
        y = np.random.randn(40).astype(np.float32)
        s_idx, q_idx = np.arange(0, 24), np.arange(24, 36)
        x_t, y_t, ts, d_t, cm, yq = build(X, y, s_idx, q_idx, None, torch.device("cpu"), True)
        assert x_t.shape == (1, 36, 3)
        assert y_t.dtype == torch.float32
        assert yq.shape == (12,)

    def test_episode_tensors_feed_the_model(self):
        build = self._builder()
        m = _tiny_model()
        X = np.random.randn(60, 5).astype(np.float32)
        y = np.random.randint(0, 3, 60)
        x_t, y_t, ts, d_t, cm, yq = build(X, y, np.arange(40), np.arange(40, 55), None, torch.device("cpu"), False)
        out = m(x_t, y_t, ts, cat_mask=cm, d=d_t)
        assert out.shape == (1, 55, 3)
        query = out[:, int(ts.item()):, :].reshape(-1, out.size(-1))
        assert query.shape == (15, 3)


@pytest.mark.unit
@pytest.mark.model_tabfm
@pytest.mark.finetuning
class TestLoRA:
    def test_lora_wraps_and_freezes(self):
        m = _tiny_model()
        n_linear_before = sum(isinstance(mod, torch.nn.Linear) and not isinstance(mod, LoRALinear)
                              for mod in m.modules())
        m = apply_tabular_lora("TabFM", m, {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0})
        wrapped = [mod for mod in m.modules() if isinstance(mod, LoRALinear)]
        assert len(wrapped) > 0
        for w in wrapped:
            assert all(not p.requires_grad for p in w.base.parameters())
            assert w.lora_A.weight.requires_grad and w.lora_B.weight.requires_grad
        assert n_linear_before > 0

    def test_forward_still_works_after_lora(self):
        m = _tiny_model()
        m = apply_tabular_lora("TabFM", m, {"r": 4})
        x = torch.randn(2, 10, 5)
        y = torch.randint(0, 3, (2, 10))
        ts = torch.full((2,), 6, dtype=torch.long)
        out = m(x, y, ts)
        assert out.shape == (2, 10, 3) and torch.isfinite(out).all()


@pytest.mark.unit
@pytest.mark.model_tabfm
@pytest.mark.finetuning
class TestFinetuneLoopReducesLoss:
    def _run(self, model, steps=40):
        """SFT-style: repeatedly forward one fixed episode and descend."""
        torch.manual_seed(3)
        H, s, q, K = 5, 16, 8, 3
        X = torch.randn(s + q, H)
        y = torch.randint(0, K, (s + q,))
        x_t = X.unsqueeze(0)
        y_t = torch.cat([y[:s], torch.zeros(q, dtype=torch.long)]).unsqueeze(0)
        ts = torch.full((1,), s, dtype=torch.long)
        yq = y[s:]
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        loss_fn = torch.nn.CrossEntropyLoss()
        losses = []
        for _ in range(steps):
            opt.zero_grad()
            logits = model(x_t, y_t, ts)[:, s:, :].reshape(-1, K)
            loss = loss_fn(logits, yq)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        return losses

    def test_full_finetune_reduces_loss(self):
        m = _tiny_model().train()
        losses = self._run(m)
        assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"

    def test_lora_finetune_reduces_loss(self):
        m = _tiny_model().train()
        m = apply_tabular_lora("TabFM", m, {"r": 8, "lora_alpha": 16})
        losses = self._run(m)
        assert losses[-1] < losses[0], f"LoRA loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"
