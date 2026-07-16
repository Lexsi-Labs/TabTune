"""
Functional tests for the VENDORED TabFM architecture
(tabtune/models/tabfm/model/model.py).

The architecture instantiates tiny and runs on random tensors, so these tests
exercise the **real forward** -- shapes, finiteness, gradient flow, the
classification vs regression heads, the categorical-mask path and the active-
feature-count (`d`) path -- **without downloading any Hugging Face weights and
without a GPU**. They only require torch.

Run:  pytest tests/test_tabfm_model.py -v
"""
import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_standalone(relpath, name):
    """Load a TabFM module by path (model.py / peft_utils.py have no relative
    imports), so these tests don't depend on the whole model zoo importing.
    The module is registered in sys.modules so dataclasses resolve correctly."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    TabFM = _load_standalone("tabtune/models/tabfm/model/model.py", "tabfm_model_std").TabFM
except Exception as e:  # pragma: no cover
    pytest.skip(f"TabFM architecture import failed: {e}", allow_module_level=True)


def _tiny_model(is_classifier=True, max_classes=3):
    torch.manual_seed(0)
    m = TabFM(
        embed_dim=8, max_classes=max_classes, col_num_blocks=1, col_nhead=2,
        col_num_inds=2, row_num_blocks=1, row_nhead=2, row_num_cls=2,
        icl_num_blocks=1, icl_nhead=2, ff_factor=2, feature_group_size=3,
        num_freq=8, is_classifier=is_classifier,
    )
    # Checkpoints normally supply the Fourier frequencies; randomise them so the
    # forward genuinely depends on x (default buffers are zeros).
    with torch.no_grad():
        m.cell_embedder.fourier_frequencies.normal_(0, 0.5)
        m.cell_embedder.fourier_frequencies_cat.normal_(0, 0.5)
    return m.float().eval()


def _episode(B=2, T=12, H=5, s=8, n_classes=3, reg=False):
    torch.manual_seed(1)
    x = torch.randn(B, T, H)
    if reg:
        y = torch.randn(B, T)
    else:
        y = torch.randint(0, n_classes, (B, T))
    train_size = torch.full((B,), s, dtype=torch.long)
    return x, y, train_size


@pytest.mark.unit
@pytest.mark.model_tabfm
class TestTabFMForward:
    def test_classification_output_shape(self):
        m = _tiny_model(is_classifier=True, max_classes=3)
        x, y, ts = _episode()
        out = m(x, y, ts)
        assert out.shape == (2, 12, 3)
        assert torch.isfinite(out).all()

    def test_regression_output_shape(self):
        m = _tiny_model(is_classifier=False)
        x, y, ts = _episode(reg=True)
        out = m(x, y, ts)
        assert out.shape == (2, 12, 1)
        assert torch.isfinite(out).all()

    def test_query_slice_matches_predict_contract(self):
        # The vendored predict path slices out[:, train_size:, :]; verify the shape.
        m = _tiny_model()
        B, T, s = 2, 12, 8
        x, y, ts = _episode(B=B, T=T, s=s)
        out = m(x, y, ts)
        query = out[:, s:, :]
        assert query.shape == (B, T - s, 3)

    def test_output_depends_on_x(self):
        m = _tiny_model()
        x, y, ts = _episode()
        out1 = m(x, y, ts)
        out2 = m(x + 3.0, y, ts)
        assert not torch.allclose(out1, out2)

    def test_gradient_flows_to_attention_and_embedders(self):
        m = _tiny_model().train()
        x, y, ts = _episode()
        loss = m(x, y, ts)[:, 8:, :].float().pow(2).mean()
        loss.backward()
        grads = {n: p.grad for n, p in m.named_parameters() if p.grad is not None}
        # attention projections + Fourier cell embedder should receive gradient
        assert any("q_proj" in n for n in grads)
        assert any("in_linear" in n for n in grads)
        assert all(torch.isfinite(g).all() for g in grads.values())

    def test_cat_mask_path_runs(self):
        m = _tiny_model()
        x, y, ts = _episode(H=5)
        cat_mask = torch.zeros(2, 5, dtype=torch.bool)
        cat_mask[:, :2] = True  # first two columns categorical
        out = m(x, y, ts, cat_mask=cat_mask)
        assert out.shape == (2, 12, 3)
        assert torch.isfinite(out).all()

    def test_active_feature_count_d_path(self):
        m = _tiny_model()
        x, y, ts = _episode(H=6)
        d = torch.tensor([4, 4], dtype=torch.long)  # only 4 of 6 features "active"
        out = m(x, y, ts, d=d)
        assert out.shape == (2, 12, 3)
        assert torch.isfinite(out).all()

    def test_nan_inputs_are_handled(self):
        m = _tiny_model()
        x, y, ts = _episode()
        x[0, 0, 0] = float("nan")  # model does nan_to_num internally
        out = m(x, y, ts)
        assert torch.isfinite(out).all()

    def test_max_classes_attribute(self):
        m = _tiny_model(max_classes=5)
        assert m.max_classes == 5
        x, y, ts = _episode(n_classes=5)
        assert m(x, y, ts).shape[-1] == 5


@pytest.mark.unit
@pytest.mark.model_tabfm
class TestTabFMSubmodules:
    def test_real_lora_target_names_exist(self):
        """The peft_utils TabFM target substrings must match real Linear leaves."""
        _peft = _load_standalone("tabtune/TuningManager/peft_utils.py", "tabfm_peft_std2")
        MODEL_LORA_TARGETS, resolve_lora_targets = _peft.MODEL_LORA_TARGETS, _peft.resolve_lora_targets

        m = _tiny_model()
        cfg = MODEL_LORA_TARGETS["TabFM"]
        leaf_names = [n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)]
        # every real leaf that contains one of our substrings should resolve
        resolved = resolve_lora_targets("TabFM", m)
        assert len(resolved) > 0
        # sanity: the canonical attention projections are present and targeted
        assert any("q_proj" in n for n in leaf_names)
        assert any("q_proj" in r for r in resolved)
        assert any(tok in " ".join(leaf_names) for tok in cfg.target_substrings)
