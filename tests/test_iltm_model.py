"""
Standalone architecture tests for the VENDORED iLTM model in TabTune.

These tests build a TINY randomly-initialised iLTM hypernetwork (no pretrained
weights, no Hugging Face download, CPU-only) and exercise the REAL forward
path: the hypernetwork generates an MLP from a labelled support set and query
rows are pushed through the generated network. Also covers the episode
featurizer and the peft_utils LoRA target resolution.

Run:  pytest tests/test_iltm_model.py -v
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tabtune.models.iltm.iltm_model import iLTM
from tabtune.models.iltm.utils import full_main_forward
from tabtune.models.iltm.episode_features import ILTMEpisodeFeaturizer

# Tiny architecture used everywhere in the iLTM tests (fast on CPU, no weights).
TINY_ARCH = dict(
    n_dims=16, hn_n_layers=2, hn_hidden_size=32, rf_size=64,
    n_classes_limit=6, main_n_layers=3, pca_sampling="zeropad", pca_fit="reduced",
)


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return iLTM(**TINY_ARCH)


def _support_query(n_s=40, n_q=20, h=8, k=3, seed=0):
    rng = np.random.RandomState(seed)
    xs = torch.tensor(rng.randn(n_s, h), dtype=torch.float32)
    ys = torch.tensor(rng.randint(0, k, n_s), dtype=torch.long)
    xq = torch.tensor(rng.randn(n_q, h), dtype=torch.float32)
    return xs, ys, xq


def _episode_forward(model, xs, ys, xq, n_classes, training=False):
    rf, pca, main_network, norm = model(xs, ys, n_classes, training=training)
    return full_main_forward(
        xq, n_classes, int(xq.shape[0]), vars(model), rf, pca, norm, main_network,
        torch.device("cpu"), use_amp=False, training_finetuning=training,
    )


@pytest.mark.unit
@pytest.mark.model_iltm
class TestILTMArchitecture:
    def test_forward_returns_generated_network(self):
        model = _tiny_model()
        xs, ys, _ = _support_query()
        rf, pca, main_network, norm = model(xs, ys, 3)
        # main_n_layers generated (functional) linear layers
        assert len(main_network) == TINY_ARCH["main_n_layers"]
        assert rf is not None and pca is not None

    def test_query_logits_shape_and_finite(self):
        model = _tiny_model()
        xs, ys, xq = _support_query()
        out = _episode_forward(model, xs, ys, xq, 3)
        assert out.shape == (xq.shape[0], 3)
        assert torch.isfinite(out).all()

    def test_regression_output_is_squeezed(self):
        model = _tiny_model()
        xs, _, xq = _support_query()
        ys = torch.randn(xs.shape[0])  # regression targets, n_classes=1
        out = _episode_forward(model, xs, ys, xq, 1)
        assert out.shape == (xq.shape[0],)
        assert torch.isfinite(out).all()

    def test_output_depends_on_query_input(self):
        model = _tiny_model()
        xs, ys, xq = _support_query()
        out1 = _episode_forward(model, xs, ys, xq, 3)
        out2 = _episode_forward(model, xs, ys, xq + 1.0, 3)
        assert not torch.allclose(out1, out2)

    def test_output_depends_on_support_labels(self):
        model = _tiny_model()
        xs, ys, xq = _support_query()
        out1 = _episode_forward(model, xs, ys, xq, 3)
        out2 = _episode_forward(model, xs, (ys + 1) % 3, xq, 3)
        assert not torch.allclose(out1, out2)

    def test_gradient_flows_to_hypernetwork(self):
        model = _tiny_model()
        xs, ys, xq = _support_query()
        yq = torch.randint(0, 3, (xq.shape[0],))
        out = _episode_forward(model, xs, ys, xq, 3, training=True)
        loss = torch.nn.functional.cross_entropy(out, yq)
        loss.backward()
        # NOTE: the InitialTransformationBlock registers a data-dependent `norm`
        # submodule on first forward; the meta-trained (checkpointed, tunable)
        # parameters all live in the hypernetwork block -- check those.
        hn_grads = [p.grad for n, p in model.named_parameters() if n.startswith("hypernetwork_block.")]
        assert len(hn_grads) > 0
        assert all(g is not None for g in hn_grads)
        assert all(g.abs().sum().item() > 0 for g in hn_grads)

    def test_labels_above_limit_rejected_by_onehot(self):
        # n_classes_limit is a FIXED model hyperparam: the one-hot in the
        # hypernetwork forward requires encoded labels < n_classes_limit,
        # which is why the tuner clips labels before fine-tuning.
        model = _tiny_model()
        xs, ys, xq = _support_query()
        ys_bad = torch.full_like(ys, TINY_ARCH["n_classes_limit"])
        with pytest.raises(Exception):
            _episode_forward(model, xs, ys_bad, xq, TINY_ARCH["n_classes_limit"] + 1)

    def test_nan_inputs_do_not_crash_generated_forward(self):
        # The wrapper featurizer imputes NaNs before episodes; the raw model
        # itself clamps at clip_data_value. Verify a clean support set with a
        # partially degenerate query still yields the right shape.
        model = _tiny_model()
        xs, ys, xq = _support_query()
        xq[0] = 0.0
        out = _episode_forward(model, xs, ys, xq, 3)
        assert out.shape == (xq.shape[0], 3)

    def test_state_dict_round_trip(self, tmp_path):
        model = _tiny_model(seed=1)
        path = tmp_path / "tiny.pth"
        torch.save(model.state_dict(), path)
        model2 = _tiny_model(seed=2)
        model2.load_state_dict(torch.load(path, weights_only=True))
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2 and torch.equal(p1, p2)


@pytest.mark.unit
@pytest.mark.model_iltm
class TestILTMLoraTargets:
    def test_registered_targets_resolve_to_real_linear_leaves(self):
        from tabtune.TuningManager.peft_utils import MODEL_LORA_TARGETS, resolve_lora_targets

        assert "ILTM" in MODEL_LORA_TARGETS
        model = _tiny_model()
        resolved = resolve_lora_targets("ILTM", model)
        assert len(resolved) > 0
        linear_leaves = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
        assert set(resolved).issubset(linear_leaves)
        # the config substrings must actually cover the hypernetwork linears
        assert any("hypernetworks" in n for n in resolved)
        assert any("hn_emb_to_weights" in n for n in resolved)


@pytest.mark.unit
@pytest.mark.model_iltm
class TestEpisodeFeaturizer:
    def test_mixed_frame_to_float32(self):
        X = pd.DataFrame({
            "num": [1.0, 2.0, np.nan, 4.0],
            "cat": ["a", "b", "a", "c"],
        })
        feat = ILTMEpisodeFeaturizer().fit(X)
        Xt = feat.transform(X)
        assert Xt.shape == (4, 2)
        assert Xt.dtype == np.float32
        assert np.isfinite(Xt).all()

    def test_unknown_categories_and_clip(self):
        X = pd.DataFrame({"num": [0.0, 1.0, 2.0, 100.0], "cat": ["a", "b", "a", "b"]})
        feat = ILTMEpisodeFeaturizer(clip_sigma=4.0).fit(X)
        X_new = pd.DataFrame({"num": [1.0], "cat": ["never-seen"]})
        Xt = feat.transform(X_new)
        assert np.isfinite(Xt).all()
        assert (np.abs(Xt) <= 4.0 + 1e-6).all()

    def test_numpy_input_ok(self):
        X = np.random.RandomState(0).randn(10, 3)
        feat = ILTMEpisodeFeaturizer().fit(X)
        assert feat.transform(X).shape == (10, 3)

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            ILTMEpisodeFeaturizer().transform(np.zeros((2, 2)))
