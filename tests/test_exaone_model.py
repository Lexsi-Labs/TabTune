"""
Standalone tests for the VENDORED EXAONE Tabular stack and its TabTune wrappers.

Almost everything here runs against a TINY randomly-initialised Cross-axis
Summary Transformer (width 32, 2 blocks, ~52k parameters) built directly from a
``ModelConfig`` -- no Hugging Face download, no gating env var, CPU, milliseconds.
That exercises the genuine forward, the genuine preprocessor and the genuine
autograd graph rather than a stand-in that cannot fail the same way.

Three areas get most of the attention:

* :class:`EXAONEFeatureEncoder` -- the ordinal encoder every EXAONE entry point
  shares (classifier, regressor, ``Dataprocess``), so a disagreement between them
  is impossible by construction only if this class is nailed down.
* The three **TabTune modifications** to the vendored code (see
  ``tabtune/models/exaone/__init__.py``). Nothing else in the suite guards them,
  and the dangerous one fails *silently*: upstream's unconditional detach returns
  a non-``None`` but wrong support gradient, so "grad is not None" would pass
  against the bug. It is therefore checked against central finite differences in
  float64, with the upstream behaviour monkeypatched back in to prove the check
  can fail.
* The wrapper contract the pipeline / TuningManager code relies
  on: lazy loading, strategy validation, episode features, the architectural
  class capacity, and the regression "no published weights" error.

The handful of tests that need the released 20.8M-parameter geometry build a
random-init checkpoint at that geometry inside a fixture and are marked
``slow``.

Run:  pytest tests/test_exaone_model.py -v
"""
from __future__ import annotations

import io
import logging

import joblib
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tabtune.models.exaone import backbone as _bk
from tabtune.models.exaone.classifier import EXAONETabularClassifier
from tabtune.models.exaone.config import (
    InferenceManifest,
    ModelConfig,
    RegressionConfig,
    RuntimeConfig,
)
from tabtune.models.exaone.episode_features import EXAONEFeatureEncoder
from tabtune.models.exaone.model.heads import ClassificationModel, RegressionModel
from tabtune.models.regression.exaone.regressor import EXAONETabularRegressorWrapper

pytestmark = pytest.mark.model_exaone

#: A toy geometry the CPU runs in milliseconds. Mirrors the one in
#: model.
TINY_MODEL = dict(
    width=32, head_count=2, block_count=2, feedforward_widths=64,
    columns_per_group=1, item_summary_count=2, feature_summary_count=4,
    feature_attention_repeats=1, class_capacity=3,
)

#: The output feed-forward token budget the CUDA planner would otherwise search
#: for. A fixed generous value is what the CPU path uses.
FF_CHUNK = _bk.DEFAULT_FEEDFORWARD_TOKEN_CHUNK

#: A tiny quantile head: odd, >= 3, so the median index is 2.
TINY_REGRESSION = dict(quantile_count=5, decoder_hidden_width=32)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _tiny_classification_manifest(n_ensemble=1, seed=0):
    return InferenceManifest(
        task="classification",
        model=ModelConfig(**TINY_MODEL),
        runtime=RuntimeConfig(
            compute_dtype="float32", ensemble_count=n_ensemble, seed=seed
        ),
    )


def _tiny_regression_manifest(n_ensemble=1, seed=0):
    return InferenceManifest(
        task="regression",
        model=ModelConfig(**TINY_MODEL),
        regression=RegressionConfig(**TINY_REGRESSION),
        runtime=RuntimeConfig(
            compute_dtype="float32", ensemble_count=n_ensemble, seed=seed
        ),
    )


def build_tiny_backbone(dtype=torch.float32, seed=0):
    """A tiny real ``ClassificationModel`` in ``dtype``, in eval mode."""
    torch.manual_seed(seed)
    model = ClassificationModel(
        ModelConfig(**TINY_MODEL), device=torch.device("cpu"), dtype=dtype
    )
    return model.eval()


def build_tiny_classifier(n_ensemble=1, seed=0):
    """A TabTune ``EXAONETabularClassifier`` on a toy backbone, no weights.

    ``_load_model`` returns early when ``model_`` and ``estimator_`` are already
    set, so a later ``fit`` runs the real vendored path without ever resolving a
    checkpoint.
    """
    from tabtune.models.exaone._upstream_classifier import (
        EXAONETabularClassifier as VendoredClassifier,
    )

    manifest = _tiny_classification_manifest(n_ensemble=n_ensemble, seed=seed)
    model = build_tiny_backbone(seed=seed)
    clf = EXAONETabularClassifier(
        device="cpu", n_ensemble=n_ensemble, random_state=seed
    )
    clf.model_ = model
    clf.manifest_ = manifest
    clf.estimator_ = VendoredClassifier(manifest, device="cpu", model=model)
    return clf


def build_tiny_regressor(n_ensemble=1, seed=0):
    """A TabTune ``EXAONETabularRegressorWrapper`` on a toy quantile backbone."""
    from tabtune.models.exaone._upstream_regressor import (
        EXAONETabularRegressor as VendoredRegressor,
    )

    manifest = _tiny_regression_manifest(n_ensemble=n_ensemble, seed=seed)
    torch.manual_seed(seed)
    model = RegressionModel(
        ModelConfig(**TINY_MODEL),
        RegressionConfig(**TINY_REGRESSION),
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).eval()
    reg = EXAONETabularRegressorWrapper(
        device="cpu", n_ensemble=n_ensemble, random_state=seed
    )
    reg.model_ = model
    reg.manifest_ = manifest
    reg.estimator_ = VendoredRegressor(manifest, device="cpu", model=model)
    return reg


def _episode(n_support=8, n_features=3, n_query=2, dtype=torch.float64, seed=1):
    """A single-member ``(x_support, y_support, x_query)`` episode."""
    generator = torch.Generator().manual_seed(seed)
    x_support = torch.randn(1, n_support, n_features, generator=generator, dtype=dtype)
    y_support = torch.tensor(
        [[float(i % 3) for i in range(n_support)]], dtype=dtype
    )
    x_query = torch.randn(1, n_query, n_features, generator=generator, dtype=dtype)
    return x_support, y_support, x_query


def _frame(n=40, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "cat": rng.choice(["red", "green", "blue"], n),
    })
    y = pd.Series(np.where(X["f1"] > 0, "pos", "neg"))
    return X, y


# --------------------------------------------------------------------------- #
# Released-geometry checkpoint (20.8M parameters, random init)
# --------------------------------------------------------------------------- #
def _write_released_checkpoint(task, path, seed=0):
    from safetensors.torch import save_file

    from tabtune.models.exaone.model.heads import build_model
    from tabtune.models.exaone.presets import released_manifest

    torch.manual_seed(seed)
    model = build_model(
        released_manifest(task), device=torch.device("cpu"), dtype=torch.float32
    )
    save_file(
        {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()},
        str(path),
    )
    return str(path)


@pytest.fixture(scope="module")
def released_classifier_checkpoint(tmp_path_factory):
    """Random-init weights at the RELEASED classification geometry.

    The published weights cannot be downloaded in a sandbox (and are research-only
    besides), but the *geometry* is fully determined by the frozen manifest, and
    the checkpoint loader validates key order and shapes against a freshly built
    model -- so a random-init file at that geometry exercises every code path the
    real file would, values aside.
    """
    try:
        path = tmp_path_factory.mktemp("exaone_release") / "clf.safetensors"
        return _write_released_checkpoint("classification", path)
    except Exception as exc:  # pragma: no cover - environment, not a defect
        pytest.skip(f"EXAONE released-geometry weights unavailable: {exc}")


@pytest.fixture(scope="module")
def released_regressor_checkpoint(tmp_path_factory):
    """Random-init weights at the RELEASED regression geometry (never published)."""
    try:
        path = tmp_path_factory.mktemp("exaone_release") / "reg.safetensors"
        return _write_released_checkpoint("regression", path)
    except Exception as exc:  # pragma: no cover - environment, not a defect
        pytest.skip(f"EXAONE released-geometry weights unavailable: {exc}")


# --------------------------------------------------------------------------- #
# EXAONEFeatureEncoder
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestFeatureEncoder:
    def test_mixed_dtypes_encode_to_contiguous_codes(self):
        frame = pd.DataFrame({
            "num": [1.5, 2.5, 3.5, 4.5],
            "int": [10, 20, 30, 40],
            "string": ["a", "b", "a", "c"],
            "bool": [True, False, True, False],
            "categorical": pd.Categorical(["x", "y", "x", "y"]),
            "when": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03",
                                    "2020-01-04"]),
        })
        encoder = EXAONEFeatureEncoder().fit(frame)
        out = encoder.transform(frame)

        assert out.shape == (4, 6)
        assert out.dtype == np.float64
        assert encoder.n_features_in_ == 6
        # Booleans and pandas categoricals go through the categorical path;
        # numeric, integer and datetime columns pass through as values.
        assert set(encoder.categorical_columns_) == {"string", "bool", "categorical"}
        np.testing.assert_array_equal(out[:, 0], [1.5, 2.5, 3.5, 4.5])
        np.testing.assert_array_equal(out[:, 1], [10, 20, 30, 40])
        np.testing.assert_array_equal(out[:, 2], [0, 1, 0, 2])   # a, b, a, c
        np.testing.assert_array_equal(out[:, 3], [1, 0, 1, 0])   # False < True
        np.testing.assert_array_equal(out[:, 4], [0, 1, 0, 1])   # x, y
        # Datetimes become int64 nanoseconds, strictly increasing here.
        assert (np.diff(out[:, 5]) > 0).all()

    def test_missing_values_are_preserved_as_nan(self):
        """Never imputed here: the model encodes NaN as an explicit channel."""
        frame = pd.DataFrame({
            "num": [1.0, np.nan, 3.0],
            "string": ["a", "b", None],
            "when": pd.to_datetime(["2020-01-01", None, "2020-01-03"]),
        })
        out = EXAONEFeatureEncoder().fit_transform(frame)
        assert np.isnan(out[1, 0])
        assert np.isnan(out[2, 1])
        assert np.isnan(out[1, 2])
        # ... and nothing else is NaN.
        assert int(np.isnan(out).sum()) == 3

    def test_inf_is_mapped_to_nan_with_a_warning(self, caplog):
        frame = pd.DataFrame({"a": [1.0, np.inf, -np.inf, 2.0]})
        encoder = EXAONEFeatureEncoder().fit(frame)
        with caplog.at_level(
            logging.WARNING, logger="tabtune.models.exaone.episode_features"
        ):
            out = encoder.transform(frame)
        assert not np.isinf(out).any()
        np.testing.assert_array_equal(np.isnan(out).ravel(),
                                      [False, True, True, False])
        assert any("infinite value" in record.message for record in caplog.records)

    def test_unseen_categories_become_nan(self):
        encoder = EXAONEFeatureEncoder().fit(pd.DataFrame({"s": ["a", "b"]}))
        out = encoder.transform(pd.DataFrame({"s": ["a", "never-seen"]}))
        np.testing.assert_array_equal(out.ravel()[:1], [0.0])
        assert np.isnan(out.ravel()[1])

    def test_pandas_nullable_dtypes_and_pd_na(self):
        """``convert_dtypes()`` gives Int64/string/boolean, whose missing is pd.NA.

        ``pd.NA == pd.NA`` is pd.NA, so a ``value == value`` missingness check
        would raise "boolean value of NA is ambiguous" rather than report a
        missing value. This is the shape Parquet and Arrow hand over.
        """
        frame = pd.DataFrame({
            "num": [1.0, 2.0, np.nan, 4.0],
            "int": [1, 2, 3, 4],
            "string": ["a", "b", "a", "c"],
            "bool": [True, False, True, False],
        }).convert_dtypes()
        frame.loc[0, "string"] = pd.NA
        frame.loc[1, "int"] = pd.NA
        frame.loc[2, "bool"] = pd.NA
        assert str(frame["int"].dtype) == "Int64"
        assert str(frame["string"].dtype) == "string"
        assert str(frame["bool"].dtype) == "boolean"

        out = EXAONEFeatureEncoder().fit_transform(frame)
        assert out.dtype == np.float64
        assert np.isnan(out[2, 0]) and np.isnan(out[1, 1])
        assert np.isnan(out[0, 2]) and np.isnan(out[2, 3])
        np.testing.assert_array_equal(out[1:, 2], [1.0, 0.0, 2.0])  # b, a, c

    def test_missing_column_at_transform_raises(self):
        encoder = EXAONEFeatureEncoder().fit(pd.DataFrame({"a": [1.0], "b": [2.0]}))
        with pytest.raises(ValueError, match="absent from the frame"):
            encoder.transform(pd.DataFrame({"a": [1.0]}))

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fitted before transform"):
            EXAONEFeatureEncoder().transform(pd.DataFrame({"a": [1.0]}))

    def test_joblib_round_trip_is_exact(self):
        """``TabularPipeline.save`` joblib-dumps the whole pipeline, this included."""
        frame = pd.DataFrame({
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "a"],
            "flag": [True, False, True],
        })
        encoder = EXAONEFeatureEncoder().fit(frame)
        buffer = io.BytesIO()
        joblib.dump(encoder, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        np.testing.assert_array_equal(encoder.transform(frame),
                                      restored.transform(frame))
        assert restored.categories_ == encoder.categories_

    def test_category_order_is_deterministic_across_row_order(self):
        """Codes must not depend on which row happened to come first.

        A support set encoded one way and a query set encoded another would put
        the same category on two different codes, which the model reads as two
        different categories.
        """
        frame = pd.DataFrame({"m": [1, "a", 2, "b", None] * 2})
        forward = EXAONEFeatureEncoder().fit(frame)
        backward = EXAONEFeatureEncoder().fit(frame.iloc[::-1].reset_index(drop=True))
        # Mixed types are ordered by (type name, str(value)): ints before strs.
        assert forward.categories_ == {"m": [1, 2, "a", "b"]}
        assert forward.categories_ == backward.categories_
        np.testing.assert_array_equal(forward.transform(frame),
                                      backward.transform(frame))

    def test_numpy_input_gets_positional_column_names(self):
        array = np.random.RandomState(0).randn(6, 3)
        encoder = EXAONEFeatureEncoder().fit(array)
        assert encoder.columns_ == ["f0", "f1", "f2"]
        assert encoder.transform(array).shape == (6, 3)


# --------------------------------------------------------------------------- #
# The three TabTune modifications to the vendored code
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestSupportGradientPatch:
    """``model/attention.py``: the retained K/V pair is detached only under no-grad."""

    def test_supports_grad_probe_is_true(self):
        assert _bk.supports_grad(build_tiny_backbone()) is True

    def test_support_row_gradient_matches_central_finite_differences(self):
        """The load-bearing test for the whole integration.

        Upstream detaches the retained key/value pair unconditionally, which severs
        the support-to-query path in *every* layer while still returning a
        non-``None`` gradient. A test that merely asserted "grad is not None"
        would pass against that bug, so the analytic gradient is checked against
        central differences in float64 instead.
        """
        model = build_tiny_backbone(dtype=torch.float64)
        x_support, y_support, x_query = _episode()

        def scalar(support):
            return model(
                support, y_support, x_query, feedforward_token_chunk=FF_CHUNK
            )[0, 0, 1]

        leaf = x_support.clone().requires_grad_(True)
        scalar(leaf).backward()
        analytic = leaf.grad.detach().clone()

        step = 1e-6
        numeric = torch.zeros_like(x_support)
        with torch.no_grad():
            for row in range(x_support.shape[1]):
                for column in range(x_support.shape[2]):
                    plus = x_support.clone()
                    plus[0, row, column] += step
                    minus = x_support.clone()
                    minus[0, row, column] -= step
                    numeric[0, row, column] = (scalar(plus) - scalar(minus)) / (2 * step)

        assert numeric.norm().item() > 1e-6, "degenerate episode: no signal to check"
        relative = ((analytic - numeric).norm() / numeric.norm()).item()
        assert relative < 1e-5, f"support gradient disagrees with FD: {relative:.3e}"

    def test_upstream_unconditional_detach_would_fail_that_check(self, monkeypatch):
        """Proof the finite-difference check can fail -- and how quietly.

        With upstream's version monkeypatched back in, the gradient is still
        non-``None`` and still finite; it is simply wrong, by tens of percent and
        with flipped signs on individual rows.
        """
        from tabtune.models.exaone.model import attention

        model = build_tiny_backbone(dtype=torch.float64)
        x_support, y_support, x_query = _episode()

        def scalar(support):
            return model(
                support, y_support, x_query, feedforward_token_chunk=FF_CHUNK
            )[0, 0, 1]

        step = 1e-6
        numeric = torch.zeros_like(x_support)
        with torch.no_grad():
            for row in range(x_support.shape[1]):
                for column in range(x_support.shape[2]):
                    plus = x_support.clone()
                    plus[0, row, column] += step
                    minus = x_support.clone()
                    minus[0, row, column] -= step
                    numeric[0, row, column] = (scalar(plus) - scalar(minus)) / (2 * step)

        monkeypatch.setattr(
            attention, "_retain_or_detach", lambda key, value: (key.detach(), value.detach())
        )
        leaf = x_support.clone().requires_grad_(True)
        scalar(leaf).backward()
        upstream = leaf.grad.detach().clone()

        assert upstream is not None and torch.isfinite(upstream).all()
        relative = ((upstream - numeric).norm() / numeric.norm()).item()
        assert relative > 1e-2, (
            "the upstream detach no longer changes the support gradient; the "
            "finite-difference test above may have stopped discriminating"
        )
        assert int(((upstream * numeric) < 0).sum()) > 0, "expected sign flips"

    def test_every_parameter_receives_a_gradient(self):
        """Upstream left two attention parameters permanently gradient-free.

        ``cache_context_projections`` is the only consumer of the final layer's
        item-attention key/value projections, so detaching there means they never
        appear in any backward graph -- a fine-tune would silently freeze them.
        """
        model = build_tiny_backbone(dtype=torch.float64)
        x_support, y_support, x_query = _episode()
        model(
            x_support, y_support, x_query, feedforward_token_chunk=FF_CHUNK
        ).sum().backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert missing == []

    def test_upstream_detach_starves_two_attention_parameters(self, monkeypatch):
        """The counterfactual for the test above, so it cannot rot into a tautology."""
        from tabtune.models.exaone.model import attention

        monkeypatch.setattr(
            attention, "_retain_or_detach", lambda key, value: (key.detach(), value.detach())
        )
        model = build_tiny_backbone(dtype=torch.float64)
        x_support, y_support, x_query = _episode()
        model(
            x_support, y_support, x_query, feedforward_token_chunk=FF_CHUNK
        ).sum().backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert missing, "expected upstream to starve at least one parameter"
        assert all("attention" in name for name in missing)


@pytest.mark.unit
class TestChunkedFeatureAttention:
    """``runtime.run_in_chunks``: no longer forces ``no_grad``, inference unchanged."""

    @staticmethod
    def _upstream_run_in_chunks(operation, primary, *aligned, chunk_size, axis=0):
        """Upstream's version: unconditional ``no_grad``, preallocate + copy_, detach."""
        with torch.no_grad():
            row_count = primary.shape[axis]
            if row_count <= chunk_size:
                return operation(primary, *aligned).detach()
            destination = None
            for start in range(0, row_count, chunk_size):
                length = min(chunk_size, row_count - start)
                source = torch.narrow(primary, axis, start, length)
                companions = tuple(
                    torch.narrow(value, axis, start, length) for value in aligned
                )
                result = operation(source, *companions)
                if destination is None:
                    shape = list(result.shape)
                    shape[axis] = row_count
                    destination = torch.empty(
                        shape, dtype=result.dtype, device=result.device
                    )
                torch.narrow(destination, axis, start, length).copy_(result)
            return destination.detach()

    @pytest.mark.parametrize("guard", ["no_grad", "inference_mode"])
    def test_matches_upstream_bit_for_bit_under_a_grad_guard(self, guard):
        from tabtune.models.exaone.runtime import run_in_chunks

        def operation(block):
            return torch.tanh(block * 1.5) + 0.25

        context = torch.no_grad if guard == "no_grad" else torch.inference_mode
        with context():
            source = torch.randn(11, 4)
            for chunk_size in (1, 3, 11, 64):
                patched = run_in_chunks(operation, source, chunk_size=chunk_size)
                upstream = self._upstream_run_in_chunks(
                    operation, source, chunk_size=chunk_size
                )
                assert torch.equal(patched, upstream)
                assert patched.requires_grad is False

    def test_is_differentiable_when_a_caller_enabled_gradients(self):
        """Upstream returned ``x.grad is None`` here, decided by GPU memory pressure."""
        from tabtune.models.exaone.runtime import run_in_chunks

        leaf = torch.randn(10, 3, requires_grad=True)
        chunked = run_in_chunks(lambda block: block * 2.0, leaf, chunk_size=3)
        unchunked = run_in_chunks(lambda block: block * 2.0, leaf, chunk_size=100)
        assert torch.equal(chunked, unchunked)
        assert chunked.requires_grad
        chunked.sum().backward()
        assert leaf.grad is not None
        torch.testing.assert_close(leaf.grad, torch.full_like(leaf, 2.0))

    @pytest.mark.parametrize("guard", ["no_grad", "inference_mode"])
    def test_model_output_is_bit_identical_with_and_without_row_chunking(self, guard):
        """The planner turns row chunking on by itself under memory pressure.

        Predictions must not depend on that decision, so the two paths are compared
        bit-for-bit rather than approximately.
        """
        model = build_tiny_backbone()
        x_support, y_support, x_query = _episode(
            n_support=16, n_features=5, n_query=4, dtype=torch.float32, seed=2
        )
        context = torch.no_grad if guard == "no_grad" else torch.inference_mode
        with context():
            full = model(
                x_support, y_support, x_query, feedforward_token_chunk=FF_CHUNK
            )
            chunked = model(
                x_support, y_support, x_query,
                feedforward_token_chunk=FF_CHUNK,
                feature_attention_row_chunk=3,
            )
            assert torch.equal(full, chunked)

    def test_chunked_path_agrees_with_the_full_path_under_autograd(self):
        model = build_tiny_backbone(dtype=torch.float64)
        x_support, y_support, x_query = _episode(n_support=16, n_features=5)
        full = model(x_support, y_support, x_query, feedforward_token_chunk=FF_CHUNK)
        chunked = model(
            x_support, y_support, x_query,
            feedforward_token_chunk=FF_CHUNK,
            feature_attention_row_chunk=3,
        )
        # Both paths stay in the graph -- the parameters require grad.
        assert full.requires_grad and chunked.requires_grad
        torch.testing.assert_close(full, chunked)


# --------------------------------------------------------------------------- #
# Wrapper contract -- classification
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestClassifierContract:
    def test_construction_is_lazy(self):
        """Importing and constructing must not touch torch weights or the Hub."""
        clf = EXAONETabularClassifier(device="cpu")
        assert clf.model_ is None
        assert clf.estimator_ is None
        assert clf.manifest_ is None
        assert clf.feature_encoder_ is None
        assert list(clf.parameters()) == []

    @pytest.mark.parametrize("strategy", ["inference", "finetune", "peft"])
    def test_accepts_the_documented_strategies(self, strategy):
        assert EXAONETabularClassifier(tuning_strategy=strategy).tuning_strategy == strategy

    def test_rejects_an_unknown_tuning_strategy(self):
        with pytest.raises(ValueError, match="tuning_strategy"):
            EXAONETabularClassifier(tuning_strategy="nope")

    def test_unknown_model_params_are_absorbed_not_raised(self):
        clf = EXAONETabularClassifier(
            device="cpu", tuning_strategy="finetune",
            task_type="classification", checkpoint_dir="/tmp",
            totally_unknown_key=1,
        )
        assert clf._extra_kwargs == {"totally_unknown_key": 1}
        # Pipeline-level keys must never reach the vendored stack.
        assert "task_type" not in clf._extra_kwargs
        assert "checkpoint_dir" not in clf._extra_kwargs
        # ... and an unknown key must not leak into the manifest overrides either.
        assert set(clf._manifest_overrides()) <= {"ensemble_count", "seed", "compute_dtype"}

    def test_manifest_overrides_carry_the_three_supported_knobs(self):
        clf = EXAONETabularClassifier(
            device="cpu", n_ensemble=3, random_state=7, compute_dtype="float32"
        )
        assert clf._manifest_overrides() == {
            "ensemble_count": 3, "seed": 7, "compute_dtype": "float32"
        }

    def test_max_classes_is_architectural_capacity_not_the_dataset(self):
        # Before any load, the module-level constant (the released head width).
        assert EXAONETabularClassifier().max_classes == _bk.CLASS_CAPACITY == 10
        clf = build_tiny_classifier()
        X, y = _frame(n=24)
        clf.fit(X, y)
        assert clf.n_classes_ == 2
        assert clf.max_classes == TINY_MODEL["class_capacity"] == 3
        assert clf.max_classes > clf.n_classes_

    def test_prepare_episode_features_shape_and_dtype(self):
        clf = build_tiny_classifier()
        X, y = _frame(n=24)
        clf.fit(X, y)
        features, categorical_mask = clf.prepare_episode_features(X)
        assert features.shape == (24, 3)
        assert features.dtype == np.float32
        # EXAONE infers categoricals per ensemble member from the support set, so
        # there is no mask to hand back.
        assert categorical_mask is None

    def test_prepare_episode_features_before_fit_raises(self):
        clf = build_tiny_classifier()
        with pytest.raises(RuntimeError, match="fitted"):
            clf.prepare_episode_features(_frame(n=4)[0])

    def test_predict_and_predict_proba_before_fit_raise(self):
        clf = build_tiny_classifier()
        X, _y = _frame(n=4)
        with pytest.raises(RuntimeError, match="must be fitted"):
            clf.predict(X)
        with pytest.raises(RuntimeError, match="must be fitted"):
            clf.predict_proba(X)

    def test_episode_logits_before_load_raises(self):
        clf = EXAONETabularClassifier(device="cpu")
        with pytest.raises(RuntimeError, match="not loaded"):
            clf.episode_logits(torch.zeros(1, 2, 2), torch.zeros(1, 2),
                               torch.zeros(1, 1, 2))

    def test_episode_logits_are_differentiable_and_head_wide(self):
        clf = build_tiny_classifier()
        X, y = _frame(n=24)
        clf.fit(X, y)
        features, _ = clf.prepare_episode_features(X)
        x_support = torch.from_numpy(features[:16]).unsqueeze(0).requires_grad_(True)
        y_support = torch.from_numpy(
            clf.y_encoder_.transform(y.values[:16]).astype(np.float32)
        ).unsqueeze(0)
        x_query = torch.from_numpy(features[16:]).unsqueeze(0)

        logits = clf.episode_logits(x_support, y_support, x_query)
        # (E, Q, class_capacity) -- the ARCHITECTURAL width, not n_classes_.
        assert logits.shape == (1, 8, clf.max_classes)
        logits.sum().backward()
        assert x_support.grad is not None
        assert x_support.grad.abs().sum().item() > 0

    def test_predict_returns_the_original_label_space(self):
        clf = build_tiny_classifier()
        X, y = _frame(n=24)
        clf.fit(X, y)
        predictions = clf.predict(X)
        assert set(np.unique(predictions)).issubset({"pos", "neg"})
        proba = clf.predict_proba(X)
        assert proba.shape == (24, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# Wrapper contract -- regression
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestRegressorContract:
    def test_construction_is_lazy(self):
        reg = EXAONETabularRegressorWrapper(device="cpu")
        assert reg.model_ is None and reg.estimator_ is None and reg.manifest_ is None
        assert list(reg.parameters()) == []

    def test_target_statistics_stay_at_identity(self):
        """The vendored engine un-scales internally; scaling here would double-apply."""
        reg = EXAONETabularRegressorWrapper(device="cpu")
        assert reg.y_mean_ == 0.0 and reg.y_std_ == 1.0

    @pytest.mark.parametrize("strategy", ["inference", "finetune"])
    def test_accepts_the_documented_strategies(self, strategy):
        wrapper = EXAONETabularRegressorWrapper(tuning_strategy=strategy)
        assert wrapper.tuning_strategy == strategy

    def test_rejects_peft(self):
        """PEFT is not offered for regression, matching the other regressors."""
        with pytest.raises(ValueError, match="inference"):
            EXAONETabularRegressorWrapper(tuning_strategy="peft")

    def test_rejects_an_unknown_tuning_strategy(self):
        with pytest.raises(ValueError):
            EXAONETabularRegressorWrapper(tuning_strategy="nope")

    def test_unknown_model_params_are_absorbed_not_raised(self):
        reg = EXAONETabularRegressorWrapper(
            device="cpu", task_type="regression", totally_unknown_key=1
        )
        assert reg._extra_kwargs == {"totally_unknown_key": 1}

    def test_median_quantile_index(self):
        # Without a manifest the released 999-level head's median is assumed.
        assert EXAONETabularRegressorWrapper().median_quantile_index == 499
        reg = build_tiny_regressor()
        assert reg.median_quantile_index == TINY_REGRESSION["quantile_count"] // 2 == 2

    def test_predict_before_fit_raises(self):
        reg = build_tiny_regressor()
        with pytest.raises(RuntimeError, match="must be fitted"):
            reg.predict(pd.DataFrame({"a": [1.0]}))

    def test_prepare_episode_features_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fitted"):
            build_tiny_regressor().prepare_episode_features(pd.DataFrame({"a": [1.0]}))

    def test_episode_predictions_before_load_raises(self):
        reg = EXAONETabularRegressorWrapper(device="cpu")
        with pytest.raises(RuntimeError, match="not loaded"):
            reg.episode_predictions(torch.zeros(1, 2, 2), torch.zeros(1, 2),
                                    torch.zeros(1, 1, 2))

    def test_episode_predictions_are_quantile_wide_and_differentiable(self):
        reg = build_tiny_regressor()
        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(24, 3), columns=["a", "b", "c"])
        y = pd.Series(rng.randn(24) * 10 + 100)
        reg.fit(X, y)
        features, _ = reg.prepare_episode_features(X)
        x_support = torch.from_numpy(features[:16]).unsqueeze(0).requires_grad_(True)
        y_support = torch.from_numpy(
            ((y.values[:16] - y.values.mean()) / y.values.std()).astype(np.float32)
        ).unsqueeze(0)
        x_query = torch.from_numpy(features[16:]).unsqueeze(0)

        quantiles = reg.episode_predictions(x_support, y_support, x_query)
        assert quantiles.shape == (1, 8, TINY_REGRESSION["quantile_count"])
        quantiles[..., reg.median_quantile_index].sum().backward()
        assert x_support.grad is not None and x_support.grad.abs().sum().item() > 0

    def test_missing_regression_weights_raise_a_helpful_file_not_found(self, monkeypatch):
        """LG AI Research published only the classification checkpoint."""
        monkeypatch.delenv(
            _bk.WEIGHTS_ENV_VARS["regression"], raising=False
        )
        with pytest.raises(FileNotFoundError) as excinfo:
            EXAONETabularRegressorWrapper(device="cpu")._load_model()
        message = str(excinfo.value)
        assert _bk.WEIGHTS_ENV_VARS["regression"] in message
        assert "checkpoint_path" in message
        # It must name a way forward, not just say no.
        assert any(name in message for name in ("TabICLv2", "Mitra", "LimiX", "TabPFNv26"))

    def test_classification_weights_are_resolved_without_the_regression_guard(
        self, monkeypatch, tmp_path
    ):
        """The guard is regression-only: a local classification path resolves fine."""
        path = tmp_path / "clf.safetensors"
        path.write_bytes(b"not really a checkpoint")
        source = _bk.resolve_checkpoint("classification", checkpoint_path=str(path))
        assert source.path == str(path)
        assert source.is_default is False

    def test_regression_env_var_satisfies_the_guard(self, monkeypatch, tmp_path):
        path = tmp_path / "reg.safetensors"
        path.write_bytes(b"not really a checkpoint")
        monkeypatch.setenv(_bk.WEIGHTS_ENV_VARS["regression"], str(path))
        source = _bk.resolve_checkpoint("regression")
        assert source.path == str(path)
        assert source.is_default is False


# --------------------------------------------------------------------------- #
# Backbone helpers
# --------------------------------------------------------------------------- #
@pytest.mark.unit
class TestBackboneHelpers:
    def test_cpu_manifests_are_forced_to_float32(self):
        """The released manifest asks for float16, which has no CPU kernel here."""
        manifest = _bk.build_manifest("classification", device="cpu")
        assert manifest.runtime.compute_dtype == "float32"

    def test_released_manifest_geometry_is_the_documented_one(self):
        from tabtune.models.exaone.presets import released_manifest

        classification = released_manifest("classification")
        assert classification.model.class_capacity == _bk.CLASS_CAPACITY == 10
        assert classification.runtime.support_row_limit == _bk.SUPPORT_ROW_LIMIT
        assert classification.runtime.feature_limit == _bk.FEATURE_LIMIT
        regression = released_manifest("regression")
        assert regression.regression.quantile_count == 999
        assert regression.output_width == 999

    def test_resolve_dtype_and_device(self):
        assert _bk.resolve_dtype("float32") is torch.float32
        assert _bk.resolve_dtype("bf16") is torch.bfloat16
        assert _bk.resolve_dtype(None) is None
        assert _bk.resolve_device("cpu") == "cpu"

    @pytest.mark.parametrize("requested", [None, "auto", "cpu", torch.device("cpu")])
    def test_every_device_spelling_reaches_torch(self, requested):
        """``torch.device`` must accept whatever the resolver returns.

        ``"auto"`` is the library-wide spelling for "pick a backend" and is the
        default of ``TuningConfig.device``, so the pipeline forwards it verbatim
        into ``model_params`` on the fine-tune path. A resolver that only
        special-cased ``None`` passed the literal string through and every
        EXAONE fine-tune that did not name a device crashed inside
        ``torch.device("auto")``.
        """
        resolved = _bk.resolve_device(requested)
        assert isinstance(resolved, str)
        torch.device(resolved)          # raises if the resolver leaked a sentinel

    @pytest.mark.parametrize("strategy", ["inference", "finetune", "peft"])
    def test_cpu_defers_to_the_manifest(self, strategy):
        """On CPU the manifest already forces float32; nothing to override."""
        assert EXAONETabularClassifier(
            tuning_strategy=strategy, device="cpu"
        )._resolved_dtype() is None

    @pytest.mark.parametrize(
        "bf16_supported,expected", [(True, "bfloat16"), (False, "float16")]
    )
    def test_cuda_finetuning_picks_a_half_precision(
        self, monkeypatch, bf16_supported, expected
    ):
        """CUDA fine-tuning must be half precision, and must never be float32.

        ``feature_context_attentions`` is pinned to the flash SDPA backend,
        which on CUDA accepts only ``{Half, BFloat16}`` -- float32 there is not
        slow, it raises ``RuntimeError: No available kernel``. Plain float16
        underflows on the backward pass and reaches NaN in a few steps, so
        bfloat16 is preferred wherever the card has it.

        Mocked rather than gated on a GPU, so this still runs on CI.
        """
        monkeypatch.setattr(_bk, "resolve_device", lambda device=None: "cuda")
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16_supported)
        for strategy in ("finetune", "peft"):
            resolved = EXAONETabularClassifier(
                tuning_strategy=strategy
            )._resolved_dtype()
            assert resolved == expected
            assert resolved != "float32", "float32 has no flash kernel on CUDA"

    def test_cuda_inference_keeps_the_released_precision(self, monkeypatch):
        """Inference is what the checkpoint was released for: manifest float16."""
        monkeypatch.setattr(_bk, "resolve_device", lambda device=None: "cuda")
        assert EXAONETabularClassifier(
            tuning_strategy="inference"
        )._resolved_dtype() is None

    def test_an_explicit_dtype_still_wins_for_finetuning(self, monkeypatch):
        """Anyone pairing fp16 with their own GradScaler must not be overridden."""
        monkeypatch.setattr(_bk, "resolve_device", lambda device=None: "cuda")
        assert EXAONETabularClassifier(
            tuning_strategy="finetune", dtype="float16"
        )._resolved_dtype() == "float16"
        assert EXAONETabularClassifier(
            tuning_strategy="finetune", compute_dtype="bfloat16"
        )._resolved_dtype() == "bfloat16"

    def test_device_resolution_is_the_shared_one(self):
        """EXAONE must not carry its own device rules; the library owns them."""
        from tabtune._internal.device import resolve_device as shared

        for requested in (None, "auto", "cpu", "cuda", "cuda:7"):
            assert _bk.resolve_device(requested) == shared(requested)

    def test_icl_logits_matches_a_direct_forward(self):
        model = build_tiny_backbone()
        x_support, y_support, x_query = _episode(dtype=torch.float32)
        direct = model(x_support, y_support, x_query, feedforward_token_chunk=FF_CHUNK)
        through_helper = _bk.icl_logits(model, x_support, y_support, x_query)
        assert torch.equal(direct, through_helper)


# --------------------------------------------------------------------------- #
# The released 20.8M-parameter geometry (random-init weights, no download)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.integration
class TestReleasedGeometry:
    def test_classifier_loads_and_predicts(self, released_classifier_checkpoint):
        clf = EXAONETabularClassifier(
            device="cpu", n_ensemble=2, random_state=0,
            checkpoint_path=released_classifier_checkpoint,
        )
        X, y = _frame(n=100, seed=1)
        clf.fit(X.iloc[:80], y.iloc[:80])
        assert clf.manifest_.model.class_capacity == 10
        assert clf.n_classes_ == 2
        # The head width is architectural; the dataset's class count is not.
        assert clf.max_classes == 10
        assert sum(p.numel() for p in clf.parameters()) > 20_000_000

        predictions = clf.predict(X.iloc[80:])
        assert predictions.shape == (20,)
        assert set(np.unique(predictions)).issubset({"pos", "neg"})
        proba = clf.predict_proba(X.iloc[80:])
        assert proba.shape == (20, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_regressor_loads_and_predicts_in_the_original_target_scale(
        self, released_regressor_checkpoint
    ):
        reg = EXAONETabularRegressorWrapper(
            device="cpu", n_ensemble=2, random_state=0,
            checkpoint_path=released_regressor_checkpoint,
        )
        rng = np.random.RandomState(0)
        X = pd.DataFrame({"f1": rng.randn(100), "f2": rng.randn(100)})
        y = pd.Series(30.0 * X["f1"] - 20.0 * X["f2"] + 100.0)
        reg.fit(X.iloc[:80], y.iloc[:80])
        assert reg.median_quantile_index == 499

        predictions = reg.predict(X.iloc[80:])
        assert predictions.shape == (20,)
        assert np.isfinite(predictions).all()
        # Random weights, so the values are meaningless -- but they must live in
        # y's space (mean ~100, sd ~36), not in a standardised one.
        assert abs(predictions.mean() - y.mean()) < 3 * y.std()
        assert abs(predictions.mean()) > 10.0
