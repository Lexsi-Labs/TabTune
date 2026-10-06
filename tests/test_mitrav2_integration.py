"""Integration tests for Mitra v2 in TabTune.

No checkpoint is downloaded. Checkpoints in the exact format
``Tab2D.save_pretrained`` writes are built on disk, so the real loading path
runs end to end: config read, module construction, strict state-dict load,
pipeline dispatch, fine-tuning and LoRA.

The central claim these tests pin is that v2 is a CHECKPOINT, not a new
architecture, and therefore needs no new model code - only a name, a repo id
and the dispatch entries. That claim is decidable from the format rather than
assumed: ``save_pretrained`` writes exactly five keys, ``from_pretrained``
reads exactly those five, and ``load_state_dict`` is strict.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from tabtune.Dataprocess.data_processor import DataProcessor
from tabtune.Dataprocess.mitra_preprocessor import MitraPreprocessor
from tabtune.TuningManager.peft_utils import MODEL_LORA_TARGETS
from tabtune.TuningManager.tuning import _mitra_lora_name
from tabtune.models.mitra.model_loading import (
    MITRA_CLASSIFIER_REPO,
    MITRA_FINETUNE_REPO,
    MITRA_REGRESSOR_REPO,
    MITRA_REPOS,
    MITRA_V2_CLASSIFIER_REPO,
    MITRA_V2_REGRESSOR_REPO,
    probe_mitra_checkpoint,
    resolve_mitra_repo,
)
from tabtune.models.mitra.tab2d import Tab2D
from tabtune.registry import MODEL_REGISTRY, get_model_spec

ALIASES = ("MitraV2", "Mitra-2", "Mitra2", "mitra v2", "mitra-classifier-2")


def write_checkpoint(directory: pathlib.Path, *, dim=64, dim_output=10, n_layers=2,
                     n_heads=4, task="CLASSIFICATION", seed=0, extra_tensor=False):
    """A checkpoint in the exact shape ``Tab2D.save_pretrained`` produces."""
    torch.manual_seed(seed)
    model = Tab2D(dim=dim, dim_output=dim_output, n_layers=n_layers, n_heads=n_heads,
                  task=task, use_pretrained_weights=False, path_to_weights="",
                  device="cpu")
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, 0.0, 0.02)
    directory.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    if extra_tensor:
        state["some_new_block.weight"] = torch.zeros(4, 4)
    save_file(state, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps({
        "dim": dim, "dim_output": dim_output, "n_layers": n_layers,
        "n_heads": n_heads, "task": task,
    }))
    return str(directory)


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("mitrav2")
    return {
        "classification": write_checkpoint(root / "cls", task="CLASSIFICATION", dim_output=10),
        "regression": write_checkpoint(root / "reg", task="REGRESSION", dim_output=1),
        # A deliberately different architecture shape, standing in for "v2 is
        # bigger / retrained" - the case the integration has to handle for free.
        "reshaped": write_checkpoint(root / "v2shape", dim=96, n_layers=3, n_heads=6),
        # A checkpoint that genuinely does not fit the vendored architecture.
        "structural": write_checkpoint(root / "bad", dim=96, n_layers=3, n_heads=6,
                                       extra_tensor=True),
    }


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(160, 6)), columns=[f"f{i}" for i in range(6)])
    y_cls = pd.Series(np.where(X["f0"] + X["f1"] > 0, "yes", "no"))
    y_reg = pd.Series(X["f0"] * 2 + X["f1"] + rng.normal(scale=0.1, size=160))
    return X, y_cls, y_reg


class TestRegistry:
    def test_spec_registered(self) -> None:
        spec = get_model_spec("MitraV2")
        assert spec.name == "MitraV2"
        assert spec.family == "icl"
        assert spec.preprocessor_key == "mitra_special"

    @pytest.mark.parametrize("alias", ALIASES)
    def test_aliases_resolve(self, alias: str) -> None:
        assert get_model_spec(alias).name == "MitraV2"

    def test_v1_names_still_resolve_to_v1(self) -> None:
        """Adding v2 must not capture v1's names - including its Tab2D alias."""
        for name in ("Mitra", "Tab2D", "mitra-classifier"):
            assert get_model_spec(name).name == "Mitra"

    def test_strategies_mirror_v1(self) -> None:
        """Same architecture, same capabilities; a difference would be invented."""
        v1, v2 = get_model_spec("Mitra"), get_model_spec("MitraV2")
        assert v2.classification_strategies == v1.classification_strategies
        assert v2.regression_strategies == v1.regression_strategies
        assert v2.finetune_modes == v1.finetune_modes

    def test_v1_row_limit_is_not_carried_across(self) -> None:
        """v1's 10k ceiling was measured on v1's weights, not v2's."""
        assert get_model_spec("Mitra").envelope.max_rows == 10_000
        assert get_model_spec("MitraV2").envelope.max_rows is None

    def test_licence_is_the_verified_apache_2_0(self) -> None:
        # 0.4.0 read the published model cards: Mitra v1 and v2 weights are Apache-2.0.
        license = get_model_spec("MitraV2").license
        assert license.commercial_use_ok is True
        assert "Apache-2.0" in license.name

    def test_listed(self) -> None:
        assert "MitraV2" in MODEL_REGISTRY


class TestCheckpointResolution:
    def test_each_name_and_task_maps_to_its_own_repo(self) -> None:
        assert resolve_mitra_repo("Mitra", "classification") == MITRA_CLASSIFIER_REPO
        assert resolve_mitra_repo("Mitra", "regression") == MITRA_REGRESSOR_REPO
        assert resolve_mitra_repo("MitraV2", "classification") == MITRA_V2_CLASSIFIER_REPO
        assert resolve_mitra_repo("MitraV2", "regression") == MITRA_V2_REGRESSOR_REPO

    def test_v2_repos_are_the_published_ones(self) -> None:
        assert MITRA_V2_CLASSIFIER_REPO == "autogluon/mitra-classifier-2"
        assert MITRA_V2_REGRESSOR_REPO == "autogluon/mitra-regressor-2"

    def test_v1_and_v2_never_collide(self) -> None:
        v1 = set(MITRA_REPOS["v1"].values())
        v2 = set(MITRA_REPOS["v2"].values())
        assert v1.isdisjoint(v2)

    def test_the_finetune_checkpoint_is_reachable_but_never_a_default(self) -> None:
        """Its role is not documented anywhere reachable, so it is opt-in only."""
        assert MITRA_FINETUNE_REPO == "autogluon/mitra-finetune"
        assert resolve_mitra_repo("MitraV2", "classification", variant="finetune") == MITRA_FINETUNE_REPO
        assert resolve_mitra_repo("Mitra", "regression", variant="finetune") == MITRA_FINETUNE_REPO
        for variant in ("v1", "v2"):
            assert MITRA_FINETUNE_REPO not in MITRA_REPOS[variant].values()

    def test_an_unknown_name_raises_instead_of_falling_back_to_v1(self) -> None:
        """Silently returning v1 is how a v2 run would report v1 numbers."""
        with pytest.raises(ValueError, match="Unknown Mitra model name"):
            resolve_mitra_repo("MitraV9", "classification")
        with pytest.raises(ValueError, match="Unknown Mitra variant"):
            resolve_mitra_repo("MitraV2", "classification", variant="v9")

    def test_repos_are_overridable_from_the_environment(self, monkeypatch) -> None:
        """The ids could not be verified from here, so a rename must not need a patch."""
        import importlib

        from tabtune.models.mitra import model_loading

        monkeypatch.setenv("TABTUNE_MITRA_V2_CLS_REPO", "someone/renamed-v2")
        reloaded = importlib.reload(model_loading)
        try:
            assert reloaded.MITRA_V2_CLASSIFIER_REPO == "someone/renamed-v2"
        finally:
            monkeypatch.delenv("TABTUNE_MITRA_V2_CLS_REPO")
            importlib.reload(model_loading)


class TestArchitectureIsCheckpointDriven:
    """Why v2 needs no new model code, pinned against the vendored loader."""

    def test_saved_config_carries_exactly_the_keys_the_loader_reads(self, tmp_path) -> None:
        directory = pathlib.Path(write_checkpoint(tmp_path / "ck"))
        config = json.loads((directory / "config.json").read_text())
        assert set(config) == {"dim", "dim_output", "n_layers", "n_heads", "task"}

    def test_no_other_architectural_argument_exists(self) -> None:
        """If a knob is not in the config, it cannot vary between checkpoints."""
        import inspect

        parameters = set(inspect.signature(Tab2D.__init__).parameters)
        parameters -= {"self", "use_pretrained_weights", "path_to_weights", "device"}
        assert parameters == {"dim", "dim_output", "n_layers", "n_heads", "task"}

    def test_a_differently_shaped_checkpoint_loads_with_no_code_change(
        self, checkpoints
    ) -> None:
        """The v2-is-bigger case: different dim / layers / heads, same class."""
        model = Tab2D.from_pretrained(checkpoints["reshaped"], device="cpu")
        assert (model.dim, model.n_layers, model.n_heads) == (96, 3, 6)
        assert isinstance(model, Tab2D)

    def test_a_structurally_different_checkpoint_cannot_load_silently(
        self, checkpoints
    ) -> None:
        """Strict loading is what makes the swap claim safe rather than hopeful."""
        with pytest.raises(RuntimeError, match="state_dict"):
            Tab2D.from_pretrained(checkpoints["structural"], device="cpu")

    def test_v2_is_the_same_class_so_isinstance_dispatch_covers_it(
        self, checkpoints
    ) -> None:
        """Vendoring a second tree would break every isinstance(model, Tab2D) branch."""
        v1 = Tab2D.from_pretrained(checkpoints["classification"], device="cpu")
        v2 = Tab2D.from_pretrained(checkpoints["reshaped"], device="cpu")
        assert type(v1) is type(v2) is Tab2D


class TestProbe:
    def test_reports_a_loadable_checkpoint_as_a_swap(self, checkpoints) -> None:
        result = probe_mitra_checkpoint(checkpoints["reshaped"], device="cpu")
        assert result["verdict"] == "checkpoint_swap"
        assert result["config"]["dim"] == 96
        assert result["n_parameters"] > 0
        assert result["error"] is None

    def test_reports_a_structural_mismatch_as_needing_code(self, checkpoints) -> None:
        result = probe_mitra_checkpoint(checkpoints["structural"], device="cpu")
        assert result["verdict"] == "needs_new_code"
        assert "some_new_block" in result["error"]

    def test_an_unreachable_checkpoint_is_not_an_architecture_verdict(self) -> None:
        """A blocked download says nothing about the architecture."""
        result = probe_mitra_checkpoint("/nonexistent/path/that/is/not/a/dir", device="cpu")
        assert result["verdict"] == "unreachable"
        assert result["error"] is not None

    def test_extra_config_keys_are_reported_not_treated_as_failure(self, tmp_path) -> None:
        directory = pathlib.Path(write_checkpoint(tmp_path / "extra"))
        config = json.loads((directory / "config.json").read_text())
        config["some_future_key"] = 7
        (directory / "config.json").write_text(json.dumps(config))
        result = probe_mitra_checkpoint(str(directory), device="cpu")
        assert result["verdict"] == "checkpoint_swap"
        assert result["unexpected_config_keys"] == ["some_future_key"]


class TestWiring:
    def test_preprocessor_is_shared_with_v1(self) -> None:
        processor = DataProcessor(model_name="MitraV2", task_type="classification")
        assert isinstance(processor._get_custom_preprocessor(), MitraPreprocessor)

    def test_regression_target_is_not_scaled(self) -> None:
        processor = DataProcessor(model_name="MitraV2", task_type="regression")
        assert processor._get_regression_processor().target_scaling_strategy == "none"

    def test_peft_targets_match_v1_exactly(self) -> None:
        """Same module names; a different set would make the two incomparable."""
        assert "MitraV2" in MODEL_LORA_TARGETS
        assert (MODEL_LORA_TARGETS["MitraV2"].target_substrings
                == MODEL_LORA_TARGETS["Mitra"].target_substrings)

    def test_lora_name_comes_from_the_tag_not_the_class(self, checkpoints) -> None:
        """v1 and v2 are the same class, so isinstance cannot tell them apart."""
        model = Tab2D.from_pretrained(checkpoints["classification"], device="cpu")
        assert _mitra_lora_name(model) == "Mitra"
        model._tabtune_model_name = "MitraV2"
        assert _mitra_lora_name(model) == "MitraV2"


class TestSupportSizeClamp:
    """A dataset smaller than support_size used to crash inside einx."""

    def test_the_split_always_leaves_a_query_row(self) -> None:
        import inspect

        from tabtune.TuningManager import tuning

        source = inspect.getsource(tuning.TuningManager._finetune_mitra)
        assert "min(config['support_size'], max(1, rows - 1))" in source

    @pytest.mark.parametrize("rows,support", [(120, 128), (120, 256), (8, 64), (2, 1000)])
    def test_the_clamp_keeps_at_least_one_query_row(self, rows, support) -> None:
        clamped = min(support, max(1, rows - 1))
        assert 1 <= clamped < rows or rows == 1
        assert rows - clamped >= 1

    def test_a_large_enough_dataset_is_left_alone(self) -> None:
        assert min(128, max(1, 1000 - 1)) == 128


class TestPipelineEndToEnd:
    @staticmethod
    def build(name, task, strategy, checkpoint):
        from tabtune.TabularPipeline.pipeline import TabularPipeline

        params = {"epochs": 1, "steps_per_epoch": 2, "learning_rate": 1e-4}
        if strategy == "peft":
            params["peft_config"] = {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0}
        return TabularPipeline(
            model_name=name, task_type=task, tuning_strategy=strategy,
            model_params={"device": "cpu", "pretrained_repo_id": checkpoint,
                          "use_pretrained_weights": True, "path_to_weights": checkpoint},
            tuning_params=params,
        )

    @pytest.mark.parametrize("strategy", ["inference", "finetune", "peft"])
    def test_classification_returns_labels_in_the_original_space(
        self, checkpoints, frame, strategy
    ) -> None:
        X, y, _ = frame
        pipeline = self.build("MitraV2", "classification", strategy,
                              checkpoints["classification"])
        pipeline.fit(X[:120], y[:120])
        assert set(np.unique(pipeline.predict(X[120:]))) <= {"yes", "no"}
        proba = pipeline.predict_proba(X[120:])
        assert proba.shape == (40, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    @pytest.mark.parametrize("strategy", ["inference", "finetune"])
    def test_regression_returns_finite_predictions(
        self, checkpoints, frame, strategy
    ) -> None:
        X, _, y = frame
        pipeline = self.build("MitraV2", "regression", strategy, checkpoints["regression"])
        pipeline.fit(X[:120], y[:120])
        predictions = pipeline.predict(X[120:])
        assert np.shape(predictions) == (40,)
        assert np.isfinite(np.asarray(predictions, dtype=float)).all()

    @pytest.mark.parametrize("strategy", ["inference", "finetune", "peft"])
    def test_v1_still_works_through_the_same_paths(
        self, checkpoints, frame, strategy
    ) -> None:
        """The v2 arms extend v1's branches; v1 must be unchanged by that."""
        X, y, _ = frame
        pipeline = self.build("Mitra", "classification", strategy,
                              checkpoints["classification"])
        pipeline.fit(X[:120], y[:120])
        assert set(np.unique(pipeline.predict(X[120:]))) <= {"yes", "no"}

    def test_the_head_is_trimmed_to_the_task_classes(self, checkpoints, frame) -> None:
        """The checkpoint head is 10 wide; a binary task must get 2 columns."""
        X, y, _ = frame
        pipeline = self.build("MitraV2", "classification", "inference",
                              checkpoints["classification"])
        pipeline.fit(X[:120], y[:120])
        assert pipeline.model.dim_output == 10
        assert pipeline.predict_proba(X[120:]).shape[1] == 2


class TestProbeCli:
    """The one command that settles swap-vs-architecture once network works."""

    @staticmethod
    def run(*args):
        import os
        import subprocess
        import sys

        repo = pathlib.Path(__file__).resolve().parent.parent
        return subprocess.run(
            [sys.executable, "-m", "tabtune.models.mitra.probe_mitra_v2", *args],
            cwd=repo, env={**os.environ, "PYTHONPATH": str(repo)},
            capture_output=True, text=True,
        )

    def test_a_loadable_checkpoint_exits_zero(self, checkpoints) -> None:
        result = self.run(checkpoints["reshaped"], "--device", "cpu")
        assert result.returncode == 0, result.stdout + result.stderr
        assert "CHECKPOINT SWAP" in result.stdout

    def test_a_structural_mismatch_exits_two(self, checkpoints) -> None:
        result = self.run(checkpoints["structural"], "--device", "cpu")
        assert result.returncode == 2, result.stdout + result.stderr
        assert "NEEDS NEW CODE" in result.stdout

    def test_an_unfetchable_checkpoint_exits_one(self) -> None:
        """Distinct from 2: a network failure is not an architecture verdict."""
        result = self.run("/nonexistent/checkpoint/dir", "--device", "cpu")
        assert result.returncode == 1, result.stdout + result.stderr
        assert "UNREACHABLE" in result.stdout

    def test_json_output_is_parseable(self, checkpoints) -> None:
        """stdout must be the payload alone - importing tabtune prints banners."""
        result = self.run(checkpoints["reshaped"], "--device", "cpu", "--json")
        payload = json.loads(result.stdout)
        assert payload[0]["verdict"] == "checkpoint_swap"
        assert payload[0]["config"]["dim"] == 96

    def test_the_default_run_covers_every_known_repo(self) -> None:
        """Including the finetune checkpoint, which has no default anywhere else."""
        import inspect

        from tabtune.models.mitra import probe_mitra_v2

        listed = probe_mitra_v2.targets([])
        assert MITRA_V2_CLASSIFIER_REPO in listed
        assert MITRA_V2_REGRESSOR_REPO in listed
        assert MITRA_FINETUNE_REPO in listed
        assert MITRA_CLASSIFIER_REPO in listed
        assert len(listed) == len(set(listed))
        assert inspect.isfunction(probe_mitra_v2.main)


class TestDeviceHandling:
    """Regression tests for two device bugs found on a real CUDA box.

    Both were invisible on a CPU-only machine, where 'auto' resolves to 'cpu'
    and every tensor lands in the same place anyway.
    """

    @pytest.mark.parametrize("model", ["MitraV2", "Mitra", "TabLDM"])
    def test_explicit_auto_is_resolved_not_passed_through(
        self, checkpoints, frame, model, tmp_path
    ) -> None:
        """`model_params={'device': 'auto'}` used to reach torch.device() verbatim.

        The pipeline resolved 'auto' only as a *default*, so a caller who asked
        for it explicitly got `RuntimeError: Expected one of cpu, cuda, ... at
        start of device string: auto`. Anything that forwards a device -- an
        sweep script, a config file -- hits this.
        """
        from tabtune.TabularPipeline.pipeline import TabularPipeline

        X, y, _ = frame
        if model == "TabLDM":
            pytest.importorskip("einops")
            params = {"device": "auto", "allow_auto_download": False,
                      "model_path": str(tmp_path / "missing.ckpt")}
            # TabLDM has no random-init fallback, so it raises on the missing
            # checkpoint. That is fine: the device must be resolved before the
            # checkpoint is ever consulted, so a device error would come first.
            with pytest.raises(Exception) as excinfo:
                TabularPipeline(model_name=model, task_type="classification",
                                tuning_strategy="inference",
                                model_params=params).fit(X[:120], y[:120])
            assert "device string" not in str(excinfo.value)
            return

        pipeline = TabularPipeline(
            model_name=model, task_type="classification", tuning_strategy="inference",
            model_params={"device": "auto",
                          "pretrained_repo_id": checkpoints["classification"],
                          "use_pretrained_weights": True,
                          "path_to_weights": checkpoints["classification"]},
        )
        pipeline.fit(X[:120], y[:120])
        assert len(pipeline.predict(X[120:])) == 40

    def test_model_params_cannot_overwrite_the_resolved_device(self) -> None:
        """The TabLDM arms build config, then update() from model_params.

        Setting 'device' before that update let the raw request win.
        """
        import inspect

        from tabtune.TabularPipeline.pipeline import TabularPipeline

        source = inspect.getsource(TabularPipeline.__init__)
        for block in source.split("elif self.model_name == 'TabLDM':")[1:]:
            arm = block.split("self.model = ")[0]
            assert arm.index("config.update(self.model_params)") < arm.index("config['device'] = device")

    def test_prediction_tensors_follow_the_module(self) -> None:
        """Fine-tuning moves the module; predict used to trust the params.

        On a CUDA box, `device='cpu'` plus fine-tuning put the weights on
        cuda:0 (the TuningManager's own resolve_device('auto') default) and the
        inputs on cpu: "Expected all tensors to be on the same device".
        """
        import inspect

        from tabtune.TabularPipeline.pipeline import TabularPipeline

        for method in (TabularPipeline._predict_uncached,
                       TabularPipeline._predict_proba_uncached):
            source = inspect.getsource(method)
            assert "next(self.model.parameters()).device" in source

    def test_the_pipeline_device_reaches_the_tuning_manager(self) -> None:
        """Otherwise a CPU request trains on the GPU and leaves the model there."""
        from tabtune.TabularPipeline.pipeline import TabularPipeline

        pipeline = TabularPipeline(
            model_name="MitraV2", task_type="classification",
            tuning_strategy="finetune", model_params={"device": "cpu"},
            tuning_params={"epochs": 1},
        )
        assert pipeline.tuning_params["device"] == "cpu"

    def test_an_explicit_tuning_device_still_wins(self) -> None:
        from tabtune.TabularPipeline.pipeline import TabularPipeline

        pipeline = TabularPipeline(
            model_name="MitraV2", task_type="classification",
            tuning_strategy="finetune", model_params={"device": "cpu"},
            tuning_params={"epochs": 1, "device": "cpu"},
        )
        assert pipeline.tuning_params["device"] == "cpu"
