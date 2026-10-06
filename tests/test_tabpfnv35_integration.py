"""TabPFN v3.5 wiring: registry, checkpoint sourcing, architecture, dispatch.

The vendored tree is upstream TabPFN v9.0.0, which is the first release to ship
the ``tabpfn_v3_5`` architecture and the first where one multitask checkpoint
carries both prediction heads. These tests pin both of those facts, since they
are what make v3.5 a separate tree rather than a new checkpoint for v3.
"""

from __future__ import annotations

import importlib

import pytest

from tabtune.registry import get_model_spec, list_model_names


class TestRegistry:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("TabPFNv35", "TabPFNv35"),
            ("TabPFN-3.5", "TabPFNv35"),
            ("tabpfn v3.5", "TabPFNv35"),
            ("TabPFN3.5", "TabPFNv35"),
            ("TabPFNv35Fast", "TabPFNv35Fast"),
            ("TabPFN-3.5-fast", "TabPFNv35Fast"),
        ],
    )
    def test_aliases_resolve(self, query: str, expected: str) -> None:
        assert get_model_spec(query).name == expected

    def test_older_versions_unaffected(self) -> None:
        for query, expected in [("TabPFNv3", "TabPFNv3"), ("TabPFN", "TabPFN"),
                                ("TabPFNv26", "TabPFNv26")]:
            assert get_model_spec(query).name == expected

    def test_both_variants_listed(self) -> None:
        assert {"TabPFNv35", "TabPFNv35Fast"} <= set(list_model_names())

    def test_unverified_limits_are_not_invented(self) -> None:
        envelope = get_model_spec("TabPFNv35").envelope
        assert envelope.max_classes is None
        assert envelope.max_rows is None
        assert envelope.max_cells is None
        assert envelope.native_text is True

    def test_license_is_non_commercial(self) -> None:
        # 0.4.0 checked the tabpfn 9.0.0 release: the v3.5 weights are non-commercial.
        assert get_model_spec("TabPFNv35").license.commercial_use_ok is False
        assert get_model_spec("TabPFNv35Fast").license.commercial_use_ok is False


class TestVendoredTree:
    @pytest.fixture(autouse=True)
    def _torch(self):
        pytest.importorskip("torch")

    def test_vendored_release_recorded(self) -> None:
        package = importlib.import_module("tabtune.models.tabpfnv35")
        assert package.VENDORED_FROM == "9.0.0"

    def test_v3_5_architecture_present(self) -> None:
        architectures = importlib.import_module(
            "tabtune.models.tabpfnv35.architectures"
        ).ARCHITECTURES
        assert "tabpfn_v3_5" in architectures
        # The v3 tree does not have it - that is why this is a separate vendor.
        v3 = importlib.import_module("tabtune.models.tabpfnv3.architectures").ARCHITECTURES
        assert "tabpfn_v3_5" not in v3

    def test_no_bare_upstream_imports_remain(self) -> None:
        import pathlib
        import re

        root = pathlib.Path(
            importlib.import_module("tabtune.models.tabpfnv35").__file__
        ).parent
        offenders = [
            f"{path.relative_to(root)}:{n}"
            for path in root.rglob("*.py")
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if re.match(r"^\s*(from|import)\s+tabpfn\b", line)
        ]
        assert not offenders, offenders

    def test_upstream_module_namespace_not_squatted(self) -> None:
        # Upstream registers sys.modules['tabpfn.model'] for tabpfn-extensions;
        # vendored, that would collide with a co-installed real tabpfn.
        import sys

        importlib.import_module("tabtune.models.tabpfnv35")
        assert "tabpfn.model" not in sys.modules


class TestCheckpointSourcing:
    @pytest.fixture(autouse=True)
    def _torch(self):
        pytest.importorskip("torch")

    def test_one_multitask_checkpoint_backs_both_heads(self) -> None:
        loading = importlib.import_module("tabtune.models.tabpfnv35.model_loading")
        constants = importlib.import_module("tabtune.models.tabpfnv35.constants")
        classifier = loading._get_model_source(constants.ModelVersion.V3_5, loading.ModelType.CLASSIFIER)
        regressor = loading._get_model_source(constants.ModelVersion.V3_5, loading.ModelType.REGRESSOR)
        assert classifier.default_filename == regressor.default_filename
        assert classifier.repo_id == "Prior-Labs/tabpfn_3_5"
        assert classifier.default_filename.endswith(".safetensors")

    def test_fast_is_a_distinct_checkpoint(self) -> None:
        loading = importlib.import_module("tabtune.models.tabpfnv35.model_loading")
        assert (
            loading.ModelSource.get_v3_5().default_filename
            != loading.ModelSource.get_v3_5_fast().default_filename
        )

    @pytest.mark.parametrize(
        ("filename", "attr"),
        [
            ("tabpfn-v3.5-fast-20260909.safetensors", "V3_5_FAST"),
            ("tabpfn-v3.5-20260909.safetensors", "V3_5"),
            ("tabpfn-v3-classifier-v3_default.ckpt", "V3"),
            ("tabpfn-v2.6-classifier-v2.6_default.ckpt", "V2_6"),
            ("tabpfn-v2.5-regressor-v2.5_default.ckpt", "V2_5"),
            ("tabpfn-v2-classifier.ckpt", "V2"),
        ],
    )
    def test_version_resolution_is_most_specific_first(self, filename: str, attr: str) -> None:
        # "v3.5-fast" contains "v3.5", which contains "v3"; a shortest-first
        # test order silently resolves the wrong version.
        loading = importlib.import_module("tabtune.models.tabpfnv35.model_loading")
        ModelVersion = importlib.import_module("tabtune.models.tabpfnv35.constants").ModelVersion
        assert loading.resolve_model_version(filename) is getattr(ModelVersion, attr)


class TestEstimators:
    @pytest.fixture(autouse=True)
    def _torch(self):
        pytest.importorskip("torch")

    def test_estimators_pin_their_checkpoints(self) -> None:
        package = importlib.import_module("tabtune.models.tabpfnv35")
        clf = package.TabPFNv35Classifier(device="cpu")
        fast = package.TabPFNv35FastClassifier(device="cpu")
        assert str(clf.model_path).endswith("tabpfn-v3.5-20260909.safetensors")
        assert str(fast.model_path).endswith("tabpfn-v3.5-fast-20260909.safetensors")

    def test_regression_wrappers_exist(self) -> None:
        module = importlib.import_module("tabtune.models.regression.tabpfnv35")
        assert module.TabPFNv35RegressorWrapper(device="cpu") is not None
        assert module.TabPFNv35FastRegressorWrapper(device="cpu") is not None

    def test_finetuning_resolves_the_version_itself(self) -> None:
        # Unlike v3, upstream v9 no longer hardcodes an old version in the
        # fine-tuner, so no pin module is needed for v3.5.
        base = importlib.import_module("tabtune.models.tabpfnv35.finetuning.finetuned_base")
        assert hasattr(base.FinetunedTabPFNBase, "finetune_model_version")


class TestDispatch:
    def test_pipeline_has_arms_for_both_variants(self) -> None:
        import inspect

        from tabtune.TabularPipeline import pipeline

        source = inspect.getsource(pipeline)
        for name in ("TabPFNv35", "TabPFNv35Fast"):
            assert f"self.model_name == '{name}'" in source

    def test_preprocessor_mapping(self) -> None:
        import inspect

        from tabtune.Dataprocess.data_processor import DataProcessor

        source = inspect.getsource(DataProcessor)
        assert "'TabPFNv35': {'categorical_encoding': 'tabpfn_special'}" in source
        assert "'TabPFNv35Fast': {'categorical_encoding': 'tabpfn_special'}" in source

    def test_lora_targets_registered(self) -> None:
        from tabtune.TuningManager.peft_utils import MODEL_LORA_TARGETS

        assert {"TabPFNv35", "TabPFNv35Fast"} <= set(MODEL_LORA_TARGETS)

    def test_lora_targets_match_the_v3_5_architecture(self) -> None:
        pytest.importorskip("torch")
        import inspect

        from tabtune.TuningManager.peft_utils import MODEL_LORA_TARGETS

        architecture = importlib.import_module(
            "tabtune.models.tabpfnv35.architectures.tabpfn_v3_5"
        )
        source = inspect.getsource(architecture)
        for target in ("q_projection", "k_projection", "v_projection", "out_projection"):
            assert target in MODEL_LORA_TARGETS["TabPFNv35"].target_substrings
            assert f"self.{target} = nn.Linear" in source

    def test_tuning_manager_shares_the_v3_loops(self) -> None:
        import inspect

        from tabtune.TuningManager.tuning import TuningManager

        source = inspect.getsource(TuningManager)
        # One set of episodic loops serves both trees; only the import target
        # and the LoRA key differ.
        assert "_tabpfn_data_util" in source
        assert "tree='tabpfnv35'" in source
        assert "_finetune_tabpfnv35_native_classifier" in source
