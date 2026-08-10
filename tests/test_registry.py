"""Tests for the model registry: resolution, discovery, envelopes and licenses.

These tests are deliberately dependency-free - no torch, no weights, no
network - so they run in under a second and gate every pull request.
"""

from __future__ import annotations

import warnings

import pytest

from tabtune.registry import (
    MODEL_REGISTRY,
    CapabilityEnvelope,
    EnvelopeError,
    LicenseError,
    LicenseSpec,
    ModelNotFoundError,
    ModelSpec,
    UnsupportedStrategyError,
    UnsupportedTaskError,
    check_envelope,
    check_license,
    get_model_spec,
    infer_data_shape,
    list_model_names,
    list_models,
    models_dataframe,
    normalise_name,
    register_model,
    resolve_model_name,
    validate_request,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------- catalog


def test_registry_is_populated():
    assert len(MODEL_REGISTRY) == 16
    for expected in (
        "TabPFN", "TabPFNv3", "TabICLv2", "Mitra", "TabFM", "OrionMSP", "XRFM", "ILTM",
        "EXAONETabular",
    ):
        assert expected in MODEL_REGISTRY


def test_every_spec_declares_at_least_one_task():
    for spec in MODEL_REGISTRY.values():
        assert spec.tasks, f"{spec.name} declares no supported task"


def test_every_spec_has_license_and_paper():
    for spec in MODEL_REGISTRY.values():
        assert spec.license.name, f"{spec.name} has no license name"
        assert spec.paper, f"{spec.name} has no paper URL"


def test_inference_is_always_supported():
    """Zero-shot inference is the one guarantee TabTune makes for every model."""
    for spec in MODEL_REGISTRY.values():
        if spec.supports_classification:
            assert "inference" in spec.classification_strategies
        if spec.supports_regression:
            assert "inference" in spec.regression_strategies


# ------------------------------------------------------------------ resolution


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("TabPFNv26", "TabPFNv26"),
        ("tabpfnv26", "TabPFNv26"),
        ("TabPFN-v2.6", "TabPFNv26"),
        ("TabPFN v26", "TabPFNv26"),
        ("TABPFN_V26", "TabPFNv26"),
        ("SAP-RPT-1-OSS", "ContextTab"),
        ("ConTextTab", "ContextTab"),
        ("Orion-MSP", "OrionMSP"),
        ("OrionMSPv15", "OrionMSPv1.5"),
        ("LimiX", "Limix"),
        ("Tab2D", "Mitra"),
        ("xRFM", "XRFM"),
        ("x-RFM", "XRFM"),
        ("RFM", "XRFM"),
        ("iLTM", "ILTM"),
        ("i-LTM", "ILTM"),
        # Only "EXAONE" is a declared alias; the punctuated spellings resolve
        # because normalise_name() strips '-', '_' and spaces before lookup.
        ("EXAONE", "EXAONETabular"),
        ("exaone", "EXAONETabular"),
        ("exaone-tabular", "EXAONETabular"),
        ("EXAONE_Tabular", "EXAONETabular"),
    ],
)
def test_alias_resolution(alias, expected):
    assert resolve_model_name(alias) == expected


def test_normalise_name_strips_punctuation():
    assert normalise_name("TabPFN-v2.6") == normalise_name("tabpfn v26") == "tabpfnv26"


def test_unknown_model_suggests_a_close_match():
    with pytest.raises(ModelNotFoundError) as excinfo:
        resolve_model_name("TabPFNv4")
    message = str(excinfo.value)
    assert "TabPFNv3" in message
    assert "Available models" in message


def test_normalise_name_rejects_non_strings():
    with pytest.raises(TypeError):
        normalise_name(42)


# ------------------------------------------------------------------- discovery


def test_list_models_by_task():
    regression = {s.name for s in list_models(task="regression")}
    assert "TabICLv2" in regression
    # OrionMSP is classification-only; the README's regression example using it
    # would have raised at construction time.
    assert "OrionMSP" not in regression


def test_list_models_commercial_filter_is_conservative():
    """Unverified licenses are excluded from a commercial-only listing."""
    commercial = {s.name for s in list_models(commercial_ok=True)}
    assert "Mitra" in commercial and "OrionMSP" in commercial
    # Both models added in 0.2.0 are permissively licensed: xRFM is MIT and
    # trains from scratch, iLTM is Apache-2.0 with ungated weights.
    assert "XRFM" in commercial and "ILTM" in commercial
    assert "TabPFNv3" not in commercial  # explicitly research-only
    assert "TabPFN" not in commercial  # unverified, so excluded by default
    # EXAONE's *code* is BSD-3-Clause-LG AI Research, but its weights are
    # research-only, and the registry records the weight licence.
    assert "EXAONETabular" not in commercial

    lenient = {s.name for s in list_models(commercial_ok=True, include_unverified_licenses=True)}
    assert "TabPFN" in lenient


def test_list_models_by_strategy():
    peft = {s.name for s in list_models(task="classification", strategy="peft")}
    assert "TabICL" in peft
    assert "Limix" not in peft  # inference-only


def test_models_dataframe_shape():
    frame = models_dataframe()
    assert len(frame) == len(MODEL_REGISTRY)
    for column in ("Model", "License", "Commercial", "Max classes"):
        assert column in frame.columns


def test_list_model_names_is_sorted():
    names = list_model_names()
    assert names == sorted(names)


# ------------------------------------------------------------------ validation


def test_validate_request_returns_spec():
    spec = validate_request("TabICLv2", "classification", "inference")
    assert spec.name == "TabICLv2"


def test_validate_request_rejects_unsupported_task():
    with pytest.raises(UnsupportedTaskError) as excinfo:
        validate_request("OrionMSP", "regression", "inference")
    assert "classification" in str(excinfo.value)


def test_validate_request_rejects_unsupported_strategy_and_names_alternatives():
    with pytest.raises(UnsupportedStrategyError) as excinfo:
        validate_request("Limix", "classification", "finetune")
    message = str(excinfo.value)
    assert "Supported strategies" in message
    assert "TabICL" in message  # a model that does support finetuning


def test_validate_request_warns_on_experimental_strategy():
    with pytest.warns(UserWarning, match="experimental"):
        validate_request("TabPFN", "classification", "peft")


def test_validate_request_warns_on_unknown_finetune_mode():
    with pytest.warns(UserWarning, match="does not implement finetune_mode"):
        validate_request("TabICL", "classification", "finetune", "native")


def test_validate_request_can_be_strict_about_finetune_mode():
    with pytest.raises(UnsupportedStrategyError):
        validate_request(
            "TabICL", "classification", "finetune", "native", strict_finetune_mode=True
        )


def test_validate_request_rejects_bad_task_type():
    with pytest.raises(ValueError, match="task_type"):
        validate_request("TabICL", "clustering", "inference")


# ------------------------------------------------------------------- envelopes


def test_hard_class_limit_raises_even_in_warn_mode():
    """TabFM's ten-class head is architectural: no amount of RAM fixes it."""
    with pytest.raises(EnvelopeError) as excinfo:
        check_envelope("TabFM", n_rows=1000, n_features=20, n_classes=14, mode="warn")
    message = str(excinfo.value)
    assert "at most 10 classes" in message
    assert "Compatible models" in message


def test_soft_row_limit_only_warns():
    with pytest.warns(UserWarning, match="rows"):
        violations = check_envelope("Mitra", n_rows=50_000, n_features=10, mode="warn")
    assert any(v.constraint == "max_rows" for v in violations)
    assert all(v.severity == "warn" for v in violations)


def test_error_mode_escalates_soft_limits():
    with pytest.raises(EnvelopeError):
        check_envelope("Mitra", n_rows=50_000, n_features=10, mode="error")


def test_ignore_mode_disables_all_checking():
    assert check_envelope("TabFM", n_classes=999, mode="ignore") == []


def test_envelope_accepts_data_within_limits():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_envelope("TabFM", n_rows=500, n_features=20, n_classes=3) == []


def test_cell_budget_is_checked_independently_of_row_and_feature_caps():
    """TabPFN-3 advertises a budget, not a row cap: 1k x 20k fits, 1M x 20k does not."""
    envelope = CapabilityEnvelope(max_rows=1_000_000, max_features=20_000, max_cells=200_000_000)
    assert envelope.check(n_rows=1_000, n_features=20_000) == []
    violations = envelope.check(n_rows=1_000_000, n_features=20_000)
    assert any(v.constraint == "max_cells" for v in violations)


def test_min_rows_is_a_hard_constraint():
    violations = CapabilityEnvelope(min_rows=10).check(n_rows=3)
    assert violations[0].constraint == "min_rows"
    assert violations[0].severity == "error"


def test_envelope_violations_sort_errors_first():
    envelope = CapabilityEnvelope(max_classes=2, max_rows=10)
    violations = envelope.check(n_rows=1000, n_features=5, n_classes=5)
    assert violations[0].severity == "error"


def test_envelope_describe_is_human_readable():
    text = CapabilityEnvelope(max_classes=10, native_nan=True).describe()
    assert "10 classes" in text and "native NaN" in text


def test_bad_envelope_mode_is_rejected():
    with pytest.raises(ValueError, match="envelope mode"):
        check_envelope("TabFM", n_rows=10, mode="explode")


# -------------------------------------------------------------------- licenses


def test_research_mode_never_blocks():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_license("TabPFNv3", "research").commercial_use_ok is False


def test_commercial_mode_blocks_restricted_weights_with_alternatives():
    with pytest.raises(LicenseError) as excinfo:
        check_license("TabPFNv3", "commercial")
    message = str(excinfo.value)
    assert "does not permit commercial use" in message
    assert "Mitra" in message  # a suggested alternative
    assert "TabDistiller" in message  # the distillation escape hatch
    assert "not legal advice" in message


def test_commercial_mode_allows_permissive_weights():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_license("Mitra", "commercial").commercial_use_ok is True


def test_commercial_mode_warns_on_unverified_weights():
    """TabTune must not invent a restriction it has not verified."""
    with pytest.warns(UserWarning, match="has not verified"):
        check_license("TabPFN", "commercial")


def test_license_badge_reflects_attribution():
    assert LicenseSpec("X", True).badge == "yes"
    assert LicenseSpec("X", True, requires_attribution=True).badge == "yes (attribution)"
    assert LicenseSpec("X", False).badge == "no"
    assert LicenseSpec("X", None).badge == "unverified"


def test_bad_license_mode_is_rejected():
    with pytest.raises(ValueError, match="license mode"):
        check_license("Mitra", "whatever")


# ------------------------------------------------------------ data-shape probe


def test_infer_data_shape_from_dataframe():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    shape = infer_data_shape(X, y, "classification")
    assert shape == {"n_rows": 3, "n_features": 2, "n_classes": 2}


def test_infer_data_shape_skips_classes_for_regression():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"a": [1.0, 2.0]})
    shape = infer_data_shape(X, pd.Series([1.5, 2.5]), "regression")
    assert shape["n_classes"] is None


def test_infer_data_shape_tolerates_exotic_inputs():
    shape = infer_data_shape(object(), None, "classification")
    assert shape["n_rows"] is None and shape["n_features"] is None


# ------------------------------------------------------------- extensibility


def test_third_party_model_can_be_registered():
    spec = ModelSpec(
        name="_TestModel",
        family="test",
        aliases=("_tm",),
        classification_strategies=frozenset({"inference"}),
        envelope=CapabilityEnvelope(max_classes=3),
        license=LicenseSpec("MIT", True),
        paper="https://example.invalid",
    )
    try:
        register_model(spec)
        assert resolve_model_name("_tm") == "_TestModel"
        assert get_model_spec("_TestModel").family == "test"
        with pytest.raises(EnvelopeError):
            check_envelope("_TestModel", n_classes=9)
    finally:
        MODEL_REGISTRY.pop("_TestModel", None)
        from tabtune.registry.registry import _ALIASES

        _ALIASES.pop("testmodel", None)
        _ALIASES.pop("tm", None)


def test_duplicate_registration_is_rejected():
    with pytest.raises(ValueError, match="already registered"):
        register_model(MODEL_REGISTRY["TabICL"])


def test_register_model_type_checks_its_argument():
    with pytest.raises(TypeError):
        register_model({"name": "nope"})


def test_spec_to_dict_is_json_serialisable():
    import json

    payload = get_model_spec("TabFM").to_dict()
    assert json.loads(json.dumps(payload))["envelope"]["max_classes"] == 10


# ------------------------------------------------------- xRFM and iLTM (0.2.0)


def test_xrfm_needs_no_pretrained_weights():
    """xRFM trains from scratch, which makes it the air-gapped-safe option."""
    spec = get_model_spec("XRFM")
    assert "none" in spec.weights.lower()
    assert spec.license.name == "MIT"
    assert spec.license.commercial_use_ok is True


def test_xrfm_has_no_class_ceiling():
    """Kernel methods do not have a fixed-width output head."""
    assert get_model_spec("XRFM").envelope.max_classes is None
    assert check_envelope("XRFM", n_rows=100_000, n_features=500, n_classes=500) == []


def test_xrfm_supports_both_tasks():
    spec = get_model_spec("XRFM")
    assert set(spec.tasks) == {"classification", "regression"}
    assert "peft" in spec.classification_strategies


def test_iltm_enforces_its_hundred_class_ceiling():
    """The limit is architectural and upstream does not guard it.

    iLTM sizes the hypernetwork's first linear layer from ``n_classes_limit``,
    so the released checkpoints are frozen at 100. Upstream has no check: a
    101-class target fails inside ``F.one_hot`` with a bare torch error deep in
    the forward pass. TabTune therefore enforces it before any weights load.
    """
    spec = get_model_spec("ILTM")
    assert spec.envelope.max_classes == 100

    assert check_envelope("ILTM", n_rows=5_000, n_features=50, n_classes=100) == []

    with pytest.raises(EnvelopeError) as excinfo:
        check_envelope("ILTM", n_rows=5_000, n_features=50, n_classes=101)
    message = str(excinfo.value)
    assert "at most 100 classes" in message
    assert "Compatible models" in message


def test_iltm_is_apache_licensed_with_attribution():
    """Apache-2.0 permits commercial use but carries attribution obligations."""
    spec = get_model_spec("ILTM")
    assert spec.license.name == "Apache-2.0"
    assert spec.license.commercial_use_ok is True
    assert spec.license.requires_attribution is True
    assert spec.license.badge == "yes (attribution)"
    assert spec.weights == "dbonet/iLTM"


def test_iltm_and_xrfm_pass_commercial_licence_checks():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_license("XRFM", "commercial").commercial_use_ok is True
        assert check_license("ILTM", "commercial").commercial_use_ok is True


@pytest.mark.parametrize("name", ["XRFM", "ILTM"])
def test_new_models_support_regression_finetuning(name):
    """Both replaced entries in the hardcoded allowlist the registry now owns."""
    spec = validate_request(name, "regression", "finetune")
    assert spec.name == name


def test_high_class_count_suggests_only_capable_models():
    """A 120-class problem must not suggest TabFM (10) or iLTM (100)."""
    with pytest.raises(EnvelopeError) as excinfo:
        check_envelope("TabFM", n_rows=1_000, n_features=20, n_classes=120)
    suggestions = str(excinfo.value).split("Compatible models for this data:")[1]
    assert "ILTM" not in suggestions
    assert "TabPFN" not in suggestions  # 10-class ceiling


def test_truncated_suggestions_say_so():
    """An error listing six options must not imply those are the only six.

    More models than fit in the message usually qualify, and silently cutting
    the list reads as a complete answer.
    """
    with pytest.raises(EnvelopeError) as excinfo:
        check_envelope("TabFM", n_rows=1_000, n_features=20, n_classes=120)
    assert "more - see tabtune.registry.list_models" in str(excinfo.value)


def test_suggestions_prefer_headroom_over_alphabet():
    """Ordering is by capability, not by name.

    Sorting alphabetically and then truncating meant a well-suited model late
    in the alphabet was never suggested.
    """
    from tabtune.registry.registry import _compatible_models

    names = _compatible_models(get_model_spec("TabFM"), n_classes=50)
    unlimited = [n for n in names if not n.startswith("(+")]
    for name in unlimited:
        limit = get_model_spec(name).envelope.max_classes
        assert limit is None or limit >= 50


# ------------------------------------------------------------ EXAONE Tabular


def test_exaone_weights_are_research_only_even_though_the_code_is_not():
    """The two licences differ, and ``LicenseSpec`` describes the weights.

    The code is BSD-3-Clause-LG AI Research and permits commercial use; the
    released weights are granted "solely for research purposes" under the EXAONE
    AI Model License Agreement 1.1 - NC. Recording the code licence here would
    clear the model for a deployment its weights forbid.
    """
    spec = get_model_spec("EXAONE")
    assert spec.name == "EXAONETabular"
    assert spec.license.name == "EXAONE AI Model License Agreement 1.1 - NC"
    assert spec.license.commercial_use_ok is False
    assert spec.license.badge == "no"
    assert "BSD-3-Clause" in spec.license.notes  # the code/weights split is stated

    with pytest.raises(LicenseError) as excinfo:
        check_license("EXAONETabular", "commercial")
    assert "does not permit commercial use" in str(excinfo.value)

    # Every suggested alternative must actually be deployable.
    assert spec.commercial_alternatives
    for name in spec.commercial_alternatives:
        assert get_model_spec(name).license.commercial_use_ok is True


def test_exaone_class_limit_is_soft_and_must_not_be_declared_hard():
    """Above ten classes EXAONE runs an ECOC decomposition; it does not fail.

    ``max_classes`` is a *hard* constraint that raises even in the default
    ``envelope_mode='warn'``, so declaring the 10-class head capacity here would
    reject datasets the model handles by design.
    """
    spec = get_model_spec("EXAONETabular")
    assert spec.envelope.max_classes is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_envelope("EXAONETabular", n_rows=5_000, n_features=20, n_classes=40) == []


@pytest.mark.parametrize(
    "kwargs,constraint",
    [
        ({"n_rows": 250_000, "n_features": 20}, "max_rows"),
        ({"n_rows": 5_000, "n_features": 400}, "max_features"),
    ],
)
def test_exaone_row_and_feature_limits_only_warn(kwargs, constraint):
    """Both degrade rather than fail: subsampling and attention-based selection."""
    with pytest.warns(UserWarning):
        violations = check_envelope("EXAONETabular", mode="warn", **kwargs)
    assert [v.constraint for v in violations] == [constraint]
    assert all(v.severity == "warn" for v in violations)


def test_exaone_envelope_mirrors_the_model_package():
    """The model package stays authoritative; the registry only mirrors it."""
    backbone = pytest.importorskip("tabtune.models.exaone.backbone")
    envelope = get_model_spec("EXAONETabular").envelope
    assert envelope.max_rows == backbone.SUPPORT_ROW_LIMIT
    assert envelope.max_features == backbone.FEATURE_LIMIT
    # CLASS_CAPACITY exists but is deliberately not declared - see the test above.
    assert backbone.CLASS_CAPACITY == 10 and envelope.max_classes is None


def test_exaone_regression_is_experimental_because_no_weights_are_published():
    """The code path is complete; LG AI Research released only the classifier."""
    spec = get_model_spec("EXAONETabular")
    assert "regression" in spec.experimental
    assert spec.supports_regression
    assert spec.regression_strategies == frozenset({"inference", "finetune"})
    # The regression wrapper rejects peft in its constructor.
    assert "peft" not in spec.regression_strategies
    assert "peft" in spec.classification_strategies




