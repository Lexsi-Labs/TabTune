"""The catalog of tabular foundation models TabTune ships support for.

This module is the single place where model metadata lives. Adding a model to
TabTune starts here: write a :class:`~tabtune.registry.spec.ModelSpec`, and the
registry, validation, error messages, model cards, CLI listings and generated
documentation tables all pick it up automatically.

On epistemic honesty
--------------------
``LicenseSpec.commercial_use_ok`` is tri-state. Where upstream terms are
unambiguous we record ``True``/``False``. Where they are ambiguous, have
changed recently, or we simply have not verified them, we record ``None``,
which makes TabTune warn rather than block. Inventing a restriction is as
wrong as ignoring one, and a library that silently guesses about licensing is
worse than one that says "check this yourself".

Likewise, envelope fields are left as ``None`` unless the limit is documented.
An invented row cap would produce spurious warnings and train users to ignore
them.
"""

from __future__ import annotations

from .spec import CapabilityEnvelope, LicenseSpec, ModelSpec

__all__ = ["MODEL_SPECS", "PRIOR_LABS_LICENSE_NOTE"]

# Reused notes -------------------------------------------------------------

PRIOR_LABS_LICENSE_NOTE = (
    "Prior Labs revised its weight licensing during 2025-2026 and terms differ "
    "per checkpoint. TabTune does not assert commercial permissibility for this "
    "checkpoint; confirm the current terms at https://docs.priorlabs.ai/models "
    "before deploying."
)

_COMMERCIAL_FALLBACKS = ("Mitra", "TabICLv2", "OrionMSP", "OrionMSPv1.5", "OrionBix")

# Common strategy bundles --------------------------------------------------

_ICL_CLS = frozenset({"inference", "finetune", "peft"})
_INFERENCE_ONLY = frozenset({"inference"})
_REG_FT = frozenset({"inference", "finetune"})

MODEL_SPECS: tuple[ModelSpec, ...] = (
    # ------------------------------------------------------------ TabPFN v2
    ModelSpec(
        name="TabPFN",
        family="pfn",
        aliases=("TabPFNv2", "TabPFN-v2", "TabPFN2"),
        summary="Prior-data fitted network approximating Bayesian inference on synthetic priors.",
        classification_strategies=_ICL_CLS,
        regression_strategies=_REG_FT,
        finetune_modes=frozenset({"meta-learning", "sft", "turn_by_turn"}),
        experimental=frozenset({"peft"}),
        preprocessor_key="tabpfn_special",
        envelope=CapabilityEnvelope(
            max_classes=10,
            max_features=500,
            max_rows=10_000,
            native_nan=True,
            notes=(
                "TabTune passes ignore_pretraining_limits=True, so exceeding the "
                "row/feature limits degrades accuracy rather than raising."
            ),
        ),
        license=LicenseSpec(
            name="Prior Labs License (Apache-2.0 + attribution)",
            commercial_use_ok=True,
            requires_attribution=True,
            url="https://github.com/PriorLabs/TabPFN",
            notes=(
                "TabPFN v2 weights are under the Prior Labs License: Apache-2.0 plus an "
                "attribution clause (display 'Built with PriorLabs-TabPFN'; derived models "
                "you distribute must be named starting with 'TabPFN'). Later checkpoints "
                "(v2.5, v2.6, v3, v3.5) are non-commercial. " + "Checked 2026-09-28" + " against the "
                "tabpfn 9.0.0 release and AutoGluon's model table ('Prior Labs License "
                "(commercial use permitted)')."
            ),
        ),
        commercial_alternatives=_COMMERCIAL_FALLBACKS,
        paper="https://doi.org/10.1038/s41586-024-08328-6",
        weights="Prior-Labs/TabPFN-v2",
    ),
    # ---------------------------------------------------------- TabPFN v2.6
    ModelSpec(
        name="TabPFNv26",
        family="pfn",
        aliases=("TabPFN-v2.6", "TabPFN2.6", "TabPFNv2.6", "TabPFN26"),
        summary="Prior Labs release adding a native fine-tuning API with bar-distribution loss.",
        # No "peft": the v2.6 meta-learning and SFT loops take no LoRA config and
        # there is no v2.6 LoRA target table, so a PEFT request would not train.
        classification_strategies=frozenset({"inference", "finetune"}),
        regression_strategies=_REG_FT,
        finetune_modes=frozenset({"meta-learning", "sft", "native", "turn_by_turn"}),
        preprocessor_key="tabpfn_special",
        envelope=CapabilityEnvelope(
            max_classes=10,
            max_features=500,
            max_rows=10_000,
            native_nan=True,
        ),
        license=LicenseSpec(
            name="TabPFN-2.6 license (non-commercial)",
            commercial_use_ok=False,
            url="https://github.com/PriorLabs/TabPFN",
            notes=(
                "The v2.6 weights are non-commercial; a commercial license from Prior Labs "
                "is required. " + "Checked 2026-09-28" + " against the tabpfn 9.0.0 release and "
                "AutoGluon's model table ('Commercial license required')."
            ),
        ),
        commercial_alternatives=_COMMERCIAL_FALLBACKS,
        paper="https://arxiv.org/abs/2511.08667",
        weights="Prior-Labs/tabpfn_2_5",
    ),
    # ------------------------------------------------------------ TabPFN v3
    ModelSpec(
        name="TabPFNv3",
        family="pfn",
        aliases=("TabPFN-v3", "TabPFN3", "TabPFNV3"),
        summary=(
            "Re-architected PFN: column distribution embedding, row aggregation, "
            "then in-context learning over compressed row embeddings."
        ),
        classification_strategies=_ICL_CLS,
        regression_strategies=_REG_FT,
        finetune_modes=frozenset({"meta-learning", "sft", "native", "turn_by_turn"}),
        preprocessor_key="tabpfn_special",
        envelope=CapabilityEnvelope(
            max_classes=160,
            max_features=20_000,
            max_rows=1_000_000,
            max_cells=200_000_000,
            native_nan=True,
            notes=(
                "TabPFN-3 advertises a cell budget rather than a row cap: roughly "
                "1M x 200, 100k x 2,000 or 1k x 20,000. An 8-estimator ensemble "
                "needs on the order of 56 GB of KV cache at 1M rows."
            ),
        ),
        license=LicenseSpec(
            name="TABPFN-3.0 License v1.0",
            commercial_use_ok=False,
            url="https://docs.priorlabs.ai/models",
            notes=(
                "Licensed for research and internal evaluation only. Upstream "
                "treats evaluation that informs commercial decisions, and "
                "fine-tuning for commercial purposes, as commercial use."
            ),
        ),
        commercial_alternatives=_COMMERCIAL_FALLBACKS,
        paper="https://arxiv.org/abs/2605.13986",
        weights="Prior-Labs/tabpfn_3",
    ),
    # ---------------------------------------------------------- TabPFN v3.5
    ModelSpec(
        name="TabPFNv35",
        family="pfn",
        aliases=("TabPFN-v3.5", "TabPFN3.5", "TabPFNV35", "TabPFN-3.5"),
        summary=(
            "Prior Labs' v3.5 PFN. A distinct architecture from v3, and the first "
            "release where one multitask checkpoint carries both the "
            "classification and the regression head."
        ),
        classification_strategies=_ICL_CLS,
        regression_strategies=_REG_FT,
        finetune_modes=frozenset({"meta-learning", "sft", "native", "turn_by_turn"}),
        preprocessor_key="tabpfn_special",
        # Limits are left undeclared rather than copied from v3: this module's
        # rule is that an unverified limit is not invented, and v3.5's class
        # ceiling, feature count and cell budget have not been measured here.
        envelope=CapabilityEnvelope(
            native_nan=True,
            native_text=True,
            native_categorical=True,
            notes=(
                "v3.5 preprocessing handles text and datetime columns natively "
                "through skrub. Row/feature/cell limits unverified for this "
                "release; v3 advertises a cell budget (~1M x 200, 100k x 2,000, "
                "1k x 20,000) rather than a row cap."
            ),
        ),
        license=LicenseSpec(
            name="tabpfn-3-5-license-v1.0 (non-commercial)",
            commercial_use_ok=False,
            url="https://github.com/PriorLabs/TabPFN",
            notes=(
                "TabPFN-3.5 weights (gated repo Prior-Labs/tabpfn_3_5) are non-commercial; "
                "the tabpfn code is Apache-2.0 since tabpfn 9.0.0 (2026-09-14). "
                + "Checked 2026-09-28" + " against the tabpfn 9.0.0 wheel and AutoGluon's model "
                "table ('Commercial license required')."
            ),
        ),
        commercial_alternatives=_COMMERCIAL_FALLBACKS,
        paper="https://docs.priorlabs.ai/models",
        weights="Prior-Labs/tabpfn_3_5",
    ),
    # ----------------------------------------------------- TabPFN v3.5-fast
    ModelSpec(
        name="TabPFNv35Fast",
        family="pfn",
        aliases=("TabPFN-v3.5-fast", "TabPFN3.5Fast", "TabPFN-3.5-fast"),
        summary=(
            "The v3.5-fast checkpoint: a separate, smaller-cost model trained "
            "alongside v3.5, not a re-export of it."
        ),
        classification_strategies=_ICL_CLS,
        regression_strategies=_REG_FT,
        finetune_modes=frozenset({"meta-learning", "sft", "native", "turn_by_turn"}),
        preprocessor_key="tabpfn_special",
        envelope=CapabilityEnvelope(
            native_nan=True,
            native_text=True,
            native_categorical=True,
            notes="Shares the v3.5 runtime; limits unverified.",
        ),
        license=LicenseSpec(
            name="tabpfn-3-5-license-v1.0 (non-commercial)",
            commercial_use_ok=False,
            url="https://github.com/PriorLabs/TabPFN",
            notes=(
                "A separate checkpoint (tabpfn-v3.5-fast-20260909.safetensors) in the same "
                "gated repo as TabPFN-3.5, under the same non-commercial license. "
                + "Checked 2026-09-28" + "."
            ),
        ),
        commercial_alternatives=_COMMERCIAL_FALLBACKS,
        paper="https://docs.priorlabs.ai/models",
        weights="Prior-Labs/tabpfn_3_5",
    ),
    # -------------------------------------------------------------- Causilo
    ModelSpec(
        name="Causilo",
        family="icl",
        aliases=("Causilo-v1", "causilo"),
        summary=(
            "Nums AI in-context model: column attention, row mixing and a pooled "
            "prediction stage over a fixed pretrained context. Reports rank 1 on "
            "TabArena Full for classification, regression and overall."
        ),
        classification_strategies=_ICL_CLS,
        regression_strategies=frozenset({"inference", "finetune", "peft"}),
        # Causilo publishes no fine-tuning recipe, so TabTune supplies the
        # episodic loop it uses for the other in-context models. There is no
        # upstream "native" mode to offer.
        finetune_modes=frozenset({"meta-learning"}),
        preprocessor_key="causilo_special",
        envelope=CapabilityEnvelope(
            native_nan=True,
            native_categorical=True,
            notes=(
                "Missing values, categoricals and normalisation are handled inside "
                "the model's own PreparedDataset, so TabTune passes features "
                "through untouched. The classification head has a native 10-class "
                "capacity; more classes are decomposed with error-correcting "
                "output codes, costing one ensemble pass per codebook row. "
                "max_classes is therefore deliberately left None - it is a hard "
                "constraint that would reject datasets this model handles by "
                "design."
            ),
        ),
        license=LicenseSpec(
            name="Apache-2.0 code / Causilo License v1.0 weights (non-commercial)",
            commercial_use_ok=False,
            url="https://github.com/nums-ai/causilo",
            notes=(
                "Code Apache-2.0; weights under the Causilo License v1.0: non-commercial "
                "research, evaluation and modification, and free research redistribution, "
                "are permitted; commercial or production use of the model, derivatives or "
                "outputs, and any hosted/API/SaaS service (paid or free), need a separate "
                "license from Nums AI. " + "Checked 2026-09-28" + " against the vendor README, "
                "AutoGluon and TabArena metadata ('non-commercial weights')."
            ),
        ),
        commercial_alternatives=_COMMERCIAL_FALLBACKS,
        paper="https://huggingface.co/nums-ai/causilo",
        weights="nums-ai/causilo",
    ),
    # -------------------------------------------------------------- TabLDM
    ModelSpec(
        name="TabLDM",
        family="icl",
        aliases=("Xiaomi-TabLDM", "XiaomiTabLDM", "TabLDM-v1"),
        summary=(
            "Xiaomi tabular foundation model: dual-stream feature grouping, a "
            "lightweight attention residual and a sparse mixture of experts, "
            "pretrained only on structural-causal-model synthetic data. Reports "
            "rank 1 on OpenML-CTR23 and rank 2 on regression across TALENT, "
            "TabArena and BCCO."
        ),
        classification_strategies=_ICL_CLS,
        regression_strategies=frozenset({"inference", "finetune", "peft"}),
        # TabLDM ships inference-only, so there is no upstream "native" mode.
        # Its forward() keeps the supervised branch, though, so the episodic
        # loop TabTune runs is the vendor's own training path.
        finetune_modes=frozenset({"meta-learning"}),
        preprocessor_key="tabldm_special",
        envelope=CapabilityEnvelope(
            native_nan=True,
            native_categorical=True,
            notes=(
                "Categorical detection, missing-value handling, outlier clipping "
                "and normalisation all happen inside the model's own "
                "TransformToNumerical/EnsembleGenerator, so TabTune passes "
                "features through untouched. No envelope limit is declared: the "
                "classification head has a native 10-class capacity but more "
                "classes are handled by hierarchical grouping "
                "(support_many_classes), and the 300-feature max_num_features "
                "default triggers per-member feature subsampling rather than "
                "rejecting the dataset. Declaring either as a limit would reject "
                "datasets this model handles by design. Upstream states the "
                "pretraining distribution covers hundreds to tens of thousands of "
                "rows and up to roughly a hundred columns, and that accuracy may "
                "decline beyond it, but publishes no hard bound. KV caching is "
                "unavailable above 10 classes and cannot be combined with the "
                "candidate-enhancement path."
            ),
        ),
        license=LicenseSpec(
            name="Apache-2.0 code; weight license conflicting",
            commercial_use_ok=None,
            url="https://github.com/XiaomiMiMo/Xiaomi-TabLDM",
            notes=(
                "The repository LICENSE and README License section say Apache-2.0 (as does "
                "TabArena), but a README news line added 2026-09-28 says weight usage 'is "
                "subject to Xiaomi-TabLDM Non-Commercial License' while linking to the Apache "
                "LICENSE. Until Xiaomi resolves this, TabTune does not assert commercial use. "
                + "Checked 2026-09-28" + "."
            ),
        ),
        paper="https://arxiv.org/abs/2609.03880",
        weights="occams/Xiaomi-TabLDM",
    ),
    # -------------------------------------------------------------- TabICL
    ModelSpec(
        name="TabICL",
        family="icl",
        aliases=("TabICLv1", "TabICL-v1"),
        summary="Scalable tabular in-context learning with column-then-row attention.",
        classification_strategies=_ICL_CLS,
        finetune_modes=frozenset({"meta-learning", "sft"}),
        preprocessor_key="tabicl_special",
        envelope=CapabilityEnvelope(native_nan=True),
        license=LicenseSpec(
            name="BSD-3-Clause",
            commercial_use_ok=True,
            url="https://github.com/soda-inria/tabicl",
        ),
        paper="https://arxiv.org/abs/2502.05564",
        weights="soda-inria/tabicl",
    ),
    # ------------------------------------------------------------ TabICL v2
    ModelSpec(
        name="TabICLv2",
        family="icl",
        aliases=("TabICL-v2", "TabICL2"),
        summary="Improved column-then-row attention with QASSMax and quantile regression.",
        classification_strategies=frozenset({"inference", "finetune"}),
        regression_strategies=_REG_FT,
        finetune_modes=frozenset({"meta-learning", "sft", "turn_by_turn"}),
        preprocessor_key="tabiclv2_special",
        envelope=CapabilityEnvelope(
            max_features=2_000,
            max_rows=500_000,
            native_nan=True,
            notes="Upstream supports 500k rows via CPU/disk offloading of the KV cache.",
        ),
        license=LicenseSpec(
            name="BSD-3-Clause",
            commercial_use_ok=True,
            url="https://github.com/soda-inria/tabicl",
        ),
        paper="https://arxiv.org/abs/2602.11139",
        weights="soda-inria/tabicl",
    ),
    # ------------------------------------------------------------ OrionMSP
    ModelSpec(
        name="OrionMSP",
        family="icl",
        aliases=("Orion-MSP", "OrionMSPv1", "OrionMSPv1.0"),
        summary="Multi-scale sparse attention for tabular in-context learning.",
        classification_strategies=_ICL_CLS,
        finetune_modes=frozenset({"meta-learning", "sft"}),
        preprocessor_key="orion_msp_special",
        envelope=CapabilityEnvelope(native_nan=True),
        license=LicenseSpec(
            name="MIT",
            commercial_use_ok=True,
            url="https://github.com/Lexsi-Labs/OrionMSP",
        ),
        paper="https://arxiv.org/abs/2511.02818",
        weights="Lexsi-Labs/OrionMSP",
    ),
    # -------------------------------------------------------- OrionMSP v1.5
    ModelSpec(
        name="OrionMSPv1.5",
        family="icl",
        aliases=("Orion-MSP-v1.5", "OrionMSP1.5", "OrionMSPv15"),
        summary="OrionMSP with stabilized prototype refinement.",
        classification_strategies=_ICL_CLS,
        finetune_modes=frozenset({"meta-learning", "sft"}),
        preprocessor_key="orion_msp_special",
        envelope=CapabilityEnvelope(native_nan=True),
        license=LicenseSpec(
            name="MIT",
            commercial_use_ok=True,
            url="https://github.com/Lexsi-Labs/OrionMSP",
        ),
        paper="https://arxiv.org/abs/2511.02818",
        weights="Lexsi-Labs/OrionMSP",
    ),
    # ------------------------------------------------------------- OrionBix
    ModelSpec(
        name="OrionBix",
        family="icl",
        aliases=("Orion-BiX", "OrionBiX"),
        summary="Bi-axial in-context learning for tabular data.",
        classification_strategies=_ICL_CLS,
        finetune_modes=frozenset({"meta-learning", "sft"}),
        preprocessor_key="orion_bix_special",
        envelope=CapabilityEnvelope(native_nan=True),
        license=LicenseSpec(
            name="MIT",
            commercial_use_ok=True,
            url="https://github.com/Lexsi-Labs/OrionBix",
        ),
        paper="https://arxiv.org/abs/2512.00181",
        weights="Lexsi-Labs/OrionBix",
    ),
    # ---------------------------------------------------------------- Mitra
    ModelSpec(
        name="Mitra",
        family="icl",
        aliases=("Tab2D", "mitra-classifier"),
        summary="Mixed synthetic priors with 2D row-and-column attention (AWS).",
        classification_strategies=_ICL_CLS,
        regression_strategies=_REG_FT,
        finetune_modes=frozenset({"meta-learning", "sft", "turn_by_turn"}),
        preprocessor_key="mitra_special",
        envelope=CapabilityEnvelope(
            max_rows=10_000,
            notes=(
                "Upstream documents a practical ceiling near 10k context rows; "
                "larger inputs typically exhaust GPU memory."
            ),
        ),
        license=LicenseSpec(
            name="Apache-2.0",
            commercial_use_ok=True,
            url="https://huggingface.co/autogluon/mitra-classifier",
            notes=(
                "AutoGluon's release notes: Mitra's 'weights are fully open-sourced under the "
                "Apache-2.0 license'. (Earlier TabTune releases recorded CC-BY-4.0; that was "
                "wrong.) " + "Checked 2026-09-28" + " against autogluon@6f9ebe9 docs."
            ),
        ),
        paper="https://arxiv.org/abs/2510.21204",
        weights="autogluon/mitra-classifier",
    ),
    # -------------------------------------------------------------- Mitra 2
    ModelSpec(
        name="MitraV2",
        family="icl",
        aliases=("Mitra-2", "Mitra2", "mitra-classifier-2", "mitra-regressor-2"),
        summary=(
            "Second-generation Mitra checkpoint on the same 2D row-and-column "
            "attention architecture (AWS). Shares TabTune's vendored Tab2D with "
            "Mitra v1; only the weights differ."
        ),
        classification_strategies=_ICL_CLS,
        regression_strategies=_REG_FT,
        finetune_modes=frozenset({"meta-learning", "sft", "turn_by_turn"}),
        preprocessor_key="mitra_special",
        envelope=CapabilityEnvelope(
            notes=(
                "No limit is declared. v1's entry documents a practical ceiling "
                "near 10k context rows, but that was measured on v1's weights; "
                "v2's dim / n_layers / n_heads come from its own config.json and "
                "were not readable when this entry was written, so carrying v1's "
                "number across would be asserting a limit nobody measured. "
                "Classification head width is architectural, not the dataset's "
                "class count, and is sliced to the task at prediction time."
            ),
        ),
        license=LicenseSpec(
            name="Apache-2.0",
            commercial_use_ok=True,
            url="https://huggingface.co/autogluon/mitra-classifier-2",
            notes=(
                "The mitra-classifier-2 / mitra-regressor-2 model cards state Apache-2.0 (read "
                "through search-result text; huggingface.co was not reachable) and TabArena's "
                "metadata agrees. AutoGluon itself still defaults to the v1 weights. "
                + "Checked 2026-09-28" + "."
            ),
        ),
        paper="https://arxiv.org/abs/2510.21204",
        weights="autogluon/mitra-classifier-2",
    ),
    # ----------------------------------------------------------- ContextTab
    ModelSpec(
        name="ContextTab",
        family="semantic-icl",
        # Upstream renamed the release to SAP-RPT-1-OSS; both names resolve here
        # so users who find the model under either name land in the right place.
        aliases=("ConTextTab", "SAP-RPT-1-OSS", "SAPRPT1OSS", "sap-rpt-1"),
        summary="Semantics-aware in-context learning with modality-specific embeddings.",
        classification_strategies=_ICL_CLS,
        regression_strategies=_REG_FT,
        finetune_modes=frozenset({"sft", "turn_by_turn"}),
        experimental=frozenset({"peft"}),
        preprocessor_key="contexttab_special",
        envelope=CapabilityEnvelope(
            native_text=True,
            native_categorical=True,
            notes="The only bundled model with first-class text and datetime handling.",
        ),
        license=LicenseSpec(
            name="SAP-RPT-1-OSS (research use)",
            commercial_use_ok=False,
            url="https://huggingface.co/SAP/sap-rpt-1-oss",
            notes=(
                "Code is Apache-2.0 but the released checkpoints are restricted to "
                "research use and inherit upstream dataset restrictions."
            ),
        ),
        commercial_alternatives=_COMMERCIAL_FALLBACKS,
        paper="https://arxiv.org/abs/2506.10707",
        weights="SAP/sap-rpt-1-oss",
    ),
    # --------------------------------------------------------------- TabDPT
    ModelSpec(
        name="TabDPT",
        family="denoising",
        aliases=("Tab-DPT",),
        summary="Denoising pre-training transformer with retrieval-based context.",
        classification_strategies=_ICL_CLS,
        regression_strategies=_REG_FT,
        finetune_modes=frozenset({"meta-learning", "sft", "turn_by_turn"}),
        preprocessor_key="tabdpt_special",
        envelope=CapabilityEnvelope(),
        license=LicenseSpec(
            name="see upstream",
            commercial_use_ok=None,
            url="https://github.com/layer6ai-labs/TabDPT-inference",
            notes="TabTune has not verified the weight license; confirm upstream.",
        ),
        paper="https://arxiv.org/abs/2410.18164",
        weights="layer6ai-labs/TabDPT",
    ),
    # ---------------------------------------------------------------- LimiX
    ModelSpec(
        name="Limix",
        family="probabilistic-icl",
        aliases=("LimiX", "LimiX-16M"),
        summary="Likelihood-based mixture modelling with uncertainty-aware inference.",
        classification_strategies=_INFERENCE_ONLY,
        regression_strategies=_REG_FT,
        finetune_modes=frozenset({"turn_by_turn"}),
        preprocessor_key="limix_special",
        envelope=CapabilityEnvelope(native_nan=True),
        license=LicenseSpec(
            name="Stable AI Technology License 1.0 (Apache-2.0 + attribution)",
            commercial_use_ok=True,
            requires_attribution=True,
            url="https://github.com/limix-ldm-ai/LimiX/blob/main/LICENSE.txt",
            notes=(
                "TabTune loads LimiX-16M (stableai-org/LimiX-16M). The LimiX README lists "
                "LimiX-16M and LimiX-2M under the Stable AI Technology Co., Ltd. License 1.0 "
                "(Sept 2026): Apache-2.0 sections 1-9 plus section 10, which requires "
                "displaying 'Built with StableAI LimiX' and a name starting with 'LimiX' for "
                "derived models you distribute; internal research use triggers neither. "
                "LimiX-2 weights are non-commercial and are not used by TabTune. "
                "Checked 2026-09-28 against the GitHub LICENSE.txt and README; the Hugging "
                "Face license file itself could not be fetched."
            ),
        ),
        commercial_alternatives=_COMMERCIAL_FALLBACKS,
        paper="https://arxiv.org/abs/2509.03505",
        weights="limix-ldm-ai/LimiX",
    ),
    # ---------------------------------------------------------------- TabFM
    ModelSpec(
        name="TabFM",
        family="hybrid-attention-icl",
        aliases=("Tab-FM", "google-tabfm"),
        summary=(
            "Google Research hybrid attention: alternating row/column blocks, row "
            "compression to CLS tokens, then a causal ICL transformer."
        ),
        classification_strategies=_ICL_CLS,
        regression_strategies=_REG_FT,
        finetune_modes=frozenset({"meta-learning", "sft", "turn_by_turn"}),
        preprocessor_key="tabfm_special",
        envelope=CapabilityEnvelope(
            max_classes=10,
            max_features=500,
            notes=(
                "The ten-class limit is architectural: the pretrained output head "
                "has ten slots and cannot be widened without retraining."
            ),
        ),
        license=LicenseSpec(
            name="TabFM Non-Commercial License v1.0",
            commercial_use_ok=False,
            url="https://huggingface.co/google/tabfm-1.0.0-pytorch",
            notes="Code is Apache-2.0; the released weights are non-commercial.",
        ),
        commercial_alternatives=_COMMERCIAL_FALLBACKS,
        paper="https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/",
        weights="google/tabfm-1.0.0-pytorch",
    ),
    # ----------------------------------------------------------------- xRFM
    ModelSpec(
        name="XRFM",
        family="kernel-feature-learning",
        aliases=("xRFM", "x-RFM", "RFM", "RecursiveFeatureMachine"),
        summary=(
            "Recursive Feature Machine: a kernel method that learns features via the "
            "average gradient outer product, partitioned by a tree and solved with "
            "EigenPro so it scales to large tabular data."
        ),
        classification_strategies=_ICL_CLS,
        regression_strategies=_REG_FT,
        # xRFM has no gradient-descent fine-tuning. "finetune" refits or refines
        # the RFM, and "peft" is low-rank adaptation of the learned M matrix
        # rather than LoRA over linear layers, so it has no LoRA target table.
        finetune_modes=frozenset({"refit", "refine"}),
        preprocessor_key="xrfm_special",
        envelope=CapabilityEnvelope(
            native_categorical=True,
            notes=(
                "Trains from scratch on every dataset: there are no pretrained "
                "weights and no download, which also makes it the only bundled "
                "model that works air-gapped out of the box. No hard row or "
                "feature cap - the practical ceiling is GPU memory per tree leaf, "
                "and max_leaf_size (default 60,000) is auto-rescaled from the "
                "device's memory, so results can differ across hardware. Upstream "
                "reports it becomes competitive from roughly 60k rows upward."
            ),
        ),
        license=LicenseSpec(
            name="MIT",
            commercial_use_ok=True,
            url="https://github.com/dmbeaglehole/xRFM",
            notes="Copyright (c) 2025 Daniel Beaglehole. No weights to license.",
        ),
        paper="https://arxiv.org/abs/2508.10053",
        weights="(none - trained from scratch)",
    ),
    # ----------------------------------------------------------------- iLTM
    ModelSpec(
        name="ILTM",
        family="hypernetwork",
        aliases=("iLTM", "i-LTM", "IntegratedLargeTabularModel"),
        summary=(
            "Integrated Large Tabular Model: a hypernetwork generates MLP ensembles "
            "conditioned on dataset embeddings, combining GBDT tree embeddings with "
            "retrieval over the training set."
        ),
        classification_strategies=_ICL_CLS,
        regression_strategies=_REG_FT,
        finetune_modes=frozenset({"meta-learning", "sft", "turn_by_turn"}),
        preprocessor_key="iltm_special",
        envelope=CapabilityEnvelope(
            max_classes=100,
            native_categorical=True,
            notes=(
                "The 100-class limit is architectural: the hypernetwork's first "
                "linear layer is sized from n_classes_limit, so the released "
                "checkpoints are frozen at 100. Upstream does not guard it - a "
                "101-class target fails inside F.one_hot with a bare torch error "
                "deep in the forward pass - so TabTune enforces it here. "
                "Dimensionality-agnostic by construction (evaluated from 4 to "
                "~20,000 features); the retrieval context is capped at 8,192 rows. "
                "Pretrained on classification only; regression is reached by "
                "transfer plus light fine-tuning."
            ),
        ),
        license=LicenseSpec(
            name="Apache-2.0",
            commercial_use_ok=True,
            requires_attribution=True,
            url="https://github.com/AI-sandbox/iLTM",
            notes=(
                "Code and weights are both Apache-2.0 and the Hugging Face "
                "repository is ungated. TabTune vendors a modified copy, so the "
                "licence text and change notices are retained under Apache-2.0 "
                "sections 4(b) and 4(c)."
            ),
        ),
        paper="https://arxiv.org/abs/2511.15941",
        weights="dbonet/iLTM",
    ),
    # ------------------------------------------------------- EXAONE Tabular
    ModelSpec(
        name="EXAONETabular",
        family="cross-axis-icl",
        # normalise_name() strips '-', '_', '.' and whitespace, so "exaone-tabular",
        # "EXAONE_Tabular" and "exaone tabular" already resolve to the canonical
        # name. Only the bare "EXAONE" spelling needs an entry of its own - it is
        # what the model package and LG AI Research's own materials call it.
        aliases=("EXAONE",),
        summary=(
            "LG AI Research in-context learner built on the Cross-axis Summary "
            "Transformer (CAST): per-row feature summaries and per-column row "
            "summaries exchanged across both table axes under SSMax-normalised "
            "attention. Only the classification checkpoint is published; "
            "regression needs a locally supplied weights file."
        ),
        classification_strategies=_ICL_CLS,
        regression_strategies=_REG_FT,
        finetune_modes=frozenset({"meta-learning", "sft", "turn_by_turn"}),
        # Not a strategy caveat: the regression code path is complete and tested,
        # but LG AI Research has published no regression checkpoint, so nothing
        # downloads and the wrapper raises FileNotFoundError without a local file.
        experimental=frozenset({"regression"}),
        preprocessor_key="exaone_special",
        envelope=CapabilityEnvelope(
            max_features=100,
            max_rows=100_000,
            native_nan=True,
            notes=(
                "All three of this model's ceilings are soft, and none of them "
                "raises: above 100,000 support rows the vendored fit randomly "
                "subsamples down to the limit, above 100 features it runs its "
                "attention-based selector and keeps the 100 highest-scoring "
                "columns, and above the classification head's 10-class capacity "
                "it decomposes the problem with an ECOC codebook, costing one "
                "full ensemble forward per codebook row. max_classes is "
                "therefore deliberately left None: it is a hard constraint that "
                "raises even in envelope_mode='warn', so declaring it would "
                "reject datasets this model handles by design. Values mirror "
                "SUPPORT_ROW_LIMIT / FEATURE_LIMIT / CLASS_CAPACITY in "
                "tabtune/models/exaone/backbone.py."
            ),
        ),
        license=LicenseSpec(
            name="EXAONE AI Model License Agreement 1.1 - NC",
            commercial_use_ok=False,
            url="https://huggingface.co/LG-AI-Research/EXAONE-Tabular",
            notes=(
                "Code and weights are licensed separately. The code is "
                "BSD-3-Clause-LG AI Research and permits commercial use; the "
                "weights are granted 'solely for research purposes' and the "
                "agreement expressly prohibits using the model, derivatives or "
                "output for any commercial purpose. LicenseSpec describes the "
                "weights, hence commercial_use_ok=False. A fine-tuned checkpoint "
                "is a Derivative under that agreement: still research-only, and "
                "its name must begin with 'EXAONE'."
            ),
        ),
        commercial_alternatives=_COMMERCIAL_FALLBACKS,
        # No paper yet - LG AI Research says a technical report will follow, so
        # this points at the source repository rather than inventing a citation.
        paper="https://github.com/LGAI-Research/EXAONE-Tabular",
        weights="LG-AI-Research/EXAONE-Tabular",
    ),
)
