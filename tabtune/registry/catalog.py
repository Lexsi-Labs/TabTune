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
            name="Prior Labs License",
            commercial_use_ok=None,
            url="https://docs.priorlabs.ai/models",
            notes=PRIOR_LABS_LICENSE_NOTE,
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
        classification_strategies=_ICL_CLS,
        regression_strategies=_REG_FT,
        finetune_modes=frozenset({"meta-learning", "sft", "native", "turn_by_turn"}),
        experimental=frozenset({"peft"}),
        preprocessor_key="tabpfn_special",
        envelope=CapabilityEnvelope(
            max_classes=10,
            max_features=500,
            max_rows=10_000,
            native_nan=True,
        ),
        license=LicenseSpec(
            name="Prior Labs License",
            commercial_use_ok=None,
            url="https://docs.priorlabs.ai/models",
            notes=PRIOR_LABS_LICENSE_NOTE,
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
            name="CC-BY-4.0",
            commercial_use_ok=True,
            requires_attribution=True,
            url="https://huggingface.co/autogluon/mitra-classifier",
        ),
        paper="https://arxiv.org/abs/2510.21204",
        weights="autogluon/mitra-classifier",
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
            name="LimiX (academic use free)",
            commercial_use_ok=False,
            url="https://github.com/limix-ldm-ai/LimiX",
            notes="Free for academic use; commercial deployment requires authorization.",
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
