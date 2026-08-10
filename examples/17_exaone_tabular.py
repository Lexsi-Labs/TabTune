"""
Example 17: EXAONE Tabular (LG AI Research) — Registry, Limits and the Full Surface
===================================================================================

EXAONE Tabular (model name `EXAONETabular`, alias `EXAONE`) is an in-context
learner built on the Cross-axis Summary Transformer: 3 summary tokens per row
pool that row's columns, 32 summary tokens per feature group pool that column
across rows, and the two axes alternate for 12 blocks under SSMax-normalised
attention. At ~21M parameters it is the smallest bundled foundation model, which
is why the released default is an 8-member ensemble.

What runs here, and what does not
---------------------------------
Everything that needs no checkpoint runs for real: the registry entry, the three
soft limits, the licence split, and the feature encoding the model consumes.

The sections that need pretrained weights — inference, fine-tuning, PEFT,
regression and support-set attribution — are *printed rather than executed* when
no checkpoint is staged, the same way `16_attribution.py` handles the models it
cannot download. The printed text is the real source of the functions in this
file, so it cannot drift: stage a checkpoint and this same file runs them.

Two things to know before planning a run:

* **Only the classification checkpoint is published.** The regression path is
  fully implemented and tested, but there is nothing to download; supply a local
  file through `model_params={'checkpoint_path': ...}` or the
  `EXAONETABULAR_REGRESSOR_WEIGHTS` environment variable.
* **The weights are research-only** (EXAONE AI Model License Agreement 1.1 - NC)
  even though the code is BSD-3-Clause-LG AI Research and permits commercial use.
  The registry records the weights, so `license_mode='commercial'` blocks.

Run:
    python examples/17_exaone_tabular.py

Staging weights (optional, enables the executed path):
    export EXAONETABULAR_CLASSIFIER_WEIGHTS=/path/to/exaone-tabular-classifier.safetensors
"""

from __future__ import annotations

import inspect
import logging
import os

import numpy as np
import pandas as pd

from tabtune import TabularPipeline
from tabtune.models.exaone.backbone import (
    CLASS_CAPACITY,
    EXAONE_HF_REPO,
    FEATURE_LIMIT,
    SUPPORT_ROW_LIMIT,
    WEIGHTS_ENV_VARS,
)
from tabtune.models.exaone.episode_features import EXAONEFeatureEncoder
from tabtune.registry import (
    EnvelopeError,
    LicenseError,
    check_envelope,
    check_license,
    get_model_spec,
    resolve_model_name,
)

# The pipeline narrates every forward pass at INFO; this example is about the
# model's metadata and its edges, not the plumbing.
logging.getLogger("tabtune").setLevel(logging.WARNING)


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def make_data(n: int = 400, seed: int = 0):
    """A small mixed-type frame: two numeric signals, one categorical, one noisy.

    Synthetic on purpose — this example must run with no network access, so it
    does not reach for OpenML the way the older model examples do.
    """
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "signal_a": rng.normal(size=n),
            "signal_b": rng.normal(size=n),
            "noise": rng.normal(size=n),
            "region": rng.choice(["north", "south", "east"], size=n),
        }
    )
    logit = 2.4 * frame["signal_a"] - 1.8 * frame["signal_b"] + rng.normal(scale=0.35, size=n)
    return frame, pd.Series((logit > 0).astype(int), name="target")


X, y = make_data()
X_train, y_train = X.iloc[:320], y.iloc[:320]
X_test, y_test = X.iloc[320:], y.iloc[320:]


def weights_are_staged() -> bool:
    """Whether a classification checkpoint is reachable without downloading one.

    True when the environment override points at an existing file, or when the
    released file is already in the Hugging Face cache. Anything else means this
    run prints the weight-dependent code instead of executing it.
    """
    override = os.environ.get(WEIGHTS_ENV_VARS["classification"])
    if override and os.path.isfile(os.path.expanduser(override)):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache

        from tabtune.models.exaone.presets import released_checkpoint

        record = released_checkpoint("classification")
        cached = try_to_load_from_cache(record.repo_id, record.filename)
        return isinstance(cached, str) and os.path.isfile(cached)
    except Exception:  # pragma: no cover - cache probing is best-effort
        return False


# --------------------------------------------------------------------------
# The weight-dependent surface. Executed when a checkpoint is staged, printed
# verbatim otherwise — the printed text is this source, so it cannot go stale.
# --------------------------------------------------------------------------


def run_classification():
    """Inference, episodic meta-learning, SFT and PEFT — the same unified API."""
    results = {}

    # --- Strategy 1: Inference (zero-shot, in-context) ---
    pipe = TabularPipeline(
        model_name="EXAONE",
        task_type="classification",
        tuning_strategy="inference",
        model_params={"n_ensemble": 8},
    )
    pipe.fit(X_train, y_train)
    results["Inference"] = pipe.evaluate(X_test, y_test)

    # --- Strategy 2: Episodic meta-learning fine-tuning (default) ---
    # Episodes call the model's real differentiable forward,
    # episode_logits(x_support, y_support, x_query), not a surrogate.
    pipe = TabularPipeline(
        model_name="EXAONE",
        task_type="classification",
        tuning_strategy="finetune",
        finetune_mode="meta-learning",
        tuning_params={"epochs": 2, "learning_rate": 1e-5, "show_progress": True},
    )
    pipe.fit(X_train, y_train)
    results["Meta-Learning"] = pipe.evaluate(X_test, y_test)

    # --- Strategy 3: SFT (one fixed support/query split, trained repeatedly) ---
    # finetune_mode goes in the constructor: for finetune/peft runs the pipeline
    # writes the constructor value into tuning_params, so a mode set only in the
    # dict is overwritten.
    pipe = TabularPipeline(
        model_name="EXAONE",
        task_type="classification",
        tuning_strategy="finetune",
        finetune_mode="sft",
        tuning_params={"epochs": 2, "learning_rate": 1e-5},
    )
    pipe.fit(X_train, y_train)
    results["SFT"] = pipe.evaluate(X_test, y_test)

    # --- Strategy 4: PEFT / LoRA ---
    # Honest caveat: EXAONE's projections are raw nn.Parameters applied through
    # F.linear, not nn.Linear submodules, so the injector currently wraps zero
    # adapters and logs a warning saying so. The run is a full fine-tune.
    pipe = TabularPipeline(
        model_name="EXAONE",
        task_type="classification",
        tuning_strategy="peft",
        tuning_params={"epochs": 2, "peft_config": {"r": 8, "lora_alpha": 16}},
    )
    pipe.fit(X_train, y_train)
    results["PEFT"] = pipe.evaluate(X_test, y_test)

    for strategy, metrics in results.items():
        print(f"   {strategy:15s} accuracy={metrics.get('accuracy', float('nan')):.4f}")
    return results


def run_regression(checkpoint_path: str):
    """Regression needs a checkpoint you supply: none is published."""
    X_reg = X.drop(columns=["region"])
    y_reg = 3.0 * X_reg["signal_a"] - 1.5 * X_reg["signal_b"]

    pipe = TabularPipeline(
        model_name="EXAONE",
        task_type="regression",
        tuning_strategy="inference",
        model_params={"checkpoint_path": checkpoint_path},
    )
    pipe.fit(X_reg.iloc[:320], y_reg.iloc[:320])
    print("   inference:", pipe.evaluate(X_reg.iloc[320:], y_reg.iloc[320:]))

    # Regression fine-tuning is episodic turn-by-turn on the quantile head.
    pipe = TabularPipeline(
        model_name="EXAONE",
        task_type="regression",
        tuning_strategy="finetune",
        finetune_mode="turn_by_turn",
        model_params={"checkpoint_path": checkpoint_path},
        tuning_params={"epochs": 2, "learning_rate": 1e-5},
    )
    pipe.fit(X_reg.iloc[:320], y_reg.iloc[:320])
    print("   turn-by-turn:", pipe.evaluate(X_reg.iloc[320:], y_reg.iloc[320:]))





# --------------------------------------------------------------------------
# Sections that need nothing but the registry and a few hundred bytes of numpy.
# --------------------------------------------------------------------------


def describe_registry_entry() -> None:
    section("1. The registry entry (no weights, no network)")

    spec = get_model_spec("EXAONE")
    print(f"  name              {spec.name}")
    print(f"  family            {spec.family}")
    print(f"  tasks             {', '.join(spec.tasks)}")
    print(f"  classification    {sorted(spec.classification_strategies)}")
    print(f"  regression        {sorted(spec.regression_strategies)}")
    print(f"  finetune modes    {sorted(spec.finetune_modes)}")
    print(f"  experimental      {sorted(spec.experimental)}")
    print(f"  preprocessor      {spec.preprocessor_key}")
    print(f"  envelope          {spec.envelope.describe()}")
    print(f"  weights           {spec.weights}  (public, no token)")
    print(f"  licence           {spec.license.name}  -> commercial: {spec.license.badge}")

    print("\n  Name resolution ignores case, hyphens, underscores and spaces:")
    for alias in ["EXAONE", "exaone", "exaone-tabular", "EXAONE_Tabular", "EXAONE Tabular"]:
        print(f"    {alias:<18} -> {resolve_model_name(alias)}")

    print(
        "\n  'regression' is experimental for one reason only: the code path is\n"
        "  complete and tested, but LG AI Research published only the\n"
        f"  classification checkpoint ({EXAONE_HF_REPO}). Supply a regression file\n"
        f"  via model_params={{'checkpoint_path': ...}} or {WEIGHTS_ENV_VARS['regression']}."
    )


def demonstrate_soft_limits() -> None:
    section("2. Three limits, and not one of them is an error")

    print(
        f"  support rows > {SUPPORT_ROW_LIMIT:,}  -> random subsample down to the limit\n"
        f"  features     > {FEATURE_LIMIT}      -> attention-based selection of the top {FEATURE_LIMIT}\n"
        f"  classes      > {CLASS_CAPACITY}       -> ECOC decomposition, one full ensemble\n"
        f"                              forward per codebook row\n"
    )

    print("  Rows and features warn (they degrade quality, not correctness):")
    violations = check_envelope(
        "EXAONE", n_rows=250_000, n_features=120, n_classes=14, mode="warn"
    )
    for violation in violations:
        print(f"    [{violation.severity}] {violation.message}")

    print(
        "\n  Note what is NOT in that list: 14 classes is not a violation at all.\n"
        "  max_classes is a *hard* constraint - it raises even in the default\n"
        "  envelope_mode='warn' - so declaring the 10-class head capacity would\n"
        "  reject datasets this model handles by design via ECOC."
    )
    print(f"    declared max_classes: {get_model_spec('EXAONE').envelope.max_classes}")

    print("\n  envelope_mode='error' is how you turn the soft limits into a stop:")
    try:
        check_envelope("EXAONE", n_rows=250_000, n_features=20, mode="error")
    except EnvelopeError as exc:
        print("    " + str(exc).splitlines()[1].strip())


def demonstrate_licensing() -> None:
    section("3. The code and the weights are licensed separately")

    spec = get_model_spec("EXAONE")
    print("  code    BSD-3-Clause-LG AI Research  -> commercial use permitted")
    print(f"  weights {spec.license.name}  -> research only")
    print(
        "\n  LicenseSpec describes the *weights*, because those are what decide\n"
        "  whether you can ship. Research mode (the default) never blocks:"
    )
    print(f"    check_license('EXAONE', 'research') -> {check_license('EXAONE', 'research').badge}")

    print("\n  Commercial mode does:")
    try:
        check_license("EXAONE", "commercial")
    except LicenseError as exc:
        for line in str(exc).splitlines():
            print("    " + line)


def demonstrate_feature_encoding() -> None:
    section("4. What the model actually consumes (runs for real)")

    encoder = EXAONEFeatureEncoder().fit(X_train)
    encoded = encoder.transform(X_test.head(3))
    print(f"  columns              {encoder.columns_}")
    print(f"  categorical columns  {encoder.categorical_columns_}")
    print(f"  encoded dtype/shape  {encoded.dtype}, {encoded.shape}")
    print(f"  first row            {np.round(encoded[0], 3)}")
    print(
        "\n  There is no categorical encoder inside the network, so this ordinal\n"
        "  encoding happens first: the model's ensemble then treats any column\n"
        "  with fewer than ten distinct support values as categorical and gives\n"
        "  each member its own permutation of the codes. Missing values stay NaN\n"
        "  on purpose - the model encodes missingness as an explicit channel."
    )


if __name__ == "__main__":
    print("=" * 72)
    print("EXAMPLE 17: EXAONE Tabular (LG AI Research)")
    print("=" * 72)

    describe_registry_entry()
    demonstrate_soft_limits()
    demonstrate_licensing()
    demonstrate_feature_encoding()

    section("5. Inference, fine-tuning, PEFT, regression and attribution")

    if weights_are_staged():
        print("  A classification checkpoint is staged; running for real.\n")
        run_classification()

        regression_weights = os.environ.get(WEIGHTS_ENV_VARS["regression"])
        if regression_weights:
            run_regression(regression_weights)
        else:
            print(
                f"\n  Skipping regression: set {WEIGHTS_ENV_VARS['regression']} to a local\n"
                "  .safetensors file. LG AI Research publishes no regression checkpoint."
            )
    else:
        print(
            "  Not executed: no EXAONE checkpoint is staged in this environment.\n"
            "  Every line below is real API, and is the literal source of the\n"
            "  functions in this file - stage a checkpoint and it runs unchanged.\n"
        )
        for function in (run_classification, run_regression):
            print(inspect.getsource(function))

    section("6. When to reach for EXAONE Tabular")
    print("   Classification, zero-shot   : small vendored model, 8-member ensemble")
    print("   Fine-tuning                 : meta-learning (default) or sft")
    print("   Wide or long tables         : degrades by selection/subsampling, never fails")
    print("   Many classes                : ECOC handles it; budget the extra forwards")
    print("   Regression                  : only with a checkpoint you supply yourself")
    print("   Commercial deployment       : no - distil it, or use Mitra / TabICLv2")
