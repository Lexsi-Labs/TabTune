"""Model registry: discovery, capability envelopes and weight licensing.

Answers three questions without downloading a single checkpoint:

1. Which models can do my task?
2. Will this model accept my data?
3. Am I allowed to deploy it?

Run:
    python examples/13_registry_and_licensing.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabtune.registry import (
    EnvelopeError,
    LicenseError,
    ModelNotFoundError,
    check_envelope,
    check_license,
    get_model_spec,
    list_models,
    models_dataframe,
    resolve_model_name,
)


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# --------------------------------------------------------------------------
section("1. Discovery")

print("Models supporting regression:")
for spec in list_models(task="regression"):
    print(f"  {spec.name:<14} {spec.family:<20} {spec.license.name}")

print("\nCommercially deployable classification models:")
for spec in list_models(task="classification", commercial_ok=True):
    print(f"  {spec.name:<14} {spec.license.name}")

print("\nModels supporting PEFT:")
print("  " + ", ".join(s.name for s in list_models(strategy="peft")))


# --------------------------------------------------------------------------
section("2. Name resolution is forgiving")

for alias in ["tabpfn-v2.6", "TABPFN_V26", "SAP-RPT-1-OSS", "Orion-MSP", "Tab2D"]:
    print(f"  {alias:<18} -> {resolve_model_name(alias)}")

# ContextTab was renamed upstream to SAP-RPT-1-OSS; both names resolve, so
# users who find the model under either name land in the right place.
try:
    resolve_model_name("TabPFNv4")
except ModelNotFoundError as exc:
    print(f"\n  Unknown names suggest a match:\n  {str(exc).splitlines()[0]}")


# --------------------------------------------------------------------------
section("3. Capability envelopes")

spec = get_model_spec("TabFM")
print(f"TabFM envelope: {spec.envelope.describe()}")
print(f"  {spec.envelope.notes}\n")

# A 14-class problem cannot run on a checkpoint with a ten-slot output head.
# This is caught in microseconds, before any weights are fetched.
try:
    check_envelope("TabFM", n_rows=1_000, n_features=20, n_classes=14)
except EnvelopeError as exc:
    print(exc)

# Resource limits warn rather than raise: exceeding them degrades quality or
# memory, but the run is still meaningful.
print("\nSoft limit (warns, does not raise):")
violations = check_envelope("Mitra", n_rows=50_000, n_features=10, mode="warn")
for violation in violations:
    print(f"  [{violation.severity}] {violation.message}")


# --------------------------------------------------------------------------
section("4. Two models that break the pattern (new in 0.2.0)")

xrfm = get_model_spec("xRFM")
print(f"xRFM   weights={xrfm.weights}")
print(f"       licence={xrfm.license.name}, commercial={xrfm.license.badge}")
print("       -> trains from scratch, so it is the only bundled model that")
print("          works air-gapped with nothing pre-staged.\n")

iltm = get_model_spec("iLTM")
print(f"iLTM   weights={iltm.weights}")
print(f"       licence={iltm.license.name}, commercial={iltm.license.badge}")
print(f"       envelope={iltm.envelope.describe()}")
# The 100-class limit is architectural and upstream does not guard it: a
# 101-class target fails inside F.one_hot with a bare torch error, deep in the
# forward pass, after the checkpoint has already downloaded.
try:
    check_envelope("iLTM", n_rows=5_000, n_features=50, n_classes=101)
except EnvelopeError as exc:
    print("       " + str(exc).splitlines()[1].strip())


# --------------------------------------------------------------------------
section("5. Weight licensing")

for name in ["TabPFNv3", "Mitra", "TabPFN"]:
    spec = get_model_spec(name)
    print(f"  {name:<12} {spec.license.name:<28} commercial: {spec.license.badge}")

print()
try:
    check_license("TabPFNv3", "commercial")
except LicenseError as exc:
    print(exc)

print(
    "\nNote the tri-state: TabPFN is 'unverified', not 'forbidden'. TabTune warns\n"
    "rather than blocking, because inventing a restriction is as wrong as\n"
    "ignoring one. Always confirm the current terms upstream."
)


# --------------------------------------------------------------------------
section("6. The full table")

frame = models_dataframe()
print(frame[["Model", "Tasks", "Max classes", "License", "Commercial"]].to_string(index=False))


# --------------------------------------------------------------------------
section("7. Enforcement inside the pipeline")

print(
    "TabularPipeline runs these checks automatically:\n\n"
    "    TabularPipeline('TabFM', envelope_mode='warn')       # default\n"
    "    TabularPipeline('TabPFNv3', license_mode='commercial')\n"
    "    TabularPipeline('MyModel', validate=False)           # unregistered model\n"
)

rng = np.random.default_rng(0)
X = pd.DataFrame(rng.normal(size=(200, 6)), columns=[f"f{i}" for i in range(6)])
y = pd.Series(rng.integers(0, 14, size=200))
print(f"Example data: {X.shape[0]} rows x {X.shape[1]} features, {y.nunique()} classes")
print("Compatible models:")
for spec in list_models(task="classification"):
    limit = spec.envelope.max_classes
    if limit is None or limit >= y.nunique():
        print(f"  {spec.name}")
