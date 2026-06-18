"""Compatibility shims for the vendored TabPFN v3 tree.

Replaces the external `tabpfn_common_utils` dependency with internal no-ops so
the vendored package is hermetic (no telemetry, no network callbacks).
"""
