"""
Distillation Exporters
========================
ONNX export and inference benchmarking for distilled students.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def export_onnx(distiller, path: str, opset_version: int = 13) -> dict:
    """Export MLP student to ONNX format.

    Parameters
    ----------
    distiller : TabDistiller (fitted, with MLP student)
    path : str
        Output .onnx file path.
    opset_version : int

    Returns
    -------
    dict with size_mb, n_params, path
    """
    try:
        import torch
        import onnx
    except ImportError:
        raise ImportError(
            "ONNX export requires: pip install onnx onnxruntime\n"
            "These are optional dependencies not included in base TabTune."
        )

    student = distiller.student_
    if not hasattr(student, 'model_'):
        raise ValueError("ONNX export only supports MLP students.")

    model = student.model_
    model.eval()

    n_features = student.n_features
    dummy = torch.randn(1, n_features)

    torch.onnx.export(
        model, dummy, path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=opset_version,
    )

    # Compute stats
    n_params = sum(p.numel() for p in model.parameters())
    import os
    size_mb = os.path.getsize(path) / (1024 * 1024)

    logger.info(f"[ONNX] Exported to {path}: {size_mb:.2f}MB, {n_params:,} params")
    return {"path": path, "size_mb": size_mb, "n_params": n_params}


def quantize_student(distiller):
    """INT8 dynamic quantization of MLP student (PyTorch).

    Replaces the student's model_ in-place with quantized version.
    """
    try:
        import torch
        import torch.quantization
    except ImportError:
        raise ImportError("Quantization requires PyTorch.")

    student = distiller.student_
    if not hasattr(student, 'model_'):
        raise ValueError("Quantization only supports MLP students.")

    model = student.model_
    model.eval()
    model.cpu()

    quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    student.model_ = quantized
    student.device = "cpu"
    student.use_amp = False

    logger.info("[Quantize] INT8 dynamic quantization applied")
    return quantized


def export_torchscript(distiller, path: str) -> dict:
    """Export MLP student to TorchScript format via tracing.

    Parameters
    ----------
    distiller : TabDistiller (fitted, with MLP student)
    path : str
        Output .pt file path.

    Returns
    -------
    dict with size_mb, n_params, path
    """
    try:
        import torch
    except ImportError:
        raise ImportError("TorchScript export requires PyTorch.")

    student = distiller.student_
    if not hasattr(student, 'model_'):
        raise ValueError("TorchScript export only supports MLP students.")

    model = student.model_
    model.eval()
    model.cpu()

    n_features = student.n_features
    dummy = torch.randn(2, n_features)  # batch=2 for BatchNorm

    traced = torch.jit.trace(model, dummy)
    traced.save(path)

    n_params = sum(p.numel() for p in model.parameters())
    import os
    size_mb = os.path.getsize(path) / (1024 * 1024)

    logger.info(f"[TorchScript] Exported to {path}: {size_mb:.2f}MB, {n_params:,} params")
    return {"path": path, "size_mb": size_mb, "n_params": n_params}


def benchmark_inference(distiller, X_test, n_runs: int = 50) -> dict:
    """Benchmark student inference latency.

    Returns
    -------
    dict with mean_ms, std_ms, p50_ms, p99_ms, throughput_per_sec
    """
    import pandas as pd

    X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
    n_samples = len(X_test)

    # Warmup
    for _ in range(3):
        distiller.predict(X_test)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        distiller.predict(X_test)
        times.append((time.perf_counter() - t0) * 1000)  # ms

    times = np.array(times)
    stats = {
        "mean_ms": round(float(np.mean(times)), 2),
        "std_ms": round(float(np.std(times)), 2),
        "p50_ms": round(float(np.median(times)), 2),
        "p99_ms": round(float(np.percentile(times, 99)), 2),
        "throughput_per_sec": round(n_samples / (np.mean(times) / 1000), 0),
        "n_samples": n_samples,
        "n_runs": n_runs,
    }
    logger.info(f"[Benchmark] {stats['mean_ms']:.1f}ms mean, "
                f"{stats['throughput_per_sec']:,.0f} samples/sec")
    return stats