"""
Distillation Analysis
=====================================================
Comprehensive diagnostics for distillation quality:

1. DistillationProfiler: fidelity, diversity, agreement, latency
2. Calibration: ECE, MCE, Brier Score, reliability diagrams
3. Fairness: demographic parity, equalized odds, per-group AUC
4. Memory profiling: peak RAM, VRAM, model size
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, brier_score_loss,
    mean_squared_error, r2_score,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 1. Calibration Metrics
# ═══════════════════════════════════════════════════════════════

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error.

    Parameters
    ----------
    y_true : ndarray (N,) — binary labels (0/1)
    y_prob : ndarray (N,) — predicted P(y=1)
    n_bins : int

    Returns
    -------
    float : ECE (lower = better, 0 = perfect)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        ece += (n_bin / total) * abs(y_true[mask].mean() - y_prob[mask].mean())

    return float(ece)


def compute_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Maximum Calibration Error — worst-case bin miscalibration."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    max_ce = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        max_ce = max(max_ce, abs(y_true[mask].mean() - y_prob[mask].mean()))

    return float(max_ce)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Full calibration suite: ECE, MCE, Brier Score."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float).clip(0, 1)
    return {
        "ece": round(compute_ece(y_true, y_prob, n_bins), 4),
        "mce": round(compute_mce(y_true, y_prob, n_bins), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
    }


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibration curve data for reliability diagrams.

    Returns (bin_centers, bin_accuracies, bin_counts).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, counts = [], [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        n_bin = mask.sum()
        counts.append(n_bin)
        if n_bin > 0:
            centers.append(y_prob[mask].mean())
            accs.append(y_true[mask].mean())
        else:
            centers.append((lo + hi) / 2)
            accs.append(0.0)

    return np.array(centers), np.array(accs), np.array(counts)


def plot_reliability_diagram(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_bins: int = 10,
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram",
) -> None:
    """Plot reliability diagram comparing multiple models.

    Parameters
    ----------
    models_data : dict
        {model_name: (y_true, y_prob)} for each model
    save_path : str or None
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )
    colors = ["#534AB7", "#D85A30", "#1D9E75", "#378ADD", "#D4537E", "#639922"]

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")

    for idx, (model_name, (y_true, y_prob)) in enumerate(models_data.items()):
        centers, accs, counts = compute_calibration_curve(y_true, y_prob, n_bins)
        ece = compute_ece(y_true, y_prob, n_bins)
        color = colors[idx % len(colors)]
        ax1.plot(centers, accs, "o-", color=color, markersize=5,
                 label=f"{model_name} (ECE={ece:.3f})")
        ax2.bar(centers, counts, width=0.08, alpha=0.4, color=color)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(title)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Count")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"[Plot] Reliability diagram saved to {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# 2. Fairness Metrics
# ═══════════════════════════════════════════════════════════════

def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_feature: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute fairness metrics across sensitive attribute groups.

    Returns demographic_parity_diff, equalized_odds_diff, per_group_accuracy,
    per_group_auc, max_auc_gap.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_feature = np.asarray(sensitive_feature)
    result = {}

    # Try fairlearn
    try:
        from fairlearn.metrics import (
            demographic_parity_difference,
            equalized_odds_difference,
        )
        result["demographic_parity_diff"] = round(float(
            demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_feature)
        ), 4)
        result["equalized_odds_diff"] = round(float(
            equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_feature)
        ), 4)
    except ImportError:
        # Manual fallback
        groups = np.unique(sensitive_feature)
        rates = []
        for g in groups:
            mask = sensitive_feature == g
            if mask.sum() > 0:
                rates.append(float(np.mean(y_pred[mask] == 1)))
        result["demographic_parity_diff"] = round(max(rates) - min(rates), 4) if len(rates) >= 2 else 0.0
        result["equalized_odds_diff"] = None  # Requires fairlearn for proper computation

    # Per-group AUC
    groups = np.unique(sensitive_feature)
    per_group_acc = {}
    per_group_auc = {}
    for g in groups:
        mask = sensitive_feature == g
        if mask.sum() < 5:
            continue
        g_key = str(g)
        per_group_acc[g_key] = round(float(np.mean(y_pred[mask] == y_true[mask])), 4)
        if y_prob is not None:
            try:
                per_group_auc[g_key] = round(float(roc_auc_score(y_true[mask], y_prob[mask])), 4)
            except ValueError:
                per_group_auc[g_key] = None

    result["per_group_accuracy"] = per_group_acc
    result["per_group_auc"] = per_group_auc

    valid_aucs = [v for v in per_group_auc.values() if v is not None]
    result["max_auc_gap"] = round(max(valid_aucs) - min(valid_aucs), 4) if len(valid_aucs) >= 2 else None
    result["n_groups"] = len(groups)

    return result


# ═══════════════════════════════════════════════════════════════
# 3. Memory & Size Profiling
# ═══════════════════════════════════════════════════════════════

def profile_memory(predict_fn, X_test: pd.DataFrame, measure_gpu: bool = True) -> Dict[str, float]:
    """Profile peak memory during inference."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        baseline_ram = process.memory_info().rss / (1024 * 1024)
    except ImportError:
        baseline_ram = 0

    gpu_baseline = 0.0
    if measure_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                gpu_baseline = torch.cuda.memory_allocated() / (1024 * 1024)
        except ImportError:
            measure_gpu = False

    _ = predict_fn(X_test)

    result = {}
    try:
        import psutil
        peak_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        result["peak_cpu_ram_mb"] = round(peak_ram - baseline_ram, 2)
    except ImportError:
        pass

    if measure_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
                result["peak_gpu_vram_mb"] = round(peak_vram - gpu_baseline, 2)
        except ImportError:
            pass

    return result


def get_model_size_mb(model_or_path) -> float:
    """Get model size in MB."""
    if isinstance(model_or_path, str) and os.path.exists(model_or_path):
        return round(os.path.getsize(model_or_path) / (1024 * 1024), 3)
    import pickle
    buf = io.BytesIO()
    pickle.dump(model_or_path, buf)
    return round(buf.tell() / (1024 * 1024), 3)


# ═══════════════════════════════════════════════════════════════
# 4. DistillationProfiler (enhanced)
# ═══════════════════════════════════════════════════════════════

class DistillationProfiler:
    """Profile a fitted TabDistiller: metrics, latency, calibration, agreement.

    Usage:
        profiler = DistillationProfiler(distiller)
        report = profiler.profile(X_test, y_test)
        profiler.print_report(report)
    """

    def __init__(self, distiller):
        self.distiller = distiller

    def profile(self, X_test, y_test, sensitive_feature=None) -> dict:
        """Run full profiling suite.

        Parameters
        ----------
        X_test, y_test : test data
        sensitive_feature : ndarray or None
            If provided, adds fairness metrics to report.
        """
        X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
        y_test = np.asarray(y_test)
        d = self.distiller

        report = {"task_type": d.task_type, "teacher_names": d.teacher_names}

        # Student metrics
        report["student_metrics"] = d.evaluate(X_test, y_test)

        # Teacher metrics
        report["teacher_metrics"] = {}
        for pipe, name in zip(d.teacher_pipelines_ or [], d.teacher_names):
            try:
                t_preds = pipe.predict(X_test)
                if d.task_type == "classification":
                    report["teacher_metrics"][name] = {
                        "accuracy": round(float(accuracy_score(y_test, t_preds)), 4)
                    }
                else:
                    report["teacher_metrics"][name] = {
                        "r2": round(float(r2_score(y_test, t_preds)), 4)
                    }
            except Exception as e:
                report["teacher_metrics"][name] = {"error": str(e)}

        # Latency
        from ..exporters import benchmark_inference
        try:
            report["latency"] = benchmark_inference(d, X_test, n_runs=20)
        except Exception:
            report["latency"] = {}

        # Agreement analysis
        if d.task_type == "classification" and d.teacher_pipelines_:
            try:
                s_preds = d.predict(X_test)
                for pipe, name in zip(d.teacher_pipelines_, d.teacher_names):
                    t_preds = pipe.predict(X_test)
                    agreement = np.mean(np.asarray(s_preds) == np.asarray(t_preds))
                    report[f"agreement_{name}"] = round(float(agreement), 4)
            except Exception:
                pass

        # Soft label stats
        if d.soft_labels_ is not None and d.soft_labels_.ndim == 2:
            sl = d.soft_labels_
            entropy = -np.sum(sl * np.log(np.clip(sl, 1e-10, 1.0)), axis=1)
            report["soft_label_stats"] = {
                "mean_entropy": round(float(np.mean(entropy)), 4),
                "std_entropy": round(float(np.std(entropy)), 4),
                "mean_max_prob": round(float(np.mean(np.max(sl, axis=1))), 4),
            }

        # Calibration (binary classification only)
        if d.task_type == "classification" and d.n_classes_ == 2:
            try:
                probas = d.predict_proba(X_test)
                y_enc = y_test
                if not np.all(np.isin(y_test, [0, 1])):
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_enc = le.fit_transform(y_test)
                report["calibration"] = compute_calibration_metrics(y_enc, probas[:, 1])
            except Exception as e:
                report["calibration"] = {"error": str(e)}

        # Fairness (if sensitive feature provided)
        if sensitive_feature is not None and d.task_type == "classification":
            try:
                preds = d.predict(X_test)
                y_enc = y_test
                if not np.all(np.isin(y_test, [0, 1])):
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_enc = le.fit_transform(y_test)
                    preds_enc = le.transform(preds)
                else:
                    preds_enc = preds
                probas = d.predict_proba(X_test)
                y_prob = probas[:, 1] if probas.shape[1] == 2 else None
                report["fairness"] = compute_fairness_metrics(
                    y_enc, preds_enc, sensitive_feature, y_prob=y_prob
                )
            except Exception as e:
                report["fairness"] = {"error": str(e)}

        # Model size
        try:
            report["model_size_mb"] = get_model_size_mb(d.student_)
        except Exception:
            pass

        report["fit_time_s"] = d._fit_time
        return report

    def print_report(self, report: dict):
        """Pretty-print profiling report."""
        print("\n" + "=" * 60)
        print("  TabDistiller Profiling Report")
        print("=" * 60)

        if "student_metrics" in report:
            print("\n  Student Metrics:")
            for k, v in report["student_metrics"].items():
                print(f"    {k}: {v}")

        if "teacher_metrics" in report:
            print("\n  Teacher Metrics:")
            for name, metrics in report["teacher_metrics"].items():
                print(f"    {name}: {metrics}")

        if "latency" in report and report["latency"]:
            lat = report["latency"]
            print(f"\n  Latency: {lat.get('mean_ms', '?')}ms mean, "
                  f"{lat.get('throughput_per_sec', '?')} samples/sec")

        for key in report:
            if key.startswith("agreement_"):
                name = key.replace("agreement_", "")
                print(f"  Agreement with {name}: {report[key] * 100:.1f}%")

        if "soft_label_stats" in report:
            sl = report["soft_label_stats"]
            print(f"\n  Soft Labels: entropy={sl['mean_entropy']:.4f}, "
                  f"max_prob={sl['mean_max_prob']:.4f}")

        if "calibration" in report and "error" not in report["calibration"]:
            cal = report["calibration"]
            print(f"\n  Calibration: ECE={cal.get('ece')}, MCE={cal.get('mce')}, "
                  f"Brier={cal.get('brier_score')}")

        if "fairness" in report and "error" not in report.get("fairness", {}):
            fair = report["fairness"]
            print(f"\n  Fairness: DP_diff={fair.get('demographic_parity_diff')}, "
                  f"EO_diff={fair.get('equalized_odds_diff')}, "
                  f"AUC_gap={fair.get('max_auc_gap')}")

        if "model_size_mb" in report:
            print(f"\n  Model size: {report['model_size_mb']:.2f} MB")

        if report.get("fit_time_s"):
            print(f"  Fit time: {report['fit_time_s']:.1f}s")

        print("\n" + "=" * 60)

    def save_report(self, report: dict, path: str):
        """Save report to JSON."""
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        clean = json.loads(json.dumps(report, default=_convert))
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        logger.info(f"[Profiler] Report saved to {path}")


# ═══════════════════════════════════════════════════════════════
# 5. LaTeX Helpers
# ═══════════════════════════════════════════════════════════════

def format_mean_std(mean: float, std: float, precision: int = 4) -> str:
    """Format as 'mean ± std' for LaTeX."""
    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"


def results_to_latex(
    df: pd.DataFrame,
    metric_col: str,
    index_col: str = "model",
    columns_col: str = "dataset",
    caption: str = "Results",
    label: str = "tab:results",
    bold_best: bool = True,
) -> str:
    """Generate LaTeX table from pivoted results, bolding best per column."""
    pivot = df.pivot_table(values=metric_col, index=index_col, columns=columns_col, aggfunc="mean")

    formatted = pivot.copy()
    for col in formatted.columns:
        best_val = formatted[col].max()
        for idx in formatted.index:
            val = formatted.loc[idx, col]
            if pd.isna(val):
                formatted.loc[idx, col] = "—"
            elif bold_best and abs(val - best_val) < 0.0001:
                formatted.loc[idx, col] = f"\\textbf{{{val:.4f}}}"
            else:
                formatted.loc[idx, col] = f"{val:.4f}"

    return formatted.to_latex(
        escape=False, caption=caption, label=label,
        column_format="l" + "c" * len(formatted.columns),
    )
