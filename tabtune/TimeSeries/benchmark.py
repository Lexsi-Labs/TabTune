"""Benchmark time series pipelines across datasets.

Each dataset is a :class:`~tabtune.TimeSeries.leaderboard.TimeSeriesLeaderboard`
in backtest mode: every configuration is fitted without the last ``windows``
forecast windows and scored on rolling-origin forecasts over them. Results
are aggregated as fev-bench does (Shchur et al., 2025):

* **skill** = ``1 - geomean(clip(error / baseline error, 0.01, 100))`` over
  tasks, relative to seasonal naive, with a bootstrap interval over tasks;
* **win rate**: the fraction of tasks a model beats each other model (ties
  count one half), averaged over opponents;
* **pairwise tests**: Wilcoxon signed-rank tests on task errors with Holm's
  correction.

A task counts towards the ranking only if every model succeeded on it.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..logger import logged_operation, track
from .data import make_panel
from .metrics import LOWER_IS_BETTER, season_length
from .schema import TimeSeriesSchema

logger = logging.getLogger(__name__)

__all__ = ["TimeSeriesBenchmark", "BenchmarkResults", "builtin_datasets"]

_MIN_TASKS = 5
_SCHEMA = TimeSeriesSchema(target="target", item_id="item_id")


def builtin_datasets(n_series: int = 8, length: int = 400, seed: int = 0) -> dict[str, dict[str, Any]]:
    """Synthetic smoke-test datasets (not a substitute for GIFT-Eval or fev-bench)."""
    kinds = [
        ("seasonal_hourly", "h", "seasonal", 24),
        ("trend_daily", "D", "trend", 7),
        ("random_walk_daily", "D", "random_walk", 7),
        ("multi_seasonal_hourly", "h", "multi_seasonal", 24),
        ("intermittent_daily", "D", "intermittent", 7),
    ]
    return {
        f"synthetic:{name}": {
            "df": make_panel(n_series, length, freq=freq, kind=kind, season=season, seed=seed + i),
            "schema": _SCHEMA,
            "prediction_length": min(48, max(4, season_length(freq, natural=True))),
        }
        for i, (name, freq, kind, season) in enumerate(kinds)
    }


@dataclass
class BenchmarkResults:
    """Per-task records with aggregation.

    Attributes:
        raw: One row per (dataset, model): status, metrics and runtimes.
        baseline: Model label that skill scores are relative to.
        config: The benchmark settings.
    """

    raw: pd.DataFrame
    baseline: str = "SeasonalNaive"
    config: dict[str, Any] = field(default_factory=dict)

    def task_scores(self, metric: str = "mase") -> pd.DataFrame:
        """Datasets by models; failed runs are ``NaN``."""
        ok = self.raw[self.raw["status"] == "ok"]
        if metric not in ok:
            raise KeyError(f"Metric {metric!r} was not recorded")
        return ok.pivot_table(index="dataset", columns="model", values=metric, aggfunc="mean")

    def complete_tasks(self, metric: str = "mase") -> pd.DataFrame:
        """Tasks on which every model that ever succeeded has a score."""
        scores = self.task_scores(metric)
        ranked = [m for m in scores.columns if scores[m].notna().any()]
        return scores[ranked].dropna(axis=0, how="any")

    def leaderboard(
        self,
        metric: str = "mase",
        *,
        clip: tuple[float, float] = (0.01, 100.0),
        n_bootstrap: int = 1000,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Skill, win rate, mean rank, failure rate and runtime per model, best first."""
        complete = self.complete_tasks(metric)
        n_tasks = len(complete)
        if n_tasks < _MIN_TASKS:
            warnings.warn(
                f"Only {n_tasks} task(s) on which every model succeeded: the ranking is unreliable.",
                UserWarning,
                stacklevel=2,
            )
        models = list(complete.columns)
        sign = 1.0 if LOWER_IS_BETTER.get(metric, True) else -1.0
        values = complete.to_numpy(dtype=float) * sign
        rng = np.random.default_rng(seed)
        boots = rng.integers(0, max(n_tasks, 1), size=(n_bootstrap, max(n_tasks, 1)))

        def skill(col: np.ndarray, base: np.ndarray) -> float:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(base == 0, np.where(col == 0, 1.0, np.inf), col / base)
            ratio = np.clip(ratio, *clip)
            return float(1.0 - np.exp(np.mean(np.log(ratio)))) if ratio.size else float("nan")

        failure = self.raw.assign(failed=self.raw["status"] != "ok").groupby("model")["failed"].mean()
        runtime = self.raw.groupby("model")[["fit_seconds", "predict_seconds"]].median()
        ranks = pd.DataFrame(values, columns=models).rank(axis=1) if n_tasks else None
        base = models.index(self.baseline) if self.baseline in models else None
        rows = []
        for j, model in enumerate(models):
            row: dict[str, Any] = {"model": model, "n_tasks": n_tasks}
            if base is not None and n_tasks:
                col, ref = np.abs(values[:, j]), np.abs(values[:, base])
                if sign < 0:
                    col, ref = ref, col
                row["skill"] = skill(col, ref)
                samples = [skill(col[b], ref[b]) for b in boots]
                row["skill_ci_low"], row["skill_ci_high"] = np.nanquantile(samples, [0.025, 0.975])
            others = [k for k in range(len(models)) if k != j]
            if others and n_tasks:
                row["win_rate"] = float(
                    np.mean([np.mean((values[:, j] < values[:, k]) + 0.5 * (values[:, j] == values[:, k])) for k in others])
                )
                row["mean_rank"] = float(ranks[model].mean())
            row[f"mean_{metric}"] = float(complete[model].mean()) if n_tasks else float("nan")
            row["failure_rate"] = float(failure.get(model, np.nan))
            row["median_fit_s"] = float(runtime.loc[model, "fit_seconds"])
            row["median_predict_s"] = float(runtime.loc[model, "predict_seconds"])
            rows.append(row)
        for model in sorted(set(self.raw["model"]) - set(models)):
            rows.append({"model": model, "n_tasks": n_tasks, "failure_rate": float(failure.get(model, 1.0))})
        board = pd.DataFrame(rows)
        key = "skill" if "skill" in board else "win_rate" if "win_rate" in board else "model"
        return board.sort_values(key, ascending=key == "model", na_position="last").reset_index(drop=True)

    def pairwise_tests(self, metric: str = "mase") -> pd.DataFrame:
        """Wilcoxon signed-rank p-values for every pair of models, Holm-adjusted."""
        from scipy.stats import wilcoxon

        complete = self.complete_tasks(metric)
        models = list(complete.columns)
        pairs, p_values = [], []
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                diff = complete[a] - complete[b]
                p = float(wilcoxon(complete[a], complete[b]).pvalue) if (diff != 0).any() else 1.0
                pairs.append((a, b, float(diff.median())))
                p_values.append(p)
        order = np.argsort(p_values)
        adjusted = np.empty(len(p_values))
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
            adjusted[index] = running
        return pd.DataFrame(
            [
                {"model_a": a, "model_b": b, f"median_diff_{metric}": d, "p_value": p, "p_holm": h}
                for (a, b, d), p, h in zip(pairs, p_values, adjusted, strict=True)
            ]
        )

    def failures(self) -> pd.DataFrame:
        """The failed runs and their errors."""
        return self.raw[self.raw["status"] != "ok"][["dataset", "model", "error"]].reset_index(drop=True)

    def to_markdown(self, metric: str = "mase") -> str:
        """The leaderboard and per-dataset scores as Markdown."""
        board = self.leaderboard(metric).to_markdown(index=False, floatfmt=".3f")
        per_dataset = self.task_scores(metric).to_markdown(floatfmt=".3f")
        return f"## Leaderboard ({metric})\n\n{board}\n\n## Per dataset ({metric})\n\n{per_dataset}\n"

    def save(self, path: str | Path, metric: str = "mase") -> None:
        """Write ``raw.csv``, ``leaderboard.csv``, ``report.md`` and ``config.json`` to ``path``."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self.raw.to_csv(out / "raw.csv", index=False)
        self.leaderboard(metric).to_csv(out / "leaderboard.csv", index=False)
        (out / "report.md").write_text(self.to_markdown(metric))
        (out / "config.json").write_text(json.dumps(self.config, indent=2, default=str))


class TimeSeriesBenchmark:
    """Run several configurations over several datasets.

    Args:
        models: Model names, or dicts of ``TimeSeriesPipeline`` arguments
            with ``model_name`` and an optional ``label``.
        datasets: Name to ``{"df", "schema", "prediction_length"}`` (plus
            optional ``quantile_levels``). ``None`` uses
            :func:`builtin_datasets`.
        windows: Backtest windows per series.
        baseline: Model label skill is relative to; added when missing.
        quantile_levels: Default quantile levels.
    """

    def __init__(
        self,
        models: Sequence[str | Mapping[str, Any]],
        datasets: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        windows: int = 2,
        baseline: str = "SeasonalNaive",
        quantile_levels: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    ) -> None:
        self.models = [dict(m) if isinstance(m, Mapping) else {"model_name": m} for m in models]
        labels = [m.get("label", m["model_name"]) for m in self.models]
        if baseline not in labels:
            self.models.append({"model_name": baseline})
        self.datasets = dict(datasets) if datasets is not None else builtin_datasets()
        self.windows = windows
        self.baseline = baseline
        self.quantile_levels = list(quantile_levels)

    @logged_operation("benchmark")
    def run(self, *, progress: Callable[[str, str], None] | None = None) -> BenchmarkResults:
        """Run every configuration on every dataset."""
        from .leaderboard import TimeSeriesLeaderboard

        records = []
        logger.info(
            "[TimeSeriesBenchmark] Starting benchmark: %d dataset(s) x %d model(s)",
            len(self.datasets),
            len(self.models),
        )
        for name, spec in track(self.datasets.items(), logger=logger, description="Datasets completed"):
            logger.info("[TimeSeriesBenchmark] Running dataset %s", name)
            forecast_params = {
                "prediction_length": int(spec["prediction_length"]),
                "quantile_levels": list(spec.get("quantile_levels", self.quantile_levels)),
            }
            board = TimeSeriesLeaderboard(
                spec["df"], spec["schema"], forecast_params=forecast_params, windows=self.windows
            )
            for config in self.models:
                kwargs = {k: v for k, v in config.items() if k not in ("model_name", "label")}
                board.add_model(config["model_name"], label=config.get("label", config["model_name"]), **kwargs)
            board.run(
                display=False,
                progress=None if progress is None else (lambda i, n, label, _n=name: progress(_n, label)),
            )
            for entry in board.entries:
                records.append(
                    {
                        "dataset": name,
                        "model": entry.label,
                        "status": "ok" if entry.ok else "failed",
                        "fit_seconds": entry.fit_seconds,
                        "predict_seconds": entry.predict_seconds,
                        "error": entry.error,
                        **entry.metrics,
                    }
                )
            failed = sum(not entry.ok for entry in board.entries)
            logger.info(
                "[TimeSeriesBenchmark] %s complete (%d ok, %d failed)",
                name,
                len(board.entries) - failed,
                failed,
            )
        logger.info("[TimeSeriesBenchmark] Benchmark complete: %d result rows", len(records))
        return BenchmarkResults(
            pd.DataFrame(records),
            baseline=self.baseline,
            config={
                "models": self.models,
                "datasets": list(self.datasets),
                "windows": self.windows,
                "quantile_levels": self.quantile_levels,
            },
        )
