"""The real-weight benchmark harness (``benchmarks/``), exercised without weights.

Baselines, the GBM forecaster and xRFM need no downloads, so every suite can
run end to end here. The fev suite uses a local parquet task (fev reads local
paths), which runs fev's own scoring on TabTune forecasts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from harness import load_config, run_suite  # noqa: E402

pytestmark = [pytest.mark.time_series]


def _write(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_shipped_configs_are_valid():
    for path in sorted((ROOT / "benchmarks" / "configs").glob("*.yaml")):
        config = load_config(path)
        assert config.models and config.suite in ("timeseries", "fev", "tabular"), path.name


@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"suite": "nope", "models": ["SeasonalNaive"]}, "suite"),
        ({"suite": "timeseries", "models": []}, "at least one model"),
        ({"suite": "timeseries", "models": ["SeasonalNaive", "SeasonalNaive"]}, "unique"),
        ({"suite": "tabular", "models": [{"label": "x"}]}, "model_name"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path, config, match):
    with pytest.raises(ValueError, match=match):
        load_config(_write(tmp_path, config))


def test_timeseries_suite_records_results_failures_and_resumes(tmp_path):
    config = {
        "suite": "timeseries",
        "name": "smoke",
        "output": str(tmp_path / "out"),
        "device": "cpu",
        "datasets": ["synthetic:seasonal_hourly"],
        "builtin": {"n_series": 3, "length": 120},
        "horizons": [6],
        "n_windows": 1,
        "models": ["SeasonalNaive", "Drift", {"label": "Broken", "model": "NoSuchModel", "requires": "nothing"}],
    }
    out = run_suite(load_config(_write(tmp_path, config)))
    records = pd.read_csv(out / "records.csv").set_index("model")
    assert records.loc["SeasonalNaive", "status"] == "ok" and records.loc["Broken", "status"] == "failed"
    report = (out / "report.md").read_text()
    assert "SeasonalNaive" in report and "Broken" in report.split("## Failures")[1]
    assert {"config.json", "environment.json", "raw.csv"} <= {p.name for p in out.iterdir()}
    before = (out / "raw.csv").stat().st_mtime_ns
    raw_rows = len(pd.read_csv(out / "raw.csv"))
    run_suite(load_config(_write(tmp_path, {**config, "models": ["SeasonalNaive", "Drift"]})), resume=True)
    assert len(pd.read_csv(out / "raw.csv")) == raw_rows
    assert (out / "raw.csv").stat().st_mtime_ns == before
    fixed = {**config, "models": ["SeasonalNaive", "Drift", {"label": "Broken", "model": "Drift"}]}
    run_suite(load_config(_write(tmp_path, fixed)), resume=True)
    raw = pd.read_csv(out / "raw.csv")
    assert (raw.loc[raw["model"] == "Broken", "status"] == "ok").all()
    assert pd.read_csv(out / "records.csv").set_index("model").loc["Broken", "status"] == "ok"


def test_tabular_suite_scores_xrfm(tmp_path):
    config = {
        "suite": "tabular",
        "name": "tab",
        "output": str(tmp_path / "out"),
        "repeats": 1,
        "datasets": [{"sklearn": "breast_cancer"}],
        "models": [{"label": "xRFM", "model_name": "XRFM"}, {"label": "Missing", "model_name": "NoSuchModel"}],
    }
    out = run_suite(load_config(_write(tmp_path, config)))
    records = pd.read_csv(out / "records.csv").set_index("model")
    assert records.loc["xRFM", "status"] == "ok" and records.loc["xRFM", "accuracy"] > 0.85
    assert 0 < records.loc["xRFM", "roc_auc"] <= 1
    assert records.loc["Missing", "status"] == "failed"


def test_fev_suite_scores_tabtune_forecasts_with_fev(tmp_path):
    fev = pytest.importorskip("fev")
    from tabtune.TimeSeries import make_panel

    frame = make_panel(3, 24 * 8, freq="h", covariates=["promo"], seed=0).rename(columns={"item_id": "id"})
    dataset = fev.utils.convert_long_df_to_hf_dataset(frame, id_column="id", timestamp_column="timestamp")
    dataset.to_parquet(str(tmp_path / "data.parquet"))
    config = {
        "suite": "fev",
        "name": "fev_local",
        "output": str(tmp_path / "out"),
        "device": "cpu",
        "baseline": "Seasonal Naive",
        "tasks": [
            {
                "dataset_path": str(tmp_path / "data.parquet"),
                "horizon": 12,
                "num_windows": 2,
                "seasonality": 24,
                "target": "target",
                "id_column": "id",
                "known_dynamic_columns": ["promo"],
                "eval_metric": "SQL",
                "extra_metrics": ["MASE", "WQL"],
                "quantile_levels": [0.1, 0.5, 0.9],
            }
        ],
        "models": [
            {"label": "Seasonal Naive", "model": "SeasonalNaive"},
            {"label": "Drift", "model": "Drift"},
            {"label": "xRFM lags", "model": "TabularTS-XRFM", "model_params": {"features": "lags"}},
        ],
    }
    out = run_suite(load_config(_write(tmp_path, config)))
    records = pd.read_csv(out / "records.csv")
    assert set(records["model_name"]) == {"Seasonal Naive", "Drift", "xRFM lags"}
    assert (records["status"] == "ok").all() and records["SQL"].notna().all()
    quantiles = records.set_index("model_name")["quantiles"]
    assert quantiles["xRFM lags"].startswith("point") and quantiles["Drift"] == "native"
    report = (out / "report.md").read_text()
    assert "fev.leaderboard" in report and "Point-only models" in report and "xRFM lags" in report
