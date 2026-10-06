"""TimeSeriesLeaderboard, TimeSeriesEnsemble, TimeSeriesBenchmark, model selection and the CLI."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tabtune.models.TimeSeries.base import AdapterOutput, TSFMAdapter
from tabtune.registry import (
    TimeSeriesModelSpec,
    get_time_series_model_spec,
    register_time_series_model,
)
from tabtune.TimeSeries import (
    TimeSeriesBenchmark,
    TimeSeriesEnsemble,
    TimeSeriesLeaderboard,
    TimeSeriesPipeline,
    TimeSeriesSchema,
    make_panel,
    split_horizon,
)
from tabtune.TimeSeries.cli import main as cli

pytestmark = [pytest.mark.unit, pytest.mark.time_series]

SCHEMA = TimeSeriesSchema(target="target", item_id="item_id")
PARAMS = {"prediction_length": 24}


class BrokenAdapter(TSFMAdapter):
    def load(self):
        raise RuntimeError("weights are missing")

    def forecast(self, panel, config):
        raise AssertionError


class OffsetAdapter(TSFMAdapter):
    """Seasonal naive shifted by ``model_params['offset']``; point forecasts only."""

    def load(self):
        self._model = "ok"

    def forecast(self, panel, config):
        offset = float(self.model_params.get("offset", 0.0))
        point = np.stack([np.resize(v[-24:], config.prediction_length) for v in panel.values]) + offset
        return AdapterOutput(point=point)


@pytest.fixture(autouse=True)
def fake_models(isolated_ts_registry):
    for name, adapter in (("Broken", BrokenAdapter), ("Offset", OffsetAdapter)):
        register_time_series_model(
            TimeSeriesModelSpec(name=name, family="test", adapter=adapter, native_missing=True)
        )


@pytest.fixture(scope="module")
def data():
    frame = make_panel(4, 240, freq="h", seed=2)
    return split_horizon(frame, SCHEMA, 24)


def test_backtest_leaderboard_ranks_and_records_failures(data):
    history, _ = data
    board = TimeSeriesLeaderboard(history, SCHEMA, forecast_params=PARAMS, windows=2)
    board.add_models(["SeasonalNaive", "Naive", "Broken"])
    table = board.run(display=False)
    assert list(table["Model"]) == ["SeasonalNaive", "Naive", "Broken"]
    assert list(table["Rank"]) == [1, 2, 3]
    assert table.loc[2, "Status"] == "failed" and "weights are missing" in table.loc[2, "Error"]
    assert board.best().model_name == "SeasonalNaive"
    assert board.mode == "backtest" and board._rank_by == "wql"
    assert {"mase", "wql", "coverage_80", "fit_s", "predict_s", "License"} <= set(table.columns)


def test_backtest_mode_trains_without_the_scored_windows(data):
    history, _ = data
    board = TimeSeriesLeaderboard(history, SCHEMA, forecast_params=PARAMS, windows=3)
    lengths = board._training_frame().groupby("item_id").size().unique().tolist()
    assert lengths == [len(history) // 4 - 3 * 24]


def test_holdout_leaderboard_scores_the_actuals(data):
    history, actual = data
    board = TimeSeriesLeaderboard(history, SCHEMA, forecast_params=PARAMS, df_actual=actual)
    table = board.add_models(["SeasonalNaive", "Mean"]).run(display=False)
    expected = TimeSeriesPipeline("SeasonalNaive", forecast_params=PARAMS).fit(history, SCHEMA).evaluate(actual)
    assert table.loc[0, "Model"] == "SeasonalNaive"
    assert table.loc[0, "mase"] == pytest.approx(expected["mase"])


def test_add_all_skips_models_that_cannot_read_the_schema(data):
    history, _ = data
    multivariate = TimeSeriesSchema(target=("a", "b"), item_id="item_id")
    board = TimeSeriesLeaderboard(history, multivariate, forecast_params=PARAMS)
    board.add_all()
    names = {c["model_name"] for c in board.models_to_run}
    assert names and all(get_time_series_model_spec(n).supports_multivariate for n in names)
    board = TimeSeriesLeaderboard(history, SCHEMA, forecast_params=PARAMS).add_all(commercial_ok=True)
    names = {c["model_name"] for c in board.models_to_run}
    assert "SeasonalNaive" in names and "TimesFM3" not in names
    assert not any(n.startswith("TabularTS-") for n in names)


def test_leaderboard_exports(data, tmp_path):
    history, _ = data
    board = TimeSeriesLeaderboard(history, SCHEMA, forecast_params=PARAMS)
    board.add_model("Offset", model_params={"offset": 1.0}, label="Offset+1").add_model("SeasonalNaive")
    table = board.run(display=False)
    assert list(table["Model"]) == ["SeasonalNaive", "Offset+1"]
    assert board._rank_by == "mase"
    assert "| SeasonalNaive" in board.to_markdown()
    assert pd.read_csv(board.to_csv(tmp_path / "board.csv")).shape[0] == 2
    payload = json.loads(board.to_json(tmp_path / "board.json").read_text())
    entries = {e["Model"]: e for e in payload["entries"]}
    assert payload["mode"] == "backtest"
    assert entries["Offset+1"]["pipeline_kwargs"]["model_params"] == {"offset": 1.0}
    with pytest.raises(ValueError, match="not a reported metric"):
        board.to_frame("crps")


@pytest.mark.parametrize("strategy", ["greedy_selection", "weighted_averaging", "stacking", "best", "mean", "median"])
def test_ensemble_strategies(data, strategy):
    history, actual = data
    ensemble = TimeSeriesEnsemble(
        ["SeasonalNaive", "Naive", "Drift"], strategy, forecast_params=PARAMS, verbose=False
    ).fit(history, SCHEMA)
    assert sum(ensemble.weights_.values()) == pytest.approx(1.0)
    forecast = ensemble.predict()
    assert forecast.point.shape == (4, 24) and forecast.quantiles.shape == (4, 24, 3)
    assert (np.diff(forecast.quantiles, axis=-1) >= 0).all()
    assert forecast.metadata["strategy"] == strategy
    assert np.isfinite(ensemble.evaluate(actual)["mase"])
    if strategy in ("greedy_selection", "best"):
        assert ensemble.weights_["SeasonalNaive"] == max(ensemble.weights_.values())


def test_ensemble_mixes_point_only_members_and_centres_the_band(data):
    history, _ = data
    ensemble = TimeSeriesEnsemble(
        ["SeasonalNaive", {"model_name": "Offset", "model_params": {"offset": 0.5}, "label": "shifted"}],
        "mean",
        forecast_params=PARAMS,
        verbose=False,
    ).fit(history, SCHEMA)
    assert ensemble.loss_name_ == "wape"
    forecast = ensemble.predict()
    np.testing.assert_allclose(forecast.quantiles[..., 1], forecast.point, atol=1e-9)
    leaderboard = ensemble.get_leaderboard()
    assert list(leaderboard.columns) == ["member", "validation_wape", "weight", "fit_s"]


def test_ensemble_survives_a_failing_member_and_rejects_bad_settings(data):
    history, _ = data
    ensemble = TimeSeriesEnsemble(["SeasonalNaive", "Broken"], forecast_params=PARAMS, verbose=False)
    ensemble.fit(history, SCHEMA)
    assert list(ensemble.pipelines_) == ["SeasonalNaive"]
    with pytest.raises(ValueError, match="ensemble_strategy"):
        TimeSeriesEnsemble(["Naive"], "vote", forecast_params=PARAMS)
    with pytest.raises(ValueError, match="unique"):
        TimeSeriesEnsemble(["Naive", "Naive"], forecast_params=PARAMS)
    with pytest.raises(RuntimeError, match="No ensemble member"):
        TimeSeriesEnsemble(["Broken"], forecast_params=PARAMS, verbose=False).fit(history, SCHEMA)


def test_select_returns_the_best_fitted_pipeline(data):
    history, _ = data
    pipe = TimeSeriesPipeline.select(
        history, SCHEMA, PARAMS, candidates=["Naive", "SeasonalNaive", "Mean"], windows=2
    )
    assert pipe.model_name == "SeasonalNaive" and pipe._is_fitted
    assert list(pipe.selection_["Model"])[0] == "SeasonalNaive"


def test_benchmark_aggregates_over_datasets():
    results = TimeSeriesBenchmark(["Naive", "Drift"], windows=1).run()
    board = results.leaderboard("mase")
    assert set(board["model"]) == {"Naive", "Drift", "SeasonalNaive"}
    naive_row = board.set_index("model").loc["SeasonalNaive"]
    assert naive_row["skill"] == pytest.approx(0.0)
    assert board["n_tasks"].iloc[0] == 5
    tests = results.pairwise_tests("mase")
    assert len(tests) == 3 and (tests["p_holm"] >= tests["p_value"]).all()
    assert "## Leaderboard" in results.to_markdown()


def test_benchmark_reports_failures_and_saves(tmp_path):
    frame = make_panel(3, 120, freq="h", seed=0)
    datasets = {"one": {"df": frame, "schema": SCHEMA, "prediction_length": 12}}
    with pytest.warns(UserWarning, match="unreliable"):
        results = TimeSeriesBenchmark(["Naive", "Broken"], datasets, windows=1).run()
        board = results.leaderboard()
    assert board.set_index("model").loc["Broken", "failure_rate"] == 1.0
    assert list(results.failures()["model"]) == ["Broken"]
    with pytest.warns(UserWarning):
        results.save(tmp_path / "bench")
    assert {p.name for p in (tmp_path / "bench").iterdir()} == {
        "raw.csv",
        "leaderboard.csv",
        "report.md",
        "config.json",
    }


@pytest.fixture
def csv(tmp_path):
    frame = make_panel(2, 120, freq="h", seed=0)
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return path, frame


def test_cli_lists_and_describes_models(capsys):
    assert cli(["list-models", "--task", "embedding", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["model"] for r in rows} >= {"TiRex", "TiRex2", "TimeMoE"}
    assert cli(["info", "tirex-2"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "TiRex2"


def test_cli_forecast_evaluate_anomalies_impute(csv, tmp_path, capsys):
    path, frame = csv
    out = tmp_path / "forecast.csv"
    assert cli(["forecast", "--data", str(path), "--horizon", "6", "--output", str(out)]) == 0
    forecast = pd.read_csv(out)
    assert len(forecast) == 12 and {"item_id", "timestamp", "point", "0.1", "0.9"} <= set(forecast.columns)
    capsys.readouterr()
    assert cli(["evaluate", "--data", str(path), "--horizon", "6", "--json"]) == 0
    assert "mase" in json.loads(capsys.readouterr().out)
    anomalies = tmp_path / "anomalies.csv"
    assert cli(["anomalies", "--data", str(path), "--output", str(anomalies)]) == 0
    assert "is_anomaly" in pd.read_csv(anomalies).columns
    gappy = frame.copy()
    gappy.loc[[10, 11], "target"] = np.nan
    gappy_path = tmp_path / "gappy.csv"
    gappy.to_csv(gappy_path, index=False)
    filled = tmp_path / "filled.csv"
    assert cli(["impute", "--data", str(gappy_path), "--output", str(filled)]) == 0
    assert pd.read_csv(filled)["target"].notna().all()


def test_cli_benchmark_and_group_entry_point(csv, tmp_path, capsys):
    from tabtune.cli import main as tabtune_main

    path, _ = csv
    assert tabtune_main(["timeseries", "benchmark", "--models", "Naive", "--data", str(path), "--horizon", "6", "--windows", "1"]) == 0
    assert "SeasonalNaive" in capsys.readouterr().out
    assert tabtune_main(["timeseries", "benchmark", "--models", "Naive", "--data", str(path)]) == 2
    assert tabtune_main(["nope"]) == 2
