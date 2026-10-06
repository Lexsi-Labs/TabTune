"""TiRex adapter on a tiny random checkpoint (2 blocks, patch size 4, 32-step context)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabtune.registry import ConfigError, get_time_series_model_spec
from tabtune.TimeSeries import TimeSeriesPipeline, TimeSeriesSchema, make_panel

pytestmark = [pytest.mark.unit, pytest.mark.time_series]

torch = pytest.importorskip("torch")

SCHEMA = TimeSeriesSchema(target="target", item_id="item_id")


def _pipe(checkpoint, horizon=6, **kwargs):
    model_params = {"checkpoint": checkpoint, "device": "cpu", **kwargs.pop("model_params", {})}
    return TimeSeriesPipeline(
        "TiRex", model_params=model_params, forecast_params={"prediction_length": horizon}, **kwargs
    )


def _series(frame, item):
    return frame.loc[frame["item_id"] == item, "target"].to_numpy(np.float32)


def test_registry_entry():
    spec = get_time_series_model_spec("TiRex-1.1")
    assert spec.name == "TiRex" and spec.dependency_extra is None
    assert spec.native_missing and not spec.supports_multivariate
    assert {"embedding", "forecasting", "anomaly_detection", "imputation"} <= spec.tasks
    assert spec.strategies == {"inference", "finetune", "peft"}
    assert spec.license.requires_attribution and spec.license.commercial_use_ok is None


def test_no_third_party_model_package_is_imported():
    import sys

    import tabtune.models.tirex  # noqa: F401

    assert not {"xlstm", "flashrnn", "mlstm_kernels", "tirex"} & set(sys.modules)


def test_forecast_matches_the_model(tiny_tirex_checkpoint):
    frame = make_panel(3, 60, freq="h", seed=0)
    pipe = _pipe(tiny_tirex_checkpoint).fit(frame, SCHEMA)
    forecast = pipe.predict()
    assert forecast.point.shape == (3, 6) and forecast.quantiles.shape == (3, 6, 3)
    assert forecast.metadata["point_forecast"] == "median"
    grid, median = pipe.adapter_._model.forecast([_series(frame, "series_1")], prediction_length=6)
    np.testing.assert_allclose(forecast.point[1], median[0].numpy(), atol=1e-6)
    np.testing.assert_allclose(forecast.quantiles[1], np.sort(grid[0].numpy()[:, [0, 4, 8]], axis=-1), atol=1e-6)


def test_levels_between_native_ones_are_interpolated(tiny_tirex_checkpoint):
    frame = make_panel(1, 40, freq="h", seed=0)
    pipe = TimeSeriesPipeline(
        "TiRex",
        model_params={"checkpoint": tiny_tirex_checkpoint, "device": "cpu"},
        forecast_params={"prediction_length": 5, "quantile_levels": [0.25, 0.5]},
    ).fit(frame, SCHEMA)
    grid, _ = pipe.adapter_._model.forecast([_series(frame, "series_0")], prediction_length=5)
    grid = grid[0].numpy()
    expected = np.sort(np.stack([(grid[:, 1] + grid[:, 2]) / 2, grid[:, 4]], axis=-1), axis=-1)
    np.testing.assert_allclose(pipe.predict().quantiles[0], expected, atol=1e-6)


def test_levels_outside_the_grid_warn(tiny_tirex_checkpoint):
    frame = make_panel(1, 40, freq="h", seed=0)
    pipe = TimeSeriesPipeline(
        "TiRex",
        model_params={"checkpoint": tiny_tirex_checkpoint, "device": "cpu"},
        forecast_params={"prediction_length": 5, "quantile_levels": [0.025, 0.5, 0.975]},
    ).fit(frame, SCHEMA)
    with pytest.warns(UserWarning, match="nearest"):
        pipe.predict()


def test_missing_values_are_passed_through(tiny_tirex_checkpoint):
    frame = make_panel(2, 50, freq="h", seed=0)
    frame.loc[[5, 6, 7, 60], "target"] = np.nan
    pipe = _pipe(tiny_tirex_checkpoint).fit(frame, SCHEMA)
    forecast = pipe.predict()
    assert np.isfinite(forecast.point).all()
    _, median = pipe.adapter_._model.forecast([_series(frame, "series_0")], prediction_length=6)
    np.testing.assert_allclose(forecast.point[0], median[0].numpy(), atol=1e-6)


def test_ragged_batch_matches_single_series(tiny_tirex_checkpoint):
    frame = make_panel(2, 60, freq="h", seed=0)
    frame = frame[~((frame["item_id"] == "series_1") & (frame.index % 60 < 25))]
    together = _pipe(tiny_tirex_checkpoint).fit(frame, SCHEMA).predict()
    alone = _pipe(tiny_tirex_checkpoint).fit(frame[frame["item_id"] == "series_1"], SCHEMA).predict()
    np.testing.assert_allclose(alone.point[0], together.point[1], atol=1e-5)


def test_embeddings_match_upstream_and_do_not_depend_on_the_batch(tiny_tirex_checkpoint):
    frame = make_panel(3, 50, freq="h", seed=0)
    frame = frame[~((frame["item_id"] == "series_2") & (frame.index % 50 < 13))]
    pipe = TimeSeriesPipeline(
        "TiRex", task_type="embedding", model_params={"checkpoint": tiny_tirex_checkpoint, "device": "cpu"}
    ).fit(frame, SCHEMA)
    result = pipe.predict()
    assert result.embeddings.shape == (3, 16) and np.isfinite(result.embeddings).all()
    for row, item in enumerate(("series_0", "series_2")):
        reference = pipe.adapter_._model.embed(torch.as_tensor(_series(frame, item))[None])
        np.testing.assert_allclose(result.embeddings[[0, 2][row]], reference[0].numpy(), atol=1e-5)


@pytest.mark.parametrize("strategy", ["peft", "finetune"])
def test_fine_tuning_round_trips_through_save_and_load(tiny_tirex_checkpoint, tmp_path, strategy):
    frame = make_panel(4, 80, freq="h", seed=1)
    zero_shot = _pipe(tiny_tirex_checkpoint).fit(frame, SCHEMA).predict().point
    pipe = _pipe(
        tiny_tirex_checkpoint,
        tuning_strategy=strategy,
        tuning_params={"epochs": 1, "steps_per_epoch": 4, "batch_size": 8, "learning_rate": 1e-2, "seed": 0},
    ).fit(frame, SCHEMA)
    report = pipe.training_report_
    assert report["steps_run"] == 4 and np.isfinite(report["final_train_loss"])
    assert "pinball" in report["objective"]
    tuned = pipe.predict().point
    assert not np.allclose(tuned, zero_shot, atol=1e-7)
    pipe.save(str(tmp_path / "tirex.joblib"))
    restored = TimeSeriesPipeline.load(str(tmp_path / "tirex.joblib")).predict().point
    np.testing.assert_allclose(restored, tuned, atol=1e-6)
    again = _pipe(tiny_tirex_checkpoint).fit(frame, SCHEMA).predict().point
    np.testing.assert_allclose(again, zero_shot, atol=1e-6)


def test_fine_tuning_gates_match_upstream_with_finite_gradients():
    from tabtune.models.tirex.models.slstm.cell import sLSTMCellTorch
    from tabtune.models.TimeSeries.tirex import _safe_pointwise

    torch.manual_seed(0)
    B, H = 3, 5

    raw = torch.randn(B, 4 * H)
    raw[:, :H] = -100.0
    states = [torch.zeros(B, H) for _ in range(4)]
    second = [torch.randn(B, H), torch.randn(B, H), torch.rand(B, H) + 0.5, torch.randn(B, H)]
    for start in (states, second):
        expected = sLSTMCellTorch.slstm_forward_pointwise(raw, torch.zeros_like(raw), torch.zeros(1, 4 * H), start)
        actual = _safe_pointwise(raw, torch.zeros_like(raw), torch.zeros(1, 4 * H), start)
        for e, a in zip(expected, actual):
            torch.testing.assert_close(a, e)

    grads = {}
    for name, fn in (("upstream", sLSTMCellTorch.slstm_forward_pointwise), ("safe", _safe_pointwise)):
        x = raw.clone().requires_grad_(True)
        fn(x, torch.zeros_like(raw), torch.zeros(1, 4 * H), states)[0].sum().backward()
        grads[name] = x.grad
    assert not torch.isfinite(grads["upstream"]).all()
    assert torch.isfinite(grads["safe"]).all()


def test_fine_tuning_restores_the_upstream_gates(tiny_tirex_checkpoint):
    from tabtune.models.tirex.models.slstm.cell import sLSTMCellTorch

    upstream = sLSTMCellTorch.slstm_forward_pointwise
    _pipe(
        tiny_tirex_checkpoint,
        tuning_strategy="finetune",
        tuning_params={"epochs": 1, "steps_per_epoch": 1, "batch_size": 4, "seed": 0},
    ).fit(make_panel(2, 60, freq="h", seed=0), SCHEMA)
    assert sLSTMCellTorch.slstm_forward_pointwise is upstream


def test_anomalies_and_imputation(tiny_tirex_checkpoint):
    frame = make_panel(1, 60, freq="h", seed=2)
    frame.loc[[20, 21], "target"] = np.nan
    base = {"checkpoint": tiny_tirex_checkpoint, "device": "cpu"}
    filled = TimeSeriesPipeline("TiRex", task_type="imputation", model_params=base).fit(frame, SCHEMA).predict()
    assert np.isfinite(filled.to_pandas()["target"]).all()
    scores = TimeSeriesPipeline(
        "TiRex", task_type="anomaly_detection", model_params=base, task_params={"min_context": 16}
    ).fit(frame, SCHEMA).predict()
    assert len(scores.to_pandas()) > 0


def test_invalid_settings_are_rejected(tiny_tirex_checkpoint):
    frame = pd.DataFrame(
        {"item_id": "a", "timestamp": pd.date_range("2024-01-01", periods=20, freq="D"), "target": 1.0}
    )
    with pytest.raises(ConfigError, match="float32"):
        _pipe(tiny_tirex_checkpoint, model_params={"dtype": "bfloat16"}).fit(frame, SCHEMA)
    with pytest.raises(ConfigError, match="batch_size"):
        _pipe(tiny_tirex_checkpoint, model_params={"batch_size": 0}).fit(frame, SCHEMA)


def test_no_quantiles_requested(tiny_tirex_checkpoint):
    """No requested levels means no quantile output."""
    pipe = TimeSeriesPipeline(
        "TiRex",
        model_params={"checkpoint": tiny_tirex_checkpoint, "device": "cpu"},
        forecast_params={"prediction_length": 4, "quantile_levels": []},
    )
    result = pipe.fit(make_panel(2, 60, freq="h", seed=0), SCHEMA).predict()
    assert result.quantiles is None
    assert result.quantile_levels == ()
    assert result.point.shape == (2, 4)
