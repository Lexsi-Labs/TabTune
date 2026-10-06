"""Tests for TimeSeriesSchema validation and the canonical TimeSeriesPanel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabtune._internal.deprecation import reset_warning_cache
from tabtune.TimeSeries import TimeSeriesPanel, TimeSeriesSchema

pytestmark = [pytest.mark.unit, pytest.mark.time_series]


@pytest.fixture(autouse=True)
def _fresh_warnings():
    reset_warning_cache()
    yield
    reset_warning_cache()


def _panel(n=6, items=("a", "b"), freq="D"):
    frames = []
    for offset, item in enumerate(items):
        frames.append(
            pd.DataFrame(
                {
                    "store": item,
                    "date": pd.date_range("2024-01-01", periods=n, freq=freq),
                    "sales": np.arange(n, dtype=float) + 10 * offset,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


SCHEMA = TimeSeriesSchema(target="sales", timestamp="date", item_id="store")


def test_frame_to_panel():
    panel = SCHEMA.to_panel(_panel())
    assert isinstance(panel, TimeSeriesPanel)
    assert panel.item_ids == ("a", "b")
    assert panel.freq == "D"
    np.testing.assert_array_equal(panel.values[1], np.arange(6.0) + 10)
    assert panel.last_timestamps[0] == pd.Timestamp("2024-01-06")


def test_single_series_without_item_column():
    df = _panel(items=("a",)).drop(columns="store")
    panel = TimeSeriesSchema(target="sales", timestamp="date").to_panel(df)
    assert panel.item_ids == (None,)
    assert len(panel.values[0]) == 6


def test_row_order_does_not_matter():
    df = _panel()
    shuffled = df.sample(frac=1.0, random_state=0)
    a, b = SCHEMA.to_panel(df), SCHEMA.to_panel(shuffled)
    assert a.item_ids == b.item_ids
    for x, y in zip(a.values, b.values, strict=True):
        np.testing.assert_array_equal(x, y)
    assert a.fingerprint() == b.fingerprint()


def test_input_is_not_mutated():
    df = _panel().sample(frac=1.0, random_state=0)
    df["date"] = df["date"].astype(str)
    before = df.copy()
    SCHEMA.to_panel(df)
    pd.testing.assert_frame_equal(df, before)


def test_string_timestamps_are_parsed():
    df = _panel()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    assert SCHEMA.to_panel(df).freq == "D"


def test_tail_keeps_origin():
    panel = SCHEMA.to_panel(_panel())
    short = panel.tail(2)
    np.testing.assert_array_equal(short.values[0], [4.0, 5.0])
    assert short.last_timestamps == panel.last_timestamps
    assert panel.max_length == 6


def test_unused_categories_are_not_series():
    df = _panel()
    df["store"] = pd.Categorical(df["store"], categories=["a", "b", "unused"])
    assert SCHEMA.to_panel(df).item_ids == ("a", "b")


def test_extra_columns_warn_once_and_do_not_affect_the_panel():
    df = _panel()
    df["promo"] = np.arange(len(df))
    with pytest.warns(UserWarning, match="play no role in this schema"):
        panel = SCHEMA.to_panel(df)
    assert panel.fingerprint() == SCHEMA.to_panel(_panel()).fingerprint()


def test_fingerprint_changes_with_values_origin_and_ids():
    base = SCHEMA.to_panel(_panel()).fingerprint()
    changed_value = _panel()
    changed_value.loc[0, "sales"] = 99.0
    later = _panel(n=7)
    renamed = _panel(items=("a", "c"))
    fingerprints = {base} | {
        SCHEMA.to_panel(df).fingerprint() for df in (changed_value, later, renamed)
    }
    assert len(fingerprints) == 4


def test_check_observed_flags_all_missing_window():
    df = _panel(n=10)
    df.loc[(df["store"] == "a") & (df["date"] > "2024-01-03"), "sales"] = np.nan
    panel = SCHEMA.to_panel(df, native_missing=True)
    panel.check_observed()  # the full history has observations
    with pytest.raises(ValueError, match="last 4 observations"):
        panel.tail(4).check_observed(context=4)

@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": "sales", "timestamp": "sales"},
        {"target": "sales", "timestamp": "date", "item_id": "date"},
        {"target": ""},
    ],
)
def test_schema_rejects_invalid_roles(kwargs):
    with pytest.raises(ValueError):
        TimeSeriesSchema(**kwargs)


def test_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        SCHEMA.to_panel(np.zeros((3, 3)))


def test_rejects_missing_columns():
    with pytest.raises(ValueError, match="not in the frame"):
        SCHEMA.to_panel(_panel().drop(columns="sales"))


def test_rejects_non_numeric_target():
    df = _panel()
    df["sales"] = df["sales"].astype(str)
    with pytest.raises(ValueError, match="must be numeric"):
        SCHEMA.to_panel(df)


def test_rejects_infinite_targets():
    df = _panel()
    df.loc[3, "sales"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        SCHEMA.to_panel(df, native_missing=True)


def test_rejects_duplicate_keys():
    df = _panel()
    with pytest.raises(ValueError, match="duplicate"):
        SCHEMA.to_panel(pd.concat([df, df.iloc[[0]]]))


def test_rejects_missing_item_ids():
    df = _panel()
    df.loc[0, "store"] = None
    with pytest.raises(ValueError, match="Item column"):
        SCHEMA.to_panel(df)


def test_rejects_irregular_grid():
    df = _panel().drop(index=2)
    with pytest.raises(ValueError, match="Resample"):
        SCHEMA.to_panel(df)


def test_rejects_mixed_frequencies():
    daily = _panel(items=("a",))
    hourly = _panel(items=("b",), freq="h")
    with pytest.raises(ValueError, match="different frequencies"):
        SCHEMA.to_panel(pd.concat([daily, hourly]))


def test_short_series_needs_explicit_freq():
    df = _panel(n=2)
    with pytest.raises(ValueError, match="too few"):
        SCHEMA.to_panel(df)
    schema = TimeSeriesSchema(target="sales", timestamp="date", item_id="store", freq="D")
    assert schema.to_panel(df).freq == "D"


def test_explicit_freq_is_checked():
    schema = TimeSeriesSchema(target="sales", timestamp="date", item_id="store", freq="h")
    with pytest.raises(ValueError, match="regular 'h' grid"):
        schema.to_panel(_panel())


def test_missing_targets_follow_native_missing():
    df = _panel()
    df.loc[1, "sales"] = np.nan
    with pytest.raises(ValueError, match="missing target"):
        SCHEMA.to_panel(df, native_missing=False)
    panel = SCHEMA.to_panel(df, native_missing=True)
    assert np.isnan(panel.values[0][1])


def test_all_missing_series_is_rejected_even_when_nan_allowed():
    df = _panel()
    df.loc[df["store"] == "a", "sales"] = np.nan
    with pytest.raises(ValueError, match="no observed"):
        SCHEMA.to_panel(df, native_missing=True)


def _rich(n=6, items=("a", "b")):
    frame = _panel(n, items)
    frame["traffic"] = frame["sales"] * 10
    frame["temp"] = np.linspace(15.0, 25.0, len(frame))
    frame["weather"] = ["sun", "rain"] * (len(frame) // 2)
    return frame


def _future(horizon=2, items=("a", "b"), start="2024-01-07"):
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "store": item,
                    "date": pd.date_range(start, periods=horizon, freq="D"),
                    "temp": np.arange(horizon, dtype=float),
                    "weather": ["sun"] * horizon,
                }
            )
            for item in items
        ],
        ignore_index=True,
    )


MULTI = TimeSeriesSchema(target=("sales", "traffic"), timestamp="date", item_id="store")
COVARIATES = TimeSeriesSchema(
    target="sales",
    timestamp="date",
    item_id="store",
    past_covariates=("temp",),
    known_covariates=("weather",),
)


def test_several_targets_become_consecutive_rows_per_item():
    panel = MULTI.to_panel(_rich())
    assert len(panel) == 4
    assert panel.item_ids == ("a", "a", "b", "b")
    assert panel.target_names == ("sales", "traffic", "sales", "traffic")
    assert panel.n_targets == 2
    assert panel.item_groups() == ((0, 1), (2, 3))
    np.testing.assert_array_equal(panel.values[1], panel.values[0] * 10)


def test_a_single_target_is_unchanged_by_the_multivariate_support():
    """Regression guard: univariate panels must be exactly what they always were."""
    frame = _panel()
    panel = SCHEMA.to_panel(frame)
    assert len(panel) == 2
    assert panel.target_names == ("sales", "sales")
    assert panel.n_targets == 1
    assert panel.item_groups() == ((0,), (1,))
    assert panel.past_covariates == ({}, {})
    assert panel.future_covariates == ({}, {})


def test_target_may_be_given_as_a_sequence_of_one():
    assert TimeSeriesSchema(target=["sales"]).target_names == ("sales",)
    with pytest.raises(ValueError, match="at least one column"):
        TimeSeriesSchema(target=())


def test_roles_must_still_be_distinct_across_covariates():
    with pytest.raises(ValueError, match="only one role"):
        TimeSeriesSchema(target="sales", past_covariates=("sales",))
    with pytest.raises(ValueError, match="only one role"):
        TimeSeriesSchema(target="sales", past_covariates=("temp",), known_covariates=("temp",))
    with pytest.raises(ValueError, match="only one role"):
        TimeSeriesSchema(target=("sales", "sales"))


def test_covariates_are_kept_and_typed_by_dtype():
    panel = COVARIATES.to_panel(_rich(), future_df=_future(), horizon=2)
    past = panel.past_covariates[0]
    assert sorted(past) == ["temp", "weather"]
    assert past["temp"].dtype == np.float64
    assert past["weather"].dtype.kind in "UO"
    assert panel.covariate_names == ("temp", "weather")
    # Only the known covariate is carried into the future.
    assert list(panel.future_covariates[0]) == ["weather"]
    assert len(panel.future_covariates[0]["weather"]) == 2


def test_covariates_must_be_fully_observed():
    frame = _rich()
    frame.loc[2, "temp"] = np.nan
    with pytest.raises(ValueError, match="Covariates must be fully observed"):
        COVARIATES.to_panel(frame, future_df=_future(), horizon=2)

    frame = _rich()
    frame.loc[2, "temp"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        COVARIATES.to_panel(frame, future_df=_future(), horizon=2)


def test_future_frame_must_line_up_with_the_history():
    frame = _rich()

    with pytest.raises(ValueError, match="no known_covariates"):
        SCHEMA.to_panel(frame[["store", "date", "sales"]], future_df=_future(), horizon=2)

    with pytest.raises(ValueError, match="exactly the 2 timestamp"):
        COVARIATES.to_panel(frame, future_df=_future(horizon=3), horizon=2)

    with pytest.raises(ValueError, match="exactly the 2 timestamp"):
        COVARIATES.to_panel(frame, future_df=_future(start="2024-01-09"), horizon=2)

    with pytest.raises(ValueError, match="no rows for item 'b'"):
        COVARIATES.to_panel(frame, future_df=_future(items=("a",)), horizon=2)

    with pytest.raises(ValueError, match="not in the history"):
        COVARIATES.to_panel(frame, future_df=_future(items=("a", "b", "c")), horizon=2)

    with pytest.raises(ValueError, match=r"Columns \['weather'\] are not in future_df"):
        COVARIATES.to_panel(frame, future_df=_future().drop(columns="weather"), horizon=2)

    with pytest.raises(ValueError, match="must not carry the target"):
        COVARIATES.to_panel(frame, future_df=_future().assign(sales=1.0), horizon=2)


def test_tail_trims_past_covariates_but_not_future_ones():
    panel = COVARIATES.to_panel(_rich(), future_df=_future(), horizon=2)
    cut = panel.tail(3)
    assert len(cut.values[0]) == 3
    assert len(cut.past_covariates[0]["temp"]) == 3
    assert len(cut.future_covariates[0]["weather"]) == 2
    assert cut.last_timestamps == panel.last_timestamps


def test_fingerprint_covers_targets_and_covariate_values():
    base = COVARIATES.to_panel(_rich(), future_df=_future(), horizon=2)
    assert base.fingerprint() == COVARIATES.to_panel(
        _rich(), future_df=_future(), horizon=2
    ).fingerprint()

    warmer = _rich()
    warmer["temp"] += 1.0
    assert COVARIATES.to_panel(
        warmer, future_df=_future(), horizon=2
    ).fingerprint() != base.fingerprint()

    rainy = _future()
    rainy["weather"] = "rain"
    assert COVARIATES.to_panel(
        _rich(), future_df=rainy, horizon=2
    ).fingerprint() != base.fingerprint()

    assert MULTI.to_panel(_rich()).fingerprint() != SCHEMA.to_panel(_rich()[
        ["store", "date", "sales"]
    ]).fingerprint()


def test_attach_future_can_be_applied_after_the_history_is_validated():
    """fit validates the history once; each forecast brings its own future values."""
    panel = COVARIATES.to_panel(_rich())
    assert all(not row for row in panel.future_covariates)
    with_future = COVARIATES.attach_future(panel, _future(), horizon=2)
    assert list(with_future.future_covariates[0]) == ["weather"]
    # The original panel is untouched.
    assert not panel.future_covariates[0]


def test_check_observed_names_the_target_when_multivariate():
    frame = _rich()
    frame.loc[frame["store"] == "b", "traffic"] = np.nan
    with pytest.raises(ValueError, match=r"item 'b' target 'traffic'"):
        MULTI.to_panel(frame, native_missing=True)
