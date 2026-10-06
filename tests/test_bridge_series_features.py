"""Time series histories as tabular features (``tabtune.bridge``).

The central property is leakage: a row with an as-of cutoff must never see
an observation at or after it. It is tested by corrupting everything from the
cutoff on and checking that the features do not move. Embeddings use a tiny
random Time-MoE checkpoint; end to end, the features feed xRFM (no pretrained
weights) through ``TabularPipeline``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from tabtune.bridge import SUMMARY_NAMES, SeriesFeaturizer, add_series_features, summarise_history
from tabtune.registry.errors import ConfigError, UnsupportedTaskError

pytestmark = [pytest.mark.unit, pytest.mark.time_series]


def _transactions(n_customers=60, days=60, seed=0):
    """Daily spend per customer; half are trending down (the churners)."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for c in range(n_customers):
        churn = c % 2
        base = rng.uniform(20, 40)
        slope = -0.4 if churn else 0.05
        t = np.arange(days)
        spend = base + slope * t + rng.normal(scale=2.0, size=days)
        rows.append(pd.DataFrame({"customer": c, "date": pd.date_range("2024-01-01", periods=days), "spend": spend}))
        labels.append(churn)
    series = pd.concat(rows, ignore_index=True)
    table = pd.DataFrame(
        {
            "customer": np.arange(n_customers),
            "age": rng.integers(18, 80, n_customers),  # uninformative static feature
            "as_of": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(30, 60, n_customers), "D"),
        }
    )
    return series, table, np.asarray(labels)


def test_summaries_of_a_known_history():
    values = summarise_history(np.array([1.0, 2.0, np.nan, 4.0, np.nan]), season=2)
    stats = dict(zip(SUMMARY_NAMES, values, strict=True))
    assert stats["count"] == 3 and stats["missing_frac"] == pytest.approx(0.4)
    assert (stats["first"], stats["last"], stats["max"]) == (1.0, 4.0, 4.0)
    assert stats["mean"] == pytest.approx(7 / 3)
    assert stats["slope"] == pytest.approx(np.polyfit([0, 1, 3], [1, 2, 4], 1)[0])
    assert stats["steps_since_last"] == 1  # one trailing gap
    assert np.isnan(stats["acf_season"])  # too short for lag 2


def test_empty_history_is_count_zero_and_nan_elsewhere():
    stats = summarise_history(np.zeros(0))
    assert stats[0] == 0 and np.isnan(stats[1:]).all()


def test_rows_never_see_observations_at_or_after_their_cutoff():
    series, table, _ = _transactions(n_customers=10)
    featurizer = SeriesFeaturizer(series, id_col="customer", time_col="date", cutoff="as_of")
    before = featurizer.fit_transform(table)
    corrupted = series.copy()
    for _, row in table.iterrows():
        late = (corrupted["customer"] == row["customer"]) & (corrupted["date"] >= row["as_of"])
        corrupted.loc[late, "spend"] = 1e9
    after = SeriesFeaturizer(corrupted, id_col="customer", time_col="date", cutoff="as_of").fit_transform(table)
    pd.testing.assert_frame_equal(before, after)
    # and the cutoff is strict: count == number of days strictly before as_of
    expected = (table["as_of"] - pd.Timestamp("2024-01-01")).dt.days
    np.testing.assert_array_equal(before["ts_spend_count"], expected)


def test_global_cutoff_and_window():
    series, table, _ = _transactions(n_customers=4)
    out = SeriesFeaturizer(
        series, id_col="customer", time_col="date", cutoff="2024-01-11", window=5
    ).fit_transform(table)
    assert (out["ts_spend_count"] == 5).all()  # 10 days before the cutoff, last 5 kept


def test_unknown_entities_get_empty_histories():
    series, table, _ = _transactions(n_customers=3)
    table = pd.concat([table, pd.DataFrame({"customer": [999], "age": [40], "as_of": [pd.Timestamp("2024-02-15")]})])
    out = SeriesFeaturizer(series, id_col="customer", time_col="date", cutoff="as_of").fit_transform(table)
    assert out["ts_spend_count"].iloc[-1] == 0 and np.isnan(out["ts_spend_mean"].iloc[-1])


def test_is_a_well_behaved_sklearn_transformer():
    series, table, y = _transactions()
    featurizer = SeriesFeaturizer(series, id_col="customer", time_col="date", cutoff="as_of", keep_input=False)
    assert clone(featurizer).get_params()["cutoff"] == "as_of"
    features = featurizer.fit(table).transform(table)
    assert list(features.columns) == [f"ts_spend_{name}" for name in SUMMARY_NAMES]
    assert list(featurizer.get_feature_names_out()) == list(features.columns)
    pipe = Pipeline(
        [
            ("history", SeriesFeaturizer(series, id_col="customer", time_col="date", cutoff="as_of", keep_input=False)),
            ("model", HistGradientBoostingClassifier(max_iter=50, random_state=0)),
        ]
    )
    pipe.fit(table, y)
    assert pipe.predict(table).shape == (len(table),)


def test_history_features_lift_a_tabular_model_on_a_history_driven_label():
    series, table, y = _transactions(n_customers=200, seed=3)
    train, test, y_train, y_test = train_test_split(table, y, test_size=0.4, random_state=0, stratify=y)
    static_only = HistGradientBoostingClassifier(max_iter=50, random_state=0).fit(train[["age"]], y_train)
    featurizer = SeriesFeaturizer(series, id_col="customer", time_col="date", cutoff="as_of")
    X_train = featurizer.fit_transform(train).drop(columns=["customer", "as_of"])
    X_test = featurizer.transform(test).drop(columns=["customer", "as_of"])
    with_history = HistGradientBoostingClassifier(max_iter=50, random_state=0).fit(X_train, y_train)
    assert with_history.score(X_test, y_test) >= static_only.score(test[["age"]], y_test) + 0.3


def test_settings_are_validated():
    series, table, _ = _transactions(n_customers=3)
    with pytest.raises(ConfigError, match="id column"):
        SeriesFeaturizer(series, id_col="nope", time_col="date").fit(table)
    with pytest.raises(ConfigError, match="Nothing to compute"):
        SeriesFeaturizer(series, id_col="customer", time_col="date", summaries=False).fit(table)
    with pytest.raises(ConfigError, match="value_cols"):
        SeriesFeaturizer(series, id_col="customer", time_col="date", value_cols=["x"]).fit(table)
    with pytest.raises(ConfigError, match="fit"):
        SeriesFeaturizer(series, id_col="customer", time_col="date").transform(table)


def test_model_embeddings_are_reduced_and_leak_safe(tiny_timemoe_checkpoint):
    series, table, _ = _transactions(n_customers=12)
    featurizer = SeriesFeaturizer(
        series, id_col="customer", time_col="date", cutoff="as_of",
        model="TimeMoE", checkpoint=tiny_timemoe_checkpoint,
        n_components=3, summaries=False, keep_input=False,
    )
    features = featurizer.fit_transform(table)
    assert list(features.columns) == ["ts_spend_emb0", "ts_spend_emb1", "ts_spend_emb2"]
    assert np.isfinite(features.to_numpy()).all()
    again = featurizer.transform(table)
    pd.testing.assert_frame_equal(features, again)
    corrupted = series.copy()
    for _, row in table.iterrows():
        late = (corrupted["customer"] == row["customer"]) & (corrupted["date"] >= row["as_of"])
        corrupted.loc[late, "spend"] = -1e6
    leak_check = SeriesFeaturizer(
        corrupted, id_col="customer", time_col="date", cutoff="as_of",
        model="TimeMoE", checkpoint=tiny_timemoe_checkpoint,
        n_components=3, summaries=False, keep_input=False,
    ).fit_transform(table)
    np.testing.assert_allclose(leak_check.to_numpy(), features.to_numpy(), atol=1e-6)


def test_gaps_in_a_history_do_not_zero_its_embedding(tiny_timemoe_checkpoint):
    series, table, _ = _transactions(n_customers=4)
    gappy = series.copy()
    gappy.loc[gappy.index[5], "spend"] = np.nan  # one missing day for customer 0
    kwargs = dict(
        id_col="customer", time_col="date", summaries=False, keep_input=False,
        model="TimeMoE", checkpoint=tiny_timemoe_checkpoint,
    )
    clean = SeriesFeaturizer(series, **kwargs)
    holey = SeriesFeaturizer(gappy, **kwargs)
    a, b = clean.fit_transform(table).iloc[0].to_numpy(), holey.fit_transform(table).iloc[0].to_numpy()
    assert np.isfinite(b).all() and np.abs(b).sum() > 0
    assert np.corrcoef(a, b)[0, 1] > 0.9


def test_columns_are_fixed_at_fit_even_for_a_batch_without_history(tiny_timemoe_checkpoint):
    series, table, _ = _transactions(n_customers=6)
    featurizer = SeriesFeaturizer(
        series, id_col="customer", time_col="date", cutoff="as_of",
        model="TimeMoE", checkpoint=tiny_timemoe_checkpoint,
    ).fit(table)
    trained = featurizer.transform(table)
    newcomers = pd.DataFrame({"customer": [900, 901], "age": [30, 50], "as_of": [pd.Timestamp("2024-02-01")] * 2})
    scored = featurizer.transform(newcomers)
    assert list(scored.columns) == list(trained.columns)
    assert scored.filter(like="_emb").isna().all().all()


def test_a_missing_cutoff_sees_no_history():
    series, table, _ = _transactions(n_customers=3)
    table = table.copy()
    table.loc[table.index[1], "as_of"] = pd.NaT
    out = SeriesFeaturizer(series, id_col="customer", time_col="date", cutoff="as_of").fit_transform(table)
    assert out["ts_spend_count"].iloc[1] == 0 and np.isnan(out["ts_spend_last"].iloc[1])


def test_feature_names_include_kept_inputs_and_support_set_output():
    series, table, _ = _transactions(n_customers=5)
    featurizer = SeriesFeaturizer(series, id_col="customer", time_col="date", cutoff="as_of").fit(table)
    names = list(featurizer.get_feature_names_out())
    assert names[: table.shape[1]] == list(table.columns)
    assert names[table.shape[1]:] == [f"ts_spend_{name}" for name in SUMMARY_NAMES]
    featurizer.set_output(transform="pandas")
    out = featurizer.transform(table)
    assert list(out.columns) == names


def test_models_without_embeddings_are_refused():
    series, table, _ = _transactions(n_customers=3)
    with pytest.raises(UnsupportedTaskError):
        SeriesFeaturizer(series, id_col="customer", time_col="date", model="SeasonalNaive").fit(table)
    with pytest.raises(ConfigError, match="model name or a TimeSeriesPipeline"):
        SeriesFeaturizer(series, id_col="customer", time_col="date", model=object()).fit(table)


def test_end_to_end_with_a_tabtune_tabular_model():
    from tabtune import TabularPipeline

    series, table, y = _transactions(n_customers=120, seed=5)
    train, test, y_train, y_test = train_test_split(table, y, test_size=0.3, random_state=0, stratify=y)
    featurizer = SeriesFeaturizer(series, id_col="customer", time_col="date", cutoff="as_of")
    X_train = featurizer.fit_transform(train).drop(columns=["customer", "as_of"])
    X_test = featurizer.transform(test).drop(columns=["customer", "as_of"])
    pipeline = TabularPipeline("XRFM", task_type="classification").fit(X_train, pd.Series(y_train))
    accuracy = float((pipeline.predict(X_test) == y_test).mean())
    assert accuracy >= 0.8
    assert add_series_features(test, series, id_col="customer", time_col="date", cutoff="as_of").shape[1] == (
        test.shape[1] + len(SUMMARY_NAMES)
    )
