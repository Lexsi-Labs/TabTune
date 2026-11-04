"""Time series histories as features for tabular models.

:class:`SeriesFeaturizer` turns the history of each row's entity into
fixed-length columns: summary statistics (count, missing fraction, first, last,
mean, standard deviation, min, max, median, recent mean and its ratio to the
mean, OLS slope, lag-1 and seasonal autocorrelation, steps since the last
observation) and, optionally, embeddings from a time series model with an
embedding task via ``TimeSeriesPipeline(task_type="embedding")``, PCA-reduced
on the training rows only.

With ``cutoff`` set to a table column (each row's as-of time) a row sees only
observations strictly before its own cutoff; a timestamp cuts every row there;
``None`` uses the whole history. The transformer is scikit-learn compatible;
:func:`add_series_features` is the one-call form.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..registry.errors import ConfigError

__all__ = ["SUMMARY_NAMES", "SeriesFeaturizer", "add_series_features", "summarise_history"]

SUMMARY_NAMES: tuple[str, ...] = (
    "count",
    "missing_frac",
    "first",
    "last",
    "mean",
    "std",
    "min",
    "max",
    "median",
    "recent_mean",
    "recent_ratio",
    "slope",
    "acf1",
    "acf_season",
    "steps_since_last",
)

_RECENT = 7


def summarise_history(values: np.ndarray, *, season: int = 7) -> np.ndarray:
    """Summary statistics of one history (see :data:`SUMMARY_NAMES`).

    ``values`` is the history in time order, ``NaN`` for missing
    observations. An empty history gives ``count = 0`` and ``NaN`` elsewhere.
    ``steps_since_last`` counts trailing missing values. Statistics that need
    more observations than there are (a slope from one point, a seasonal
    autocorrelation from fewer than ``season + 2`` points) are ``NaN``.
    """
    x = np.asarray(values, dtype=float)
    out = np.full(len(SUMMARY_NAMES), np.nan)
    observed = np.isfinite(x)
    n = int(observed.sum())
    out[0] = n
    if x.size:
        out[1] = 1.0 - n / x.size
    if n == 0:
        return out
    obs = x[observed]
    idx = np.flatnonzero(observed)
    out[2], out[3] = obs[0], obs[-1]
    out[4], out[5] = obs.mean(), obs.std()
    out[6], out[7], out[8] = obs.min(), obs.max(), np.median(obs)
    recent = obs[-_RECENT:]
    out[9] = recent.mean()
    out[10] = recent.mean() / obs.mean() if abs(obs.mean()) > 1e-12 else np.nan
    if n >= 2 and np.ptp(idx) > 0:
        out[11] = np.polyfit(idx.astype(float), obs, 1)[0]

    def acf(lag: int) -> float:
        if n < lag + 2 or obs.std() == 0:
            return np.nan
        z = (obs - obs.mean()) / obs.std()
        return float(np.mean(z[lag:] * z[:-lag]))

    out[12] = acf(1)
    out[13] = acf(max(1, int(season)))
    out[14] = x.size - 1 - idx[-1]
    return out


class SeriesFeaturizer(TransformerMixin, BaseEstimator):
    """Append history features of each row's entity to a table.

    Args:
        series: Long-format histories: ``id_col``, ``time_col`` and value columns.
        id_col: Entity column, present in both ``series`` and the table.
        time_col: Timestamp column of ``series``.
        value_cols: Columns of ``series`` to featurise (default: every numeric
            column other than the id and time columns).
        cutoff: ``None`` (whole history), a timestamp (history strictly before
            it), or the name of a table column holding each row's as-of time
            (history strictly before that row's time: leak-safe for temporal
            prediction).
        window: Keep only the last ``window`` observations of each history.
        summaries: Add :data:`SUMMARY_NAMES` columns.
        model: ``None`` (no embeddings), a model name, or a
            :class:`~tabtune.TimeSeries.TimeSeriesPipeline` built with
            ``task_type="embedding"``.
        checkpoint: Checkpoint for ``model`` when it is a name.
        n_components: Reduce embeddings to this many PCA components, fitted
            on the rows passed to :meth:`fit` (``None`` keeps all dimensions).
        season: Lag of the seasonal autocorrelation summary.
        prefix: Prefix of the new columns.
        keep_input: Return the table with the new columns appended (default),
            or the new columns only.
        device, seed: Passed to the pipeline built from a model name.
    """

    def __init__(
        self,
        series: pd.DataFrame,
        *,
        id_col: str = "item_id",
        time_col: str = "timestamp",
        value_cols: Sequence[str] | None = None,
        cutoff: Any = None,
        window: int | None = None,
        summaries: bool = True,
        model: Any = None,
        checkpoint: str | None = None,
        n_components: int | None = None,
        season: int = 7,
        prefix: str = "ts",
        keep_input: bool = True,
        device: str = "cpu",
        seed: int | None = 0,
    ) -> None:
        self.series = series
        self.id_col = id_col
        self.time_col = time_col
        self.value_cols = value_cols
        self.cutoff = cutoff
        self.window = window
        self.summaries = summaries
        self.model = model
        self.checkpoint = checkpoint
        self.n_components = n_components
        self.season = season
        self.prefix = prefix
        self.keep_input = keep_input
        self.device = device
        self.seed = seed


    def _validate(self) -> None:
        if self.id_col not in self.series.columns or self.time_col not in self.series.columns:
            raise ConfigError(
                f"series needs the id column {self.id_col!r} and the time column {self.time_col!r}."
            )
        if not self.summaries and self.model is None:
            raise ConfigError("Nothing to compute: set summaries=True or pass a model.")
        if self.window is not None and int(self.window) < 1:
            raise ConfigError(f"window must be >= 1, got {self.window}.")

    def _columns(self) -> list[str]:
        if self.value_cols is not None:
            missing = [c for c in self.value_cols if c not in self.series.columns]
            if missing:
                raise ConfigError(f"value_cols not in series: {missing}")
            return list(self.value_cols)
        return [
            c
            for c in self.series.columns
            if c not in (self.id_col, self.time_col) and pd.api.types.is_numeric_dtype(self.series[c])
        ]

    def _index(self) -> dict[Hashable, tuple[np.ndarray, dict[str, np.ndarray]]]:
        """Entity -> (sorted timestamps, value column -> values)."""
        frame = self.series.copy()
        frame[self.time_col] = pd.to_datetime(frame[self.time_col])
        frame = frame.sort_values([self.id_col, self.time_col], kind="stable")
        index = {}
        for entity, group in frame.groupby(self.id_col, sort=False):
            times = group[self.time_col].to_numpy(dtype="datetime64[ns]")
            index[entity] = (times, {c: group[c].to_numpy(dtype=float) for c in self._columns()})
        return index

    def _cutoffs(self, X: pd.DataFrame) -> np.ndarray | None:
        if self.cutoff is None:
            return None
        if isinstance(self.cutoff, str) and self.cutoff in X.columns:
            return pd.to_datetime(X[self.cutoff]).to_numpy(dtype="datetime64[ns]")
        stamp = pd.Timestamp(self.cutoff).to_datetime64()
        return np.full(len(X), stamp, dtype="datetime64[ns]")

    def _histories(self, X: pd.DataFrame) -> dict[str, list[np.ndarray]]:
        """Per value column, the history of every row (empty when the entity is unknown)."""
        if self.id_col not in X.columns:
            raise ConfigError(f"The table needs the id column {self.id_col!r}.")
        index = self._index_cache
        cutoffs = self._cutoffs(X)
        columns = self._columns()
        out: dict[str, list[np.ndarray]] = {c: [] for c in columns}
        for row, entity in enumerate(X[self.id_col].to_numpy()):
            entry = index.get(entity)
            if entry is None:
                for c in columns:
                    out[c].append(np.zeros(0))
                continue
            times, values = entry
            if cutoffs is None:
                stop = len(times)
            elif np.isnat(cutoffs[row]):
                stop = 0
            else:
                stop = int(np.searchsorted(times, cutoffs[row], "left"))
            start = 0 if self.window is None else max(0, stop - int(self.window))
            for c in columns:
                out[c].append(values[c][start:stop])
        return out

    def _encoder(self) -> Any:
        from ..TimeSeries import TimeSeriesPipeline

        if isinstance(self.model, TimeSeriesPipeline):
            if self.model.task_type != "embedding":
                raise ConfigError("model must be a TimeSeriesPipeline with task_type='embedding'.")
            return self.model
        if not isinstance(self.model, str):
            raise ConfigError("model must be a model name or a TimeSeriesPipeline.")
        model_params = {"device": self.device}
        if self.checkpoint:
            model_params["checkpoint"] = self.checkpoint
        return TimeSeriesPipeline(
            self.model,
            task_type="embedding",
            model_params=model_params,
            tuning_params={"seed": self.seed} if self.seed is not None else None,
        )

    def _encode(self, histories: list[np.ndarray]) -> np.ndarray:
        """``[len(histories), dim]`` embeddings of gap-free histories."""
        from ..TimeSeries import TimeSeriesSchema

        end = pd.Timestamp("2000-01-01")
        frame = pd.concat(
            [
                pd.DataFrame(
                    {
                        "item": i,
                        "timestamp": pd.date_range(end=end, periods=len(h), freq=self._freq),
                        "value": h,
                    }
                )
                for i, h in enumerate(histories)
            ],
            ignore_index=True,
        )
        encoder = self._encoder_cache
        if not encoder._is_fitted:
            schema = TimeSeriesSchema(target="value", item_id="item", freq=self._freq)
            return encoder.fit(frame, schema).predict().embeddings
        return encoder.predict(frame).embeddings

    def _embeddings(self, histories: list[np.ndarray], width: int | None = None) -> np.ndarray:
        """Embeddings ``[rows, dim]``; ``NaN`` rows for empty histories.

        Gaps are filled by linear interpolation (the nearest value at the ends)
        before encoding; a ``NaN`` in the context makes the whole embedding NaN.
        """
        present = [i for i, h in enumerate(histories) if np.isfinite(h).any()]
        if not present:
            return np.full((len(histories), width or 0), np.nan)
        histories = [
            pd.Series(h, dtype=float).interpolate(limit_direction="both").to_numpy()
            if np.isnan(h).any() else h
            for h in histories
        ]
        keys: dict[bytes, int] = {}
        unique: list[np.ndarray] = []
        slot = []
        for i in present:
            key = np.ascontiguousarray(histories[i]).tobytes()
            if key not in keys:
                keys[key] = len(unique)
                unique.append(histories[i])
            slot.append(keys[key])
        vectors = self._encode(unique)
        out = np.full((len(histories), vectors.shape[1]), np.nan)
        out[present] = vectors[slot]
        return out


    def fit(self, X: pd.DataFrame, y: Any = None) -> SeriesFeaturizer:
        """Index the histories and fit the embedding PCA on ``X``'s rows (labels are not used)."""
        self._validate()
        self._index_cache = self._index()
        self._freq = _frequency(self.series, self.id_col, self.time_col)
        self._encoder_cache = self._encoder() if self.model is not None else None
        self.pca_: dict[str, Any] = {}
        self.feature_names_in_ = np.asarray([str(c) for c in pd.DataFrame(X).columns], dtype=object)
        if self._encoder_cache is not None and self.n_components is not None:
            from sklearn.decomposition import PCA

            histories = self._histories(pd.DataFrame(X))
            for column, rows in histories.items():
                vectors = self._embeddings(rows)
                usable = vectors[np.isfinite(vectors).all(axis=1)] if vectors.size else vectors
                if len(usable) >= 2:
                    k = max(1, min(int(self.n_components), usable.shape[0], usable.shape[1]))
                    self.pca_[column] = PCA(n_components=k, random_state=0).fit(usable)
        self.embedding_dims_: dict[str, int] = {}
        if self._encoder_cache is not None:
            raw = self._embeddings([np.linspace(0.0, 1.0, 64)]).shape[1]
            for column in self._columns():
                self.embedding_dims_[column] = (
                    int(self.pca_[column].n_components_) if column in self.pca_ else int(raw)
                )
        names = []
        for column in self._columns():
            if self.summaries:
                names += [f"{self.prefix}_{column}_{name}" for name in SUMMARY_NAMES]
            names += [f"{self.prefix}_{column}_emb{j}" for j in range(self.embedding_dims_.get(column, 0))]
        self.feature_names_out_ = names
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` with the history features appended (or the features only)."""
        if not hasattr(self, "_index_cache"):
            raise ConfigError("Call fit() before transform().")
        X = pd.DataFrame(X)
        histories = self._histories(X)
        blocks: dict[str, np.ndarray] = {}
        for column, rows in histories.items():
            if self.summaries:
                stats = np.stack([summarise_history(h, season=self.season) for h in rows]) if rows else np.zeros((0, len(SUMMARY_NAMES)))
                for j, name in enumerate(SUMMARY_NAMES):
                    blocks[f"{self.prefix}_{column}_{name}"] = stats[:, j]
            if self._encoder_cache is not None:
                width = self.embedding_dims_[column]
                vectors = self._embeddings(rows)
                if column in self.pca_:
                    reduced = np.full((len(rows), width), np.nan)
                    ok = np.isfinite(vectors).all(axis=1) if vectors.shape[1] else np.zeros(len(rows), bool)
                    if ok.any():
                        reduced[ok] = self.pca_[column].transform(vectors[ok])
                    vectors = reduced
                elif vectors.shape[1] != width:
                    vectors = np.full((len(rows), width), np.nan)
                for j in range(width):
                    blocks[f"{self.prefix}_{column}_emb{j}"] = vectors[:, j]
        features = pd.DataFrame(blocks, index=X.index)[self.feature_names_out_]
        if not self.keep_input:
            return features
        clash = [c for c in features.columns if c in X.columns]
        if clash:
            raise ConfigError(f"Feature columns already exist in the table: {clash[:5]}; change prefix.")
        return pd.concat([X, features], axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """The output columns: the new features, after the input columns when ``keep_input``."""
        if not hasattr(self, "feature_names_out_"):
            raise ConfigError("Call fit() before get_feature_names_out().")
        names = list(self.feature_names_out_)
        if self.keep_input:
            inputs = self.feature_names_in_ if input_features is None else input_features
            names = [*map(str, inputs), *names]
        return np.asarray(names, dtype=object)


def add_series_features(table: pd.DataFrame, series: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """``SeriesFeaturizer(series, **kwargs).fit_transform(table)``.

    When embeddings are PCA-reduced, fit on the training table and ``transform``
    the test table with the same featurizer.
    """
    return SeriesFeaturizer(series, **kwargs).fit_transform(table)


def _frequency(series: pd.DataFrame, id_col: str, time_col: str) -> str:
    """Frequency of the longest history (``"D"`` when it cannot be inferred)."""
    stamps = pd.to_datetime(series[time_col])
    longest = None
    for _, group in stamps.groupby(series[id_col]):
        if longest is None or len(group) > len(longest):
            longest = group
    if longest is None or len(longest) < 3:
        return "D"
    freq = pd.infer_freq(pd.DatetimeIndex(longest.sort_values().unique()))
    return freq or "D"
