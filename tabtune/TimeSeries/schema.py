"""Input contract for time series pipelines.

Time series data is accepted in long format: one row per ``(item, timestamp)``
with the observed target in its own column; extra targets and covariates are
extra columns named by the schema.

:class:`TimeSeriesSchema` names the roles of the columns and validates a frame
against them. :class:`TimeSeriesPanel` is the validated form the pipeline
hands to model adapters. A schema may name several targets and two kinds of
covariate: ``past_covariates`` are observed only over the history,
``known_covariates`` are also known over the forecast horizon and are supplied
at predict time through a separate future frame. A model that does not support
a schema feature rejects the schema.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from .._internal.deprecation import warn_once
from .forecast import future_timestamps

logger = logging.getLogger(__name__)

__all__ = ["TimeSeriesSchema", "TimeSeriesPanel"]


@dataclass(frozen=True, eq=False)
class TimeSeriesPanel:
    """A validated panel of series, ready for a model adapter.

    One row is one ``(item, target)`` pair: a univariate panel has one row per
    item, a panel with three targets three consecutive rows per item in schema
    order. An adapter that forecasts variates jointly recovers the grouping
    from :meth:`item_groups`.

    Attributes:
        item_ids: One identifier per row. ``(None,)`` for a single series
            given without an item column. Repeated across a multivariate item.
        values: Per row, the target history as a 1-D ``float64`` array in
            time order. ``NaN`` marks a missing observation; every other value
            is finite.
        last_timestamps: Per row, the timestamp of the final observation;
            forecasts start one step after it.
        freq: Pandas frequency string shared by every series.
        target_names: Per row, the name of the target column it holds.
        past_covariates: Per row, a mapping of covariate name to a 1-D array
            as long as that row's history. Shared by the rows of one item.
        future_covariates: Per row, a mapping of covariate name to a 1-D array
            as long as the forecast horizon. Shared by the rows of one item.
            Empty unless the schema names ``known_covariates``.
    """

    item_ids: tuple[Hashable, ...]
    values: tuple[np.ndarray, ...]
    last_timestamps: tuple[pd.Timestamp, ...]
    freq: str
    target_names: tuple[str, ...]
    past_covariates: tuple[Mapping[str, np.ndarray], ...] = ()
    future_covariates: tuple[Mapping[str, np.ndarray], ...] = ()

    def __post_init__(self) -> None:
        rows = len(self.item_ids)
        for name in ("values", "last_timestamps", "target_names"):
            if len(getattr(self, name)) != rows:
                raise ValueError(
                    f"TimeSeriesPanel.{name} has {len(getattr(self, name))} entries "
                    f"but there are {rows} rows"
                )
        for name in ("past_covariates", "future_covariates"):
            given = getattr(self, name)
            if not given:
                object.__setattr__(self, name, tuple({} for _ in range(rows)))
            elif len(given) != rows:
                raise ValueError(
                    f"TimeSeriesPanel.{name} has {len(given)} entries but there "
                    f"are {rows} rows"
                )

    def __len__(self) -> int:
        return len(self.item_ids)

    @property
    def max_length(self) -> int:
        """Length of the longest series history."""
        return max((len(v) for v in self.values), default=0)

    @property
    def n_targets(self) -> int:
        """Number of target variates per item."""
        return len(dict.fromkeys(self.target_names))

    @property
    def covariate_names(self) -> tuple[str, ...]:
        """Names of the past covariates carried by the panel."""
        return tuple(self.past_covariates[0]) if self.past_covariates else ()

    def item_groups(self) -> tuple[tuple[int, ...], ...]:
        """Return the row indices of each item, in panel order.

        A univariate panel gives one index per group. An adapter that forecasts
        an item's variates jointly uses these groups to assemble its inputs.
        """
        groups: dict[Hashable, list[int]] = {}
        for row, item in enumerate(self.item_ids):
            groups.setdefault(item, []).append(row)
        return tuple(tuple(rows) for rows in groups.values())

    def tail(self, n: int) -> TimeSeriesPanel:
        """Return a copy keeping only the most recent ``n`` observations per row.

        ``last_timestamps`` is unchanged: truncating the context does not move
        the forecast origin. Past covariates are cut to the same window; future
        covariates describe the horizon and are left alone.
        """
        return replace(
            self,
            values=tuple(v[-n:] for v in self.values),
            past_covariates=tuple(
                {name: values[-n:] for name, values in row.items()}
                for row in self.past_covariates
            ),
        )

    def check_observed(self, *, context: int | None = None) -> None:
        """Raise if any row has no observed value.

        Run on the window a model actually sees: cutting to the context length
        can leave an all-missing window.

        Raises:
            ValueError: Naming the first row with no observed value.
        """
        for item, target, values in zip(
            self.item_ids, self.target_names, self.values, strict=True
        ):
            if np.isnan(values).all():
                window = f"last {context} observations" if context else "history"
                raise ValueError(
                    f"{_label(item, target if self.n_targets > 1 else None)} has no "
                    f"observed target values in its {window}, so there is nothing to "
                    f"forecast from. Drop the series or supply recent observations."
                )

    def fingerprint(self) -> str:
        """Return a content hash of the panel, used as the forecast cache key.

        Covers everything that changes a forecast input: item ids, target names,
        values, covariate values, forecast origins and frequency.
        """
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(self.freq.encode("utf-8"))
        for item, target, values, last, past, future in zip(
            self.item_ids,
            self.target_names,
            self.values,
            self.last_timestamps,
            self.past_covariates,
            self.future_covariates,
            strict=True,
        ):
            hasher.update(repr((item, target, last, len(values))).encode("utf-8"))
            hasher.update(np.ascontiguousarray(values, dtype="float64").tobytes())
            for group in (past, future):
                for name in sorted(group):
                    hasher.update(repr(name).encode("utf-8"))
                    hasher.update(_covariate_bytes(group[name]))
        return hasher.hexdigest()


@dataclass(frozen=True)
class TimeSeriesSchema:
    """Column roles of a long-format time series frame.

    Attributes:
        target: Column holding the value to forecast, or a sequence of columns
            for a multivariate target forecast jointly.
        timestamp: Column holding the observation time. Anything
            :func:`pandas.to_datetime` accepts.
        item_id: Column identifying the series in a panel. ``None`` treats the
            whole frame as a single series.
        freq: Pandas frequency of the series (``"D"``, ``"h"``, ``"MS"``...).
            ``None`` infers it per series and requires the inferences to agree.
        past_covariates: Columns observed alongside the target over the history.
        known_covariates: Columns also known over the forecast horizon.
            Their future values are supplied to ``predict`` through ``future_df``.

    Covariate columns may be numeric or categorical; a model that does not
    support covariates, several targets, or categorical covariates rejects the
    schema when the pipeline is fitted.
    """

    target: str | Sequence[str]
    timestamp: str = "timestamp"
    item_id: str | None = None
    freq: str | None = None
    past_covariates: tuple[str, ...] = ()
    known_covariates: tuple[str, ...] = ()
    target_names: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.target, str):
            targets: tuple[str, ...] = (self.target,)
        elif isinstance(self.target, Sequence):
            targets = tuple(self.target)
            if not targets:
                raise ValueError("TimeSeriesSchema.target must name at least one column")
        else:
            raise ValueError(
                f"TimeSeriesSchema.target must be a column name or a sequence of "
                f"column names, got {self.target!r}"
            )
        object.__setattr__(self, "target_names", targets)
        object.__setattr__(self, "past_covariates", tuple(self.past_covariates))
        object.__setattr__(self, "known_covariates", tuple(self.known_covariates))

        roles: list[tuple[str, str | None]] = [
            *((f"target[{i}]", name) for i, name in enumerate(targets)),
            ("timestamp", self.timestamp),
            ("item_id", self.item_id),
            *((f"past_covariates[{i}]", n) for i, n in enumerate(self.past_covariates)),
            *((f"known_covariates[{i}]", n) for i, n in enumerate(self.known_covariates)),
        ]
        for role, name in roles:
            if name is None and role == "item_id":
                continue
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"TimeSeriesSchema.{role} must be a non-empty column name, got {name!r}"
                )
        named = [name for _, name in roles if name is not None]
        if len(set(named)) != len(named):
            duplicates = sorted({n for n in named if named.count(n) > 1})
            raise ValueError(
                f"Each column can play only one role; {duplicates} appear more than once "
                f"across target={self.target!r}, timestamp={self.timestamp!r}, "
                f"item_id={self.item_id!r}, past_covariates={self.past_covariates!r}, "
                f"known_covariates={self.known_covariates!r}"
            )

    @property
    def covariate_names(self) -> tuple[str, ...]:
        """Every covariate column, past-only first then known-future."""
        return (*self.past_covariates, *self.known_covariates)

    @property
    def is_multivariate(self) -> bool:
        """Whether the schema names more than one target."""
        return len(self.target_names) > 1

    @property
    def columns(self) -> tuple[str, ...]:
        """The columns this schema reads, in canonical order."""
        keys = (self.item_id,) if self.item_id is not None else ()
        return (*keys, self.timestamp, *self.target_names, *self.covariate_names)

    @property
    def future_columns(self) -> tuple[str, ...]:
        """The columns a ``future_df`` must carry, in canonical order."""
        keys = (self.item_id,) if self.item_id is not None else ()
        return (*keys, self.timestamp, *self.known_covariates)

    def to_panel(
        self,
        df: pd.DataFrame,
        *,
        native_missing: bool = False,
        future_df: pd.DataFrame | None = None,
        horizon: int | None = None,
    ) -> TimeSeriesPanel:
        """Validate ``df`` against this schema and return the canonical panel.

        The input frames are never modified.

        Args:
            df: Long-format history frame.
            native_missing: Whether the model accepts ``NaN`` in the target
                history. When ``False``, missing targets are rejected.
            future_df: Known-future covariate values covering the horizon, one
                row per ``(item, timestamp)``. Optional: the history is valid
                without it, and :meth:`attach_future` can supply it later.
            horizon: Forecast length, needed to check ``future_df`` covers
                exactly the steps that follow each series.

        Returns:
            The validated panel, one row per ``(item, target)``, items ordered
            by id and each series in time order.

        Raises:
            TypeError: If ``df`` is not a DataFrame.
            ValueError: On missing columns, a non-numeric target, unparseable
                timestamps, infinite target values, duplicate
                ``(item, timestamp)`` keys, an irregular or inconsistent
                frequency, disallowed missing values, a series with no
                observations, or a ``future_df`` that does not line up with the
                history.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"time series data must be a pandas DataFrame in long format, "
                f"got {type(df).__name__}"
            )

        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} named by the schema are not in the frame. "
                f"Available columns: {list(df.columns)}"
            )

        extra = [c for c in df.columns if c not in self.columns]
        if extra:
            warn_once(
                f"Columns {extra} are ignored: they play no role in this schema. "
                f"Name them in past_covariates or known_covariates to use them.",
                UserWarning,
                key=f"ts-extra-columns:{','.join(map(str, extra))}",
            )

        frame = df.loc[:, list(self.columns)].copy()

        for target in self.target_names:
            if not pd.api.types.is_numeric_dtype(frame[target]) or pd.api.types.is_bool_dtype(
                frame[target]
            ):
                raise ValueError(
                    f"Target column {target!r} must be numeric, got dtype {frame[target].dtype}"
                )
            frame[target] = frame[target].astype("float64")
            infinite = np.isinf(frame[target].to_numpy())
            if infinite.any():
                raise ValueError(
                    f"Target column {target!r} contains {int(infinite.sum())} infinite "
                    f"value(s). Only NaN may mark a missing observation; replace or drop them."
                )

        for name in self.covariate_names:
            _check_covariate(frame[name], name)

        try:
            frame[self.timestamp] = pd.to_datetime(frame[self.timestamp])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Timestamp column {self.timestamp!r} could not be parsed as datetimes: {exc}"
            ) from exc
        if frame[self.timestamp].isna().any():
            raise ValueError(f"Timestamp column {self.timestamp!r} contains missing values")
        if self.item_id is not None and frame[self.item_id].isna().any():
            raise ValueError(f"Item column {self.item_id!r} contains missing values")

        keys = [c for c in (self.item_id, self.timestamp) if c is not None]
        duplicated = frame.duplicated(subset=keys)
        if duplicated.any():
            example = frame.loc[duplicated, keys].iloc[0].to_dict()
            raise ValueError(
                f"Found {int(duplicated.sum())} duplicate {tuple(keys)} key(s), "
                f"e.g. {example}. Each series needs one row per timestamp."
            )

        frame = frame.sort_values(keys, kind="mergesort").reset_index(drop=True)

        if self.item_id is None:
            groups: list[tuple[Hashable, pd.DataFrame]] = [(None, frame)]
        else:
            groups = list(frame.groupby(self.item_id, sort=False, observed=True))

        freq = self._resolve_freq(groups)

        item_ids: list[Hashable] = []
        values: list[np.ndarray] = []
        last_timestamps: list[pd.Timestamp] = []
        target_names: list[str] = []
        past_covariates: list[Mapping[str, np.ndarray]] = []
        for item, group in groups:
            past = {name: _covariate_array(group[name]) for name in self.covariate_names}
            for target in self.target_names:
                series = group[target].to_numpy(dtype="float64")
                if not native_missing and np.isnan(series).any():
                    raise ValueError(
                        f"{_label(item, target if self.is_multivariate else None)} has "
                        f"{int(np.isnan(series).sum())} missing target value(s) and this "
                        f"model does not accept missing values. Impute or drop them "
                        f"before fitting."
                    )
                item_ids.append(item)
                values.append(series)
                last_timestamps.append(group[self.timestamp].iloc[-1])
                target_names.append(target)
                past_covariates.append(past)

        panel = TimeSeriesPanel(
            item_ids=tuple(item_ids),
            values=tuple(values),
            last_timestamps=tuple(last_timestamps),
            freq=freq,
            target_names=tuple(target_names),
            past_covariates=tuple(past_covariates),
        )
        panel.check_observed()
        if future_df is not None:
            panel = self.attach_future(panel, future_df, horizon=horizon)
        return panel

    def attach_future(
        self, panel: TimeSeriesPanel, future_df: pd.DataFrame, *, horizon: int | None
    ) -> TimeSeriesPanel:
        """Return ``panel`` with known-future covariate values attached.

        ``future_df`` must cover exactly the ``horizon`` steps after each
        series' last observation.

        Raises:
            ValueError: If the schema names no ``known_covariates``, or if
                ``future_df`` does not line up with the panel.
        """
        if not self.known_covariates:
            raise ValueError(
                "future_df was given but the schema names no known_covariates; "
                "list the columns known over the horizon in "
                "TimeSeriesSchema(known_covariates=...)."
            )
        if not isinstance(future_df, pd.DataFrame):
            raise TypeError(
                f"future_df must be a pandas DataFrame in long format, "
                f"got {type(future_df).__name__}"
            )
        if horizon is None:
            raise ValueError("horizon is required to validate future_df")

        missing = [c for c in self.future_columns if c not in future_df.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} are not in future_df. It needs "
                f"{list(self.future_columns)}; available: {list(future_df.columns)}"
            )
        forbidden = [c for c in self.target_names if c in future_df.columns]
        if forbidden:
            raise ValueError(
                f"future_df must not carry the target column(s) {forbidden}: it describes "
                f"what is known before the forecast, not the values being forecast."
            )

        frame = future_df.loc[:, list(self.future_columns)].copy()
        for name in self.known_covariates:
            _check_covariate(frame[name], name, where="future_df")
        try:
            frame[self.timestamp] = pd.to_datetime(frame[self.timestamp])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"future_df timestamp column {self.timestamp!r} could not be parsed "
                f"as datetimes: {exc}"
            ) from exc

        keys = [c for c in (self.item_id, self.timestamp) if c is not None]
        if frame.duplicated(subset=keys).any():
            raise ValueError(
                f"future_df has duplicate {tuple(keys)} key(s); it needs one row per "
                f"item and horizon step."
            )
        frame = frame.sort_values(keys, kind="mergesort")

        if self.item_id is None:
            future_groups: dict[Hashable, pd.DataFrame] = {None: frame}
        else:
            future_groups = {
                item: rows
                for item, rows in frame.groupby(self.item_id, sort=False, observed=True)
            }

        origins = dict(zip(panel.item_ids, panel.last_timestamps, strict=True))
        unknown = [i for i in future_groups if i not in origins]
        if unknown:
            raise ValueError(
                f"future_df covers item(s) {unknown[:5]} that are not in the history."
            )

        per_item: dict[Hashable, Mapping[str, np.ndarray]] = {}
        for item, last in origins.items():
            rows = future_groups.get(item)
            if rows is None:
                raise ValueError(f"future_df has no rows for {_label(item)}.")
            expected = future_timestamps(last, panel.freq, horizon)
            actual = pd.DatetimeIndex(rows[self.timestamp])
            if not actual.equals(expected):
                raise ValueError(
                    f"future_df for {_label(item)} must cover exactly the {horizon} "
                    f"timestamp(s) after its last observation "
                    f"({expected[0]} ... {expected[-1]} on a {panel.freq!r} grid), "
                    f"got {len(actual)} row(s) starting at "
                    f"{actual[0] if len(actual) else 'nothing'}."
                )
            per_item[item] = {
                name: _covariate_array(rows[name]) for name in self.known_covariates
            }

        return replace(
            panel,
            future_covariates=tuple(per_item[item] for item in panel.item_ids),
        )

    def _resolve_freq(self, groups: list[tuple[Hashable, pd.DataFrame]]) -> str:
        """Verify or infer one regular frequency shared by every series."""
        if self.freq is not None:
            try:
                offset = to_offset(self.freq)
            except ValueError as exc:
                raise ValueError(f"Invalid frequency {self.freq!r}: {exc}") from exc
            for item, group in groups:
                stamps = pd.DatetimeIndex(group[self.timestamp])
                expected = pd.date_range(start=stamps[0], periods=len(stamps), freq=offset)
                if not stamps.equals(expected):
                    raise ValueError(
                        f"{_label(item)} is not on a regular {offset.freqstr!r} grid. "
                        f"Resample it to a regular frequency first, e.g. "
                        f"df.set_index({self.timestamp!r}).resample({offset.freqstr!r})."
                    )
            return offset.freqstr

        inferred: dict[str, Hashable] = {}
        for item, group in groups:
            stamps = pd.DatetimeIndex(group[self.timestamp])
            if len(stamps) < 3:
                raise ValueError(
                    f"{_label(item)} has {len(stamps)} observation(s), too few to infer "
                    f"its frequency. Set TimeSeriesSchema(freq=...) explicitly."
                )
            freq = pd.infer_freq(stamps)
            if freq is None:
                raise ValueError(
                    f"Could not infer a regular frequency for {_label(item)}: its "
                    f"timestamps are irregular or have gaps. Resample it to a regular "
                    f"frequency first, or set TimeSeriesSchema(freq=...) to check a "
                    f"specific grid."
                )
            inferred.setdefault(to_offset(freq).freqstr, item)

        if len(inferred) > 1:
            details = ", ".join(f"{f!r} ({_label(i)})" for f, i in inferred.items())
            raise ValueError(
                f"Series have different frequencies: {details}. Every series in one "
                f"pipeline must share a frequency."
            )
        return next(iter(inferred))


def _label(item: Hashable, target: str | None = None) -> str:
    base = "the series" if item is None else f"item {item!r}"
    return base if target is None else f"{base} target {target!r}"


def _check_covariate(column: pd.Series, name: str, *, where: str = "the frame") -> None:
    """Reject covariates a model could not read: nulls, or non-finite numbers."""
    if column.isna().any():
        raise ValueError(
            f"Covariate column {name!r} in {where} contains {int(column.isna().sum())} "
            f"missing value(s). Covariates must be fully observed; impute them first."
        )
    if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
        if np.isinf(column.to_numpy(dtype="float64")).any():
            raise ValueError(f"Covariate column {name!r} in {where} contains infinite value(s)")


def _covariate_array(column: pd.Series) -> np.ndarray:
    """Numeric covariates become float64; everything else stays categorical."""
    if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
        return column.to_numpy(dtype="float64")
    return column.astype(str).to_numpy()


def _covariate_bytes(values: np.ndarray) -> bytes:
    """Stable bytes for a covariate array, numeric or categorical."""
    if values.dtype.kind in "fiu":
        return np.ascontiguousarray(values, dtype="float64").tobytes()
    return repr(values.tolist()).encode("utf-8")
