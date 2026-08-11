"""Turn a raw mixed-type table into the numeric array EXAONE Tabular requires.

The vendored stack has a hard input contract: ``np.ndarray``, real numeric dtype,
rank 2, no ``inf``. It rejects ``object`` columns, strings, pandas DataFrames and
pandas categoricals at the estimator boundary, and its own preprocessor raises
``TypeError: features must have a real numeric dtype`` on anything else. There is
no categorical encoder inside the model.

That is not a gap so much as a division of labour — EXAONE *does* handle
categoricals, but downstream and implicitly: the ensemble builder treats any
column with fewer than ten distinct support values as categorical and gives each
ensemble member its own random permutation of the codes. For that to mean
anything, the codes have to arrive as small contiguous integers. This encoder
produces exactly that.

It is deliberately minimal, and shared by the classifier, the regressor and the
``Dataprocess`` preprocessor so all three agree on the encoding. Anything richer
(target encoding, feature crosses) would fight the model's own preprocessing,
which standardises, selects features by attention, and re-inserts NaN at
originally-missing positions so missingness survives as an explicit signal.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Missing values are passed through as NaN rather than imputed here. The model
#: imputes twice on purpose — once in its preprocessor (support-only column
#: means) and again inside the feature encoder, which additionally emits a
#: separate non-finite code channel (0 finite, -2 NaN, 2 +inf, 4 -inf) through
#: the same affine as the values. Filling NaN early would destroy that signal.
_MISSING = np.nan


class EXAONEFeatureEncoder:
    """Ordinal-encode a mixed table to ``float64``, fit once and frozen.

    Categorical and boolean columns become contiguous integer codes in first-seen
    (sorted) order; numeric columns pass through as floats. Unseen categories at
    transform time map to NaN, which the model's own encoder then treats as
    missing — the honest behaviour, since an unseen category genuinely carries no
    information the support set can condition on.

    Picklable by construction (no lambdas, no closures): ``TabularPipeline.save``
    joblib-dumps the entire pipeline including this object.

    Attributes:
        columns_: Column names seen at fit, in order.
        categorical_columns_: Names treated as categorical.
        categories_: ``column -> ordered list of category values``.
        n_features_in_: Number of columns at fit.

    Example:
        >>> enc = EXAONEFeatureEncoder().fit(df_train)
        >>> X = enc.transform(df_test)          # (n, K) float64, NaN allowed
        >>> X.dtype
        dtype('float64')
    """

    def __init__(self, max_cardinality: int = 10_000) -> None:
        self.max_cardinality = int(max_cardinality)
        self.columns_: Optional[List[str]] = None
        self.categorical_columns_: Optional[List[str]] = None
        self.categories_: Optional[dict] = None
        self.n_features_in_: Optional[int] = None

    # ------------------------------------------------------------------ #
    def fit(self, X, y=None) -> "EXAONEFeatureEncoder":
        frame = _as_frame(X)
        self.columns_ = [str(c) for c in frame.columns]
        self.categorical_columns_ = []
        self.categories_ = {}

        for name in self.columns_:
            column = frame[name]
            if not _is_categorical(column):
                continue
            values = column.dropna().unique()
            if len(values) > self.max_cardinality:
                logger.warning(
                    "[EXAONE] column %r has %d distinct values (> max_cardinality=%d); "
                    "encoding it anyway, but a high-cardinality identifier column is "
                    "rarely useful and the model's feature selection may drop it.",
                    name, len(values), self.max_cardinality,
                )
            self.categorical_columns_.append(name)
            self.categories_[name] = sorted(values, key=_sort_key)

        self.n_features_in_ = len(self.columns_)
        return self

    def transform(self, X) -> np.ndarray:
        import pandas as pd

        if self.columns_ is None:
            raise RuntimeError("EXAONEFeatureEncoder must be fitted before transform().")
        frame = _as_frame(X)

        missing = [c for c in self.columns_ if c not in frame.columns]
        if missing:
            raise ValueError(
                f"EXAONEFeatureEncoder was fitted on columns {self.columns_} but "
                f"{missing} are absent from the frame passed to transform()."
            )

        out = np.empty((len(frame), len(self.columns_)), dtype=np.float64)
        for j, name in enumerate(self.columns_):
            column = frame[name]
            if name in self.categorical_columns_:
                lookup = {value: code for code, value in enumerate(self.categories_[name])}
                # pd.isna, not ``value == value``: pandas' nullable dtypes
                # (Int64/string/boolean, which is what convert_dtypes(), Parquet
                # and Arrow all produce) use pd.NA, and ``pd.NA == pd.NA`` is
                # pd.NA -- truth-testing it raises "boolean value of NA is
                # ambiguous" rather than reporting missingness.
                out[:, j] = [
                    _MISSING if pd.isna(value) else lookup.get(value, _MISSING)
                    for value in column.to_numpy(dtype=object)
                ]
            else:
                out[:, j] = _to_numeric(column)

        # The vendored preprocessor raises on inf. Map it to NaN, which the model
        # already has a channel for, instead of failing a whole fit on one row.
        non_finite = np.isinf(out)
        if non_finite.any():
            logger.warning(
                "[EXAONE] %d infinite value(s) mapped to NaN; the model rejects inf "
                "outright but encodes NaN as an explicit missingness signal.",
                int(non_finite.sum()),
            )
            out[non_finite] = np.nan
        return out

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


# --------------------------------------------------------------------------- #
def _as_frame(X):
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        return X
    array = np.asarray(X)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return pd.DataFrame(array, columns=[f"f{i}" for i in range(array.shape[1])])


def _is_categorical(column) -> bool:
    import pandas as pd

    if isinstance(column.dtype, pd.CategoricalDtype):
        return True
    kind = getattr(column.dtype, "kind", None)
    # 'O' object, 'U'/'S' strings, 'b' bool. Booleans go through the categorical
    # path so they arrive as 0/1 codes rather than as a numeric column the
    # ensemble's numeric transforms (Yeo-Johnson, rank-gaussianisation) would
    # pointlessly reshape.
    return kind in {"O", "U", "S", "b"}


def _to_numeric(column) -> np.ndarray:
    import pandas as pd

    if getattr(column.dtype, "kind", None) in {"M", "m"}:
        # datetimes / timedeltas -> int64 nanoseconds, NaT preserved as NaN
        return column.astype("int64").where(column.notna(), np.nan).to_numpy(dtype=np.float64)
    return pd.to_numeric(column, errors="coerce").to_numpy(dtype=np.float64)


def _sort_key(value):
    """Order categories deterministically across mixed types."""
    return (str(type(value).__name__), str(value))


__all__ = ["EXAONEFeatureEncoder"]
