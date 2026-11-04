# Series Histories as Features

The other bridge. A customer table, a sensor table, a store table: each row is an entity
with a history somewhere else (transactions, readings, sales). `SeriesFeaturizer` turns that
history into features of the row, leak-safe, for any `TabularPipeline` model.

```python
from tabtune.bridge import SeriesFeaturizer

feat = SeriesFeaturizer(
    transactions,                 # long frame: one row per (customer, date)
    id_col="customer",
    time_col="date",
    cutoff="as_of",               # column of the entity table: the row's own timestamp
    model="TiRex",                # optional: a time series model for embeddings
    n_components=4,               # PCA of the embeddings
)
X = feat.fit_transform(customers)
```

Each row of `customers` receives summary statistics of its history (count, mean, standard
deviation, trend, last value, recency, and so on) and, when a model is named, a compressed
embedding of that history, all computed only from observations at or before the row's own
`cutoff`. A row dated in March never sees April. That is the leak-safety guarantee, and it
is why `cutoff` is required.

`add_series_features(entities, history, ...)` is the one-call form. The featurizer is a
scikit-learn transformer, so it composes with `Pipeline` and with
[Custom Preprocessing](../advanced/custom-preprocessing.md).

Then it is an ordinary tabular problem:

```python
from tabtune import TabularPipeline

pipe = TabularPipeline("Kumo", task_type="classification", tuning_strategy="inference")
pipe.fit(X_train, y_train)
```

`examples/timeseries/embeddings_and_series_features.py` builds the whole thing end to end.
