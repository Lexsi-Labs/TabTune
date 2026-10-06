"""Time series foundation model adapters.

Deliberately empty: each backend (Chronos, ...) is imported only when a
``TimeSeriesPipeline`` is fitted with it, via the adapter path declared in its
:class:`~tabtune.registry.TimeSeries.TimeSeriesModelSpec`. Importing this
package never pulls in torch or a model library.
"""
