"""Datadog's Toto time series models, one subpackage per generation.

Both generations live here for the same reason the three Chronos models share
``tabtune/models/chronos/``: one upstream family, one folder. Unlike Chronos they
cannot share a *namespace*, because they are genuinely different codebases that
happen to share a name -- Toto 1.0 ships a ``model/`` subpackage where Toto 2.0
ships a ``model.py``, so flattening them would collide. Hence two subpackages:

* :mod:`tabtune.models.toto.toto1` -- the ``toto-ts`` 0.2.0 inference closure
  (13 upstream modules in upstream's own layout, plus ``_gluonts.py``, the four
  gluonts symbols its core needs). Sample-based.
* :mod:`tabtune.models.toto.toto2` -- the ``toto-2`` 2.0.0 model
  (``configuration.py`` verbatim and ``model.py`` with three GluonTS-only
  deletions). Quantile-native.

Provenance constants and re-exports live in each subpackage's own
``__init__.py``, since the two were vendored from different releases.

**This module deliberately imports nothing.** Both subpackages need torch (and
Toto 2.0 its own pip extra, ``tabtune[toto2]``), and the adapters
in :mod:`tabtune.models.TimeSeries` import them inside ``load()`` so that
``import tabtune`` stays torch-free. Re-exporting either one here would import
torch as a side effect of touching the parent package.
"""
