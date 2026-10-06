"""Toto 2.0 (Datadog) vendored for TabTune: the inference model.

Vendored from the ``toto-2`` PyPI release rather than installed as that package,
the same way TabTune ships Chronos, TimesFM, TabFM and TabPFN v3. Unlike those,
Toto 2.0 cannot be vendored *completely*: it is built on Datadog's u-microP
parameterisation and imports ``dd_unit_scaling`` (and through it Graphcore's
``unit_scaling``) plus ``jaxtyping`` at module level. Those are general-purpose
libraries rather than model code, and ``dd-unit-scaling`` ships no license file,
so they are declared as the ``toto2`` pip extra instead:

    pip install 'tabtune[toto2]'

That makes Toto 2.0 the only model that uses ``TimeSeriesModelSpec.dependency_extra``.
Both libraries declare ``requires-python >= 3.12``; they run correctly on 3.11,
so an install there needs ``--ignore-requires-python``.

Provenance (the constants below record it in code):

* ``configuration.py`` is a **verbatim** copy of upstream.
* ``model.py`` is upstream with **three deletions and nothing else**, listed in
  its header: the ``gluonts`` import block, the ``_FnImputation`` helper, and the
  "GluonTS Integration" section (``Toto2GluonTSModel``). All three are
  GluonTS-only code belonging to upstream's benchmark harness, and dropping them
  means ``gluonts`` and ``matplotlib`` are not needed at all. 1336 of upstream's
  1509 lines are kept, and no import rewrites were necessary.
* This ``__init__.py`` is the one file TabTune wrote rather than copied.
* Toto is published as a PyPI sdist rather than a tagged release we vendored
  from, so provenance is the release plus that sdist's digest rather than a git
  commit.

The ``toto2.model`` module also exposes ``ffill_imputation`` and
``linear_imputation``; the adapter calls them so that missing values are filled
exactly the way upstream's own inference harness fills them.

Nothing here is imported by ``import tabtune``: the adapter in
``tabtune.models.TimeSeries.toto2`` loads this package when a
``TimeSeriesPipeline`` is fitted.
"""

from .configuration import Toto2ModelConfig
from .model import Toto2Model, ffill_imputation, linear_imputation

#: Upstream release these files were taken from.
UPSTREAM_VERSION = "2.0.0"
#: sha256 of ``toto_2-2.0.0.tar.gz`` on PyPI, in place of a git commit.
UPSTREAM_SDIST_SHA256 = "ce23ce328c593b8baaedcd679dfc49c4296cb06c4a46654caab7ce66ef4087a7"
UPSTREAM_URL = "https://github.com/DataDog/toto"

__all__ = [
    "Toto2Model",
    "Toto2ModelConfig",
    "ffill_imputation",
    "linear_imputation",
    "UPSTREAM_VERSION",
    "UPSTREAM_SDIST_SHA256",
    "UPSTREAM_URL",
]
