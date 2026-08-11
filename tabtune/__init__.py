"""TabTune - a unified library for inference and fine-tuning of tabular foundation models.

TabTune provides one scikit-learn-style API across thirteen tabular foundation
models and four adaptation strategies, plus the evaluation machinery those
models actually need: calibration diagnostics, fairness auditing, shift-aware
splits, distillation and ensembling.

Quick start:
    >>> from tabtune import TabularPipeline
    >>> pipe = TabularPipeline("TabICLv2", task_type="classification")   # doctest: +SKIP
    >>> pipe.fit(X_train, y_train)                                       # doctest: +SKIP
    >>> pipe.evaluate(X_test, y_test)                                    # doctest: +SKIP

Discover what is available and what you may deploy:
    >>> from tabtune.registry import list_models
    >>> ", ".join(s.name for s in list_models(task="regression", commercial_ok=True))
    'Mitra, TabICLv2'

Imports are lazy. ``import tabtune`` costs milliseconds and pulls in neither
torch nor transformers; the heavy modules load on first attribute access, so a
CLI listing models or a docs build reading the registry never pays for them.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING, Any

# Version is sourced from installed package metadata so it never drifts from
# pyproject.toml. The literal below is the single fallback for an uninstalled
# source tree; setup.py and this module previously carried their own copies and
# disagreed with pyproject.toml, giving three different answers for __version__.
__version__ = "0.2.0"
try:
    __version__ = _pkg_version("tabtune")
except PackageNotFoundError:  # running from a source checkout
    pass

__author__ = (
    "Aditya Tanna, Pratinav Seth, Mohamed Bouadi, Utsav Avaiya, Yash Desai, "
    "Nassim Bouarour, Vinay Kumar Sankarapu"
)
__maintainer__ = "Aditya Tanna"
__email__ = "contact@lexsi.ai"
__maintainer_email__ = "contact@lexsi.ai"

from .logger import setup_logger

setup_logger()

# The registry is torch-free and cheap, and is the entry point for discovery
# ("which models can I use?"), so it stays eager. `config` pulls pydantic and is
# resolved lazily below.
from . import registry  # noqa: E402
from .registry.errors import (  # noqa: E402
    ConfigError,
    EnvelopeError,
    LicenseError,
    ModelNotFoundError,
    TabTuneError,
    UnsupportedStrategyError,
    UnsupportedTaskError,
)

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from .caching import PredictionCache
    from .Dataprocess.data_processor import DataProcessor
    from .ensemble.tabular_ensemble import TabularEnsemble
    from .evaluation import GroupedSplit, ShiftEvaluator, StratifiedGroupedSplit, TemporalSplit
    from .TabularLeaderboard.leaderboard import LeaderboardEntry, TabularLeaderboard
    from .TabularPipeline.pipeline import TabularPipeline
    from .TuningManager.tuning import TuningManager

# name -> (module path, attribute). Resolved on first access by __getattr__.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "TabularPipeline": (".TabularPipeline.pipeline", "TabularPipeline"),
    "TabularLeaderboard": (".TabularLeaderboard.leaderboard", "TabularLeaderboard"),
    "LeaderboardEntry": (".TabularLeaderboard.leaderboard", "LeaderboardEntry"),
    "DataProcessor": (".Dataprocess.data_processor", "DataProcessor"),
    "TuningManager": (".TuningManager.tuning", "TuningManager"),
    "TabularEnsemble": (".ensemble.tabular_ensemble", "TabularEnsemble"),
    "PredictionCache": (".caching", "PredictionCache"),
    "ShiftEvaluator": (".evaluation", "ShiftEvaluator"),
    "TemporalSplit": (".evaluation", "TemporalSplit"),
    "GroupedSplit": (".evaluation", "GroupedSplit"),
    "StratifiedGroupedSplit": (".evaluation", "StratifiedGroupedSplit"),
}

__all__ = [
    "TabularPipeline",
    "TabularLeaderboard",
    "LeaderboardEntry",
    "DataProcessor",
    "TuningManager",
    "TabularEnsemble",
    "PredictionCache",
    "ShiftEvaluator",
    "TemporalSplit",
    "GroupedSplit",
    "StratifiedGroupedSplit",
    "registry",
    "config",
    "uncertainty",
    # errors
    "TabTuneError",
    "ConfigError",
    "ModelNotFoundError",
    "UnsupportedTaskError",
    "UnsupportedStrategyError",
    "EnvelopeError",
    "LicenseError",
    "__version__",
]


# Two public class names collide with subpackage names of the same spelling:
# `tabtune.TabularPipeline` is both the class and the package containing it,
# and likewise for `tabtune.TabularLeaderboard`. Importing the submodule binds
# the *module* onto the parent package, which would shadow the class for any
# code that touched the submodule first. The eager imports used before 0.2.0
# hid this by always running last; lazy loading exposes it, so these two names
# are resolved through __getattribute__ on a module subclass instead.
_SHADOWED_BY_SUBPACKAGE = frozenset({"TabularPipeline", "TabularLeaderboard"})

# Lazily-imported submodules exposed as attributes of the package.
_LAZY_SUBMODULES = frozenset(
    {
        "config",
        "caching",
        "evaluation",
        "distillation",
        "ensemble",
        "resampling",
        # Conformal prediction + recalibration. Pure numpy/scipy over
        # predict_proba, but lazy for symmetry and import-time hygiene.
        "uncertainty",
    }
)


def _resolve_lazy(name: str) -> Any:
    """Import and return the object behind a lazily-exposed public name."""
    from importlib import import_module

    if name in _LAZY_SUBMODULES:
        return import_module(f".{name}", __name__)

    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attribute = target
    try:
        module = import_module(module_path, __name__)
    except ImportError as exc:
        raise ImportError(
            f"Could not import {name} from {module_path}: {exc}\n"
            f"This usually means an optional dependency is missing. Install the "
            f"full stack with `pip install 'tabtune[all]'`."
        ) from exc
    return getattr(module, attribute)


def __getattr__(name: str) -> Any:
    """Resolve public names lazily.

    Importing ``TabularPipeline`` pulls in all thirteen vendored model families
    plus torch, transformers and sentence-transformers - several seconds and
    around a gigabyte of RAM. Deferring that to first use keeps ``import
    tabtune`` cheap for the many entry points that only need the registry or
    the config schemas.

    Raises:
        AttributeError: If ``name`` is not a public TabTune symbol.
        ImportError: If the backing module exists but a dependency is missing.
    """
    value = _resolve_lazy(name)
    globals()[name] = value  # cache so later accesses skip __getattr__
    return value


def __dir__() -> list[str]:
    """Expose lazy names to tab completion and ``dir()``."""
    return sorted(set(globals()) | set(__all__))


def _install_shadow_guard() -> None:
    """Make the two shadowed names always resolve to their classes.

    Without this, ``import tabtune.TabularPipeline.pipeline`` binds the
    subpackage onto ``tabtune.TabularPipeline``, and every later
    ``tabtune.TabularPipeline(...)`` raises ``TypeError: 'module' object is not
    callable``. The subpackages stay importable as normal - ``import
    tabtune.TabularPipeline.pipeline`` and ``from tabtune.TabularPipeline
    import leaderboard`` both go through ``sys.modules`` and are unaffected.
    """
    import sys
    import types

    package = sys.modules[__name__]

    class _TabTunePackage(types.ModuleType):
        __slots__ = ()

        def __getattribute__(self, name: str) -> Any:
            if name in _SHADOWED_BY_SUBPACKAGE:
                cache_key = f"_class_{name}"
                cached = self.__dict__.get(cache_key)
                if cached is None:
                    cached = _resolve_lazy(name)
                    self.__dict__[cache_key] = cached
                return cached
            return super().__getattribute__(name)

    package.__class__ = _TabTunePackage


_install_shadow_guard()
