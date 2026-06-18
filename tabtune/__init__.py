"""
TabTune - An Advanced Library for Tabular Model Training and Adaptation

A powerful and flexible Python library designed to simplify the training and fine-tuning 
of modern foundation models on tabular data.
"""

# Version is sourced from the installed package metadata (pyproject.toml) so it
# never drifts. The literal is only a fallback when running from a source tree
# that hasn't been installed.
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("tabtune")
except PackageNotFoundError:  # running from an uninstalled source checkout
    __version__ = "0.1.17"

__author__ = "Aditya Tanna, Pratinav Seth, Mohamed Bouadi, Utsav Avaiya, Yash Desai, Nassim Bouarour, Vinay Kumar Sankarapu"
__maintainer__ = "Aditya Tanna"
__email__ = "contact@lexsi.ai"
__maintainer_email__ = "contact@lexsi.ai"

from .logger import setup_logger
setup_logger()

# Direct imports for better compatibility
try:
    from .TabularPipeline.pipeline import TabularPipeline
    from .TabularLeaderboard.leaderboard import TabularLeaderboard
    from .Dataprocess.data_processor import DataProcessor
    from .TuningManager.tuning import TuningManager
    from .ensemble.tabular_ensemble import TabularEnsemble
except ImportError as e:
    # Fallback to lazy imports if direct imports fail
    def __getattr__(name):
        try:
            if name == "TabularPipeline":
                from .TabularPipeline.pipeline import TabularPipeline
                return TabularPipeline
            elif name == "TabularLeaderboard":
                from .TabularLeaderboard.leaderboard import TabularLeaderboard
                return TabularLeaderboard
            elif name == "TabularEnsemble":
                from .ensemble.tabular_ensemble import TabularEnsemble
                return TabularEnsemble
            elif name == "DataProcessor":
                from .Dataprocess.data_processor import DataProcessor
                return DataProcessor
            elif name == "TuningManager":
                from .TuningManager.tuning import TuningManager
                return TuningManager
            else:
                raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
        except ImportError as e:
            raise ImportError(f"Failed to import {name}: {e}. Please ensure all dependencies are installed.")

__all__ = [
    "TabularPipeline",
    "TabularLeaderboard", 
    "DataProcessor",
    "TuningManager",
    "TabularEnsemble",
]

