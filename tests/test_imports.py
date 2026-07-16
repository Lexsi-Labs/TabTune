"""
Test module imports and package structure.

These tests ensure that all public APIs can be imported correctly.
"""
import re
import pytest
import sys


class TestPackageImports:
    """Test basic package imports."""

    def test_tabtune_module_import(self):
        """Test that the main tabtune module can be imported."""
        import tabtune
        assert hasattr(tabtune, '__version__')
        # Assert a valid, non-empty semantic-version string rather than a
        # hard-coded literal, so the test does not go stale on every release.
        assert isinstance(tabtune.__version__, str) and tabtune.__version__
        assert re.match(r"^\d+\.\d+", tabtune.__version__), (
            f"unexpected __version__: {tabtune.__version__!r}"
        )
    
    def test_tabular_pipeline_import(self):
        """Test TabularPipeline can be imported via package."""
        from tabtune import TabularPipeline
        assert TabularPipeline is not None
    
    def test_tabular_leaderboard_import(self):
        """Test TabularLeaderboard can be imported via package."""
        from tabtune import TabularLeaderboard
        assert TabularLeaderboard is not None
    
    def test_data_processor_import(self):
        """Test DataProcessor can be imported via package."""
        from tabtune import DataProcessor
        assert DataProcessor is not None
    
    def test_tuning_manager_import(self):
        """Test TuningManager can be imported via package."""
        from tabtune import TuningManager
        assert TuningManager is not None
    
    def test_direct_imports(self):
        """Test direct imports from submodules work."""
        from tabtune.TabularPipeline.pipeline import TabularPipeline
        from tabtune.TabularLeaderboard.leaderboard import TabularLeaderboard
        from tabtune.Dataprocess.data_processor import DataProcessor
        from tabtune.TuningManager.tuning import TuningManager
        
        assert TabularPipeline is not None
        assert TabularLeaderboard is not None
        assert DataProcessor is not None
        assert TuningManager is not None

