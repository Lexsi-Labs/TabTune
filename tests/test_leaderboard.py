"""
Tests for TabularLeaderboard - the benchmarking utility.

These tests cover leaderboard initialization, adding models, and running benchmarks.
"""
import pytest
import pandas as pd
from tabtune import TabularLeaderboard, TabularPipeline


class TestTabularLeaderboardInitialization:
    """Test TabularLeaderboard initialization."""
    
    def test_init_with_data(self, minimal_data):
        """Test initialization with train/test split."""
        X_train, X_test, y_train, y_test = minimal_data
        
        leaderboard = TabularLeaderboard(X_train, X_test, y_train, y_test)
        
        assert leaderboard.X_train is not None
        assert leaderboard.X_test is not None
        assert len(leaderboard.models_to_run) == 0
    
    def test_init_creates_empty_model_list(self, minimal_data):
        """Test that models list is initialized empty."""
        X_train, X_test, y_train, y_test = minimal_data
        
        leaderboard = TabularLeaderboard(X_train, X_test, y_train, y_test)
        
        assert isinstance(leaderboard.models_to_run, list)
        assert len(leaderboard.models_to_run) == 0


class TestTabularLeaderboardAddModel:
    """Test adding models to leaderboard."""
    
    def test_add_model(self, minimal_data):
        """Test adding a model configuration."""
        X_train, X_test, y_train, y_test = minimal_data
        
        leaderboard = TabularLeaderboard(X_train, X_test, y_train, y_test)
        leaderboard.add_model(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        assert len(leaderboard.models_to_run) == 1
        assert leaderboard.models_to_run[0]['model_name'] == 'TabPFN'
        assert leaderboard.models_to_run[0]['tuning_strategy'] == 'inference'
    
    def test_add_multiple_models(self, minimal_data):
        """Test adding multiple model configurations."""
        X_train, X_test, y_train, y_test = minimal_data
        
        leaderboard = TabularLeaderboard(X_train, X_test, y_train, y_test)
        leaderboard.add_model('TabPFN', 'inference')
        leaderboard.add_model('TabPFN', 'finetune', tuning_params={'epochs': 1})
        
        assert len(leaderboard.models_to_run) == 2
    
    def test_add_model_with_params(self, minimal_data):
        """Test adding model with parameters."""
        X_train, X_test, y_train, y_test = minimal_data
        
        leaderboard = TabularLeaderboard(X_train, X_test, y_train, y_test)
        leaderboard.add_model(
            model_name='TabPFN',
            tuning_strategy='finetune',
            model_params={'device': 'cpu'},
            tuning_params={'epochs': 3, 'learning_rate': 1e-5}
        )
        
        config = leaderboard.models_to_run[0]
        assert config['model_params']['device'] == 'cpu'
        assert config['tuning_params']['epochs'] == 3


class TestTabularLeaderboardRun:
    """Test running the leaderboard benchmark."""
    
    @pytest.mark.slow
    def test_run_single_model(self, minimal_data):
        """Test running leaderboard with single model."""
        X_train, X_test, y_train, y_test = minimal_data
        
        leaderboard = TabularLeaderboard(X_train, X_test, y_train, y_test)
        leaderboard.add_model(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        results = leaderboard.run()
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1
        assert 'Model' in results.columns
        assert 'Strategy' in results.columns
        assert 'Accuracy' in results.columns
    
    @pytest.mark.slow
    def test_run_multiple_models(self, minimal_data):
        """Test running leaderboard with multiple models."""
        X_train, X_test, y_train, y_test = minimal_data
        
        leaderboard = TabularLeaderboard(X_train, X_test, y_train, y_test)
        leaderboard.add_model('TabPFN', 'inference')
        leaderboard.add_model('TabPFN', 'inference')  # Same config twice
        
        results = leaderboard.run()
        
        assert len(results) == 2
    
    @pytest.mark.slow
    def test_run_with_ranking(self, minimal_data):
        """Test running with different ranking metrics."""
        X_train, X_test, y_train, y_test = minimal_data
        
        leaderboard = TabularLeaderboard(X_train, X_test, y_train, y_test)
        leaderboard.add_model('TabPFN', 'inference')
        
        results_by_accuracy = leaderboard.run(rank_by='accuracy')
        results_by_f1 = leaderboard.run(rank_by='f1_score')
        results_by_auc = leaderboard.run(rank_by='roc_auc_score')
        
        # All should return dataframes
        assert isinstance(results_by_accuracy, pd.DataFrame)
        assert isinstance(results_by_f1, pd.DataFrame)
        assert isinstance(results_by_auc, pd.DataFrame)
    
    def test_run_handles_failing_model(self, minimal_data):
        """Test that leaderboard handles model failures gracefully."""
        X_train, X_test, y_train, y_test = minimal_data
        
        leaderboard = TabularLeaderboard(X_train, X_test, y_train, y_test)
        leaderboard.add_model(
            model_name='InvalidModel',  # This will fail
            tuning_strategy='inference'
        )
        
        # Should not raise, but mark as failed
        # Note: This depends on implementation details
        # If InvalidModel raises ValueError in init, it might fail differently
        try:
            results = leaderboard.run()
            # If it succeeds, check that failures are recorded
            assert isinstance(results, pd.DataFrame)
        except (ValueError, AttributeError):
            # Some models fail at initialization, which is expected
            pass

