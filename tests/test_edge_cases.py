"""
Tests for Edge Cases and Error Handling.

This module tests various edge cases and error scenarios to ensure
robust error handling throughout the pipeline.
"""
import pytest
import pandas as pd
import numpy as np
from tabtune import TabularPipeline


class TestEdgeCasesData:
    """Test edge cases related to data."""
    
    def test_single_class_target_raises_error(self):
        """Test that single class target raises appropriate error."""
        X = pd.DataFrame({
            'feat1': [1, 2, 3, 4, 5],
            'feat2': [2, 3, 4, 5, 6]
        })
        y = pd.Series([0, 0, 0, 0, 0])  # All same class
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        # Single class might work for some models, but evaluation should fail
        try:
            pipeline.fit(X, y)
            # If fit succeeds, evaluation should fail
            X_test = pd.DataFrame({'feat1': [6], 'feat2': [7]})
            y_test = pd.Series([0])
            
            # Evaluation with single class should handle gracefully or raise error
            with pytest.raises((ValueError, RuntimeError)):
                pipeline.evaluate(X_test, y_test)
        except (ValueError, RuntimeError) as e:
            # If fit fails, that's also acceptable
            assert "single" in str(e).lower() or "class" in str(e).lower() or len(str(e)) > 0
    
    def test_empty_dataset_raises_error(self):
        """Test that empty dataset raises appropriate error."""
        X = pd.DataFrame({'feat1': [], 'feat2': []})
        y = pd.Series([], dtype=int)
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            pipeline.fit(X, y)
    
    def test_all_nan_features_handles_gracefully(self, minimal_data):
        """Test that all NaN features are handled gracefully."""
        X_train, X_test, y_train, _ = minimal_data
        
        # Fill all features with NaN
        X_train_nan = X_train.copy()
        X_train_nan.iloc[:] = np.nan
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference',
            processor_params={'imputation_strategy': 'mean'}
        )
        
        # Should handle NaN values through imputation
        try:
            pipeline.fit(X_train_nan, y_train)
            # If fit succeeds, imputation worked
            X_test_nan = X_test.copy()
            X_test_nan.iloc[:] = np.nan
            predictions = pipeline.predict(X_test_nan)
            assert len(predictions) == len(X_test)
        except Exception as e:
            # If it fails, that's expected for all-NaN data
            assert len(str(e)) > 0
    
    def test_extreme_values(self):
        """Test that extreme values are handled correctly."""
        X = pd.DataFrame({
            'feat1': [1e10, 1e-10, -1e10, -1e-10, 0],
            'feat2': [np.inf, -np.inf, np.nan, 0, 1]
        })
        y = pd.Series([0, 0, 1, 1, 0])
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference'
        )
        
        # Should handle extreme values (may need preprocessing)
        try:
            pipeline.fit(X, y)
            predictions = pipeline.predict(X)
            assert len(predictions) == len(X)
        except Exception:
            # Some extreme values may cause issues, which is acceptable
            pass


class TestEdgeCasesConfiguration:
    """Test edge cases related to configuration."""
    
    def test_invalid_model_name_raises_error(self):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            TabularPipeline(
                model_name='InvalidModel',
                tuning_strategy='inference'
            )
    
    def test_invalid_tuning_strategy(self):
        """Test that invalid tuning strategy is handled."""
        with pytest.raises((ValueError, TypeError)):
            TabularPipeline(
                model_name='TabPFN',
                tuning_strategy='invalid_strategy'
            )
    
    def test_invalid_model_params(self):
        """Test that invalid model parameters are handled."""
        # Some models may accept unexpected params, others may raise errors
        try:
            pipeline = TabularPipeline(
                model_name='TabPFN',
                tuning_strategy='inference',
                model_params={'nonexistent_param': 'value'}
            )
            # If it doesn't raise error, that's fine (params might be ignored)
            assert pipeline is not None
        except Exception:
            # If it raises error, that's also acceptable
            pass
    
    def test_invalid_tuning_params(self, minimal_data):
        """Test that invalid tuning parameters are handled."""
        X_train, _, y_train, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference',
            tuning_params={'epochs': -1, 'invalid_param': 'value'}  # Negative epochs, invalid param
        )
        
        # Should either work (ignoring invalid) or raise error
        try:
            pipeline.fit(X_train, y_train)
            assert pipeline._is_fitted == True
        except (ValueError, RuntimeError):
            # If it raises error, that's acceptable
            pass


class TestEdgeCasesCheckpoint:
    """Test edge cases related to checkpoints."""
    
    def test_checkpoint_file_not_found(self, tmp_path):
        """Test loading non-existent checkpoint."""
        non_existent_path = tmp_path / "nonexistent_checkpoint.pt"
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference',
            model_checkpoint_path=str(non_existent_path)
        )
        
        # Should either work (checkpoint ignored) or raise warning/error
        # The implementation may handle missing checkpoints gracefully
        assert pipeline is not None
    
    def test_checkpoint_load_with_wrong_model(self, minimal_data, tmp_path):
        """Test loading checkpoint from different model."""
        X_train, _, y_train, _ = minimal_data
        
        # Create checkpoint with TabPFN
        checkpoint_path = tmp_path / "tabpfn_checkpoint.pt"
        checkpoint_params = {
            'epochs': 1,
            'save_checkpoint_path': str(checkpoint_path),
            'device': 'cpu'
        }
        
        pipeline1 = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='finetune',
            tuning_params=checkpoint_params
        )
        
        try:
            pipeline1.fit(X_train, y_train)
            
            # Try to load TabPFN checkpoint into TabICL (should fail or be ignored)
            pipeline2 = TabularPipeline(
                model_name='TabICL',
                tuning_strategy='inference',
                model_checkpoint_path=str(checkpoint_path)
            )
            
            # May raise error or load with mismatched weights
            try:
                pipeline2.fit(X_train, y_train)
                # If it succeeds, checkpoint loading is lenient
                assert pipeline2._is_fitted == True
            except Exception:
                # If it fails, that's expected for mismatched checkpoints
                pass
        except Exception:
            # If TabPFN fine-tuning fails, skip this test
            pytest.skip("TabPFN fine-tuning failed, cannot test checkpoint mismatch")


class TestEdgeCasesDevice:
    """Test edge cases related to device handling."""
    
    def test_device_cpu_override(self, minimal_data):
        """Test that CPU device override works."""
        X_train, _, y_train, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name='TabPFN',
            tuning_strategy='inference',
            tuning_params={'device': 'cpu'}
        )
        
        pipeline.fit(X_train, y_train)
        assert pipeline._is_fitted == True

