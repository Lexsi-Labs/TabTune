"""
Tests for Fine-tuning - All models with fine-tuning strategies.

This module tests all 7 supported models with fine-tuning strategies:
- finetune with base-ft (meta-learning mode - default)
- finetune with SFT mode

All tests use 1 epoch and minimal_data for speed.
"""
import pytest
import pandas as pd
import numpy as np
from tabtune import TabularPipeline

# All supported models
ALL_MODELS = [
    'TabPFN',
    'TabICL',
    'OrionMSP',
    'OrionBix',
    'Mitra',
    'ContextTab',
    'TabDPT',
]


class TestFineTuningMetaLearning:
    """Test fine-tuning with meta-learning mode (base-ft, default)."""
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    @pytest.mark.finetuning
    def test_finetune_meta_learning_fit(self, minimal_data, model_name, fast_finetune_params):
        """Test fitting each model with fine-tuning in meta-learning mode."""
        X_train, _, y_train, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='finetune',
            tuning_params=fast_finetune_params
        )
        
        pipeline.fit(X_train, y_train)
        
        assert pipeline._is_fitted == True
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    @pytest.mark.finetuning
    def test_finetune_meta_learning_predict(self, minimal_data, model_name, fast_finetune_params):
        """Test prediction after fine-tuning in meta-learning mode."""
        X_train, X_test, y_train, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='finetune',
            tuning_params=fast_finetune_params
        )
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        assert predictions is not None
        assert len(predictions) == len(X_test)
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    @pytest.mark.finetuning
    def test_finetune_meta_learning_evaluate(self, minimal_data, model_name, fast_finetune_params):
        """Test evaluation after fine-tuning in meta-learning mode."""
        X_train, X_test, y_train, y_test = minimal_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='finetune',
            tuning_params=fast_finetune_params
        )
        
        pipeline.fit(X_train, y_train)
        metrics = pipeline.evaluate(X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics


class TestFineTuningSFT:
    """Test fine-tuning with SFT (Supervised Fine-Tuning) mode."""
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    @pytest.mark.finetuning
    def test_finetune_sft_fit(self, minimal_data, model_name, fast_finetune_params):
        """Test fitting each model with fine-tuning in SFT mode."""
        X_train, _, y_train, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='finetune',
            tuning_params=fast_finetune_params,
            finetune_mode='sft'  # Pass as constructor parameter, not in tuning_params
        )
        
        pipeline.fit(X_train, y_train)
        
        assert pipeline._is_fitted == True
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    @pytest.mark.finetuning
    def test_finetune_sft_predict(self, minimal_data, model_name, fast_finetune_params):
        """Test prediction after fine-tuning in SFT mode."""
        X_train, X_test, y_train, _ = minimal_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='finetune',
            tuning_params=fast_finetune_params,
            finetune_mode='sft'  # Pass as constructor parameter, not in tuning_params
        )
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        assert predictions is not None
        assert len(predictions) == len(X_test)
    
    @pytest.mark.parametrize("model_name", ALL_MODELS)
    @pytest.mark.slow
    @pytest.mark.finetuning
    def test_finetune_sft_evaluate(self, minimal_data, model_name, fast_finetune_params):
        """Test evaluation after fine-tuning in SFT mode."""
        X_train, X_test, y_train, y_test = minimal_data
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='finetune',
            tuning_params=fast_finetune_params,
            finetune_mode='sft'  # Pass as constructor parameter, not in tuning_params
        )
        
        pipeline.fit(X_train, y_train)
        metrics = pipeline.evaluate(X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics

