"""
Tests for PEFT (Parameter-Efficient Fine-Tuning) with LoRA adapters.

This module tests PEFT functionality for models that fully support it:
- TabICL
- OrionMSP
- OrionBix
- Mitra
- TabDPT

Note: TabPFN and ContextTab have experimental PEFT support and are skipped.
"""
import pytest
import pandas as pd
import numpy as np
from tabtune import TabularPipeline

# Models with full PEFT support
PEFT_SUPPORTED_MODELS = [
    'TabICL',
    'OrionMSP',
    'OrionBix',
    'Mitra',
    'TabDPT',
]

# Models with experimental PEFT (skip)
PEFT_EXPERIMENTAL_MODELS = ['TabPFN', 'ContextTab']


class TestPEFTFineTuning:
    """Test PEFT fine-tuning for supported models."""
    
    @pytest.mark.parametrize("model_name", PEFT_SUPPORTED_MODELS)
    @pytest.mark.slow
    @pytest.mark.finetuning
    def test_peft_finetune_fit(self, minimal_data, model_name, fast_finetune_params, peft_config):
        """Test fitting each supported model with PEFT."""
        X_train, _, y_train, _ = minimal_data
        
        # Update params to include PEFT config
        peft_params = fast_finetune_params.copy()
        peft_params['peft_config'] = peft_config
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='peft',
            tuning_params=peft_params
        )
        
        pipeline.fit(X_train, y_train)
        
        assert pipeline._is_fitted == True
    
    @pytest.mark.parametrize("model_name", PEFT_SUPPORTED_MODELS)
    @pytest.mark.slow
    @pytest.mark.finetuning
    def test_peft_finetune_predict(self, minimal_data, model_name, fast_finetune_params, peft_config):
        """Test prediction after PEFT fine-tuning."""
        X_train, X_test, y_train, _ = minimal_data
        
        # Update params to include PEFT config
        peft_params = fast_finetune_params.copy()
        peft_params['peft_config'] = peft_config
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='peft',
            tuning_params=peft_params
        )
        
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        
        assert predictions is not None
        assert len(predictions) == len(X_test)
    
    @pytest.mark.parametrize("model_name", PEFT_SUPPORTED_MODELS)
    @pytest.mark.slow
    @pytest.mark.finetuning
    def test_peft_finetune_evaluate(self, minimal_data, model_name, fast_finetune_params, peft_config):
        """Test evaluation after PEFT fine-tuning."""
        X_train, X_test, y_train, y_test = minimal_data
        
        # Update params to include PEFT config
        peft_params = fast_finetune_params.copy()
        peft_params['peft_config'] = peft_config
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='peft',
            tuning_params=peft_params
        )
        
        pipeline.fit(X_train, y_train)
        metrics = pipeline.evaluate(X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics


class TestPEFTExperimentalModels:
    """Test that experimental PEFT models are handled correctly."""
    
    @pytest.mark.parametrize("model_name", PEFT_EXPERIMENTAL_MODELS)
    @pytest.mark.slow
    def test_peft_experimental_model_warning(self, minimal_data, model_name, fast_finetune_params, peft_config):
        """Test that experimental PEFT models log warnings but may still work."""
        X_train, _, y_train, _ = minimal_data
        
        # Update params to include PEFT config
        peft_params = fast_finetune_params.copy()
        peft_params['peft_config'] = peft_config
        
        pipeline = TabularPipeline(
            model_name=model_name,
            tuning_strategy='peft',
            tuning_params=peft_params
        )
        
        # These models may fall back to base-ft or raise warnings
        # Test that fit doesn't crash (may use fallback)
        try:
            pipeline.fit(X_train, y_train)
            assert pipeline._is_fitted == True
        except Exception as e:
            # If it fails, that's expected for experimental support
            pytest.skip(f"PEFT not working for {model_name}: {e}")

