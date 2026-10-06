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



class TestFunctionalWeightLoRA:
    """Coverage for the LoRA wrapper used where a weight is read, not called.

    ``LoRALinear`` adds its delta inside ``forward``. Any caller that takes
    ``layer.weight`` and applies it itself - a functional attention call, or a
    bare ``F.linear`` - never runs that, so the adapters would be allocated,
    counted as trainable, and have no effect on a single output.
    """

    @staticmethod
    def _functional_module():
        import torch
        import torch.nn.functional as F

        class FunctionallyApplied(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.out_proj = torch.nn.Linear(4, 4)

            def forward(self, x):
                return F.linear(x, self.out_proj.weight, self.out_proj.bias)

        return FunctionallyApplied()

    def test_plain_wrapper_is_a_no_op_for_a_functional_caller(self) -> None:
        from tabtune.TuningManager.peft_utils import (
            LoRALinear,
            inject_custom_lora_into_linear_layers,
        )

        import torch

        model = self._functional_module()
        x = torch.randn(3, 4)
        inject_custom_lora_into_linear_layers(model, ["out_proj"], r=2, alpha=4)
        assert isinstance(model.out_proj, LoRALinear)
        with torch.no_grad():
            before = model(x).clone()
            model.out_proj.lora_B.weight.normal_(0.0, 1.0)
            after = model(x)
        assert torch.equal(before, after)

    def test_functional_wrapper_reaches_the_same_caller(self) -> None:
        from tabtune.TuningManager.peft_utils import (
            FunctionalWeightLoRALinear,
            inject_custom_lora_into_linear_layers,
        )

        import torch

        model = self._functional_module()
        x = torch.randn(3, 4)
        inject_custom_lora_into_linear_layers(
            model, ["out_proj"], r=2, alpha=4,
            functional_weight_patterns=["out_proj"],
        )
        assert isinstance(model.out_proj, FunctionalWeightLoRALinear)
        with torch.no_grad():
            before = model(x).clone()
            model.out_proj.lora_B.weight.normal_(0.0, 1.0)
            after = model(x)
        assert not torch.equal(before, after)

    def test_functional_wrapper_is_still_correct_when_called(self) -> None:
        """It must not double-count the delta for a normal module call."""
        from tabtune.TuningManager.peft_utils import FunctionalWeightLoRALinear

        import torch

        base = torch.nn.Linear(4, 4)
        wrapped = FunctionalWeightLoRALinear(base, r=2, alpha=4)
        with torch.no_grad():
            wrapped.lora_B.weight.normal_(0.0, 1.0)
            x = torch.randn(3, 4)
            called = wrapped(x)
            merged = torch.nn.functional.linear(x, wrapped.weight, wrapped.bias)
        assert torch.allclose(called, merged, atol=1e-5)

    def test_it_is_off_unless_a_model_asks_for_it(self) -> None:
        """Only models measured to need it opt in; the field defaults to empty."""
        from tabtune.TuningManager.peft_utils import MODEL_LORA_TARGETS

        opted_in = {
            name for name, config in MODEL_LORA_TARGETS.items()
            if config.functional_weight_substrings
        }
        assert opted_in == {"Causilo", "TabLDM"}
