import torch
from torch.optim import Adam
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from functools import partial
import numpy as np
import pandas as pd
import logging
import os
import warnings
from sklearn.metrics import r2_score


def ensure_device_consistency(model, device):
    """Ensure all model parameters and buffers are on the same device"""
    model.to(device)
    for param in model.parameters():
        param.data = param.data.to(device)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(device)
    return model


from ..models.tabpfn.classifier import TabPFNClassifier
from ..models.tabpfn.utils import meta_dataset_collator
from ..models.tabicl.sklearn.classifier import TabICLClassifier
from ..models.tabicl.sklearn.preprocessing import TabICLMetaDataset
from ..models.tabiclv2.sklearn.classifier import TabICLClassifier as TabICLv2Classifier
from ..models.orion_msp.sklearn.classifier import OrionMSPClassifier
from ..models.orionmsp_v15.sklearn.classifier import OrionMSPv15Classifier
from ..models.orion_msp.sklearn.preprocessing import OrionMSPMetaDataset
from ..models.contexttab.contexttab import ConTextTabClassifier
from ..models.mitra.tab2d import Tab2D
from torch.utils.data import TensorDataset
from ..models.orion_bix.sklearn.classifier import OrionBixClassifier
from ..models.tabdpt.classifier import TabDPTClassifier
from ..models.tabdpt.utils import pad_x
from ..models.tabdpt.model import TabDPTModel
from ..models.limix.classifier import LimixClassifier
from ..models.tabfm.classifier import TabFMClassifier
from ..models.xrfm.classifier import XRFMClassifier
from ..models.iltm.classifier import ILTMClassifier
from ..models.exaone.classifier import EXAONETabularClassifier

from ..models.regression.tabpfn.regressor import TabPFNRegressorWrapper
from ..models.regression.contexttab.regressor import ConTextTabRegressorWrapper
from ..models.regression.tabdpt.regressor import TabDPTRegressorWrapper
from ..models.regression.mitra.regressor import MitraRegressorWrapper
from ..models.regression.limix.regressor_wrapper import LimixRegressorWrapper
from ..models.tabiclv2.sklearn.regressor import TabICLRegressor as TabICLv2Regressor
from ..models.regression.tabfm.regressor import TabFMRegressorWrapper
from ..models.regression.xrfm.regressor import XRFMRegressorWrapper
from ..models.regression.iltm.regressor import ILTMRegressorWrapper
from ..models.regression.exaone.regressor import EXAONETabularRegressorWrapper

from ..models.tabpfnv26 import TabPFNv26Classifier
from ..models.tabpfnv3 import TabPFNv3Classifier
from ..models.tabpfnv26.regressor import TabPFNRegressor as TabPFNv26Regressor
from ..models.regression.tabpfnv26.regressor import TabPFNv26RegressorWrapper
from ..models.regression.tabpfnv3.regressor import TabPFNv3RegressorWrapper


from ..models.contexttab.contexttab import to_device

from .peft_utils import apply_tabular_lora

from .._internal.device import resolve_device

logger = logging.getLogger(__name__)




def _normalise_device_param(params: dict) -> dict:
    """Resolve ``params['device']`` to a concrete device string, in place.

    Every ``_finetune_*`` method does ``torch.device(config["device"])``, and
    ``config`` is the method's own defaults updated with the caller's params.
    The defaults come from ``resolve_device('auto')`` and are therefore already
    concrete -- but a caller-supplied value went straight through.

    That mattered because ``'auto'`` is the library-wide spelling for "pick a
    backend" and is the default of :class:`~tabtune.config.schemas.TuningConfig`,
    so it is exactly what arrives when someone does not name a device. Every
    fine-tune, for every model, then died on
    ``torch.device("auto")``. Normalising once here fixes all of them without
    touching the twenty-odd call sites, and additionally clamps an out-of-range
    CUDA index and falls back to CPU with a warning rather than failing later
    inside ``.to()``.
    """
    requested = params.get("device")
    if requested is not None:
        params["device"] = resolve_device(requested)
    return params


class TuningManager:
    """
    Handles the model adaptation process
    """
    def tune(self, model, X_train, y_train, strategy='inference', params=None, processor=None):
        
        params_copy = dict(params) if isinstance(params, dict) else {}
        _normalise_device_param(params_copy)
        finetune_mode = params_copy.get('finetune_mode', 'turn_by_turn')

        # --- Regression wrappers: allow ContextTab finetune (turn-by-turn) ---
        if isinstance(model, (TabPFNRegressorWrapper, ConTextTabRegressorWrapper,
                              TabDPTRegressorWrapper, MitraRegressorWrapper,LimixRegressorWrapper,TabPFNv26RegressorWrapper, TabPFNv3RegressorWrapper, TabICLv2Regressor, TabFMRegressorWrapper, XRFMRegressorWrapper, ILTMRegressorWrapper, EXAONETabularRegressorWrapper)):
            if strategy == 'inference':
                logger.info("[TuningManager] Regression model - inference path")
                model.fit(X_train, y_train)
                return model

            # ContextTab regression finetune (turn-by-turn)
            if isinstance(model, ConTextTabRegressorWrapper) and strategy == 'finetune':
                if finetune_mode not in ('turn_by_turn', 'tbt'):
                    raise ValueError(
                        f"ContextTab regression finetune currently supports finetune_mode "
                        f"'turn_by_turn' (or 'tbt'). Got '{finetune_mode}'."
                    )
                return self._finetune_contexttab_regression_turn_by_turn(model, X_train, y_train, params_copy)

            # Limix regression finetune
            if isinstance(model, LimixRegressorWrapper) and strategy == "finetune":
                return self._finetune_limix_regression(model, X_train, y_train, params_copy)

            if isinstance(model, TabDPTRegressorWrapper) and strategy == "finetune":
                params_copy = dict(params) if isinstance(params, dict) else {}
                _normalise_device_param(params_copy)
                return self._finetune_tabdpt_regression_turn_by_turn(model, X_train, y_train, params_copy)

            # Mitra regression finetune (turn-by-turn)
            if isinstance(model, MitraRegressorWrapper) and strategy == "finetune":
                if finetune_mode not in ("turn_by_turn", "tbt"):
                    raise ValueError(f"Mitra regression finetune supports finetune_mode 'turn_by_turn' (or 'tbt'). Got '{finetune_mode}'.")
                return self._finetune_mitra_regression_turn_by_turn(model, X_train, y_train, params_copy)


            if isinstance(model, TabPFNRegressorWrapper) and strategy == "finetune":
                if finetune_mode not in ("turn_by_turn", "tbt"):
                    logger.warning(
                        f"[TuningManager] TabPFN regression finetune received "
                        f"finetune_mode='{finetune_mode}' — only 'turn_by_turn'/'tbt' "
                        f"is supported. Auto-correcting to 'turn_by_turn'."
                    )
                    finetune_mode = "turn_by_turn"
                return self._finetune_tabpfn_regression_turn_by_turn(model, X_train, y_train, params_copy)

            # TabPFNv26 regression finetune
            if isinstance(model, TabPFNv26RegressorWrapper) and strategy == "finetune":
                logger.info("[TuningManager] Using v2.6 native FinetunedTabPFNRegressor")
                return self._finetune_tabpfnv26_native_regressor(model, X_train, y_train, params_copy)

            # TabPFNv3 regression finetune (native default; turn-by-turn alternative)
            if isinstance(model, TabPFNv3RegressorWrapper) and strategy == "finetune":
                # `finetune_mode` defaults to 'turn_by_turn' at the top of tune() for
                # the regression dispatch block; for TabPFNv3 the documented default is
                # 'native'. Treat the regression-block default as "unspecified" so that
                # only an explicit turn_by_turn/tbt request uses the TBT loop.
                v3_reg_mode = params_copy.get("finetune_mode", None)
                if v3_reg_mode in ("turn_by_turn", "tbt"):
                    logger.info("[TuningManager] Fine-tuning TabPFNv3 regressor (turn-by-turn)")
                    return self._finetune_tabpfnv3_regression_turn_by_turn(model, X_train, y_train, params_copy)
                logger.info("[TuningManager] Using v3 native FinetunedTabPFNRegressor (V3-pinned)")
                return self._finetune_tabpfnv3_native_regressor(model, X_train, y_train, params_copy)

            if isinstance(model, TabICLv2Regressor) and strategy == "finetune":
                logger.info("[TuningManager] Fine-tuning TabICLv2 regressor")
                return self._finetune_tabiclv2_regression(model, X_train, y_train, params_copy)

            # TabFM regression finetune (episodic turn-by-turn)
            if isinstance(model, TabFMRegressorWrapper) and strategy == "finetune":
                if finetune_mode not in ("turn_by_turn", "tbt"):
                    logger.warning(
                        f"[TuningManager] TabFM regression finetune supports finetune_mode "
                        f"'turn_by_turn' (or 'tbt'). Got '{finetune_mode}'. Auto-correcting."
                    )
                return self._finetune_tabfm_regression_turn_by_turn(model, X_train, y_train, params_copy)

            # XRFM regression finetune (full RFM training / warm-started refinement)
            if isinstance(model, XRFMRegressorWrapper) and strategy == "finetune":
                # xRFM is a kernel/RFM method: there are no gradient episodes, so
                # `finetune_mode` ('turn_by_turn', ...) does not apply and is ignored.
                if finetune_mode not in ("turn_by_turn", "tbt"):
                    logger.info(
                        f"[TuningManager] XRFM regression finetune ignores finetune_mode "
                        f"('{finetune_mode}'): xRFM refits/refines its Mahalanobis matrix M directly."
                    )
                return self._finetune_xrfm_regression(model, X_train, y_train, params_copy)

            # ILTM regression finetune (episodic turn-by-turn on the real hypernetwork)
            if isinstance(model, ILTMRegressorWrapper) and strategy == "finetune":
                if finetune_mode not in ("turn_by_turn", "tbt"):
                    logger.warning(
                        f"[TuningManager] ILTM regression finetune supports finetune_mode "
                        f"'turn_by_turn' (or 'tbt'). Got '{finetune_mode}'. Auto-correcting."
                    )
                return self._finetune_iltm_regression_turn_by_turn(model, X_train, y_train, params_copy)

            # EXAONE Tabular regression finetune (episodic turn-by-turn on the
            # real Cross-axis Summary Transformer + quantile head)
            if isinstance(model, EXAONETabularRegressorWrapper) and strategy == "finetune":
                if finetune_mode not in ("turn_by_turn", "tbt"):
                    logger.warning(
                        f"[TuningManager] EXAONETabular regression finetune supports finetune_mode "
                        f"'turn_by_turn' (or 'tbt'). Got '{finetune_mode}'. Auto-correcting."
                    )
                return self._finetune_exaone_regression_turn_by_turn(model, X_train, y_train, params_copy)


            raise NotImplementedError(
                f"Regression fine-tuning not implemented yet for model={type(model).__name__}. "
                f"Currently implemented: ContextTab + strategy='finetune' + finetune_mode='turn_by_turn'."
            )
        
        params_copy = dict(params) if isinstance(params, dict) else {}
        _normalise_device_param(params_copy)
        finetune_mode = params_copy.pop('finetune_mode', 'meta-learning')
        save_checkpoint_path = params_copy.pop('save_checkpoint_path', None)
        if save_checkpoint_path is None:
            # Only create a checkpoint directory when the caller asked for one.
            # This block used to run unconditionally, so every fine-tune created
            # ./checkpoints in the caller's working directory and wrote a
            # weights file there - surprising in notebooks, and a stray artefact
            # in CI. Opt-in now.
            default_dir = params_copy.get("checkpoint_dir")
            if default_dir:
                os.makedirs(default_dir, exist_ok=True)
                save_checkpoint_path = os.path.join(
                    default_dir, f"{type(model).__name__}_latest.pt"
                )
            else:
                logger.debug(
                    "[TuningManager] Neither save_checkpoint_path nor checkpoint_dir "
                    "was given; fine-tuned weights stay in memory."
                )

        # Strategy selection: accept either explicit 'peft' strategy or finetune_method='peft'
        finetune_method = params_copy.pop('finetune_method', None)
        peft_config = params_copy.pop('peft_config', None)
        selected_strategy = strategy
        if strategy == 'finetune' and finetune_method == 'peft':
            selected_strategy = 'peft'
        elif strategy == 'finetune':
            selected_strategy = 'finetune'

        is_finetuned = False
        original_is_tab2d = isinstance(model, Tab2D)


        if (isinstance(model, Tab2D) or original_is_tab2d) and selected_strategy in ('finetune', 'peft'):
            if finetune_mode == 'sft':
                logger.info("[TuningManager] Using Pure SFT for Mitra (task-optimized)")
                self._finetune_mitra_pure_sft(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            else:  # default: 'meta-learning'
                logger.info("[TuningManager] Using Episodic Meta-Learning for Mitra (default)")
                self._finetune_mitra(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            is_finetuned = True

        elif isinstance(model, TabPFNv26Classifier) and selected_strategy in ('finetune'):
            if finetune_mode == 'native':
                logger.info("[TuningManager] Using v2.6 native FinetunedTabPFNClassifier")
                model = self._finetune_tabpfnv26_native_classifier(model, X_train, y_train, params=params_copy)
            elif finetune_mode == 'sft':
                logger.info("[TuningManager] Using Pure SFT for TabPFNv26 (task-optimized)")
                self._finetune_tabpfnv26_sft(model, X_train, y_train, params=params_copy)
            else:  # default: 'meta-learning'
                logger.info("[TuningManager] Using Episodic Meta-Learning for TabPFNv26 (default)")
                self._finetune_tabpfnv26_meta(model, X_train, y_train, params=params_copy)
            is_finetuned = True

        elif isinstance(model, TabPFNv3Classifier) and selected_strategy in ('finetune', 'peft'):
            # PEFT is routed through meta/sft loops by passing peft_config (LoRA injection).
            # Native mode does not support PEFT (the upstream FinetunedTabPFN trains full weights).
            if finetune_mode == 'native':
                if selected_strategy == 'peft' or peft_config is not None:
                    logger.warning(
                        "[TuningManager] PEFT is not supported with finetune_mode='native' for "
                        "TabPFNv3 (native uses the upstream full-weight FinetunedTabPFN). "
                        "Use finetune_mode='meta-learning' or 'sft' for LoRA/PEFT. "
                        "Proceeding with native full fine-tuning."
                    )
                logger.info("[TuningManager] Using v3 native FinetunedTabPFNClassifier (V3-pinned)")
                model = self._finetune_tabpfnv3_native_classifier(model, X_train, y_train, params=params_copy)
            elif finetune_mode == 'sft':
                logger.info("[TuningManager] Using Pure SFT for TabPFNv3 (task-optimized)")
                self._finetune_tabpfnv3_sft(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            else:  # default: 'meta-learning'
                logger.info("[TuningManager] Using Episodic Meta-Learning for TabPFNv3 (default)")
                self._finetune_tabpfnv3_meta(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            is_finetuned = True

        elif isinstance(model, (TabPFNClassifier)) and selected_strategy in ('finetune', 'peft'):
            if finetune_mode == 'sft':
                logger.info("[TuningManager] Using Pure SFT for TabPFN (task-optimized)")
                self._finetune_tabpfn_pure_sft(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            else:  # default: 'meta-learning'
                logger.info("[TuningManager] Using Episodic Meta-Learning for TabPFN (default)")
                self._finetune_tabpfn(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            is_finetuned = True
        
        elif isinstance(model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, OrionMSPv15Classifier, TabICLv2Classifier)) and selected_strategy in ('finetune', 'peft'):
            if finetune_mode == 'meta-learning':
                logger.info("[TuningManager] Meta Learning based FT")
                self._finetune_tabicl(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            else:
                logger.info("[TuningManager] Performing SFT")
                self._finetune_tabicl_simple_sft(model, X_train, y_train, params=params_copy, peft_config=peft_config)
            is_finetuned = True
        
        elif isinstance(model, ConTextTabClassifier) and selected_strategy in ('finetune', 'peft'):
            self._full_finetune_model(model, X_train, y_train, params=params_copy, processor=processor, peft_config=peft_config)
            is_finetuned = True
        
        elif isinstance(model, TabDPTClassifier) and selected_strategy in ('finetune','peft'):
            if finetune_mode == 'sft':
                logger.info("[TuningManager] Using Pure SFT for TabDPT (task-optimized)")
                self._finetune_tabdpt_pure_sft(model, X_train, y_train, params=params_copy, processor=processor, peft_config=peft_config)
            else:  # default: 'meta-learning'
                logger.info("[TuningManager] Using Episodic Meta-Learning for TabDPT (default)")
                self._finetune_tabdpt(model, X_train, y_train, params=params_copy, processor=processor, peft_config=peft_config)
            is_finetuned = True


        elif isinstance(model, TabFMClassifier) and selected_strategy in ('finetune', 'peft'):
            if finetune_mode == 'sft':
                logger.info("[TuningManager] Using Pure SFT for TabFM (task-optimized)")
                self._finetune_tabfm(model, X_train, y_train, params=params_copy, peft_config=peft_config, mode='sft')
            else:  # default: 'meta-learning' (matches TabFM's in-context paradigm)
                logger.info("[TuningManager] Using Episodic Meta-Learning for TabFM (default)")
                self._finetune_tabfm(model, X_train, y_train, params=params_copy, peft_config=peft_config, mode='meta-learning')
            is_finetuned = True

        elif isinstance(model, XRFMClassifier) and selected_strategy in ('finetune', 'peft'):
            # xRFM is a kernel/RFM method (no gradient-trained nn.Module, no
            # pretrained checkpoint): 'finetune' = full RFM (re)training with
            # user-controlled hyperparameters (+ warm-started refinement when
            # already fitted); 'peft' = frozen-base low-rank update of the
            # AGOP-learned Mahalanobis matrix M. Checkpoints are saved via
            # joblib inside the methods (xRFM is not state_dict-compatible).
            if selected_strategy == 'peft' or peft_config is not None:
                logger.info("[TuningManager] Using low-rank M-matrix adaptation (kernel PEFT) for XRFM")
                model = self._peft_xrfm(model, X_train, y_train, params=params_copy,
                                        peft_config=peft_config, save_path=save_checkpoint_path)
            else:
                logger.info("[TuningManager] Using full RFM refit/refinement fine-tuning for XRFM")
                model = self._finetune_xrfm(model, X_train, y_train, params=params_copy,
                                            save_path=save_checkpoint_path)
            is_finetuned = False  # joblib checkpointing handled inside; skip torch state_dict round-trip

        elif isinstance(model, ILTMClassifier) and selected_strategy in ('finetune', 'peft'):
            if finetune_mode == 'sft':
                logger.info("[TuningManager] Using Pure SFT for ILTM (task-optimized)")
                self._finetune_iltm(model, X_train, y_train, params=params_copy, peft_config=peft_config, mode='sft')
            else:  # default: 'meta-learning' (matches the hypernetwork's meta-trained paradigm)
                logger.info("[TuningManager] Using Episodic Meta-Learning for ILTM (default)")
                self._finetune_iltm(model, X_train, y_train, params=params_copy, peft_config=peft_config, mode='meta-learning')
            is_finetuned = True

        elif isinstance(model, EXAONETabularClassifier) and selected_strategy in ('finetune', 'peft'):
            if finetune_mode == 'sft':
                logger.info("[TuningManager] Using Pure SFT for EXAONETabular (task-optimized)")
                self._finetune_exaone(model, X_train, y_train, params=params_copy, peft_config=peft_config, mode='sft')
            else:  # default: 'meta-learning' (matches EXAONE's in-context paradigm)
                logger.info("[TuningManager] Using Episodic Meta-Learning for EXAONETabular (default)")
                self._finetune_exaone(model, X_train, y_train, params=params_copy, peft_config=peft_config, mode='meta-learning')
            is_finetuned = True

        elif isinstance(model, LimixClassifier) and selected_strategy in ('finetune', 'peft'):
            msg = "[TuningManager] Limix fine-tuning not supported; falling back to inference-mode fit (.fit) only."
            print(msg)
            logger.warning(msg)
            logger.info("falling back to inference mode")
            # Fall back to the inference behavior (your existing inference branch calls .fit)
            model.fit(X_train, y_train)

            # Not finetuned -> don't save/reload checkpoint
            is_finetuned = False


        
        elif isinstance(model, (Tab2D)) and selected_strategy == 'inference':
            logger.info("[TuningManager] In-context learning model in inference mode. No training needed.")
            pass
        elif isinstance(model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, LimixClassifier, OrionMSPv15Classifier, TabICLv2Classifier)) and selected_strategy == 'inference':
            logger.info("[TuningManager] Applying standard .fit() for TabICL setup (inference mode)")
            model.fit(X_train, y_train)
        elif isinstance(model, TabPFNv26Classifier) and selected_strategy == 'inference':
            logger.info("[TuningManager] Applying standard .fit() for TabPFNv26 (inference mode)")
            model.fit(X_train, y_train)
        elif isinstance(model, TabPFNv3Classifier) and selected_strategy == 'inference':
            logger.info("[TuningManager] Applying standard .fit() for TabPFNv3 (inference mode)")
            model.fit(X_train, y_train)
        elif isinstance(model, TabFMClassifier) and selected_strategy == 'inference':
            logger.info("[TuningManager] Applying standard .fit() for TabFM (zero-shot inference mode)")
            model.fit(X_train, y_train)
        elif isinstance(model, XRFMClassifier) and selected_strategy == 'inference':
            logger.info("[TuningManager] Applying standard .fit() for XRFM (kernel training with default hyperparameters)")
            model.fit(X_train, y_train)
        elif isinstance(model, ILTMClassifier) and selected_strategy == 'inference':
            logger.info("[TuningManager] Applying standard .fit() for ILTM (hypernetwork-generated ensemble, no gradient training)")
            model.fit(X_train, y_train)
        elif isinstance(model, EXAONETabularClassifier) and selected_strategy == 'inference':
            logger.info("[TuningManager] Applying standard .fit() for EXAONETabular (in-context inference mode)")
            model.fit(X_train, y_train)
        else:
            logger.info("[TuningManager] Applying standard model fitting (.fit)")
            model.fit(X_train, y_train)


        if is_finetuned and save_checkpoint_path:
            self._save_checkpoint(model, save_checkpoint_path)
            logger.info(f"[TuningManager] Saved fine-tuned checkpoint to {save_checkpoint_path}")

            # The save/reload round-trip exists to guarantee the in-memory model
            # matches what was written. It is skipped when nothing was written.
            model = self.load_checkpoint(
                model, save_checkpoint_path, map_location=resolve_device('auto')
            )
            logger.info("[TuningManager] Reloaded fine-tuned weights into model for inference")

        if is_finetuned:
            if isinstance(model, torch.nn.Module):
                model.eval()
            elif hasattr(model, 'model'):
                model.model.eval()
            elif hasattr(model, 'model_'):
                model.model_.eval()
            logger.info("[TuningManager] Fine-tuning complete; model set to eval mode")

        return model
        


    def _maybe_save_epoch_ckpt(self, model, ckpt_dir, ckpt_epochs, epoch, prefix):
        if ckpt_dir and (epoch in ckpt_epochs):
            fname = f"{prefix}_epoch{epoch}.pt"
            path = os.path.join(ckpt_dir, fname)
            self._save_checkpoint(model, path)
            
    def _save_checkpoint(self, model, path: str):
        logger.info(f"[TuningManager] Saving model checkpoint to {path}")

        torch_model = None
        if hasattr(model, 'model_'):  # For TabPFN, TabICL, OrionMSP, OrionBix
            torch_model = model.model_
        elif hasattr(model, 'model'):  # For ContextTab, TabDPT
            torch_model = model.model
        elif isinstance(model, torch.nn.Module):  # For Mitra
            torch_model = model

        if torch_model:
            try:
            # Ensure path is a string here!
                if not isinstance(path, str):
                    raise ValueError("Checkpoint path must be a string")
                torch.save(torch_model.state_dict(), path)
                logger.info(f"[TuningManager] Checkpoint saved successfully to {path}")
            except Exception as e:
                logger.error(f"[TuningManager] Failed to save checkpoint: {e}")
        else:
            logger.warning(f"[TuningManager] No compatible torch model found to save checkpoint")



    def load_checkpoint(self, model, ckpt_path: str, map_location='cpu'):
        """Loads a checkpoint automatically to correct submodule."""
        if not os.path.exists(ckpt_path):
            logger.warning(f"[TuningManager] Checkpoint path {ckpt_path} not found")
            return model

        state = torch.load(ckpt_path, map_location=map_location, weights_only=True)
        state_dict = state.get('model_state_dict', state)
        candidates = [getattr(model, 'model_', None), getattr(model, 'model', None), model]

        for candidate in candidates:
            if isinstance(candidate, torch.nn.Module):
                try:
                    candidate.load_state_dict(state_dict, strict=False)
                    logger.info(f"[TuningManager] Loaded checkpoint weights into {type(candidate).__name__}")
                    return model
                except Exception as e:
                    logger.warning(f"[TuningManager] Could not load into {type(candidate).__name__}: {e}")
        logger.error("[TuningManager] Failed to load weights into model")
        return model
        
            
    def _full_finetune_model(self, model, X_train, y_train, params=None, processor=None, peft_config=None):
        """
        Performs a standard full fine-tuning loop. This has been refactored to
        use the model's own tokenizer for batch preparation, ensuring correctness.
        """
        logger.info(f"[TuningManager] Starting full fine-tuning for {type(model).__name__}")
        
        config = {
            "device": resolve_device('auto'),
            "epochs": 5,
            "learning_rate": 1e-4,
            "batch_size": 128,
            "show_progress": True
        }
        if params:
            config.update(params)
        logger.debug(f"[TuningManager] Using fine-tuning config: {config}")
            
        is_contexttab = isinstance(model, ConTextTabClassifier)
        torch_model = model.model
        
        device = torch.device(config["device"])
        torch_model.to(device)
        torch_model.train()

        for param in torch_model.parameters():
            param.data = param.data.to(device)

        if is_contexttab:
            logger.info("[TuningManager] Fitting the ConTextTab wrapper to set its data context")
            model.fit(X_train, y_train)

        if peft_config:
            logger.warning("[TuningManager] WARNING: ConTextTab PEFT support is currently experimental and may cause prediction issues")
            logger.warning("[TuningManager] ConTextTab's complex embedding pipeline may conflict with LoRA adapters")
            logger.info("[TuningManager] RECOMMENDATION: Use standard finetune strategy for ConTextTab instead of 'peft'")
            logger.info("[TuningManager] FALLBACK: Proceeding with standard base fine-tuning")
            peft_config = None  # Disable PEFT for ConTextTab
        
        optimizer = Adam(torch_model.parameters(), lr=config["learning_rate"])
        loss_fn = torch.nn.CrossEntropyLoss()

        # Create a simple dataset of indices
        dataset = TensorDataset(torch.arange(len(X_train)))
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

        for epoch in range(1, config["epochs"] + 1):
            iterable = dataloader
            if config["show_progress"]:
                iterable = tqdm(dataloader, desc=f"Finetuning Epoch {epoch}")
            
            for batch_indices in iterable:
                # Get the raw data for the current batch
                if hasattr(X_train, 'iloc'):  # DataFrame
                    X_batch_raw = X_train.iloc[batch_indices[0].numpy()]
                    y_batch_raw = y_train.iloc[batch_indices[0].numpy()]
                else:  # numpy array
                    X_batch_raw = X_train[batch_indices[0].numpy()]
                    y_batch_raw = y_train[batch_indices[0].numpy()]

                optimizer.zero_grad()
                
                if is_contexttab:
                    # Use the model's own tokenizer to prepare the batch
                    data_batch = model.get_tokenized_data(X_batch_raw, bagging_index=epoch)
                    
                    # Move tensors to the correct device
                    for k, v in data_batch.items():
                        if isinstance(v, torch.Tensor):
                            data_batch[k] = v.to(device)
                        elif isinstance(v, dict): # Handle nested dicts like ⁠ data['data'] ⁠
                             for k_inner, v_inner in v.items():
                                 if isinstance(v_inner, torch.Tensor):
                                     v[k_inner] = v_inner.to(device)
                    
                    y_batch = data_batch['data']['target']
                    # Ensure y_batch is Long type for cross-entropy loss (ContextTab may return Float)
                    if y_batch.dtype != torch.long:
                        y_batch = y_batch.long()
                    logits = torch_model(**data_batch)

                else: # Fallback for other potential models
                    X_batch_processed, y_batch_processed = processor.transform(X_batch_raw, y_batch_raw)
                    X_batch = torch.tensor(X_batch_processed, dtype=torch.float32).to(device)
                    y_batch = torch.tensor(y_batch_processed, dtype=torch.long).to(device)
                    logits = torch_model(X_batch)

                loss = loss_fn(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                if config["show_progress"]:
                    iterable.set_postfix(loss=f"{loss.item():.4f}")
        
        logger.info("[TuningManager] Full fine-tuning complete")

    def _finetune_tabpfn(self, model: TabPFNClassifier, X_train_processed: pd.DataFrame, y_train_processed: pd.Series, params: dict | None = None, peft_config=None):
        logger.info("[TuningManager] Starting advanced TabPFN fine-tuning")
        
        config = {
            "device": resolve_device('auto'),
            "epochs": 3, "learning_rate": 1e-5, "batch_size": 256, "show_progress": True 
        }
        if params:
            config.update(params)
        logger.debug(f"[TuningManager] Using fine-tuning config: {config}")

        device = torch.device(config["device"])
        model.model_.to(device)

        for param in model.model_.parameters():
            param.data = param.data.to(device)

        if peft_config:
            logger.warning("[TuningManager] WARNING: TabPFN PEFT support is currently experimental and unstable")
            logger.warning("[TuningManager] TabPFN's batched inference engine conflicts with LoRA adapter state")
            logger.info("[TuningManager] RECOMMENDATION: Use standard finetune strategy for TabPFN instead of 'peft'")
            logger.info("[TuningManager] FALLBACK: Proceeding with standard base fine-tuning")
            peft_config = None  # Disable PEFT for TabPFN

        optimizer = Adam(model.model_.parameters(), lr=config["learning_rate"])
        loss_function = torch.nn.CrossEntropyLoss()

        def stratified_splitter(X, y):
            """
            A robust splitter that attempts to stratify and falls back gracefully.
            """
            # Check if the target is multiclass and has at least 2 samples per class
            y_series = pd.Series(y)
            if y_series.nunique() > 1 and y_series.value_counts().min() > 1:
                # If stratification is possible, use it.
                return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
            else:
                # Otherwise, use a standard random split.
                return train_test_split(X, y, test_size=0.3, random_state=42)

        # Use our new, robust splitter function directly.
        splitter = stratified_splitter

        #splitter = partial(train_test_split, test_size=0.3, stratify=None)
        training_datasets = model.get_preprocessed_datasets(
            X_train_processed, y_train_processed, splitter, config["batch_size"]
        )
        finetuning_dataloader = DataLoader(
            training_datasets, batch_size=1, collate_fn=meta_dataset_collator
        )

        for epoch in range(1, config["epochs"] + 1):
            iterable = finetuning_dataloader
            if config["show_progress"]:
                iterable = tqdm(finetuning_dataloader, desc=f"Finetuning Epoch {epoch}")

            def _move_to_device(item, target_device: torch.device):
                if isinstance(item, torch.Tensor):
                    return item.to(target_device)
                if isinstance(item, list):
                    return [_move_to_device(x, target_device) for x in item]
                if isinstance(item, tuple):
                    return tuple(_move_to_device(x, target_device) for x in item)
                if isinstance(item, dict):
                    return {k: _move_to_device(v, target_device) for k, v in item.items()}
                return item
            
            for (X_train_batch, X_test_batch, y_train_batch, y_test_batch, cat_ixs, confs) in iterable:
                if len(np.unique(y_train_batch)) != len(np.unique(y_test_batch)):
                    logger.debug("[TuningManager] Skipping batch with inconsistent number of classes between train and test splits")
                    continue

                X_train_batch = _move_to_device(X_train_batch, device)
                y_train_batch = _move_to_device(y_train_batch, device)
                X_test_batch = _move_to_device(X_test_batch, device)
                y_test_batch = _move_to_device(y_test_batch, device)


                optimizer.zero_grad()
                model.fit_from_preprocessed(X_train_batch, y_train_batch, cat_ixs, confs)
                predictions = model.forward(X_test_batch, return_logits=True)
                if isinstance(predictions, torch.Tensor) and predictions.device != device:
                    predictions = predictions.to(device)
                # y_test_batch has already been moved above; in rare cases where it is a list
                # choose the first element (batch_size == 1 in our collator)
                if isinstance(y_test_batch, list) and len(y_test_batch) > 0 and isinstance(y_test_batch[0], torch.Tensor):
                    target = y_test_batch[0]
                else:
                    target = y_test_batch
                loss = loss_function(predictions, target)
                loss.backward()
                optimizer.step()
                if config["show_progress"]:
                    iterable.set_postfix(loss=f"{loss.item():.4f}")

        model.batched = False
        logger.info("[TuningManager] Fine-tuning complete")
        logger.debug("[TuningManager] Setting fine-tuned model context for inference...")
        #model.fit(X_train_processed, y_train_processed)




    def _finetune_tabpfn_pure_sft(self, model: TabPFNClassifier, X_train_processed: pd.DataFrame, y_train_processed: pd.Series, params: dict | None = None, peft_config=None):
        """
        Performs SFT-style finetuning.
        
        This is different from the meta-learning loop by:
        1. Using the *entire* dataset to create ONE single, large (Support, Query) episode.
        2. Training repeatedly over this single episode for multiple epochs.
        
        This forces the model to specialize on the single task derived from the 
        full dataset, giving the "SFT sense".
        """
        import torch
        import numpy as np
        import pandas as pd
        from torch.optim import Adam
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        # This collator is required by the TabPFN API
        try:
            from ..models.tabpfn.utils import meta_dataset_collator
        except ImportError:
            logger.error("[TuningManager] FATAL: meta_dataset_collator not found. Please fix the import path")
            # Define a minimal fallback if import fails
            def meta_dataset_collator(batch): return batch[0]
            logger.warning("[TuningManager] Using a placeholder meta_dataset_collator. This may fail")
            
        # Helper to move tensors
        def _move_to_device(item, target_device: torch.device):
            if isinstance(item, torch.Tensor):
                return item.to(target_device)
            if isinstance(item, list):
                return [_move_to_device(x, target_device) for x in item]
            if isinstance(item, tuple):
                return tuple(_move_to_device(x, target_device) for x in item)
            if isinstance(item, dict):
                return {k: _move_to_device(v, target_device) for k, v in item.items()}
            return item

        
        logger.info("[TuningManager] Starting TabPFN SFT fine-tuning")

        config = {
            "device": resolve_device('auto'),
            "epochs": 25,  # More epochs needed as we only have one "batch"
            "learning_rate": 1e-5,
            "show_progress": True,
            "max_episode_size": len(X_train_processed),
            "query_set_ratio": 0.3,
            "weight_decay": 1e-4
        }
        if params:
            # Allow user to override SFT defaults
            config.update(params)
            # Ensure max_episode_size isn't accidentally overridden by 'batch_size'
            if 'batch_size' in params:
                logger.warning("[TuningManager] Ignoring 'batch_size' param, using 'max_episode_size' for SFT")
                config.pop('batch_size', None)
            
        logger.debug(f"[TuningManager] Using SFT-style config: {config}")

        device = torch.device(config["device"])
        model.model_.to(device)
        model.model_.train() # Set to train mode

        for param in model.model_.parameters():
            param.data = param.data.to(device)

        if peft_config:
            logger.warning("[TuningManager] TabPFN PEFT not supported, falling back to base fine-tuning")
            peft_config = None

        optimizer = Adam(model.model_.parameters(), 
                         lr=config["learning_rate"], 
                         weight_decay=config["weight_decay"])
        loss_function = torch.nn.CrossEntropyLoss()
        
        # --- Data & Label Preprocessing ---
        # (This section is the same as the meta-learning function)
        if isinstance(X_train_processed, pd.DataFrame):
            X_train_processed_np = X_train_processed.to_numpy()
        else:
            X_train_processed_np = X_train_processed
            
        if isinstance(y_train_processed, (pd.Series, pd.DataFrame)):
            y_train_processed_np = y_train_processed.to_numpy()
        else:
            y_train_processed_np = y_train_processed

        if y_train_processed_np.dtype == object or not np.issubdtype(y_train_processed_np.dtype, np.number):
            logger.info("[TuningManager] Converting non-numeric labels...")
            le = LabelEncoder()
            y_train_processed_np = le.fit_transform(y_train_processed_np)
            if not hasattr(model, 'label_encoder_'):
                 model.label_encoder_ = le

        def sft_episode_splitter(X, y):
            y_series = pd.Series(y)
            test_size = config["query_set_ratio"]
            if y_series.nunique() > 1 and y_series.value_counts().min() > 1:
                return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
            else:
                return train_test_split(X, y, test_size=test_size, random_state=42)

        logger.info(f"[TuningManager] Creating a single SFT task from {len(X_train_processed_np)} samples...")
        training_datasets = model.get_preprocessed_datasets(
            X_train_processed_np, 
            y_train_processed_np, 
            sft_episode_splitter, 
            config["max_episode_size"] 
        )

        episode_dataloader = DataLoader(
            training_datasets, 
            batch_size=1, 
            collate_fn=meta_dataset_collator,
            shuffle=False
        )

        for epoch in range(1, config["epochs"] + 1):
            
            iterable = tqdm(episode_dataloader, desc=f"SFT Epoch {epoch}", leave=False)
            epoch_losses = []
            
            for (X_support, X_query, y_support, y_query, cat_ixs, confs) in iterable:
                if len(np.unique(y_support)) != len(np.unique(y_query)):
                    logger.warning("[TuningManager] Skipping epoch: Inconsistent classes in SFT split")
                    continue

                X_support = _move_to_device(X_support, device)
                y_support = _move_to_device(y_support, device)
                X_query = _move_to_device(X_query, device)
                y_query = _move_to_device(y_query, device)

                optimizer.zero_grad()
                
                # 1. Set the (large) Support Set as the prompt
                model.fit_from_preprocessed(X_support, y_support, cat_ixs, confs)
                
                # 2. Predict on the (large) Query Set
                predictions = model.forward(X_query, return_logits=True)
                
                if isinstance(predictions, torch.Tensor) and predictions.device != device:
                    predictions = predictions.to(device)
                    
                target = y_query[0] if isinstance(y_query, list) else y_query
                
                # 3. Calculate loss and backpropagate
                loss = loss_function(predictions, target)
                loss.backward()
                
                # SFT HINT 4: Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.model_.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_losses.append(loss.item())
                iterable.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
            logger.info(f"[TuningManager] Epoch [{epoch}/{config['epochs']}]: Task Loss = {avg_loss:.4f}")

        model.batched = False
        model.model_.eval()
        logger.info("[TuningManager] SFT-style finetuning complete")
        return model

    

    def _finetune_tabicl(self, model: (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, OrionMSPv15Classifier), X_train_processed: np.ndarray, y_train_processed: np.ndarray, params: dict | None = None, peft_config=None):
        logger.info("[TuningManager] Starting advanced TabICL/OrionMSP/OrionBix fine-tuning")
        
        config = {
            "device": resolve_device('auto'),
            "epochs": 5, "learning_rate": 2e-6, "show_progress": True,
            "support_size": 48, "query_size": 32, "n_episodes": 1000
        }
        if params:
            config.update(params)
            
        logger.debug(f"[TuningManager] Using fine-tuning config: {config}")
        
        model.fit(X_train_processed, y_train_processed)
        model._load_model()
        model.fit(X_train_processed, y_train_processed) 

        device = torch.device(config["device"])
        if peft_config:
            try:
                if isinstance(model, OrionBixClassifier): 
                    model_key = "OrionBix" 
                elif isinstance(model, OrionMSPClassifier):
                    model_key = "OrionMSP"
                elif isinstance(model, OrionMSPv15Classifier):
                    model_key = "OrionMSPv1.5"
                else :
                    model_key = "TabICL"
                model.model_ = apply_tabular_lora(model_key, model.model_, peft_config)
                logger.info(f"[TuningManager] PEFT SUCCESS: Applied LoRA adapters to {model_key} model")
            except Exception as e:
                logger.warning(f"[TuningManager] PEFT FAILED: TabICL/OrionMSP/OrionBix incompatible with PEFT: {e}")
                logger.info("[TuningManager] FALLBACK: Proceeding with base fine-tuning (fully supported)")
                
        
        model.model_.to(device)
        model.model_.train()
        
        # --- discover the true logits width from a safe 1-class probe ---
        C_out = None
        with torch.no_grad():
            # make one tiny 1-class episode from the first few rows
            X_np = X_train_processed if isinstance(X_train_processed, np.ndarray) else X_train_processed.to_numpy()
            y_np = y_train_processed if isinstance(y_train_processed, np.ndarray) else y_train_processed.to_numpy()

            # pick a class that has >= (support_size + query_size) examples; fall back to any class
            s_sz = int(config.get("support_size", 48))
            q_sz = int(config.get("query_size", 32))
            need = s_sz + q_sz

            cls, idx = None, None
            for c in np.unique(y_np):
                cand = np.nonzero(y_np == c)[0]
                if cand.size >= need:
                    idx = cand[:need]
                    cls = c
                    break
            if idx is None:
                idx = np.arange(min(need, len(y_np)))
                cls = y_np[idx[0]]

            X_ep = torch.from_numpy(X_np[idx]).float().unsqueeze(0).to(device)   # [1, S+Q, F]
            ys   = torch.full((s_sz,), 0, dtype=torch.long, device=device)       # all support -> class 0
            logits_probe = model.model_(X_ep, ys.unsqueeze(0))                   # [1, Q, C_eff] typically
            C_out = int(logits_probe.squeeze(0).size(-1))

        # safety
        if C_out <= 0:
            raise RuntimeError("Could not infer logits width (C_out).")



        for param in model.model_.parameters():
            param.data = param.data.to(device)
        
        optimizer = Adam(model.model_.parameters(), lr=config["learning_rate"])
        loss_fn = torch.nn.CrossEntropyLoss()

        meta_dataset = TabICLMetaDataset(
            X_train_processed, y_train_processed,
            support_size=int(config.get("support_size", 48)),
            query_size=int(config.get("query_size", 32)),
            n_episodes=int(config.get("n_episodes", 1000))
        )
        
        dataloader = DataLoader(meta_dataset, batch_size=1, shuffle=True)
        
        for epoch in range(1, config["epochs"] + 1):
            iterable = dataloader
            if config["show_progress"]:
                iterable = tqdm(dataloader, desc=f"Finetuning Epoch {epoch}")
            for X_episode, y_support, y_query in iterable:
                X_episode, y_support, y_query = X_episode.to(device), y_support.to(device), y_query.to(device)
                optimizer.zero_grad()

                ys = y_support.squeeze(0).long()
                yq = y_query.squeeze(0).long()

                supp = torch.unique(ys)
                keep = supp[:C_out]
                yq_m = torch.full_like(yq, -1)
                ys_m = torch.full_like(ys, -1)
                for i, c in enumerate(keep):
                    ys_m[ys == c] = i
                    yq_m[yq == c] = i


                keep_mask = (ys_m >= 0)
                if not keep_mask.any():
                    continue
                ys_m = ys_m[keep_mask]
                X_support_kept = X_episode[:, :ys.shape[0], :][:, keep_mask, :]
                X_query_part   = X_episode[:, ys.shape[0]:, :]
                X_episode = torch.cat([X_support_kept, X_query_part], dim=1)
                if (yq_m < 0).any():
                    continue

                logits = model.model_(X_episode, ys_m.unsqueeze(0))  
                logits = logits.squeeze(0) 
                if logits.size(-1) < yq_m.max().item() + 1:
                    continue  
                loss = loss_fn(logits, yq_m)


                
                loss.backward()
                optimizer.step()
                if config["show_progress"]:
                    iterable.set_postfix(loss=f"{loss.item():.4f}")
        logger.info("[TuningManager] Fine-tuning complete")


    @staticmethod
    def _tabfm_episode_tensors(X_np, y_np, s_idx, q_idx, cat_mask, device, is_reg=False):
        """Build one episode for the REAL TabFM forward.

        Support rows are concatenated before query rows along the sequence (T)
        axis; ``train_size`` marks the boundary. The model consumes ``y`` only
        for rows < train_size (context), so query targets are placeholders here
        and are compared against the sliced query logits by the caller.

        Returns (x[1,T,H], y[1,T], train_size[1], d[1], cat_mask[1,H]|None,
        query_targets_tensor).
        """
        s, q = len(s_idx), len(q_idx)
        H = X_np.shape[1]
        x_ep = np.concatenate([X_np[s_idx], X_np[q_idx]], axis=0).astype(np.float32)  # [T,H]
        if is_reg:
            y_ctx = y_np[s_idx].astype(np.float32)
            y_ep = np.concatenate([y_ctx, np.zeros(q, dtype=np.float32)])            # [T]
            yq = torch.from_numpy(y_np[q_idx].astype(np.float32)).to(device)
            y_t = torch.from_numpy(y_ep).float().unsqueeze(0).to(device)
        else:
            y_ctx = y_np[s_idx].astype(np.int64)
            y_ep = np.concatenate([y_ctx, np.zeros(q, dtype=np.int64)])              # [T]
            yq = torch.from_numpy(y_np[q_idx].astype(np.int64)).to(device)
            y_t = torch.from_numpy(y_ep).long().unsqueeze(0).to(device)
        x_t = torch.from_numpy(x_ep).float().unsqueeze(0).to(device)                 # [1,T,H]
        ts = torch.full((1,), s, dtype=torch.long, device=device)                   # train_size
        d_t = torch.full((1,), H, dtype=torch.long, device=device)
        cm = None
        if cat_mask is not None:
            cm = torch.from_numpy(np.asarray(cat_mask, dtype=bool)).unsqueeze(0).to(device)
        return x_t, y_t, ts, d_t, cm, yq

    def _finetune_tabfm(self, model: TabFMClassifier, X_train, y_train, params=None, peft_config=None, mode='meta-learning'):
        """Episodic fine-tuning of the REAL vendored TabFM backbone (classification).

        TabFM is an in-context learner, so -- like TabICL / Mitra -- the default
        adaptation is *episodic meta-learning*: repeatedly sample a labelled
        support set + a query set, run the model's **real** forward
        ``model_(x, y, train_size, cat_mask, d)`` (support+query concatenated
        along T; ``train_size`` marks the context boundary), and minimise
        cross-entropy on the query logits (sliced at ``[:, train_size:, :]``).
        ``mode='sft'`` fixes one large episode and trains it for many steps.

        ``X_train`` is the RAW frame: fine-tuning fits the vendored estimator
        (preprocessing + context) and then episode features come from the
        vendored normalisation (:meth:`prepare_episode_features`). LoRA/PEFT is
        injected into the real ``model_`` via :func:`apply_tabular_lora`.
        """
        logger.info("[TuningManager] Starting %s fine-tuning for TabFM (real backbone)", mode)

        config = {
            "device": resolve_device('auto'),
            "epochs": 5, "learning_rate": 2e-6, "show_progress": True,
            "support_size": 64, "query_size": 32, "steps_per_epoch": 200,
            "weight_decay": 0.0, "grad_clip": 1.0,
        }
        if params:
            config.update(params)
        device = torch.device(config["device"])

        model._load_model()
        model.fit(X_train, y_train)  # vendored preprocessing/context + y_encoder_/classes_
        if model.model_ is None:
            logger.warning("[TuningManager] TabFM backbone unavailable; skipping fine-tuning.")
            return model

        if peft_config:
            try:
                model.model_ = apply_tabular_lora("TabFM", model.model_, peft_config)
                logger.info("[TuningManager] PEFT SUCCESS: LoRA adapters applied to TabFM")
            except Exception as e:
                logger.warning(f"[TuningManager] PEFT FAILED for TabFM: {e}. Base fine-tuning instead.")

        model.model_.to(device).train()
        for p in model.model_.parameters():
            p.data = p.data.to(device)

        X_np, cat_mask = model.prepare_episode_features(X_train)
        y_np = np.clip(model.y_encoder_.transform(np.asarray(y_train).ravel()).astype(np.int64), 0, model.max_classes - 1)
        n_samples = X_np.shape[0]

        s_sz, q_sz = int(config["support_size"]), int(config["query_size"])
        if s_sz + q_sz > n_samples:
            scale = n_samples / float(s_sz + q_sz)
            s_sz, q_sz = max(1, int(s_sz * scale)), max(1, int(q_sz * scale))

        optimizer = Adam([p for p in model.model_.parameters() if p.requires_grad],
                         lr=config["learning_rate"], weight_decay=config["weight_decay"])
        loss_fn = torch.nn.CrossEntropyLoss()

        # Fixed split for SFT; fresh random split each step for meta-learning.
        fixed = np.random.permutation(n_samples) if mode == 'sft' else None

        def _step():
            if fixed is not None:
                s_idx, q_idx = fixed[:s_sz], fixed[s_sz:s_sz + q_sz]
            else:
                idx = np.random.choice(n_samples, s_sz + q_sz, replace=(s_sz + q_sz > n_samples))
                s_idx, q_idx = idx[:s_sz], idx[s_sz:]
            x_t, y_t, ts, d_t, cm, yq = self._tabfm_episode_tensors(
                X_np, y_np, s_idx, q_idx, cat_mask, device, is_reg=False)
            logits = model.icl_logits(x_t, y_t, ts, cat_mask=cm, d=d_t)  # [1,T,K]
            ql = logits[:, len(s_idx):, :].reshape(-1, logits.size(-1)).float()
            return loss_fn(ql, yq)

        steps = int(config["steps_per_epoch"])
        for epoch in range(1, config["epochs"] + 1):
            iterable = tqdm(range(steps), desc=f"TabFM {mode} Epoch {epoch}") if config["show_progress"] else range(steps)
            for _ in iterable:
                optimizer.zero_grad()
                try:
                    loss = _step()
                except Exception as e:
                    logger.debug(f"[TuningManager] TabFM episode skipped: {e}")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model_.parameters(), config["grad_clip"])
                optimizer.step()
                if config["show_progress"] and hasattr(iterable, "set_postfix"):
                    iterable.set_postfix(loss=f"{loss.item():.4f}")

        model.model_.eval()
        model.fit(X_train, y_train)  # refresh in-context context with fine-tuned weights
        logger.info("[TuningManager] TabFM fine-tuning complete")
        return model

    def _finetune_tabfm_regression_turn_by_turn(self, model, X_train, y_train, params=None):
        """Episodic turn-by-turn fine-tuning of the REAL TabFM regression backbone.

        Minimises MSE between the query predictions (real forward, sliced at the
        context boundary) and the standardised targets, mirroring the TabDPT /
        Mitra regression fine-tuners.
        """
        logger.info("[TuningManager] Starting TabFM regression fine-tuning (turn-by-turn, real backbone)")

        config = {
            "device": resolve_device('auto'),
            "epochs": 5, "learning_rate": 2e-6, "support_size": 64, "query_size": 32,
            "steps_per_epoch": 200, "show_progress": True, "grad_clip": 1.0, "peft_config": None,
        }
        if params:
            config.update(params)
        device = torch.device(config["device"])

        model._load_model()
        model.fit(X_train, y_train)
        if model.model_ is None:
            logger.warning("[TuningManager] TabFM regression backbone unavailable; skipping.")
            return model

        peft_config = config.get("peft_config")
        if peft_config:
            try:
                model.model_ = apply_tabular_lora("TabFM", model.model_, peft_config)
                logger.info("[TuningManager] LoRA adapters applied to TabFM regressor")
            except Exception as e:
                logger.warning(f"[TuningManager] TabFM regression PEFT failed: {e}")

        model.model_.to(device).train()

        X_np, cat_mask = model.prepare_episode_features(X_train)
        y_np = np.asarray(y_train, dtype=float).ravel()
        y_mean, y_std = float(np.mean(y_np)), float(np.std(y_np) + 1e-8)
        y_scaled = ((y_np - y_mean) / y_std).astype(np.float32)
        n_samples = X_np.shape[0]

        s_sz, q_sz = int(config["support_size"]), int(config["query_size"])
        if s_sz + q_sz > n_samples:
            scale = n_samples / float(s_sz + q_sz)
            s_sz, q_sz = max(1, int(s_sz * scale)), max(1, int(q_sz * scale))

        optimizer = Adam([p for p in model.model_.parameters() if p.requires_grad], lr=config["learning_rate"])
        loss_fn = torch.nn.MSELoss()

        steps = int(config["steps_per_epoch"])
        for epoch in range(1, config["epochs"] + 1):
            iterable = tqdm(range(steps), desc=f"TabFM Reg Epoch {epoch}") if config["show_progress"] else range(steps)
            for _ in iterable:
                optimizer.zero_grad()
                idx = np.random.choice(n_samples, s_sz + q_sz, replace=(s_sz + q_sz > n_samples))
                s_idx, q_idx = idx[:s_sz], idx[s_sz:]
                x_t, y_t, ts, d_t, cm, yq = self._tabfm_episode_tensors(
                    X_np, y_scaled, s_idx, q_idx, cat_mask, device, is_reg=True)
                try:
                    out = model.icl_predict(x_t, y_t, ts, cat_mask=cm, d=d_t)  # [1,T]
                    preds = out[:, len(s_idx):].reshape(-1).float()
                except Exception as e:
                    logger.debug(f"[TuningManager] TabFM reg episode skipped: {e}")
                    continue
                if preds.shape[0] != yq.shape[0]:
                    continue
                loss = loss_fn(preds, yq)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model_.parameters(), config["grad_clip"])
                optimizer.step()
                if config["show_progress"] and hasattr(iterable, "set_postfix"):
                    iterable.set_postfix(loss=f"{loss.item():.4f}")

        model.model_.eval()
        model.fit(X_train, y_train)
        logger.info("[TuningManager] TabFM regression fine-tuning complete")
        return model

    # ------------------------------------------------------------------ ILTM
    # iLTM (Integrated Large Tabular Model) is a meta-trained HYPERNETWORK: its
    # forward pass consumes a labelled support set and GENERATES the weights of
    # an MLP predictor, which is then applied to query rows. Fine-tuning is
    # therefore episodic, exactly like TabFM/TabICL: sample support+query,
    # generate the network from the support set, score the query rows and
    # backprop the query loss into the hypernetwork parameters (or their LoRA
    # adapters under PEFT -- see MODEL_LORA_TARGETS["ILTM"] in peft_utils).

    @staticmethod
    def _iltm_episode_tensors(X_np, y_np, s_idx, q_idx, device, is_reg=False):
        """Build one episode for the REAL iLTM forward.

        Unlike TabFM's concatenated sequence, iLTM takes the support set and
        query set as SEPARATE tensors: the hypernetwork generates the main
        network from ``(x_support, y_support)`` and the query rows are pushed
        through the generated network.

        Returns ``(x_support[S,H], y_support[S], x_query[Q,H], y_query[Q])``.
        """
        xs = torch.from_numpy(X_np[s_idx]).float().to(device)
        xq = torch.from_numpy(X_np[q_idx]).float().to(device)
        if is_reg:
            ys = torch.from_numpy(y_np[s_idx].astype(np.float32)).to(device)
            yq = torch.from_numpy(y_np[q_idx].astype(np.float32)).to(device)
        else:
            ys = torch.from_numpy(y_np[s_idx].astype(np.int64)).to(device)
            yq = torch.from_numpy(y_np[q_idx].astype(np.int64)).to(device)
        return xs, ys, xq, yq

    def _finetune_iltm(self, model: ILTMClassifier, X_train, y_train, params=None, peft_config=None, mode='meta-learning'):
        """Episodic fine-tuning of the REAL vendored iLTM hypernetwork (classification).

        iLTM's hypernetwork was meta-trained across datasets, so -- like TabFM /
        TabICL / Mitra -- the default adaptation is *episodic meta-learning*:
        repeatedly sample a labelled support set + a query set, run the model's
        **real** forward (``model_(x_support, y_support, n_classes)`` generates
        the MLP; the query rows go through the same data-dependent transforms +
        the generated network) and minimise cross-entropy on the query logits.
        ``mode='sft'`` fixes one support/query split and trains it repeatedly.

        ``X_train`` is the RAW frame: fine-tuning first fits the vendored engine
        (internal preprocessing + ensemble + ``y_encoder_``) and episode
        features come from :meth:`prepare_episode_features`. LoRA/PEFT is
        injected into the real ``model_`` via :func:`apply_tabular_lora`. After
        the gradient loop the engine is re-fitted so the ensemble predictors
        are regenerated from the fine-tuned hypernetwork weights.
        """
        logger.info("[TuningManager] Starting %s fine-tuning for ILTM (real hypernetwork)", mode)

        config = {
            "device": resolve_device('auto'),
            "epochs": 3, "learning_rate": 1e-4, "show_progress": True,
            "support_size": 64, "query_size": 32, "steps_per_epoch": 100,
            "weight_decay": 0.0, "grad_clip": 1.0, "finetuning_dropout": 0.0,
        }
        if params:
            config.update(params)
        device = torch.device(config["device"])

        model._load_model()
        model.fit(X_train, y_train)  # vendored preprocessing/ensemble + y_encoder_/classes_
        if model.model_ is None:
            logger.warning("[TuningManager] ILTM backbone unavailable; skipping fine-tuning.")
            return model

        if peft_config:
            try:
                model.model_ = apply_tabular_lora("ILTM", model.model_, peft_config)
                model.estimator_._model = model.model_  # keep the engine on the adapted module
                logger.info("[TuningManager] PEFT SUCCESS: LoRA adapters applied to ILTM")
            except Exception as e:
                logger.warning(f"[TuningManager] PEFT FAILED for ILTM: {e}. Base fine-tuning instead.")

        model.model_.to(device).train()
        for p in model.model_.parameters():
            p.data = p.data.to(device)

        X_np, _ = model.prepare_episode_features(X_train)
        y_np = np.clip(model.y_encoder_.transform(np.asarray(y_train).ravel()).astype(np.int64), 0, model.max_classes - 1)
        n_classes = int(min(model.n_classes_, model.max_classes))
        n_samples = X_np.shape[0]

        s_sz, q_sz = int(config["support_size"]), int(config["query_size"])
        if s_sz + q_sz > n_samples:
            scale = n_samples / float(s_sz + q_sz)
            s_sz, q_sz = max(2, int(s_sz * scale)), max(1, int(q_sz * scale))

        optimizer = torch.optim.AdamW([p for p in model.model_.parameters() if p.requires_grad],
                                      lr=config["learning_rate"], weight_decay=config["weight_decay"])
        loss_fn = torch.nn.CrossEntropyLoss()

        # Fixed split for SFT; fresh random split each step for meta-learning.
        fixed = np.random.permutation(n_samples) if mode == 'sft' else None

        def _step():
            if fixed is not None:
                s_idx, q_idx = fixed[:s_sz], fixed[s_sz:s_sz + q_sz]
            else:
                idx = np.random.choice(n_samples, s_sz + q_sz, replace=(s_sz + q_sz > n_samples))
                s_idx, q_idx = idx[:s_sz], idx[s_sz:]
            xs, ys, xq, yq = self._iltm_episode_tensors(X_np, y_np, s_idx, q_idx, device, is_reg=False)
            logits = model.episode_logits(xs, ys, xq, n_classes, training=True,
                                          dropout=config["finetuning_dropout"])  # [Q, K]
            return loss_fn(logits.reshape(-1, logits.size(-1)).float(), yq)

        steps = int(config["steps_per_epoch"])
        for epoch in range(1, config["epochs"] + 1):
            iterable = tqdm(range(steps), desc=f"ILTM {mode} Epoch {epoch}") if config["show_progress"] else range(steps)
            for _ in iterable:
                optimizer.zero_grad()
                try:
                    loss = _step()
                except Exception as e:
                    logger.debug(f"[TuningManager] ILTM episode skipped: {e}")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model_.parameters(), config["grad_clip"])
                optimizer.step()
                if config["show_progress"] and hasattr(iterable, "set_postfix"):
                    iterable.set_postfix(loss=f"{loss.item():.4f}")

        model.model_.eval()
        model.fit(X_train, y_train)  # regenerate the ensemble from the fine-tuned hypernetwork
        logger.info("[TuningManager] ILTM fine-tuning complete")
        return model

    def _finetune_iltm_regression_turn_by_turn(self, model, X_train, y_train, params=None):
        """Episodic turn-by-turn fine-tuning of the REAL iLTM regression hypernetwork.

        Minimises MSE between the query predictions (generated-network forward
        with ``n_classes=1``) and the standardised targets, mirroring the
        TabFM / TabDPT / Mitra regression fine-tuners. Reads ``peft_config``
        from the tuning config (regression PEFT routes through here).
        """
        logger.info("[TuningManager] Starting ILTM regression fine-tuning (turn-by-turn, real hypernetwork)")

        config = {
            "device": resolve_device('auto'),
            "epochs": 3, "learning_rate": 1e-4, "support_size": 64, "query_size": 32,
            "steps_per_epoch": 100, "show_progress": True, "grad_clip": 1.0,
            "weight_decay": 0.0, "finetuning_dropout": 0.0, "peft_config": None,
        }
        if params:
            config.update(params)
        device = torch.device(config["device"])

        model._load_model()
        model.fit(X_train, y_train)
        if model.model_ is None:
            logger.warning("[TuningManager] ILTM regression backbone unavailable; skipping.")
            return model

        peft_config = config.get("peft_config")
        if peft_config:
            try:
                model.model_ = apply_tabular_lora("ILTM", model.model_, peft_config)
                model.estimator_._model = model.model_
                logger.info("[TuningManager] LoRA adapters applied to ILTM regressor")
            except Exception as e:
                logger.warning(f"[TuningManager] ILTM regression PEFT failed: {e}")

        model.model_.to(device).train()
        for p in model.model_.parameters():
            p.data = p.data.to(device)

        X_np, _ = model.prepare_episode_features(X_train)
        y_np = np.asarray(y_train, dtype=float).ravel()
        y_mean, y_std = float(np.mean(y_np)), float(np.std(y_np) + 1e-8)
        y_scaled = ((y_np - y_mean) / y_std).astype(np.float32)
        n_samples = X_np.shape[0]

        s_sz, q_sz = int(config["support_size"]), int(config["query_size"])
        if s_sz + q_sz > n_samples:
            scale = n_samples / float(s_sz + q_sz)
            s_sz, q_sz = max(2, int(s_sz * scale)), max(1, int(q_sz * scale))

        optimizer = torch.optim.AdamW([p for p in model.model_.parameters() if p.requires_grad],
                                      lr=config["learning_rate"], weight_decay=config["weight_decay"])
        loss_fn = torch.nn.MSELoss()

        steps = int(config["steps_per_epoch"])
        for epoch in range(1, config["epochs"] + 1):
            iterable = tqdm(range(steps), desc=f"ILTM Reg Epoch {epoch}") if config["show_progress"] else range(steps)
            for _ in iterable:
                optimizer.zero_grad()
                idx = np.random.choice(n_samples, s_sz + q_sz, replace=(s_sz + q_sz > n_samples))
                s_idx, q_idx = idx[:s_sz], idx[s_sz:]
                xs, ys, xq, yq = self._iltm_episode_tensors(X_np, y_scaled, s_idx, q_idx, device, is_reg=True)
                try:
                    preds = model.episode_predict(xs, ys, xq, training=True,
                                                  dropout=config["finetuning_dropout"]).reshape(-1).float()
                except Exception as e:
                    logger.debug(f"[TuningManager] ILTM reg episode skipped: {e}")
                    continue
                if preds.shape[0] != yq.shape[0]:
                    continue
                loss = loss_fn(preds, yq)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model_.parameters(), config["grad_clip"])
                optimizer.step()
                if config["show_progress"] and hasattr(iterable, "set_postfix"):
                    iterable.set_postfix(loss=f"{loss.item():.4f}")

        model.model_.eval()
        model.fit(X_train, y_train)
        logger.info("[TuningManager] ILTM regression fine-tuning complete")
        return model

    # ---- EXAONE TABULAR ----
    # EXAONE Tabular (LG AI Research) is an in-context learner built on the
    # Cross-axis Summary Transformer: one forward consumes a labelled support
    # set plus a query set and scores the query rows directly. Adaptation is
    # therefore episodic, exactly like TabFM / iLTM / TabICL -- sample a support
    # set and a query set, run the model's REAL differentiable forward
    # (``episode_logits`` / ``episode_predictions``, which deliberately bypass
    # the vendored predict path because that runs under ``inference_mode`` and
    # yields tensors autograd can never accept) and backprop the query loss.
    #
    # Three model facts shape the loops below:
    #
    # 1. The forward carries an ENSEMBLE axis: ``x_support`` is (E, S, K),
    #    ``y_support`` is (E, S) and ``x_query`` is (E, Q, K). Fine-tuning uses
    #    E=1 -- the ensemble members differ only in preprocessing draws, which
    #    are an inference-time device, and one member per step keeps the
    #    gradient estimate cheap.
    # 2. Classification logits come back over the ARCHITECTURAL class capacity
    #    (ten), not the dataset's class count, so they are sliced to
    #    ``n_classes`` before cross-entropy. Without the slice the loss would
    #    spend gradient pushing down padding columns that can never be labels.
    # 3. ``y_support`` is NOT differentiable and must not be optimised: the
    #    model's label encoder turns labels into ordinal ranks with a
    #    comparison-and-count, which has zero gradient almost everywhere. Only
    #    parameters (and, in principle, support FEATURES) carry gradient.
    #
    # The vendored ``_validate_inputs`` additionally requires float support
    # labels that are finite, integral and non-negative for classification, so
    # the class codes are carried as float32 rather than int64.

    @staticmethod
    def _exaone_episode_tensors(X_np, y_np, s_idx, q_idx, device, is_reg=False):
        """Build one episode for the REAL EXAONE forward.

        EXAONE takes the support set and the query set as SEPARATE tensors, each
        with a leading ensemble axis. Fine-tuning uses a single member (E=1).

        Returns ``(x_support[1,S,K], y_support[1,S], x_query[1,Q,K], y_query[Q])``.
        ``y_support`` is float32 in both tasks -- the vendored input validator
        rejects integer label tensors -- while ``y_query`` is int64 for
        classification (cross-entropy targets) and float32 for regression.
        """
        xs = torch.from_numpy(np.ascontiguousarray(X_np[s_idx])).float().unsqueeze(0).to(device)
        xq = torch.from_numpy(np.ascontiguousarray(X_np[q_idx])).float().unsqueeze(0).to(device)
        ys = torch.from_numpy(np.ascontiguousarray(y_np[s_idx])).float().unsqueeze(0).to(device)
        if is_reg:
            yq = torch.from_numpy(np.ascontiguousarray(y_np[q_idx])).float().to(device)
        else:
            yq = torch.from_numpy(np.ascontiguousarray(y_np[q_idx])).long().to(device)
        return xs, ys, xq, yq

    def _finetune_exaone(self, model: EXAONETabularClassifier, X_train, y_train, params=None,
                         peft_config=None, mode='meta-learning'):
        """Episodic fine-tuning of the REAL vendored EXAONE backbone (classification).

        EXAONE is an in-context learner, so -- like TabFM / iLTM / TabICL -- the
        default adaptation is *episodic meta-learning*: repeatedly sample a
        labelled support set + a query set, run the model's **real** forward
        ``model.episode_logits(x_support, y_support, x_query)`` and minimise
        cross-entropy on the query logits, sliced to the dataset's class count
        (the head is as wide as the architectural class capacity).
        ``mode='sft'`` fixes one support/query split and trains it repeatedly.

        ``X_train`` is the RAW frame: fine-tuning first fits the vendored engine
        (its own preprocessing + ensemble + ``y_encoder_``/``classes_``) and
        episode features then come from :meth:`prepare_episode_features`, i.e.
        the model's own fitted encoder. LoRA/PEFT is injected into the real
        ``model_`` via :func:`apply_tabular_lora`. After the gradient loop the
        engine is re-fitted so the in-context support state is rebuilt from the
        fine-tuned weights.
        """
        logger.info("[TuningManager] Starting %s fine-tuning for EXAONETabular (real backbone)", mode)

        config = {
            "device": resolve_device('auto'),
            "epochs": 3, "learning_rate": 1e-5, "show_progress": True,
            "support_size": 64, "query_size": 32, "steps_per_epoch": 50,
            "weight_decay": 0.0, "grad_clip": 1.0,
        }
        if params:
            config.update(params)
        device = torch.device(config["device"])

        model._load_model()
        model.fit(X_train, y_train)  # vendored preprocessing/ensemble + y_encoder_/classes_
        if model.model_ is None:
            logger.warning("[TuningManager] EXAONETabular backbone unavailable; skipping fine-tuning.")
            return model

        if peft_config:
            try:
                model.model_ = apply_tabular_lora("EXAONETabular", model.model_, peft_config)
                if model.estimator_ is not None:
                    model.estimator_.model = model.model_  # keep the engine on the adapted module
                logger.info("[TuningManager] PEFT SUCCESS: LoRA adapters applied to EXAONETabular")
                self._warn_if_no_lora_adapters("EXAONETabular", model.model_)
            except Exception as e:
                logger.warning(f"[TuningManager] PEFT FAILED for EXAONETabular: {e}. Base fine-tuning instead.")

        model.model_.to(device).train()
        for p in model.model_.parameters():
            p.data = p.data.to(device)

        X_np, _ = model.prepare_episode_features(X_train)
        # Support labels ride into the model as floats (see _exaone_episode_tensors);
        # the query targets are cast back to int64 there for cross-entropy.
        y_np = np.clip(model.y_encoder_.transform(np.asarray(y_train).ravel()).astype(np.int64),
                       0, model.max_classes - 1)
        n_classes = int(min(model.n_classes_, model.max_classes))
        n_samples = X_np.shape[0]

        s_sz, q_sz = int(config["support_size"]), int(config["query_size"])
        if s_sz + q_sz > n_samples:
            scale = n_samples / float(s_sz + q_sz)
            s_sz, q_sz = max(2, int(s_sz * scale)), max(1, int(q_sz * scale))

        optimizer = torch.optim.AdamW([p for p in model.model_.parameters() if p.requires_grad],
                                      lr=config["learning_rate"], weight_decay=config["weight_decay"])
        loss_fn = torch.nn.CrossEntropyLoss()

        # Fixed split for SFT; fresh random split each step for meta-learning.
        fixed = np.random.permutation(n_samples) if mode == 'sft' else None

        def _step():
            if fixed is not None:
                s_idx, q_idx = fixed[:s_sz], fixed[s_sz:s_sz + q_sz]
            else:
                idx = np.random.choice(n_samples, s_sz + q_sz, replace=(s_sz + q_sz > n_samples))
                s_idx, q_idx = idx[:s_sz], idx[s_sz:]
            xs, ys, xq, yq = self._exaone_episode_tensors(X_np, y_np, s_idx, q_idx, device, is_reg=False)
            logits = model.episode_logits(xs, ys, xq)  # [E=1, Q, class_capacity]
            # Slice the architectural class capacity down to the dataset's real
            # class count before the loss -- the extra columns are never labels.
            ql = logits[..., :n_classes].reshape(-1, n_classes).float()
            return loss_fn(ql, yq)

        steps = int(config["steps_per_epoch"])
        for epoch in range(1, config["epochs"] + 1):
            iterable = tqdm(range(steps), desc=f"EXAONE {mode} Epoch {epoch}") if config["show_progress"] else range(steps)
            for step_index in iterable:
                optimizer.zero_grad()
                try:
                    loss = _step()
                except Exception as e:
                    logger.debug(f"[TuningManager] EXAONETabular episode skipped: {e}")
                    continue
                # Refuse to write a non-finite update. Half precision without loss
                # scaling can send this loss to NaN, and an optimizer step on a NaN
                # gradient poisons every weight it touches. The failure then shows
                # up much later and somewhere unrelated -- most confusingly inside
                # the support-cache validity check, which compares tensors with
                # torch.equal and gets False for NaN even when they are
                # bit-identical, surfacing as "inputs are incompatible with the
                # support cache". Stopping here keeps the backbone usable and
                # names the real cause.
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"[TuningManager] EXAONETabular fine-tuning produced a "
                        f"non-finite loss ({loss.item()}) at epoch {epoch}, step "
                        f"{step_index + 1}; stopped before the optimizer could "
                        f"write NaN weights. This is almost always half precision: "
                        f"the released manifest asks for float16, which is right "
                        f"for inference and unsafe for a backward pass without loss "
                        f"scaling. Fine-tuning defaults to float32 as of 0.2.0 -- if "
                        f"you overrode dtype, drop the override, or lower the "
                        f"learning rate (currently {config['learning_rate']})."
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model_.parameters(), config["grad_clip"])
                optimizer.step()
                if config["show_progress"] and hasattr(iterable, "set_postfix"):
                    iterable.set_postfix(loss=f"{loss.item():.4f}")

        model.model_.eval()
        model.fit(X_train, y_train)  # rebuild the in-context support state with fine-tuned weights
        logger.info("[TuningManager] EXAONETabular fine-tuning complete")
        return model

    def _finetune_exaone_regression_turn_by_turn(self, model, X_train, y_train, params=None):
        """Episodic turn-by-turn fine-tuning of the REAL EXAONE regression backbone.

        The regression head emits ``quantile_count`` levels per query row; the
        median level (``model.median_quantile_index``) is what ``predict``
        reports, so that is the column the loss is taken on. Targets are
        standardised for the duration of the loop because the vendored engine
        centres and scales the support targets itself before its own forward --
        the episodes must speak the same normalised space. Nothing is written
        back: ``predict`` still un-scales internally and still returns the
        original target space, which is exactly why the EXAONE regression data
        processor pins ``target_scaling_strategy='none'``.

        Mirrors the TabFM / iLTM / Mitra regression fine-tuners and reads
        ``peft_config`` from the tuning config (regression PEFT routes here).
        """
        logger.info("[TuningManager] Starting EXAONETabular regression fine-tuning (turn-by-turn, real backbone)")

        config = {
            "device": resolve_device('auto'),
            "epochs": 3, "learning_rate": 1e-5, "support_size": 64, "query_size": 32,
            "steps_per_epoch": 50, "show_progress": True, "grad_clip": 1.0,
            "weight_decay": 0.0, "peft_config": None,
        }
        if params:
            config.update(params)
        device = torch.device(config["device"])

        model._load_model()
        model.fit(X_train, y_train)
        if model.model_ is None:
            logger.warning("[TuningManager] EXAONETabular regression backbone unavailable; skipping.")
            return model

        peft_config = config.get("peft_config")
        if peft_config:
            try:
                model.model_ = apply_tabular_lora("EXAONETabular", model.model_, peft_config)
                if model.estimator_ is not None:
                    model.estimator_.model = model.model_
                logger.info("[TuningManager] LoRA adapters applied to EXAONETabular regressor")
                self._warn_if_no_lora_adapters("EXAONETabular", model.model_)
            except Exception as e:
                logger.warning(f"[TuningManager] EXAONETabular regression PEFT failed: {e}")

        model.model_.to(device).train()
        for p in model.model_.parameters():
            p.data = p.data.to(device)

        X_np, _ = model.prepare_episode_features(X_train)
        y_np = np.asarray(y_train, dtype=float).ravel()
        y_mean, y_std = float(np.mean(y_np)), float(np.std(y_np) + 1e-8)
        y_scaled = ((y_np - y_mean) / y_std).astype(np.float32)
        median_index = int(model.median_quantile_index)
        n_samples = X_np.shape[0]

        s_sz, q_sz = int(config["support_size"]), int(config["query_size"])
        if s_sz + q_sz > n_samples:
            scale = n_samples / float(s_sz + q_sz)
            s_sz, q_sz = max(2, int(s_sz * scale)), max(1, int(q_sz * scale))

        optimizer = torch.optim.AdamW([p for p in model.model_.parameters() if p.requires_grad],
                                      lr=config["learning_rate"], weight_decay=config["weight_decay"])
        loss_fn = torch.nn.MSELoss()

        steps = int(config["steps_per_epoch"])
        for epoch in range(1, config["epochs"] + 1):
            iterable = tqdm(range(steps), desc=f"EXAONE Reg Epoch {epoch}") if config["show_progress"] else range(steps)
            for _ in iterable:
                optimizer.zero_grad()
                idx = np.random.choice(n_samples, s_sz + q_sz, replace=(s_sz + q_sz > n_samples))
                s_idx, q_idx = idx[:s_sz], idx[s_sz:]
                xs, ys, xq, yq = self._exaone_episode_tensors(X_np, y_scaled, s_idx, q_idx, device, is_reg=True)
                try:
                    quantiles = model.episode_predictions(xs, ys, xq)  # [E=1, Q, quantile_count]
                    preds = quantiles[..., median_index].reshape(-1).float()
                except Exception as e:
                    logger.debug(f"[TuningManager] EXAONETabular reg episode skipped: {e}")
                    continue
                if preds.shape[0] != yq.shape[0]:
                    continue
                loss = loss_fn(preds, yq)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model_.parameters(), config["grad_clip"])
                optimizer.step()
                if config["show_progress"] and hasattr(iterable, "set_postfix"):
                    iterable.set_postfix(loss=f"{loss.item():.4f}")

        model.model_.eval()
        model.fit(X_train, y_train)
        logger.info("[TuningManager] EXAONETabular regression fine-tuning complete")
        return model

    @staticmethod
    def _warn_if_no_lora_adapters(model_name: str, torch_model) -> None:
        """Say so loudly when a LoRA request wrapped nothing.

        ``apply_tabular_lora`` only wraps ``nn.Linear`` leaves. A model whose
        projections are raw ``nn.Parameter``s applied through ``F.linear`` --
        EXAONE's attention and feed-forward blocks are exactly that -- has no
        such leaves to wrap outside its task head, so injection succeeds while
        adapting nothing. Without this the log would claim "PEFT SUCCESS" over a
        run that is really a full fine-tune.
        """
        from .peft_utils import LoRALinear

        if not any(isinstance(module, LoRALinear) for module in torch_model.modules()):
            logger.warning(
                "[TuningManager] PEFT for %s wrapped ZERO layers: its projections are raw "
                "nn.Parameters used through F.linear, which the nn.Linear-based LoRA "
                "injector cannot reach. Proceeding as a FULL fine-tune (every parameter "
                "trainable), not a parameter-efficient one.",
                model_name,
            )

    # ------------------------------------------------------------------ XRFM
    # xRFM (Recursive Feature Machines) is a kernel method: its learned state
    # is the per-leaf Mahalanobis matrix M (from AGOP iterations) plus kernel
    # ridge weights -- NOT a gradient-trained nn.Module with a pretrained
    # checkpoint. TabTune therefore maps the tuning strategies as follows:
    #
    # * 'finetune'  -> full RFM (re)training on the training split with
    #   user-controllable hyperparameters (iters, bandwidth, kernel, tree
    #   params, ...). If the wrapper was already fitted, warm-start instead:
    #   run additional RFM iterations (AGOP -> M update -> kernel refit) from
    #   the existing fitted M ("continued refinement").
    # * 'peft'      -> parameter-efficient adaptation of M: freeze the base
    #   fitted M and add a rank-r truncated update from the AGOP of the
    #   adaptation data, M_adapted = M_base + alpha * U_r diag(s_r) U_r^T
    #   (rank via tuning_params['lora_rank']/'rank' or peft_config['r'],
    #   blend via 'peft_alpha'). No LoRA nn.Linear machinery is involved.
    #
    # Checkpoints are saved via joblib (the fitted xRFM estimator is a plain
    # picklable object, not a torch state_dict).

    _XRFM_HYPERPARAM_KEYS = (
        "rfm_params", "kernel", "bandwidth", "exponent", "diag", "bandwidth_mode",
        "reg", "iters", "n_trees", "tuning_metric", "categorical_encoding",
        "val_size", "random_state", "verbose",
    )

    @staticmethod
    def _xrfm_apply_hyperparams(model, config):
        """Push user-controllable xRFM hyperparameters from the FT config onto the wrapper."""
        applied = {}
        for key in TuningManager._XRFM_HYPERPARAM_KEYS:
            if key in config:
                setattr(model, key, config[key])
                applied[key] = config[key]
        if applied:
            logger.info(f"[TuningManager] XRFM hyperparameters overridden: {applied}")

    @staticmethod
    def _xrfm_save_checkpoint(model, save_path):
        """Persist the fitted xRFM estimator via joblib (not a torch state_dict)."""
        if not save_path:
            return
        import joblib

        root, ext = os.path.splitext(str(save_path))
        path = root + ".joblib" if ext != ".joblib" else str(save_path)
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            joblib.dump(model.model_, path)
            logger.info(f"[TuningManager] Saved fitted xRFM checkpoint to {path} (joblib)")
        except Exception as e:
            logger.warning(f"[TuningManager] Failed to save xRFM checkpoint: {e}")

    @staticmethod
    def _xrfm_low_rank_M_update(M_base, agop, rank, alpha):
        """Rank-r truncated additive update of the Mahalanobis matrix M.

        The AGOP of the adaptation data is eigendecomposed and only the top-r
        components are kept: delta = U_r diag(s_r) U_r^T (top-r entries for a
        diagonal M). Both terms are max-normalised (the engine's own M
        normalisation) and blended: M_adapted = norm(M_base + alpha * delta).
        """
        eps = 1e-30
        if agop.dim() == 1:  # diagonal M: keep the top-r entries
            r = min(int(rank), agop.numel())
            delta = torch.zeros_like(agop)
            top_vals, top_idx = torch.topk(agop, r)
            delta[top_idx] = torch.clamp(top_vals, min=0.0)
        else:
            agop_sym = 0.5 * (agop + agop.T)
            S, U = torch.linalg.eigh(agop_sym)
            S = torch.clamp(S, min=0.0)
            r = min(int(rank), S.numel())
            idx = torch.argsort(S, descending=True)[:r]
            U_r, S_r = U[:, idx], S[idx]
            delta = U_r @ torch.diag(S_r) @ U_r.T
        delta = delta / (delta.max() + eps)
        M_new = M_base + float(alpha) * delta
        return M_new / (M_new.max() + eps)

    def _xrfm_refit_leaf_predictor(self, leaf, X_leaf, y_leaf):
        """Re-solve the leaf's kernel predictor (centers/weights) under its current M."""
        from ..models.xrfm.rfm_src.utils import matrix_power

        if getattr(leaf, "use_sqrtM", False):
            leaf.sqrtM = matrix_power(leaf.M, leaf.agop_power, verbose=False)
        leaf.fit_predictor(X_leaf, y_leaf, X_val=X_leaf, y_val=y_leaf)

    def _xrfm_refine_leaves(self, model, X_train, y_train, iters):
        """Continued RFM refinement of already-fitted leaves on new data.

        Per leaf and per iteration: AGOP of the adaptation data under the
        current predictor -> max-normalised M update -> kernel refit on the
        adaptation data (the standard RFM iteration, warm-started from the
        fitted M instead of the identity).
        """
        leaves = model.leaf_models()
        if not leaves:
            logger.warning("[TuningManager] XRFM has no fitted leaves to refine; skipping.")
            return
        X_num = model.transform_features(X_train)
        y_num = model.numeric_targets(y_train)
        for li, leaf in enumerate(leaves):
            X_leaf = torch.as_tensor(X_num).to(leaf.device)
            y_leaf = y_num.to(device=leaf.device, dtype=X_leaf.dtype)
            for it in range(int(iters)):
                try:
                    agop = leaf.update_M(X_leaf)
                    leaf.M = agop / (agop.max() + 1e-30)
                    self._xrfm_refit_leaf_predictor(leaf, X_leaf, y_leaf)
                except Exception as e:
                    logger.warning(f"[TuningManager] XRFM leaf {li} refinement stopped at iter {it}: {e}")
                    break

    def _finetune_xrfm(self, model, X_train, y_train, params=None, save_path=None):
        """Full-model fine-tuning for XRFM (classification): RFM (re)training.

        xRFM has no pretrained checkpoint, so 'finetune' means training the
        kernel machine on this task with user-controllable hyperparameters
        (``iters``, ``bandwidth``, ``kernel``, ``reg``, tree params, ...). When
        the wrapper is already fitted, the fitted M is warm-started instead:
        ``refine_iters`` additional RFM iterations are run on the new data
        (set ``warm_start=False`` to force a from-scratch refit).
        """
        logger.info("[TuningManager] Starting XRFM fine-tuning (full RFM training)")

        config = {
            "device": resolve_device('auto'),
            "warm_start": True,
            "refine_iters": 2,
        }
        if params:
            config.update(params)
        config.pop("peft_config", None)
        model.device = model.device or config["device"]
        self._xrfm_apply_hyperparams(model, config)

        already_fitted = bool(getattr(model, "_is_fitted", False))
        if already_fitted and config["warm_start"]:
            logger.info(
                f"[TuningManager] XRFM already fitted -> warm start: "
                f"{config['refine_iters']} continued RFM iterations from the existing M"
            )
            self._xrfm_refine_leaves(model, X_train, y_train, config["refine_iters"])
        else:
            model.fit(X_train, y_train)

        self._xrfm_save_checkpoint(model, save_path or config.get("save_checkpoint_path"))
        logger.info("[TuningManager] XRFM fine-tuning complete")
        return model

    def _peft_xrfm(self, model, X_train, y_train, params=None, peft_config=None, save_path=None):
        """Parameter-efficient adaptation for XRFM: low-rank M-matrix update.

        The base fitted Mahalanobis matrix M is frozen and a rank-r truncated
        correction from the AGOP of the adaptation data is added per leaf:
        ``M_adapted = norm(M_base + alpha * U_r diag(s_r) U_r^T)``. Only the
        r x d low-rank factors are learned from the new data (the kernel ridge
        weights are re-solved, as they must be for any M change). This is the
        kernel-method analogue of LoRA -- no nn.Linear adapters are injected
        (xRFM has no Linear layers), so ``apply_tabular_lora`` is NOT used.

        Rank comes from ``peft_config['r']`` / ``tuning_params['lora_rank']`` /
        ``'rank'`` (default 8); the blend strength from ``'peft_alpha'`` /
        ``peft_config['alpha']`` (default 0.5). If the model has not been
        fitted yet, a base fit on the training split provides the frozen M.
        """
        logger.info("[TuningManager] Starting XRFM PEFT (low-rank M-matrix adaptation)")

        config = {
            "device": resolve_device('auto'),
            "lora_rank": 8,
            "peft_alpha": 0.5,
        }
        if params:
            config.update(params)
        peft_config = dict(peft_config or config.pop("peft_config", None) or {})
        rank = int(peft_config.get("r", config.get("rank", config["lora_rank"])))
        alpha = float(peft_config.get("alpha", config["peft_alpha"]))
        model.device = model.device or config["device"]
        self._xrfm_apply_hyperparams(model, config)

        if not getattr(model, "_is_fitted", False):
            logger.info("[TuningManager] XRFM not fitted yet; fitting the frozen base model first")
            model.fit(X_train, y_train)

        leaves = model.leaf_models()
        if not leaves:
            logger.warning("[TuningManager] XRFM has no fitted leaves; PEFT adaptation skipped.")
            return model

        X_num = model.transform_features(X_train)
        y_num = model.numeric_targets(y_train)
        logger.info(f"[TuningManager] Adapting {len(leaves)} leaf M matrices (rank={rank}, alpha={alpha})")
        for li, leaf in enumerate(leaves):
            X_leaf = torch.as_tensor(X_num).to(leaf.device)
            y_leaf = y_num.to(device=leaf.device, dtype=X_leaf.dtype)
            try:
                if leaf.M is None:  # kernel never built an M (e.g. iters=0): identity base
                    d = X_leaf.shape[1]
                    leaf.M = (torch.ones(d, device=X_leaf.device, dtype=X_leaf.dtype) if leaf.diag
                              else torch.eye(d, device=X_leaf.device, dtype=X_leaf.dtype))
                M_base = leaf.M.clone()  # frozen base
                agop = leaf.update_M(X_leaf)  # AGOP of the adaptation data under the base model
                leaf.M = self._xrfm_low_rank_M_update(M_base, agop, rank, alpha)
                self._xrfm_refit_leaf_predictor(leaf, X_leaf, y_leaf)
            except Exception as e:
                logger.warning(f"[TuningManager] XRFM PEFT failed on leaf {li}: {e}. Leaf left at base state.")

        self._xrfm_save_checkpoint(model, save_path or config.get("save_checkpoint_path"))
        logger.info("[TuningManager] XRFM PEFT adaptation complete")
        return model

    def _finetune_xrfm_regression(self, model, X_train, y_train, params=None):
        """Full-model fine-tuning for XRFM regression (same mapping as classification).

        'finetune' = full RFM training with user-controllable hyperparameters;
        warm-started continued refinement of the fitted M when the wrapper was
        already fitted. Supports the same low-rank PEFT path when a
        ``peft_config`` is supplied in the tuning params.
        """
        logger.info("[TuningManager] Starting XRFM regression fine-tuning (full RFM training)")

        config = {
            "device": resolve_device('auto'),
            "warm_start": True,
            "refine_iters": 2,
        }
        if params:
            config.update(params)
        config.pop("finetune_mode", None)  # not applicable to a kernel method
        save_path = config.pop("save_checkpoint_path", None)

        peft_config = config.pop("peft_config", None)
        if peft_config:
            return self._peft_xrfm(model, X_train, y_train, params=config,
                                   peft_config=peft_config, save_path=save_path)

        model.device = model.device or config["device"]
        self._xrfm_apply_hyperparams(model, config)

        already_fitted = bool(getattr(model, "_is_fitted", False))
        if already_fitted and config["warm_start"]:
            logger.info(
                f"[TuningManager] XRFM regressor already fitted -> warm start: "
                f"{config['refine_iters']} continued RFM iterations from the existing M"
            )
            self._xrfm_refine_leaves(model, X_train, y_train, config["refine_iters"])
        else:
            model.fit(X_train, y_train)

        self._xrfm_save_checkpoint(model, save_path)
        logger.info("[TuningManager] XRFM regression fine-tuning complete")
        return model

    def _finetune_tabicl_pure_sft(self, model: (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, OrionMSPv15Classifier) , X_train_processed, y_train_processed, params=None, peft_config=None):
        """
        PURE SFT FINE-TUNING (Not Recommended for TabICL)
    
        Standard supervised fine-tuning on full batches WITHOUT episodic structure.
    
        WARNING: This ignores TabICL's meta-learning design and may:
        - Reduce generalization to new tasks
        - Increase catastrophic forgetting
        - Overfit to the specific target task
    
        Use ONLY for:
        - Benchmarking against traditional fine-tuning
        - Comparison studies
        - Tasks where you explicitly want to sacrifice generalization for accuracy
        """
        logger.warning("[TuningManager] WARNING: Pure SFT on TabICL breaks its meta-learning design")
        logger.warning("[TuningManager] This approach may reduce generalization to new tasks")
        logger.info("[TuningManager] RECOMMENDATION: Use episodic or SFT-hybrid instead")
        logger.info("[TuningManager] PROCEED: Using pure SFT (use only for comparisons)")
    
        config = {
            "device": resolve_device('auto'),
            "epochs": 10,
            "learning_rate": 1e-5,
            "batch_size": 32,
            "show_progress": True,
            "weight_decay": 1e-4,
            "warmup_epochs": 1
        }
        if params:
            config.update(params)
        logger.debug(f"[TuningManager] Using config: {config}")
    
        device = torch.device(config["device"])
        model.fit(X_train_processed, y_train_processed)
        model._load_model()
    
        model.model_.to(device)
        model.model_.train()
        
        C_out = None
        with torch.no_grad():
            X_np = X_train_processed if isinstance(X_train_processed, np.ndarray) else X_train_processed.to_numpy()
            y_np = y_train_processed if isinstance(y_train_processed, np.ndarray) else y_train_processed.to_numpy()

            
            s_sz = int(config.get("support_size", 48))
            q_sz = int(config.get("query_size", 32))
            need = s_sz + q_sz

            cls, idx = None, None
            for c in np.unique(y_np):
                cand = np.nonzero(y_np == c)[0]
                if cand.size >= need:
                    idx = cand[:need]
                    cls = c
                    break
            if idx is None:
                idx = np.arange(min(need, len(y_np)))
                cls = y_np[idx[0]]

            X_ep = torch.from_numpy(X_np[idx]).float().unsqueeze(0).to(device)   
            ys   = torch.full((s_sz,), 0, dtype=torch.long, device=device)       
            logits_probe = model.model_(X_ep, ys.unsqueeze(0))                
            C_out = int(logits_probe.squeeze(0).size(-1))


        if C_out <= 0:
            raise RuntimeError("Could not infer logits width (C_out).")
    
        for param in model.model_.parameters():
            param.data = param.data.to(device)
    
        if peft_config:
            try:
                model.model_ = apply_tabular_lora("TabICL", model.model_, peft_config)
                logger.info("[TuningManager] Applied LoRA adapters to TabICL (pure SFT)")
            except Exception as e:
                logger.warning(f"[TuningManager] LoRA failed: {e}. Proceeding with base pure SFT fine-tuning")
    
        dataset = TensorDataset(
            torch.from_numpy(X_train_processed).float(),
            torch.from_numpy(y_train_processed).long()
        )
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
        optimizer = torch.optim.Adam(model.model_.parameters(),
                                lr=config["learning_rate"],
                                weight_decay=config["weight_decay"])
        loss_fn = torch.nn.CrossEntropyLoss()
    

        total_steps = len(dataloader) * config["epochs"]
        warmup_steps = len(dataloader) * config["warmup_epochs"]
    
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
        step = 0
        for epoch in range(1, config["epochs"] + 1):
            iterable = dataloader
            if config["show_progress"]:
                iterable = tqdm(dataloader, desc=f"Pure SFT Epoch {epoch}", leave=False)
        
            epoch_loss = 0
            for X_batch, y_batch in iterable:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                mid = X_batch.size(0) // 2
                X_support, y_support = X_batch[:mid], y_batch[:mid]
                X_query,   y_query   = X_batch[mid:], y_batch[mid:]
                X_episode = torch.cat([X_support, X_query], dim=0).unsqueeze(0)

                ys = y_support.squeeze(0).long()
                yq = y_query.squeeze(0).long()

                supp = torch.unique(ys)
                # keep at most C_out classes so the head can represent them
                keep = supp[:C_out]

                # build map only for kept classes; others -> -1 (excluded)
                yq_m = torch.full_like(yq, -1)
                ys_m = torch.full_like(ys, -1)
                for i, c in enumerate(keep):
                    ys_m[ys == c] = i
                    yq_m[yq == c] = i

                # prune support rows that were dropped
                keep_mask = (ys_m >= 0)
                if not keep_mask.any():
                    continue
                ys_m = ys_m[keep_mask]
                X_support_kept = X_episode[:, :ys.shape[0], :][:, keep_mask, :]
                X_query_part   = X_episode[:, ys.shape[0]:, :]
                X_episode = torch.cat([X_support_kept, X_query_part], dim=1)

                # if any query label was excluded, skip this episode (avoids OOB gathers)
                if (yq_m < 0).any():
                    continue

                # forward with episodic labels (contiguous, ≤ C_out)
                logits = model.model_(X_episode, ys_m.unsqueeze(0))  # [1, Q, <=C_out]
                logits = logits.squeeze(0)                           # [Q, <=C_out]
                 # ensure mapping fits the actual head width (in case adapters changed it mid-run)
                if logits.size(-1) < yq_m.max().item() + 1:
                    continue  # skip this episode if it exceeds head capacity
                loss = loss_fn(logits, yq_m)


                loss.backward()
                optimizer.step()
                scheduler.step()
            
                epoch_loss += loss.item()
                step += 1
            
                if config["show_progress"]:
                    iterable.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr():.2e}")
        
            logger.info(f"[TuningManager] Epoch {epoch}: Avg Loss = {epoch_loss/len(dataloader):.4f}, "
                   f"LR = {scheduler.get_last_lr():.2e}")
    
        logger.warning("[TuningManager] Pure SFT training complete (remember: not recommended for TabICL)")



    def _finetune_mitra(self, model, X_train_processed, y_train_processed, params=None, peft_config=None):
        """
        Performs episodic fine-tuning for in-context models like Mitra (Tab2D).
        """
        logger.info(f"[TuningManager] Starting episodic fine-tuning for {type(model).__name__}")
        
        config = {
            "device": resolve_device('auto'),
            "epochs": 3,
            "learning_rate": 1e-5,
            "batch_size": 4,
            "support_size": 128,
            "query_size": 128,
            "steps_per_epoch": 50,
            "show_progress": True
        }
        if params:
            config.update(params)
        logger.debug(f"[TuningManager] Using fine-tuning config: {config}")

        device = torch.device(config["device"])
        if peft_config:
            try:
                model = apply_tabular_lora("Mitra", model, peft_config)
                logger.info("[TuningManager] PEFT SUCCESS: Applied LoRA adapters to Mitra (Tab2D) model")
            except Exception as e:
                logger.warning(f"[TuningManager] PEFT FAILED: Mitra (Tab2D) incompatible with PEFT: {e}")
                logger.info("[TuningManager] FALLBACK: Proceeding with base fine-tuning (fully supported)")
                
        model.to(device)
        model.train()

        for param in model.parameters():
            param.data = param.data.to(device)
        
        optimizer = Adam(model.parameters(), lr=config["learning_rate"])
        loss_fn = torch.nn.CrossEntropyLoss()
        
        n_samples = X_train_processed.shape[0]
        episode_size = config['support_size'] + config['query_size']

        for epoch in range(1, config["epochs"] + 1):
            iterable = range(config['steps_per_epoch'])
            if config["show_progress"]:
                iterable = tqdm(iterable, desc=f"Finetuning Epoch {epoch}")

            for step in iterable:
                optimizer.zero_grad()
                
                X_episodes, y_episodes = [], []
                for _ in range(config['batch_size']):
                    if episode_size > n_samples:
                        logger.warning(f"[TuningManager] Warning: Episode size ({episode_size}) is larger than the dataset size ({n_samples}). Using all samples")
                        indices = np.arange(n_samples)
                        np.random.shuffle(indices)
                    else:
                        indices = np.random.choice(n_samples, episode_size, replace=False)

                    X_episodes.append(X_train_processed[indices])
                    y_episodes.append(y_train_processed[indices])
                
                X_batch = torch.from_numpy(np.stack(X_episodes)).to(device)
                y_batch = torch.from_numpy(np.stack(y_episodes)).long().to(device)
                
                s_size = config['support_size']
                X_support, X_query = X_batch[:, :s_size, :], X_batch[:, s_size:, :]
                y_support, y_query = y_batch[:, :s_size], y_batch[:, s_size:]
                
                b, f = X_support.shape[0], X_support.shape[2]
                padding_features = torch.zeros(b, f, dtype=torch.bool, device=device)
                padding_obs_support = torch.zeros_like(y_support, dtype=torch.bool, device=device)
                padding_obs_query = torch.zeros(b, X_query.shape[1], dtype=torch.bool, device=device)

                logits = model(
                    x_support=X_support, y_support=y_support, x_query=X_query,
                    padding_features=padding_features, padding_obs_support=padding_obs_support,
                    padding_obs_query__=padding_obs_query
                )
                
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_query.reshape(-1))
                loss.backward()
                optimizer.step()
                
                if config["show_progress"]:
                    iterable.set_postfix(loss=f"{loss.item():.4f}")
        
        logger.info("[TuningManager] Episodic fine-tuning complete")


    def _finetune_tabdpt(self, model: TabDPTClassifier, X_train_processed: np.ndarray, y_train_processed: np.ndarray, params: dict | None = None, processor=None, peft_config=None):
        """
        Performs episodic fine-tuning for the TabDPT model.
        """
        logger.info(f"[TuningManager] Starting episodic fine-tuning for {type(model).__name__}")
        
        # Determine number of classes from training data
        num_classes = len(np.unique(y_train_processed))
        logger.info(f"[TuningManager] Detected {num_classes} classes in training data")
        
        config = {
            "device": resolve_device('auto'),
            "epochs": 5,
            "learning_rate": 1e-5,
            "batch_size": 8, 
            "support_size": 512,
            "query_size": 256,
            "steps_per_epoch": 100,
            "show_progress": True
        }
        if params:
            config.update(params)
        logger.debug(f"[TuningManager] Using fine-tuning config: {config}")

        device = torch.device(config["device"])

        if peft_config:
            try:
                model.model = apply_tabular_lora("TabDPT", model.model, peft_config)
                logger.info("[TuningManager] PEFT SUCCESS: Applied LoRA to TabDPT model")
            except Exception as e:
                logger.warning(f"[TuningManager] PEFT not compatible with TabDPT: {e}. Proceeding with base fine-tuning")
                
        model.model.to(device)
        model.model.train()

        for param in model.model.parameters():
            param.data = param.data.to(device)
        for buffer in model.model.buffers():
            buffer.data = buffer.data.to(device)
        
        # Also ensure the model's device attribute is updated
        model.device = str(device)
        
        # TabDPT now handles projection internally, so only use model parameters
        trainable_params = list(model.model.parameters())

        optimizer = torch.optim.Adam(trainable_params, lr=config["learning_rate"])
        loss_fn = torch.nn.CrossEntropyLoss()
        
        n_samples = X_train_processed.shape[0]
        #episode_size = config['support_size'] + config['query_size']
        
        # Compute PCA basis on GPU once, no autograd
        if getattr(model, "feature_reduction", "pca") == "pca" and X_train_processed.shape[1] > model.max_features:
            with torch.no_grad():
                if not hasattr(model, "V"):
                    x_dev = torch.from_numpy(X_train_processed).to(device).float()
                    q = min(x_dev.shape[0], model.max_features)
                    _, _, V = torch.pca_lowrank(x_dev, q=q)
                    model.V = V
                    model.V.requires_grad_(False)
        

        for epoch in range(1, config["epochs"] + 1):
            iterable = range(config['steps_per_epoch'])
            if config["show_progress"]:
                iterable = tqdm(iterable, desc=f"Finetuning Epoch {epoch}")

            for step in iterable:
                optimizer.zero_grad()
                
                episode_size = config['support_size'] + config['query_size']
                if episode_size > n_samples:
                    scale = n_samples / float(episode_size)
                    s = max(1, int(config['support_size'] * scale))
                    q = max(1, int(config['query_size'] * scale))
                else:
                    s, q = config['support_size'], config['query_size']

                indices = np.random.choice(n_samples, s + q, replace=False)
                X_episode = torch.from_numpy(X_train_processed[indices]).float().to(device)
                y_episode = torch.from_numpy(y_train_processed[indices]).long().to(device)
                
                 # JIT PCA projection on GPU without affecting gradients
                if getattr(model, "feature_reduction", "pca") == "pca" and X_episode.shape[-1] > model.max_features and hasattr(model, "V"):
                    with torch.no_grad():
                        X_episode = X_episode @ model.V
                
                
                X_support = X_episode[:s].unsqueeze(0)
                y_support = y_episode[:s].unsqueeze(0)
                X_query   = X_episode[s:].unsqueeze(0)
                y_query   = y_episode[s:]

                # Apply padding to match model's expected feature count
                X_support = pad_x(X_support, model.max_features)
                X_query = pad_x(X_query, model.max_features)
                
                x_src = torch.cat([X_support, X_query], dim=1)
                                
                ys = y_support.squeeze(0).long()
                yq = y_query.long()

                supp = torch.unique(ys)
                max_id = int(max(int(ys.max()), int(yq.max())))
                emap = torch.full((max_id + 1,), -1, dtype=torch.long, device=ys.device)
                for i, c in enumerate(supp):
                    emap[int(c)] = i

                ys_m = emap[ys]
                yq_m = emap[yq]
                if (yq_m < 0).any():
                    continue

                logits = model.model(x_src=x_src, y_src=ys_m.unsqueeze(0).unsqueeze(-1).float(), task='cls')

                if logits.dim() == 3:
                    if logits.size(1) == 1:
                        logits = logits[:, 0, :]
                    elif logits.size(0) == 1:
                        logits = logits[0, :, :]
                    else:
                        Q = yq_m.size(0)
                        logits = logits[-Q:, 0, :]
                elif logits.dim() == 2:
                    pass
                elif logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected logits shape {tuple(logits.shape)}; expected 2D or 3D.")

                if int(yq_m.max().item()) >= logits.size(-1):
                    continue
                loss = loss_fn(logits, yq_m)

                loss.backward()
                optimizer.step()
                
                if config["show_progress"]:
                    iterable.set_postfix(loss=f"{loss.item():.4f}")

        model.model.eval()
        model.model.to(device)

        for param in model.model.parameters():
            param.data = param.data.to(device)
        for buffer in model.model.buffers():
            buffer.data = buffer.data.to(device)
        
        logger.info("[TuningManager] Episodic fine-tuning complete")



    def _finetune_mitra_pure_sft(self, model, X_train_processed, y_train_processed, params=None, peft_config=None):
        """
        PURE SFT FOR MITRA
    
        Unlike TabICL, pure SFT works naturally for Mitra because:
        1. Forward method is flexible with sequence dimensions
        2. Padding masks handle variable-length sequences
        3. Better for task-specific optimization
    
        This is suitable when you want to fully optimize for target task accuracy.
        """
        logger.info("[TuningManager] Starting Mitra Pure SFT Fine-tuning")

        config = {
            "device": resolve_device('auto'),
            "epochs": 5,
            "learning_rate": 1e-5,
            "batch_size": 128,
            "show_progress": True,
            "weight_decay": 1e-4,
            "warmup_epochs": 1
        }
        if params:
            config.update(params)
        logger.debug(f"[TuningManager] Using config: {config}")

        device = torch.device(config["device"])
        model.to(device)
        model.train()

        for param in model.parameters():
            param.data = param.data.to(device)

        if peft_config:
            try:
                model = apply_tabular_lora("Mitra", model, peft_config)
                logger.info("[TuningManager] Applied LoRA adapters to Mitra (pure SFT)")
            except Exception as e:
                logger.warning(f"[TuningManager] LoRA failed: {e}")

        dataset = TensorDataset(
            torch.from_numpy(X_train_processed).float(),
            torch.from_numpy(y_train_processed).long()
        )
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(),
                                lr=config["learning_rate"],
                                weight_decay=config["weight_decay"])
        loss_fn = torch.nn.CrossEntropyLoss()


        total_steps = len(dataloader) * config["epochs"]
        warmup_steps = len(dataloader) * config["warmup_epochs"]

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        for epoch in range(1, config["epochs"] + 1):
            iterable = dataloader
            if config["show_progress"]:
                iterable = tqdm(dataloader, desc=f"Pure SFT Epoch {epoch}", leave=False)

            epoch_loss = 0
            for X_batch, y_batch in iterable:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                X_support = X_batch.unsqueeze(1)
                y_support = y_batch.unsqueeze(1)
                X_query = X_batch.unsqueeze(1)
            
                b, f = X_support.shape[0], X_support.shape[2]
                padding_features = torch.zeros(b, f, dtype=torch.bool, device=device)
                padding_obs_support = torch.zeros_like(y_support, dtype=torch.bool, device=device)
                padding_obs_query = torch.zeros(b, X_query.shape[1], dtype=torch.bool, device=device)

                optimizer.zero_grad()
            
                logits = model(
                    x_support=X_support, y_support=y_support, x_query=X_query,
                    padding_features=padding_features,
                    padding_obs_support=padding_obs_support,
                    padding_obs_query__=padding_obs_query
                )

                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
    
                if config["show_progress"]:
                    iterable.set_postfix(loss=f"{loss.item():.4f}")

            logger.info(f"[TuningManager] Epoch {epoch}: Avg Loss = {epoch_loss/len(dataloader):.4f}")

        logger.info("[TuningManager] Pure SFT fine-tuning complete")


    def _finetune_tabdpt_pure_sft(self, model, X_train_processed, y_train_processed, params=None, processor=None, peft_config=None):
        """
        PURE SUPERVISED FINE-TUNING FOR TabDPT
    
        Standard batch-wise supervised training without episodic sampling.
        Works similarly to Mitra's pure SFT approach.
    
        Args:
            model: TabDPTClassifier instance
            X_train_processed: Preprocessed features (numpy array)
            y_train_processed: Target labels (numpy array)
            params: Fine-tuning hyperparameters
            processor: TabDPT processor with projector
            peft_config: PEFT configuration (optional)
        """
    
        logger.info("[TuningManager] Starting TabDPT Pure Supervised Fine-Tuning")

        classes, y_train_processed = np.unique(y_train_processed, return_inverse=True)
        y_train_processed = y_train_processed.astype(np.int64)
        num_classes = len(classes)
        logger.info(f"[TuningManager] Detected {num_classes} classes in training data (contiguous remap)")
        model.classes_ = classes


        config = {
            "device": resolve_device('auto'),
            "epochs": 5,
            "learning_rate": 2e-5,
            "batch_size": 32,
            "show_progress": True,
            "weight_decay": 1e-4,
            "warmup_epochs": 1
        }
        if params:
            config.update(params)
        logger.debug(f"[TuningManager] Using config: {config}")
    
        device = torch.device(config["device"])

        if peft_config:
            try:
                model.model = apply_tabular_lora("TabDPT", model.model, peft_config)
                logger.info("[TuningManager] Applied LoRA adapters to TabDPT (Pure SFT)")
            except Exception as e:
                logger.warning(f"[TuningManager] PEFT failed: {e}. Proceeding with base fine-tuning")
    
        model.model.to(device)
        model.model.train()
    
        for param in model.model.parameters():
            param.data = param.data.to(device)
        for buffer in model.model.buffers():
            buffer.data = buffer.data.to(device)
    
        model.device = str(device)
        if getattr(model, "feature_reduction", "pca") == "pca" and X_train_processed.shape[1] > model.max_features:
            with torch.no_grad():
                if not hasattr(model, "V"):
                    x_dev = torch.from_numpy(X_train_processed).to(device).float()
                    q = min(x_dev.shape[0], model.max_features)
                    _, _, V = torch.pca_lowrank(x_dev, q=q)
                    model.V = V
                    model.V.requires_grad_(False)
    

        trainable_params = list(model.model.parameters())
        if processor and hasattr(processor, 'custom_preprocessor_') and hasattr(processor.custom_preprocessor_, 'projector_'):
            trainable_params += list(processor.custom_preprocessor_.projector_.parameters())
            logger.info("[TuningManager] Including projector parameters in optimizer")
    
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        dataset = TensorDataset(
            torch.from_numpy(X_train_processed).float(),
            torch.from_numpy(y_train_processed).long()
        )
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        total_steps = len(dataloader) * config["epochs"]
        warmup_steps = len(dataloader) * config["warmup_epochs"]
    
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
        step = 0
        for epoch in range(1, config["epochs"] + 1):
            epoch_loss = 0.0
            iterable = dataloader
        
            if config["show_progress"]:
                iterable = tqdm(dataloader, desc=f"Pure SFT Epoch {epoch}", leave=False)
        
            for X_batch, y_batch in iterable:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                if getattr(model, "feature_reduction", "pca") == "pca" and X_batch.shape[-1] > model.max_features and hasattr(model, "V"):
                    with torch.no_grad():
                        X_batch = X_batch @ model.V
            
                X_support = X_batch.unsqueeze(1)
                y_support = y_batch.unsqueeze(1)
                X_query = X_batch.unsqueeze(1)
            
                X_support = pad_x(X_support, model.max_features)
                X_query = pad_x(X_query, model.max_features)
            
                x_src = torch.cat([X_support, X_query], dim=1)
            
                optimizer.zero_grad()
                
                logits = model.model(
                    x_src=x_src,
                    y_src=y_support.unsqueeze(-1).float(),
                    task='cls'
                )
                
                logits = logits[..., :num_classes]            
                if logits.dim() == 3:
                    logits = logits.squeeze(0)                
                elif logits.dim() != 2:
                    raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

                # CE requires targets in [0, C-1]; if head width < num_classes, drop OOR rows
                C_eff = logits.size(-1)
                y_batch = y_batch.long()

                valid = (y_batch >= 0) & (y_batch < C_eff)
                if not valid.all():
                    # skip this minibatch if nothing valid remains
                    if not valid.any():
                        continue
                    logits = logits[valid]
                    y_batch = y_batch[valid]

                loss = loss_fn(logits, y_batch)

            
                loss.backward()
                optimizer.step()
                scheduler.step()
            
                epoch_loss += loss.item()
                step += 1
            
                if config["show_progress"]:
                    iterable.set_postfix(
                        loss=f"{loss.item():.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}"
                    )
        
            avg_loss = epoch_loss / len(dataloader)
            logger.info(
                f"[TuningManager] Epoch [{epoch}/{config['epochs']}]: "
                f"Avg Loss = {avg_loss:.4f}, "
                f"LR = {scheduler.get_last_lr()[0]:.2e}"
            )
    
        model.model.eval()
        logger.info("[TuningManager] TabDPT Pure Supervised Fine-Tuning Complete")
    
        return model

    def _finetune_tabicl_simple_sft(self, model, X_train_processed, y_train_processed, params=None, peft_config=None):
        """
        TabICL : Convert supervised batches to episodic format
        """

        config = {
            'device': resolve_device('auto'),
            'epochs': 5,
            'learning_rate': 1e-5,
            'batch_size': 16,
            'show_progress': True,
        }
        if params:
            config.update(params)
    
        device = torch.device(config['device'])
    
    # Initialize
        model.fit(X_train_processed, y_train_processed)
        model._load_model()
        model.model_.to(device).train()
        
        C_out = None
        with torch.no_grad():
            # make one tiny 1-class episode from the first few rows
            X_np = X_train_processed if isinstance(X_train_processed, np.ndarray) else X_train_processed.to_numpy()
            y_np = y_train_processed if isinstance(y_train_processed, np.ndarray) else y_train_processed.to_numpy()

            # pick a class that has >= (support_size + query_size) examples; fall back to any class
            s_sz = int(config.get("support_size", 48))
            q_sz = int(config.get("query_size", 32))
            need = s_sz + q_sz

            cls, idx = None, None
            for c in np.unique(y_np):
                cand = np.nonzero(y_np == c)[0]
                if cand.size >= need:
                    idx = cand[:need]
                    cls = c
                    break
            if idx is None:
                idx = np.arange(min(need, len(y_np)))
                cls = y_np[idx[0]]

            X_ep = torch.from_numpy(X_np[idx]).float().unsqueeze(0).to(device)   # [1, S+Q, F]
            ys   = torch.full((s_sz,), 0, dtype=torch.long, device=device)       # all support -> class 0
            # pack as your forward expects: first S as support, rest as query
            logits_probe = model.model_(X_ep, ys.unsqueeze(0))                   # [1, Q, C_eff] typically
            C_out = int(logits_probe.squeeze(0).size(-1))

        # safety
        if C_out <= 0:
            raise RuntimeError("Could not infer logits width (C_out).")
            
            

    
    # Standard dataset
        dataset = TensorDataset(
            torch.from_numpy(X_train_processed).float(),
            torch.from_numpy(y_train_processed).long()
        )
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    
        optimizer = torch.optim.Adam(model.model_.parameters(), lr=config['learning_rate'])
        loss_fn = torch.nn.CrossEntropyLoss()
    
        for epoch in range(1, config['epochs'] + 1):
            iterable = tqdm(dataloader, desc=f"SFT Epoch {epoch}") if config['show_progress'] else dataloader
            epoch_loss = 0
        
            for X_batch, y_batch in iterable:
                batch_size = X_batch.shape[0]
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
            
            # Split batch in half: first half = support, second half = query
                mid = batch_size // 2
                if mid == 0:  # Skip if batch too small
                    continue
                X_support = X_batch[:mid]
                y_support = y_batch[:mid]
                X_query = X_batch[mid:]
                y_query = y_batch[mid:]
            
                # Ensure X_support and X_query are 2D [samples, features] before concatenation
                if X_support.dim() > 2:
                    X_support = X_support.view(mid, -1)  # Flatten extra dimensions
                if X_query.dim() > 2:
                    X_query = X_query.view(-1, X_query.shape[-1])  # Flatten extra dimensions
                
                X_episode = torch.cat([X_support, X_query], dim=0).unsqueeze(0)  # [1, batch_size, features]

                ys = y_support.squeeze(0).long() if y_support.dim() > 1 else y_support.long()
                yq = y_query.squeeze(0).long() if y_query.dim() > 1 else y_query.long()

                supp = torch.unique(ys)
                # keep at most C_out classes so the head can represent them
                keep = supp[:C_out]

                # build map only for kept classes; others -> -1 (excluded)
                yq_m = torch.full_like(yq, -1)
                ys_m = torch.full_like(ys, -1)
                for i, c in enumerate(keep):
                    ys_m[ys == c] = i
                    yq_m[yq == c] = i

                # prune support rows that were dropped
                keep_mask = (ys_m >= 0)
                if not keep_mask.any():
                    continue
                ys_m = ys_m[keep_mask]
                X_support_all = X_episode[:, :mid, :]  # [1, mid, F]
                X_support_kept = X_support_all[:, keep_mask, :]  # [1, kept_support, F]
                X_query_part = X_episode[:, mid:, :]  # [1, query_size, F]
                X_episode = torch.cat([X_support_kept, X_query_part], dim=1)

                # if any query label was excluded, skip this episode (avoids OOB gathers)
                if (yq_m < 0).any():
                    continue

                # forward with episodic labels (contiguous, ≤ C_out)
                logits = model.model_(X_episode, ys_m.unsqueeze(0))  # [1, Q, <=C_out]
                logits = logits.squeeze(0)        # [Q, <=C_out]
                # ensure mapping fits the actual head width (in case adapters changed it mid-run)
                if logits.size(-1) < yq_m.max().item() + 1:
                    continue  # skip this episode if it exceeds head capacity

                loss = loss_fn(logits, yq_m)


                loss.backward()
                optimizer.step()
            
                epoch_loss += loss.item()
                if config['show_progress']:
                    iterable.set_postfix(loss=f"{loss.item():.4f}")
        
            logger.info(f"[TuningManager] Epoch {epoch}: Loss = {epoch_loss/len(dataloader):.4f}")
    
        model.model_.eval()
        return model


    def _finetune_tabpfnv26_meta(self, model, X_train_processed, y_train_processed,
                               params=None):
        """
        Episodic meta-learning finetuning for TabPFNv26.
     
        Creates multiple (support, query) episodes from the dataset and trains
        the model to generalize across episode splits. This is the default
        finetuning mode for classification.
     
        Improvements over v2.5 meta-learning:
          - Cosine LR schedule with warmup
          - Mixed precision (AMP) when on CUDA
          - Gradient accumulation for effective larger batch sizes
          - Per-epoch dataset re-shuffling for better generalization
        """
        import torch
        import numpy as np
        import pandas as pd
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from functools import partial
     
        logger.info("[TuningManager] Starting TabPFNv26 meta-learning fine-tuning")
     
        config = {
            "device": resolve_device('auto'),
            "epochs": 5,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "batch_size": 256,
            "grad_clip": 1.0,
            "warmup_ratio": 0.1,
            "grad_accum_steps": 1,
            "show_progress": True,
        }
        if params:
            config.update(params)
     
        device = torch.device(config["device"])
     
        # Initialize model internals if needed
        if not hasattr(model, 'model_') or model.model_ is None:
            model._initialize_model_variables()
     
        model.model_.to(device)
        model.model_.train()
     
        # Data prep
        if isinstance(X_train_processed, pd.DataFrame):
            X_np = X_train_processed.to_numpy()
        else:
            X_np = np.asarray(X_train_processed)
     
        if isinstance(y_train_processed, (pd.Series, pd.DataFrame)):
            y_np = y_train_processed.to_numpy().ravel()
        else:
            y_np = np.asarray(y_train_processed).ravel()
     
        if y_np.dtype == object or not np.issubdtype(y_np.dtype, np.number):
            le = LabelEncoder()
            y_np = le.fit_transform(y_np)
            if not hasattr(model, 'label_encoder_'):
                model.label_encoder_ = le
     
        optimizer = AdamW(model.model_.parameters(),
                          lr=config["learning_rate"],
                          weight_decay=config["weight_decay"])
     
        loss_fn = torch.nn.CrossEntropyLoss()
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler() if use_amp else None
     
        # Collator
        try:
            from ..models.tabpfnv26.finetuning.data_util import (
                meta_dataset_collator, get_preprocessed_dataset_chunks
            )
            use_v26_data_util = True
        except ImportError:
            use_v26_data_util = False
            try:
                from ..models.tabpfn.utils import meta_dataset_collator
            except ImportError:
                def meta_dataset_collator(batch):
                    return batch[0]
     
        def _move(item, dev):
            if isinstance(item, torch.Tensor):
                return item.to(dev)
            if isinstance(item, list):
                return [_move(x, dev) for x in item]
            if isinstance(item, tuple):
                return tuple(_move(x, dev) for x in item)
            if isinstance(item, dict):
                return {k: _move(v, dev) for k, v in item.items()}
            return item
     
        def _stratified_split(X, y):
            y_s = pd.Series(y)
            if y_s.nunique() > 1 and y_s.value_counts().min() > 1:
                return train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
            return train_test_split(X, y, test_size=0.3, random_state=42)
     
        # Cosine schedule with warmup
        total_steps = None  # computed after first epoch
     
        for epoch in range(1, config["epochs"] + 1):
            # Re-shuffle each epoch for diversity
            seed = 42 + epoch
            splitter = partial(train_test_split, test_size=0.3,
                               random_state=seed)
     
            if use_v26_data_util:
                training_datasets = get_preprocessed_dataset_chunks(
                    calling_instance=model,
                    X_raw=X_np, y_raw=y_np,
                    split_fn=splitter,
                    max_data_size=config["batch_size"],
                    model_type="classifier",
                    equal_split_size=False,
                    data_shuffle_seed=seed,
                    preprocessing_random_state=seed,
                )
            else:
                training_datasets = model.get_preprocessed_datasets(
                    X_np, y_np, splitter, config["batch_size"]
                )
     
            dataloader = DataLoader(
                training_datasets, batch_size=1,
                collate_fn=meta_dataset_collator, shuffle=True,
            )
     
            if total_steps is None:
                total_steps = len(dataloader) * config["epochs"]
                warmup_steps = int(total_steps * config["warmup_ratio"])
     
                def lr_lambda(step):
                    if step < warmup_steps:
                        return float(step) / max(1, warmup_steps)
                    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
                    return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))
     
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            else:
                scheduler = getattr(self, '_v26_scheduler', None)
     
            iterable = tqdm(dataloader, desc=f"TabPFNv26 Meta Epoch {epoch}",
                            disable=not config["show_progress"])
            epoch_losses = []
     
            for step_i, batch in enumerate(iterable):
                # v2.6 uses dataclass batches; v2.5 uses tuples
                if hasattr(batch, 'X_context'):
                    # v2.6 ClassifierBatch dataclass
                    X_ctx = _move(batch.X_context, device)
                    y_ctx = _move(batch.y_context, device)
                    X_qry = _move(batch.X_query, device)
                    y_qry = _move(batch.y_query, device)
                    cat_ixs = batch.cat_indices
                    confs = batch.configs
     
                    # Skip if query labels not subset of context labels
                    ctx_uniq = torch.unique(torch.cat([torch.unique(t.reshape(-1)) for t in y_ctx]))
                    qry_uniq = torch.unique(y_qry.reshape(-1))
                    if not torch.isin(qry_uniq, ctx_uniq).all():
                        continue
     
                    model.fit_from_preprocessed(X_ctx, y_ctx, cat_ixs, confs)
     
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        logits = model.forward(X_qry, return_raw_logits=True)
                        if logits.dim() == 4:
                            Q, B, E, L = logits.shape
                            logits_BLQ = logits.permute(1, 2, 3, 0).reshape(B * E, L, Q)
                            targets_BQ = y_qry.repeat(B * E, 1).to(device)
                            loss = torch.nn.functional.cross_entropy(logits_BLQ, targets_BQ)
                        else:
                            loss = loss_fn(logits, y_qry.to(device))
     
                else:
                    # v2.5-style tuple: (X_train, X_test, y_train, y_test, cat_ixs, confs)
                    X_s, X_q, y_s, y_q, cat_ixs, confs = batch
                    if len(np.unique(y_s)) != len(np.unique(y_q)):
                        continue
                    X_s = _move(X_s, device)
                    y_s = _move(y_s, device)
                    X_q = _move(X_q, device)
                    y_q = _move(y_q, device)
     
                    model.fit_from_preprocessed(X_s, y_s, cat_ixs, confs)
     
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        preds = model.forward(X_q, return_logits=True)
                        if isinstance(preds, torch.Tensor) and preds.device != device:
                            preds = preds.to(device)
                        target = y_q[0] if isinstance(y_q, list) else y_q
                        loss = loss_fn(preds, target)
     
                # Gradient accumulation
                loss = loss / config["grad_accum_steps"]
     
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    if (step_i + 1) % config["grad_accum_steps"] == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.model_.parameters(),
                                                        config["grad_clip"])
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (step_i + 1) % config["grad_accum_steps"] == 0:
                        torch.nn.utils.clip_grad_norm_(model.model_.parameters(),
                                                        config["grad_clip"])
                        optimizer.step()
                        optimizer.zero_grad()
     
                if scheduler is not None:
                    scheduler.step()
     
                epoch_losses.append(loss.item() * config["grad_accum_steps"])
                iterable.set_postfix(loss=f"{epoch_losses[-1]:.4f}",
                                      lr=f"{optimizer.param_groups[0]['lr']:.2e}")
     
            avg = np.mean(epoch_losses) if epoch_losses else float('nan')
            logger.info(f"[TuningManager] TabPFNv26 Meta Epoch [{epoch}/{config['epochs']}]: "
                        f"Loss={avg:.4f}")
     
        model.model_.eval()
        if hasattr(model, 'batched'):
            model.batched = False
        logger.info("[TuningManager] TabPFNv26 meta-learning fine-tuning complete")
        return model
 
 
 
    def _finetune_tabpfnv26_sft(self, model, X_train_processed, y_train_processed,
                                  params=None):
        """
        SFT-style finetuning for TabPFNv26.
     
        Uses the ENTIRE dataset as ONE single large (support, query) episode and
        trains over it for multiple epochs. Forces the model to specialize on
        the single task.
     
        Improvements over v2.5 SFT:
          - Cosine LR schedule with warmup
          - AMP for speed
          - Label smoothing option
          - Gradient clipping
        """
        import torch
        import numpy as np
        import pandas as pd
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
     
        logger.info("[TuningManager] Starting TabPFNv26 SFT fine-tuning")
     
        config = {
            "device": resolve_device('auto'),
            "epochs": 25,
            "learning_rate": 1e-5,
            "weight_decay": 1e-4,
            "query_set_ratio": 0.3,
            "grad_clip": 1.0,
            "label_smoothing": 0.0,
            "warmup_ratio": 0.1,
            "show_progress": True,
        }
        if params:
            config.update(params)
     
        device = torch.device(config["device"])
     
        if not hasattr(model, 'model_') or model.model_ is None:
            model._initialize_model_variables()
     
        model.model_.to(device)
        model.model_.train()
     
        # Data prep
        if isinstance(X_train_processed, pd.DataFrame):
            X_np = X_train_processed.to_numpy()
        else:
            X_np = np.asarray(X_train_processed)
     
        if isinstance(y_train_processed, (pd.Series, pd.DataFrame)):
            y_np = y_train_processed.to_numpy().ravel()
        else:
            y_np = np.asarray(y_train_processed).ravel()
     
        if y_np.dtype == object or not np.issubdtype(y_np.dtype, np.number):
            le = LabelEncoder()
            y_np = le.fit_transform(y_np)
            if not hasattr(model, 'label_encoder_'):
                model.label_encoder_ = le
     
        optimizer = AdamW(model.model_.parameters(),
                          lr=config["learning_rate"],
                          weight_decay=config["weight_decay"])
     
        loss_fn = torch.nn.CrossEntropyLoss(
            label_smoothing=config["label_smoothing"]
        )
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler() if use_amp else None
     
        try:
            from ..models.tabpfn.utils import meta_dataset_collator
        except ImportError:
            def meta_dataset_collator(batch):
                return batch[0]
     
        def _move(item, dev):
            if isinstance(item, torch.Tensor):
                return item.to(dev)
            if isinstance(item, list):
                return [_move(x, dev) for x in item]
            if isinstance(item, tuple):
                return tuple(_move(x, dev) for x in item)
            if isinstance(item, dict):
                return {k: _move(v, dev) for k, v in item.items()}
            return item
     
        def sft_splitter(X, y, **kwargs):
            test_size = kwargs.get('test_size', config["query_set_ratio"])
            random_state = kwargs.get('random_state', 42)
            stratify = kwargs.get('stratify', None)
            if stratify is None:
                y_s = pd.Series(y)
                if y_s.nunique() > 1 and y_s.value_counts().min() > 1:
                    stratify = y
            try:
                return train_test_split(X, y, test_size=test_size, stratify=stratify, random_state=random_state)
            except ValueError:
                return train_test_split(X, y, test_size=test_size, random_state=random_state)
     
        # Create ONE large episode from the entire dataset
        try:
            from ..models.tabpfnv26.finetuning.data_util import (
                get_preprocessed_dataset_chunks, meta_dataset_collator as v26_collator
            )
            if not hasattr(model, 'model_') or model.model_ is None:
                model._initialize_model_variables()
            training_datasets = get_preprocessed_dataset_chunks(
                calling_instance=model,
                X_raw=X_np, y_raw=y_np,
                split_fn=sft_splitter,
                max_data_size=len(X_np),
                model_type="classifier",
                equal_split_size=False,
                data_shuffle_seed=42,
                preprocessing_random_state=42,
            )
            meta_collator = v26_collator
        except ImportError:
            # Fallback for older API
            training_datasets = model.get_preprocessed_datasets(
                X_np, y_np, sft_splitter, len(X_np)
            )
            try:
                from ..models.tabpfn.utils import meta_dataset_collator as meta_collator
            except ImportError:
                def meta_collator(batch): return batch[0]
    
        episode_loader = DataLoader(
            training_datasets, batch_size=1,
            collate_fn=meta_collator, shuffle=False,
        )
     
        # Cosine schedule
        total_steps = config["epochs"]
        warmup_steps = max(1, int(total_steps * config["warmup_ratio"]))
     
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))
     
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
     
        for epoch in range(1, config["epochs"] + 1):
            iterable = tqdm(episode_loader,
                            desc=f"TabPFNv26 SFT Epoch {epoch}",
                            leave=False, disable=not config["show_progress"])
            epoch_losses = []
     
            for batch in iterable:
                if hasattr(batch, 'X_context'):
                    X_s = _move(batch.X_context, device)
                    y_s = _move(batch.y_context, device)
                    X_q = _move(batch.X_query, device)
                    y_q = _move(batch.y_query, device)
                    cat_ixs = batch.cat_indices
                    confs = batch.configs
                else:
                    X_s, X_q, y_s, y_q, cat_ixs, confs = batch
                    X_s = _move(X_s, device)
                    y_s = _move(y_s, device)
                    X_q = _move(X_q, device)
                    y_q = _move(y_q, device)
     
                optimizer.zero_grad()
                model.fit_from_preprocessed(X_s, y_s, cat_ixs, confs)
     
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    preds = model.forward(X_q, return_logits=True)
                    if isinstance(preds, torch.Tensor) and preds.device != device:
                        preds = preds.to(device)
                    target = y_q[0] if isinstance(y_q, list) else y_q
                    loss = loss_fn(preds, target)
     
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.model_.parameters(),
                                                    config["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.model_.parameters(),
                                                    config["grad_clip"])
                    optimizer.step()
     
                scheduler.step()
                epoch_losses.append(loss.item())
                iterable.set_postfix(loss=f"{loss.item():.4f}")
     
            avg = np.mean(epoch_losses) if epoch_losses else float('nan')
            logger.info(f"[TuningManager] TabPFNv26 SFT [{epoch}/{config['epochs']}]: "
                        f"Loss={avg:.4f}")
     
        model.model_.eval()
        if hasattr(model, 'batched'):
            model.batched = False
        logger.info("[TuningManager] TabPFNv26 SFT fine-tuning complete")
        return model
 
 
 
    def _finetune_tabpfnv26_native_classifier(self, model, X_train, y_train,
                                                params=None):
        """
        Uses PriorLabs' FinetunedTabPFNClassifier for classification finetuning.
     
        This is the most advanced finetuning mode — it uses:
          - Cosine LR with warmup
          - Mixed precision (AMP)
          - Early stopping with patience
          - Validation-based model selection
          - Gradient clipping
          - Checkpoint saving
          - Multi-GPU DDP support (if available)
     
        The finetuned model replaces model.model_ weights in place.
        """
        import torch
        import numpy as np
        import pandas as pd
     
        logger.info("[TuningManager] Starting TabPFNv26 native FinetunedTabPFNClassifier")
     
        config = {
            "device": resolve_device('auto'),
            "epochs": 30,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "early_stopping": True,
            "early_stopping_patience": 8,
            "validation_split_ratio": 0.1,
            "n_finetune_ctx_plus_query_samples": 10_000,
            "finetune_ctx_query_split_ratio": 0.2,
            "n_estimators_finetune": 2,
            "n_estimators_validation": 2,
            "n_estimators_final_inference": 8,
            "grad_clip_value": 1.0,
            "use_lr_scheduler": True,
            "use_activation_checkpointing": True,
            "random_state": 0,
            "show_progress": True,
        }
        if params:
            config.update(params)
     
        try:
            from ..models.tabpfnv26.finetuning import FinetunedTabPFNClassifier
        except ImportError:
            logger.error("[TuningManager] FinetunedTabPFNClassifier not available. "
                         "Falling back to meta-learning.")
            return self._finetune_tabpfnv26_meta(model, X_train, y_train, params)
     
        # Convert data
        if isinstance(X_train, pd.DataFrame):
            X_np = X_train.to_numpy()
        else:
            X_np = np.asarray(X_train)
     
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_np = y_train.to_numpy().ravel()
        else:
            y_np = np.asarray(y_train).ravel()
     
        finetuner = FinetunedTabPFNClassifier(
            device=config["device"],
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            early_stopping=config["early_stopping"],
            early_stopping_patience=config["early_stopping_patience"],
            validation_split_ratio=config["validation_split_ratio"],
            n_finetune_ctx_plus_query_samples=config["n_finetune_ctx_plus_query_samples"],
            finetune_ctx_query_split_ratio=config["finetune_ctx_query_split_ratio"],
            n_estimators_finetune=config["n_estimators_finetune"],
            n_estimators_validation=config["n_estimators_validation"],
            n_estimators_final_inference=config["n_estimators_final_inference"],
            grad_clip_value=config["grad_clip_value"],
            use_lr_scheduler=config["use_lr_scheduler"],
            use_activation_checkpointing=config["use_activation_checkpointing"],
            random_state=config["random_state"],
        )
     
        finetuner.fit(X_np, y_np)

        # if hasattr(finetuner, 'finetuned_estimator_') and finetuner.finetuned_estimator_ is not None:
        #     ft_est = finetuner.finetuned_estimator_
        #     # model_ is a property in v2.6 — use load_state_dict instead of direct assignment
        #     try:
        #         if not hasattr(model, 'model_') or model.model_ is None:
        #             model._initialize_model_variables()
        #         model.model_.load_state_dict(ft_est.model_.state_dict())
        #         logger.info("[TuningManager] Transferred finetuned weights via load_state_dict")
        #     except Exception as e:
        #         logger.warning(f"[TuningManager] load_state_dict failed: {e}, using finetuned estimator directly")
        #         # Fallback: replace model entirely with the finetuned estimator
        #         model = ft_est
        #     # Copy fitted attributes
        #     for attr in ('classes_', 'n_classes_', 'label_encoder_', 'is_fitted_',
        #                  'softmax_temperature_', 'executor_'):
        #         if hasattr(ft_est, attr):
        #             try:
        #                 setattr(model, attr, getattr(ft_est, attr))
        #             except (AttributeError, TypeError):
        #                 pass
    
        # # Store finetuner reference
        # model._finetuner_ = finetuner
    
        # if hasattr(model, 'model_') and model.model_ is not None:
        #     model.model_.eval()
        # logger.info("[TuningManager] TabPFNv26 native classification finetuning complete")
        # return model

        if hasattr(finetuner, 'finetuned_estimator_') and finetuner.finetuned_estimator_ is not None:
            ft_est = finetuner.finetuned_estimator_
            ft_est._finetuner_ = finetuner
    
            # Re-fit in standard mode so predict/predict_proba works
            try:
                ft_est.fit(X_np, y_np)
                logger.info("[TuningManager] Re-fitted finetuned classifier for standard inference")
            except Exception as e:
                logger.warning(f"[TuningManager] Post-finetune re-fit failed: {e}")
    
            if hasattr(ft_est, 'model_') and ft_est.model_ is not None:
                ft_est.model_.eval()
            logger.info("[TuningManager] TabPFNv26 native classification finetuning complete")
            return ft_est
    
        logger.warning("[TuningManager] No finetuned estimator found, returning original")
        return model
 
 
    def _finetune_tabpfnv26_native_regressor(self, model, X_train, y_train,
                                               params=None):
        """
        Uses PriorLabs' FinetunedTabPFNRegressor for regression finetuning.
     
        Uses bar distribution loss (better than raw MSE for TabPFN) plus
        optional CRPS/CRLS/MSE/MAE auxiliary losses.
        """
        import torch
        import numpy as np
        import pandas as pd
     
        logger.info("[TuningManager] Starting TabPFNv26 native FinetunedTabPFNRegressor")
     
        config = {
            "device": resolve_device('auto'),
            "epochs": 30,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "early_stopping": True,
            "early_stopping_patience": 8,
            "validation_split_ratio": 0.1,
            "n_finetune_ctx_plus_query_samples": 10_000,
            "finetune_ctx_query_split_ratio": 0.2,
            "n_estimators_finetune": 2,
            "n_estimators_validation": 2,
            "n_estimators_final_inference": 8,
            "grad_clip_value": 1.0,
            "use_lr_scheduler": True,
            "use_activation_checkpointing": True,
            "random_state": 0,
        }
        if params:
            config.update(params)
     
        try:
            from ..models.tabpfnv26.finetuning import FinetunedTabPFNRegressor
        except ImportError:
            logger.error("[TuningManager] FinetunedTabPFNRegressor not available. "
                         "Falling back to turn-by-turn.")
            return self._finetune_tabpfnv26_regression_turn_by_turn(model, X_train, y_train, params)
     
        if isinstance(X_train, pd.DataFrame):
            X_np = X_train.to_numpy()
        else:
            X_np = np.asarray(X_train)
     
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_np = y_train.to_numpy().ravel()
        else:
            y_np = np.asarray(y_train).ravel()
     
        finetuner = FinetunedTabPFNRegressor(
            device=config["device"],
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            early_stopping=config["early_stopping"],
            early_stopping_patience=config["early_stopping_patience"],
            validation_split_ratio=config["validation_split_ratio"],
            n_finetune_ctx_plus_query_samples=config["n_finetune_ctx_plus_query_samples"],
            finetune_ctx_query_split_ratio=config["finetune_ctx_query_split_ratio"],
            n_estimators_finetune=config["n_estimators_finetune"],
            n_estimators_validation=config["n_estimators_validation"],
            n_estimators_final_inference=config["n_estimators_final_inference"],
            grad_clip_value=config["grad_clip_value"],
            use_lr_scheduler=config["use_lr_scheduler"],
            use_activation_checkpointing=config["use_activation_checkpointing"],
            random_state=config["random_state"],
        )
     
        finetuner.fit(X_np, y_np)
     
        # if hasattr(finetuner, 'finetuned_estimator_') and finetuner.finetuned_estimator_ is not None:
        #     ft_est = finetuner.finetuned_estimator_
        #     try:
        #         if not hasattr(model, 'model_') or model.model_ is None:
        #             model._initialize_model_variables()
        #         model.model_.load_state_dict(ft_est.model_.state_dict())
        #         logger.info("[TuningManager] Transferred finetuned regressor weights via load_state_dict")
        #     except Exception as e:
        #         logger.warning(f"[TuningManager] load_state_dict failed: {e}, using finetuned estimator directly")
        #         model = ft_est
        #     for attr in ('is_fitted_', 'executor_', 'znorm_space_bardist_', 'raw_space_bardist_'):
        #         if hasattr(ft_est, attr):
        #             try:
        #                 setattr(model, attr, getattr(ft_est, attr))
        #             except (AttributeError, TypeError):
        #                 pass

        if hasattr(finetuner, 'finetuned_estimator_') and finetuner.finetuned_estimator_ is not None:
            ft_est = finetuner.finetuned_estimator_
            ft_est._finetuner_ = finetuner

            # Re-fit in standard mode so predict() works
            # (finetuning uses 'batched' mode which doesn't support standard predict)
            try:
                ft_est.fit(X_np, y_np)
                logger.info("[TuningManager] Re-fitted finetuned regressor for standard inference")
            except Exception as e:
                logger.warning(f"[TuningManager] Post-finetune re-fit failed: {e}")

            if hasattr(ft_est, 'model_') and ft_est.model_ is not None:
                ft_est.model_.eval()
            logger.info("[TuningManager] TabPFNv26 native regression finetuning complete")
            return ft_est

        logger.warning("[TuningManager] No finetuned estimator, returning original")
        return model
 



    # ==================================================================
    # TabPFN v3 fine-tuning
    # ==================================================================
    # Five entry points mirror the v2.6 surface, ported to the v3 API:
    #   _finetune_tabpfnv3_native_classifier   (mode='native')
    #   _finetune_tabpfnv3_meta                 (mode='meta-learning', default)  [+PEFT]
    #   _finetune_tabpfnv3_sft                  (mode='sft')                     [+PEFT]
    #   _finetune_tabpfnv3_native_regressor     (reg, mode='native', default)
    #   _finetune_tabpfnv3_regression_turn_by_turn (reg, mode='turn_by_turn')
    #
    # Key v3-vs-v2.6 API differences handled here:
    #   * underlying model is `models_` (list); `model_` is models_[0] (read-only property)
    #   * fit_from_preprocessed(...) requires keyword `performance_options=`
    #   * get_preprocessed_dataset_chunks(...) requires the extra keyword args
    #     equal_split_size / data_shuffle_seed / preprocessing_random_state
    #   * native FinetunedTabPFN hardcodes ModelVersion.V2_5 -> we use the
    #     V3-pinned subclasses from models.tabpfnv3.finetuning._tabtune_v3_pin
    # ------------------------------------------------------------------

    def _v3_performance_options(self, use_activation_checkpointing: bool = False):
        """Return a v3 PerformanceOptions for fine-tuning forward/fit calls.

        All fields are optional; activation checkpointing trades compute for
        memory and is useful for large context sizes during FT.
        """
        from ..models.tabpfnv3.architectures.interface import PerformanceOptions
        return PerformanceOptions(force_recompute_layer=bool(use_activation_checkpointing))

    def _v3_trainable_module(self, model):
        """Return the underlying trainable nn.Module for a v3 estimator.

        v3 exposes `models_` (list); `model_` is a convenience property == models_[0].
        Initializes model variables if needed.
        """
        if not hasattr(model, 'models_') or model.models_ is None:
            model._initialize_model_variables()
        # model_ is a property returning models_[0]; fall back to models_[0] directly.
        try:
            return model.model_
        except Exception:
            return model.models_[0]

    def _maybe_apply_v3_lora(self, model, torch_module, peft_config):
        """Inject LoRA adapters into the v3 backbone if peft_config is provided.

        Uses TabTune's custom LoRA injector with v3-specific target substrings
        (see peft_utils.MODEL_LORA_TARGETS['TabPFNv3']). Returns the (possibly
        wrapped) module and the list of parameters to optimize.
        """
        import torch
        if peft_config is None:
            return torch_module, [p for p in torch_module.parameters() if p.requires_grad]

        try:
            from .peft_utils import apply_tabular_lora
        except ImportError:
            logger.warning("[TuningManager] peft_utils unavailable; skipping LoRA injection.")
            return torch_module, [p for p in torch_module.parameters() if p.requires_grad]

        logger.info("[TuningManager] Injecting LoRA adapters into TabPFNv3 backbone")
        # Freeze all base params first; LoRALinear sets base.requires_grad_=False itself,
        # but we also freeze any non-wrapped params so only adapters train.
        for p in torch_module.parameters():
            p.requires_grad = False
        apply_tabular_lora("TabPFNv3", torch_module, peft_config=peft_config)
        trainable = [p for p in torch_module.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in torch_module.parameters())
        logger.info(f"[TuningManager] LoRA trainable params: {n_train:,} / {n_total:,} "
                    f"({100.0 * n_train / max(1, n_total):.2f}%)")
        return torch_module, trainable

    def _finetune_tabpfnv3_native_classifier(self, model, X_train, y_train, params=None):
        """Native classification FT via the V3-pinned upstream FinetunedTabPFNClassifier.

        Pins ModelVersion.V3 (upstream hardcodes V2_5). Returns a fitted estimator
        ready for predict/predict_proba.
        """
        import torch
        import numpy as np
        import pandas as pd

        logger.info("[TuningManager] Starting TabPFNv3 native FinetunedTabPFNClassifier")

        config = {
            "device": resolve_device('auto'),
            "epochs": 30,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "early_stopping": True,
            "early_stopping_patience": 8,
            "validation_split_ratio": 0.1,
            "n_finetune_ctx_plus_query_samples": 10_000,
            "finetune_ctx_query_split_ratio": 0.2,
            "n_estimators_finetune": 2,
            "n_estimators_validation": 2,
            "n_estimators_final_inference": 8,
            "grad_clip_value": 1.0,
            "use_lr_scheduler": True,
            "use_activation_checkpointing": True,
            "random_state": 0,
        }
        if params:
            config.update(params)

        try:
            from ..models.tabpfnv3.finetuning._tabtune_v3_pin import V3PinnedFinetunedClassifier
        except ImportError:
            logger.error("[TuningManager] V3PinnedFinetunedClassifier unavailable. "
                         "Falling back to meta-learning.")
            return self._finetune_tabpfnv3_meta(model, X_train, y_train, params)

        X_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_np = y_train.to_numpy().ravel()
        else:
            y_np = np.asarray(y_train).ravel()

        finetuner = V3PinnedFinetunedClassifier(
            device=config["device"],
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            early_stopping=config["early_stopping"],
            early_stopping_patience=config["early_stopping_patience"],
            validation_split_ratio=config["validation_split_ratio"],
            n_finetune_ctx_plus_query_samples=config["n_finetune_ctx_plus_query_samples"],
            finetune_ctx_query_split_ratio=config["finetune_ctx_query_split_ratio"],
            n_estimators_finetune=config["n_estimators_finetune"],
            n_estimators_validation=config["n_estimators_validation"],
            n_estimators_final_inference=config["n_estimators_final_inference"],
            grad_clip_value=config["grad_clip_value"],
            use_lr_scheduler=config["use_lr_scheduler"],
            use_activation_checkpointing=config["use_activation_checkpointing"],
            random_state=config["random_state"],
        )

        finetuner.fit(X_np, y_np)

        if hasattr(finetuner, 'finetuned_estimator_') and finetuner.finetuned_estimator_ is not None:
            ft_est = finetuner.finetuned_estimator_
            ft_est._finetuner_ = finetuner
            # Re-fit in standard mode so predict/predict_proba works
            # (FT uses 'batched' mode which doesn't support standard predict).
            try:
                ft_est.fit(X_np, y_np)
                logger.info("[TuningManager] Re-fitted finetuned v3 classifier for standard inference")
            except Exception as e:
                logger.warning(f"[TuningManager] Post-finetune re-fit failed: {e}")
            if hasattr(ft_est, 'model_') and ft_est.model_ is not None:
                try:
                    ft_est.model_.eval()
                except Exception:
                    pass
            logger.info("[TuningManager] TabPFNv3 native classification finetuning complete")
            return ft_est

        logger.warning("[TuningManager] No finetuned estimator found, returning original")
        return model

    def _finetune_tabpfnv3_native_regressor(self, model, X_train, y_train, params=None):
        """Native regression FT via the V3-pinned upstream FinetunedTabPFNRegressor.

        Uses bar-distribution + CRPS/MSE loss weights (upstream defaults).
        """
        import torch
        import numpy as np
        import pandas as pd

        logger.info("[TuningManager] Starting TabPFNv3 native FinetunedTabPFNRegressor")

        config = {
            "device": resolve_device('auto'),
            "epochs": 30,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "early_stopping": True,
            "early_stopping_patience": 8,
            "validation_split_ratio": 0.1,
            "n_finetune_ctx_plus_query_samples": 10_000,
            "finetune_ctx_query_split_ratio": 0.2,
            "n_estimators_finetune": 2,
            "n_estimators_validation": 2,
            "n_estimators_final_inference": 8,
            "grad_clip_value": 1.0,
            "use_lr_scheduler": True,
            "use_activation_checkpointing": True,
            "random_state": 0,
        }
        if params:
            config.update(params)

        try:
            from ..models.tabpfnv3.finetuning._tabtune_v3_pin import V3PinnedFinetunedRegressor
        except ImportError:
            logger.error("[TuningManager] V3PinnedFinetunedRegressor unavailable. "
                         "Falling back to turn-by-turn.")
            return self._finetune_tabpfnv3_regression_turn_by_turn(model, X_train, y_train, params)

        X_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_np = y_train.to_numpy().ravel()
        else:
            y_np = np.asarray(y_train).ravel()

        finetuner = V3PinnedFinetunedRegressor(
            device=config["device"],
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            early_stopping=config["early_stopping"],
            early_stopping_patience=config["early_stopping_patience"],
            validation_split_ratio=config["validation_split_ratio"],
            n_finetune_ctx_plus_query_samples=config["n_finetune_ctx_plus_query_samples"],
            finetune_ctx_query_split_ratio=config["finetune_ctx_query_split_ratio"],
            n_estimators_finetune=config["n_estimators_finetune"],
            n_estimators_validation=config["n_estimators_validation"],
            n_estimators_final_inference=config["n_estimators_final_inference"],
            grad_clip_value=config["grad_clip_value"],
            use_lr_scheduler=config["use_lr_scheduler"],
            use_activation_checkpointing=config["use_activation_checkpointing"],
            random_state=config["random_state"],
        )

        finetuner.fit(X_np, y_np)

        if hasattr(finetuner, 'finetuned_estimator_') and finetuner.finetuned_estimator_ is not None:
            ft_est = finetuner.finetuned_estimator_
            ft_est._finetuner_ = finetuner
            try:
                ft_est.fit(X_np, y_np)
                logger.info("[TuningManager] Re-fitted finetuned v3 regressor for standard inference")
            except Exception as e:
                logger.warning(f"[TuningManager] Post-finetune re-fit failed: {e}")
            if hasattr(ft_est, 'model_') and ft_est.model_ is not None:
                try:
                    ft_est.model_.eval()
                except Exception:
                    pass
            logger.info("[TuningManager] TabPFNv3 native regression finetuning complete")
            return ft_est

        logger.warning("[TuningManager] No finetuned v3 regressor estimator, returning original")
        return model

    def _finetune_tabpfnv3_meta(self, model, X_train_processed, y_train_processed,
                                params=None, peft_config=None):
        """Episodic meta-learning FT for TabPFN v3 (default classification mode).

        Builds (support, query) episodes per epoch and trains the backbone to
        generalize across splits. Cosine LR + warmup, AMP on CUDA, grad accumulation.
        Supports LoRA/PEFT via `peft_config` (adapters injected into the v3 backbone).
        Ported to the v3 API (models_, performance_options=, new chunk kwargs).
        """
        import torch
        import numpy as np
        import pandas as pd
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from functools import partial

        logger.info("[TuningManager] Starting TabPFNv3 meta-learning fine-tuning")

        config = {
            "device": resolve_device('auto'),
            "epochs": 5,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "batch_size": 256,
            "grad_clip": 1.0,
            "warmup_ratio": 0.1,
            "grad_accum_steps": 1,
            "use_activation_checkpointing": False,
            "show_progress": True,
        }
        if params:
            config.update(params)

        device = torch.device(config["device"])

        torch_module = self._v3_trainable_module(model)
        torch_module.to(device)
        torch_module.train()

        # Optional LoRA/PEFT: inject adapters & restrict optimized params.
        torch_module, trainable_params = self._maybe_apply_v3_lora(model, torch_module, peft_config)
        if not trainable_params:
            trainable_params = list(torch_module.parameters())

        X_np = X_train_processed.to_numpy() if isinstance(X_train_processed, pd.DataFrame) else np.asarray(X_train_processed)
        if isinstance(y_train_processed, (pd.Series, pd.DataFrame)):
            y_np = y_train_processed.to_numpy().ravel()
        else:
            y_np = np.asarray(y_train_processed).ravel()

        if y_np.dtype == object or not np.issubdtype(y_np.dtype, np.number):
            le = LabelEncoder()
            y_np = le.fit_transform(y_np)
            if not hasattr(model, 'label_encoder_'):
                model.label_encoder_ = le

        optimizer = AdamW(trainable_params, lr=config["learning_rate"],
                          weight_decay=config["weight_decay"])
        loss_fn = torch.nn.CrossEntropyLoss()
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler() if use_amp else None
        perf = self._v3_performance_options(config["use_activation_checkpointing"])

        from ..models.tabpfnv3.finetuning.data_util import (
            meta_dataset_collator, get_preprocessed_dataset_chunks,
        )

        def _move(item, dev):
            if isinstance(item, torch.Tensor):
                return item.to(dev)
            if isinstance(item, list):
                return [_move(x, dev) for x in item]
            if isinstance(item, tuple):
                return tuple(_move(x, dev) for x in item)
            if isinstance(item, dict):
                return {k: _move(v, dev) for k, v in item.items()}
            return item

        total_steps = None
        scheduler = None

        for epoch in range(1, config["epochs"] + 1):
            seed = 42 + epoch
            splitter = partial(train_test_split, test_size=0.3, random_state=seed)

            training_datasets = get_preprocessed_dataset_chunks(
                calling_instance=model,
                X_raw=X_np, y_raw=y_np,
                split_fn=splitter,
                max_data_size=config["batch_size"],
                model_type="classifier",
                equal_split_size=False,
                data_shuffle_seed=seed,
                preprocessing_random_state=seed,
            )

            dataloader = DataLoader(training_datasets, batch_size=1,
                                    collate_fn=meta_dataset_collator, shuffle=True)

            if total_steps is None:
                total_steps = max(1, len(dataloader) * config["epochs"])
                warmup_steps = int(total_steps * config["warmup_ratio"])

                def lr_lambda(step):
                    # +1: LambdaLR.__init__ calls step() before the first optimizer.step(),
                    # consuming step 0. The offset ensures the first training batch gets
                    # warmup LR > 0 rather than 0.
                    s = step + 1
                    if s < warmup_steps:
                        return float(s) / max(1, warmup_steps)
                    progress = float(s - warmup_steps) / max(1, total_steps - warmup_steps)
                    return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=UserWarning,
                        message="Detected call of `lr_scheduler.step\\(\\)`",
                    )
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            iterable = tqdm(dataloader, desc=f"TabPFNv3 Meta Epoch {epoch}",
                            disable=not config["show_progress"])
            epoch_losses = []

            for step_i, batch in enumerate(iterable):
                if not hasattr(batch, 'X_context'):
                    continue
                X_ctx = _move(batch.X_context, device)
                y_ctx = _move(batch.y_context, device)
                X_qry = _move(batch.X_query, device)
                y_qry = _move(batch.y_query, device)
                cat_ixs = batch.cat_indices
                confs = batch.configs

                # Skip episodes where query labels are not a subset of context labels.
                ctx_uniq = torch.unique(torch.cat([torch.unique(t.reshape(-1)) for t in y_ctx]))
                qry_uniq = torch.unique(y_qry.reshape(-1))
                if not torch.isin(qry_uniq, ctx_uniq).all():
                    continue

                # v3 fit_from_preprocessed REQUIRES performance_options=.
                model.fit_from_preprocessed(X_ctx, y_ctx, cat_ixs, confs,
                                            performance_options=perf)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model.forward(X_qry, return_raw_logits=True)
                    # Training-time logits shape (n_estimators, B, n_classes, n_samples)
                    # -> flatten estimators/batch and align to CE expecting (N, C, ...).
                    if logits.dim() == 4:
                        Q, B, E, L = logits.shape
                        logits_BLQ = logits.permute(1, 2, 3, 0).reshape(B * E, L, Q)
                        targets_BQ = y_qry.repeat(B * E, 1).to(device)
                        loss = torch.nn.functional.cross_entropy(logits_BLQ, targets_BQ)
                    else:
                        loss = loss_fn(logits, y_qry.to(device))

                loss = loss / config["grad_accum_steps"]

                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    if (step_i + 1) % config["grad_accum_steps"] == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, config["grad_clip"])
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if (step_i + 1) % config["grad_accum_steps"] == 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, config["grad_clip"])
                        optimizer.step()
                        optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                epoch_losses.append(loss.item() * config["grad_accum_steps"])
                iterable.set_postfix(loss=f"{epoch_losses[-1]:.4f}",
                                     lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            avg = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            logger.info(f"[TuningManager] TabPFNv3 Meta Epoch [{epoch}/{config['epochs']}]: Loss={avg:.4f}")

        torch_module.eval()
        if hasattr(model, 'batched'):
            model.batched = False
        logger.info("[TuningManager] TabPFNv3 meta-learning fine-tuning complete")
        return model

    def _finetune_tabpfnv3_sft(self, model, X_train_processed, y_train_processed,
                               params=None, peft_config=None):
        """Single-episode SFT for TabPFN v3.

        Uses the entire dataset as ONE (support, query) episode and trains over it
        for multiple epochs, forcing task specialization. Cosine LR + warmup, AMP,
        optional label smoothing, optional LoRA/PEFT.
        """
        import torch
        import numpy as np
        import pandas as pd
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        logger.info("[TuningManager] Starting TabPFNv3 SFT fine-tuning")

        config = {
            "device": resolve_device('auto'),
            "epochs": 25,
            "learning_rate": 1e-5,
            "weight_decay": 1e-4,
            "query_set_ratio": 0.3,
            "grad_clip": 1.0,
            "label_smoothing": 0.0,
            "warmup_ratio": 0.1,
            "use_activation_checkpointing": False,
            "show_progress": True,
        }
        if params:
            config.update(params)

        device = torch.device(config["device"])

        torch_module = self._v3_trainable_module(model)
        torch_module.to(device)
        torch_module.train()
        torch_module, trainable_params = self._maybe_apply_v3_lora(model, torch_module, peft_config)
        if not trainable_params:
            trainable_params = list(torch_module.parameters())

        X_np = X_train_processed.to_numpy() if isinstance(X_train_processed, pd.DataFrame) else np.asarray(X_train_processed)
        if isinstance(y_train_processed, (pd.Series, pd.DataFrame)):
            y_np = y_train_processed.to_numpy().ravel()
        else:
            y_np = np.asarray(y_train_processed).ravel()

        if y_np.dtype == object or not np.issubdtype(y_np.dtype, np.number):
            le = LabelEncoder()
            y_np = le.fit_transform(y_np)
            if not hasattr(model, 'label_encoder_'):
                model.label_encoder_ = le

        optimizer = AdamW(trainable_params, lr=config["learning_rate"],
                          weight_decay=config["weight_decay"])
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler() if use_amp else None
        perf = self._v3_performance_options(config["use_activation_checkpointing"])

        from ..models.tabpfnv3.finetuning.data_util import (
            meta_dataset_collator, get_preprocessed_dataset_chunks,
        )

        def _move(item, dev):
            if isinstance(item, torch.Tensor):
                return item.to(dev)
            if isinstance(item, list):
                return [_move(x, dev) for x in item]
            if isinstance(item, tuple):
                return tuple(_move(x, dev) for x in item)
            if isinstance(item, dict):
                return {k: _move(v, dev) for k, v in item.items()}
            return item

        def sft_splitter(X, y, **kwargs):
            test_size = kwargs.get('test_size', config["query_set_ratio"])
            random_state = kwargs.get('random_state', 42)
            stratify = kwargs.get('stratify', None)
            if stratify is None:
                y_s = pd.Series(y)
                if y_s.nunique() > 1 and y_s.value_counts().min() > 1:
                    stratify = y
            try:
                return train_test_split(X, y, test_size=test_size, stratify=stratify,
                                        random_state=random_state)
            except ValueError:
                return train_test_split(X, y, test_size=test_size, random_state=random_state)

        # One large episode covering the entire dataset.
        training_datasets = get_preprocessed_dataset_chunks(
            calling_instance=model,
            X_raw=X_np, y_raw=y_np,
            split_fn=sft_splitter,
            max_data_size=len(X_np),
            model_type="classifier",
            equal_split_size=False,
            data_shuffle_seed=42,
            preprocessing_random_state=42,
        )
        dataloader = DataLoader(training_datasets, batch_size=1,
                                collate_fn=meta_dataset_collator, shuffle=False)

        total_steps = max(1, len(dataloader) * config["epochs"])
        warmup_steps = int(total_steps * config["warmup_ratio"])

        def lr_lambda(step):
            s = step + 1  # offset: LambdaLR.__init__ consumes step 0 before first optimizer.step()
            if s < warmup_steps:
                return float(s) / max(1, warmup_steps)
            progress = float(s - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning,
                message="Detected call of `lr_scheduler.step\\(\\)`",
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        for epoch in range(1, config["epochs"] + 1):
            iterable = tqdm(dataloader, desc=f"TabPFNv3 SFT Epoch {epoch}",
                            disable=not config["show_progress"])
            epoch_losses = []
            for batch in iterable:
                if not hasattr(batch, 'X_context'):
                    continue
                X_ctx = _move(batch.X_context, device)
                y_ctx = _move(batch.y_context, device)
                X_qry = _move(batch.X_query, device)
                y_qry = _move(batch.y_query, device)
                cat_ixs = batch.cat_indices
                confs = batch.configs

                ctx_uniq = torch.unique(torch.cat([torch.unique(t.reshape(-1)) for t in y_ctx]))
                qry_uniq = torch.unique(y_qry.reshape(-1))
                if not torch.isin(qry_uniq, ctx_uniq).all():
                    continue

                model.fit_from_preprocessed(X_ctx, y_ctx, cat_ixs, confs,
                                            performance_options=perf)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model.forward(X_qry, return_raw_logits=True)
                    if logits.dim() == 4:
                        Q, B, E, L = logits.shape
                        logits_BLQ = logits.permute(1, 2, 3, 0).reshape(B * E, L, Q)
                        targets_BQ = y_qry.repeat(B * E, 1).to(device)
                        loss = torch.nn.functional.cross_entropy(logits_BLQ, targets_BQ)
                    else:
                        loss = loss_fn(logits, y_qry.to(device))

                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, config["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, config["grad_clip"])
                    optimizer.step()
                scheduler.step()

                epoch_losses.append(loss.item())
                iterable.set_postfix(loss=f"{epoch_losses[-1]:.4f}",
                                     lr=f"{optimizer.param_groups[0]['lr']:.2e}")

            avg = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            logger.info(f"[TuningManager] TabPFNv3 SFT Epoch [{epoch}/{config['epochs']}]: Loss={avg:.4f}")

        torch_module.eval()
        if hasattr(model, 'batched'):
            model.batched = False
        logger.info("[TuningManager] TabPFNv3 SFT fine-tuning complete")
        return model

    def _finetune_tabpfnv3_regression_turn_by_turn(self, model, X_train, y_train, params=None):
        """Turn-by-turn regression FT for TabPFN v3.

        Lightweight alternative to native FT: builds regression episodes and trains
        the backbone with the model's bar-distribution loss. Uses the v3 chunk/
        collator helpers and performance_options. Operates on the regression wrapper
        (which subclasses the v3 regressor, so it exposes models_/fit_from_preprocessed).
        """
        import torch
        import numpy as np
        import pandas as pd
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        from sklearn.model_selection import train_test_split
        from functools import partial

        logger.info("[TuningManager] Starting TabPFNv3 regression turn-by-turn fine-tuning")

        config = {
            "device": resolve_device('auto'),
            "epochs": 5,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "batch_size": 256,
            "grad_clip": 1.0,
            "warmup_ratio": 0.1,
            "use_activation_checkpointing": False,
            "show_progress": True,
        }
        if params:
            config.update(params)

        device = torch.device(config["device"])

        torch_module = self._v3_trainable_module(model)
        torch_module.to(device)
        torch_module.train()
        trainable_params = [p for p in torch_module.parameters() if p.requires_grad] or list(torch_module.parameters())

        X_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_np = y_train.to_numpy().ravel().astype(np.float64)
        else:
            y_np = np.asarray(y_train).ravel().astype(np.float64)

        optimizer = AdamW(trainable_params, lr=config["learning_rate"],
                          weight_decay=config["weight_decay"])
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler() if use_amp else None
        perf = self._v3_performance_options(config["use_activation_checkpointing"])

        from ..models.tabpfnv3.finetuning.data_util import (
            meta_dataset_collator, get_preprocessed_dataset_chunks,
        )

        def _move(item, dev):
            if isinstance(item, torch.Tensor):
                return item.to(dev)
            if isinstance(item, list):
                return [_move(x, dev) for x in item]
            if isinstance(item, tuple):
                return tuple(_move(x, dev) for x in item)
            if isinstance(item, dict):
                return {k: _move(v, dev) for k, v in item.items()}
            return item

        total_steps = None
        scheduler = None

        for epoch in range(1, config["epochs"] + 1):
            seed = 42 + epoch
            splitter = partial(train_test_split, test_size=0.3, random_state=seed)
            training_datasets = get_preprocessed_dataset_chunks(
                calling_instance=model,
                X_raw=X_np, y_raw=y_np,
                split_fn=splitter,
                max_data_size=config["batch_size"],
                model_type="regressor",
                equal_split_size=False,
                data_shuffle_seed=seed,
                preprocessing_random_state=seed,
            )
            dataloader = DataLoader(training_datasets, batch_size=1,
                                    collate_fn=meta_dataset_collator, shuffle=True)

            if total_steps is None:
                total_steps = max(1, len(dataloader) * config["epochs"])
                warmup_steps = int(total_steps * config["warmup_ratio"])

                def lr_lambda(step):
                    # +1: LambdaLR.__init__ calls step() before the first optimizer.step(),
                    # consuming step 0. The offset ensures the first training batch gets
                    # warmup LR > 0 rather than 0.
                    s = step + 1
                    if s < warmup_steps:
                        return float(s) / max(1, warmup_steps)
                    progress = float(s - warmup_steps) / max(1, total_steps - warmup_steps)
                    return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=UserWarning,
                        message="Detected call of `lr_scheduler.step\\(\\)`",
                    )
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            iterable = tqdm(dataloader, desc=f"TabPFNv3 Reg-TBT Epoch {epoch}",
                            disable=not config["show_progress"])
            epoch_losses = []
            for batch in iterable:
                if not hasattr(batch, 'X_context'):
                    continue
                X_ctx = _move(batch.X_context, device)
                y_ctx = _move(batch.y_context, device)
                X_qry = _move(batch.X_query, device)
                y_qry = _move(batch.y_query, device)
                cat_ixs = batch.cat_indices
                confs = batch.configs

                model.fit_from_preprocessed(X_ctx, y_ctx, cat_ixs, confs,
                                            performance_options=perf)
                optimizer.zero_grad()

                # The regressor's forward() returns (averaged_logits, outputs, borders)
                # and does NOT accept return_logits (that is a classifier-only kwarg).
                # Loss is the bar-distribution NLL in z-normalized space, mirroring the
                # TabPFN regression turn-by-turn loop.
                from tabtune.models.tabpfn.utils import translate_probs_across_borders

                target = y_qry[0] if isinstance(y_qry, list) else y_qry
                target = target.reshape(-1).to(device)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    _avg, outputs, borders = model.forward(X_qry, use_inference_mode=False)

                    std_borders = model.znorm_space_bardist_.borders.to(device)
                    transformed_probs = []
                    for probs, b in zip(outputs, borders):
                        p = probs
                        if p.dim() == 3:
                            p = p.squeeze(1)
                        p = translate_probs_across_borders(
                            p,
                            frm=torch.as_tensor(b, device=device),
                            to=std_borders,
                        )
                        transformed_probs.append(p)

                    probs_mean = torch.stack(transformed_probs, dim=0).mean(dim=0)
                    q_probs = probs_mean[-target.numel():]
                    q_log_probs = (q_probs + 1e-12).log()

                    crit = model.znorm_space_bardist_.to(device)
                    loss = crit(q_log_probs, target).mean()

                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, config["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, config["grad_clip"])
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                epoch_losses.append(loss.item())
                iterable.set_postfix(loss=f"{epoch_losses[-1]:.4f}")

            avg = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
            logger.info(f"[TuningManager] TabPFNv3 Reg-TBT Epoch [{epoch}/{config['epochs']}]: Loss={avg:.4f}")

        torch_module.eval()

        # Fine-tuning leaves the estimator in 'batched' fit_mode with a batched
        # executor, which `predict()` rejects (the InferenceEngine assertion).
        # Re-fit on the training data to rebuild a standard-inference executor,
        # mirroring how the native finetuners restore standard inference. The
        # regressor's own fit() auto-switches 'batched' -> 'fit_preprocessors'.
        try:
            if hasattr(model, "fit_mode") and model.fit_mode == "batched":
                model.fit_mode = "fit_preprocessors"
            model.fit(X_train, y_train)
            logger.info("[TuningManager] Re-fitted v3 regressor for standard inference")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[TuningManager] Post-FT re-fit failed: {e}")

        if hasattr(model, 'batched'):
            model.batched = False
        logger.info("[TuningManager] TabPFNv3 regression turn-by-turn fine-tuning complete")
        return model


    def _finetune_tabiclv2_regression(self, model, X_train, y_train, params=None):
        """
        Episodic turn-by-turn finetuning for TabICLv2 regressor.

        TabICLv2's model_ is a transformer that takes:
            input:  (batch, n_train+n_test, n_features) — concatenated support+query
            labels: (batch, n_train) — support labels only
            output: (batch, n_test, 1) — predicted query values

        We create random (support, query) episodes and backprop MSE loss
        through the transformer.
        """
        import torch
        import numpy as np
        import pandas as pd
        from torch.optim import AdamW
        from sklearn.preprocessing import StandardScaler
        from tqdm import tqdm

        logger.info("[TuningManager] Starting TabICLv2 regression fine-tuning")

        config = {
            "device": resolve_device('auto'),
            "epochs": 5,
            "learning_rate": 2e-6,
            "weight_decay": 0.01,
            "support_size": 128,
            "query_size": 64,
            "steps_per_epoch": 200,
            "grad_clip": 1.0,
            "show_progress": True,
        }
        if params:
            config.update(params)

        device = torch.device(config["device"])

        # Prepare data
        X_np = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train, dtype=np.float32)
        y_np = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train, dtype=np.float32)
        y_np = y_np.reshape(-1)

        # Ensure model is loaded
        if not hasattr(model, 'model_') or model.model_ is None:
            model._load_model()
        model.fit(X_np, y_np)  # sets up scaler, encoder, ensemble_generator

        # Scale targets (TabICLv2 regressor uses StandardScaler internally)
        y_scaler = model.y_scaler_ if hasattr(model, 'y_scaler_') else StandardScaler().fit(y_np.reshape(-1, 1))
        y_scaled = y_scaler.transform(y_np.reshape(-1, 1)).flatten()

        # Transform features through the ensemble generator
        # Get the first ensemble member's transformed data
        X_encoder = model.X_encoder_ if hasattr(model, 'X_encoder_') else None
        if X_encoder is not None:
            X_encoded = X_encoder.transform(X_np)
        else:
            X_encoded = X_np.copy()

        model.model_.to(device)
        model.model_.train()

        support_size = min(config["support_size"], len(X_encoded) // 2)
        query_size = min(config["query_size"], len(X_encoded) - support_size)

        optimizer = AdamW(model.model_.parameters(),
                          lr=config["learning_rate"],
                          weight_decay=config["weight_decay"])
        loss_fn = torch.nn.MSELoss()

        total_steps = config["epochs"] * config["steps_per_epoch"]
        step = 0

        for epoch in range(1, config["epochs"] + 1):
            epoch_losses = []

            iterator = range(config["steps_per_epoch"])
            if config["show_progress"]:
                iterator = tqdm(iterator, desc=f"TabICLv2 Reg FT Epoch {epoch}")

            for _ in iterator:
                step += 1

                # Sample random episode
                idx = np.random.permutation(len(X_encoded))
                s_idx = idx[:support_size]
                q_idx = idx[support_size:support_size + query_size]

                X_support = X_encoded[s_idx]
                y_support = y_scaled[s_idx]
                X_query = X_encoded[q_idx]
                y_query = y_scaled[q_idx]

                # Concatenate support + query features: (1, S+Q, F)
                X_episode = np.concatenate([X_support, X_query], axis=0)
                X_t = torch.from_numpy(X_episode).float().unsqueeze(0).to(device)

                # Support labels: (1, S)
                y_s = torch.from_numpy(y_support).float().unsqueeze(0).to(device)

                # Query ground truth
                y_q_true = torch.from_numpy(y_query).float().to(device)

                optimizer.zero_grad()

                # Forward: model expects (batch, S+Q, F) and (batch, S) labels
                # Returns predictions for the query portion: (batch, Q, 1) or (batch, Q)
                with torch.enable_grad():
                    raw_out = model.model_(X_t, y_s)

                # raw_out shape: (batch, Q, n_bins) or (batch, Q, 1) or (batch, Q)
                if raw_out.dim() == 3:
                    raw_out = raw_out.squeeze(0)  # (Q, n_bins) or (Q, 1)

                if raw_out.dim() == 2 and raw_out.size(-1) > 1:
                    # Distribution output — take weighted mean across bins
                    # Softmax to get probabilities, then weighted sum
                    probs = torch.softmax(raw_out, dim=-1)  # (Q, n_bins)
                    n_bins = probs.size(-1)
                    # Create evenly spaced bin centers in [-3, 3] (z-normalized space)
                    bin_centers = torch.linspace(-3, 3, n_bins, device=device)
                    preds = (probs * bin_centers.unsqueeze(0)).sum(dim=-1)  # (Q,)
                elif raw_out.dim() == 2 and raw_out.size(-1) == 1:
                    preds = raw_out.squeeze(-1)  # (Q,)
                else:
                    preds = raw_out  # already (Q,)

                # Take only query predictions if needed
                if preds.shape[0] > query_size:
                    preds = preds[-query_size:]

                loss = loss_fn(preds, y_q_true)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.model_.parameters(),
                                                config["grad_clip"])
                optimizer.step()

                epoch_losses.append(loss.item())
                if config["show_progress"]:
                    iterator.set_postfix(loss=f"{loss.item():.6f}")

            avg = np.mean(epoch_losses) if epoch_losses else float('nan')
            logger.info(f"[TuningManager] TabICLv2 Reg FT Epoch [{epoch}/{config['epochs']}]: "
                        f"MSE Loss={avg:.6f}")

        model.model_.eval()
        logger.info("[TuningManager] TabICLv2 regression fine-tuning complete")

        # Re-fit for inference (rebuilds caches etc.)
        try:
            model.fit(X_np, y_np)
        except Exception as e:
            logger.warning(f"[TuningManager] Post-finetune re-fit: {e}")

        return model


    def get_default_config(self, model, selected_strategy: str, finetune_mode: str, processor=None) -> dict:
        """
        Return the default config that would be used for this model/strategy/mode.
        This must match the dicts defined inside the _finetune_* methods.
        """
        device = resolve_device('auto')

        # TabICL / Orion MSP / Orion Bix
        if isinstance(model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, OrionMSPv15Classifier, TabICLv2Classifier)):
            if finetune_mode == "meta-learning":
                return {
                    "device": device,
                    "epochs": 5,
                    "learning_rate": 2e-6,
                    "show_progress": True,
                    "support_size": 48,
                    "query_size": 32,
                    "n_episodes": 1000,
                    # keep these visible too if you support them
                    # "finetune_method": None,
                    # "peft_config": None,
                }
            else:
                # simple SFT defaults (_finetune_tabicl_simple_sft)
                return {
                    "device": device,
                    "epochs": 5,
                    "learning_rate": 1e-5,
                    "batch_size": 16,
                    "show_progress": True,
                }

        # TabPFN
        if isinstance(model, TabPFNv26Classifier):
            if finetune_mode == "native":
                return {
                    "device": device,
                    "epochs": 30,
                    "learning_rate": 1e-5,
                    "weight_decay": 0.01,
                    "early_stopping": True,
                    "early_stopping_patience": 8,
                    "validation_split_ratio": 0.1,
                    "n_finetune_ctx_plus_query_samples": 10_000,
                    "finetune_ctx_query_split_ratio": 0.2,
                    "n_estimators_finetune": 2,
                    "n_estimators_validation": 2,
                    "n_estimators_final_inference": 8,
                    "grad_clip_value": 1.0,
                    "use_lr_scheduler": True,
                    "use_activation_checkpointing": True,
                    "random_state": 0,
                }
            elif finetune_mode == "sft":
                return {
                    "device": device,
                    "epochs": 25,
                    "learning_rate": 1e-5,
                    "weight_decay": 1e-4,
                    "query_set_ratio": 0.3,
                    "grad_clip": 1.0,
                    "label_smoothing": 0.0,
                    "warmup_ratio": 0.1,
                    "show_progress": True,
                }
            else:  # meta-learning
                return {
                    "device": device,
                    "epochs": 5,
                    "learning_rate": 1e-5,
                    "weight_decay": 0.01,
                    "batch_size": 256,
                    "grad_clip": 1.0,
                    "warmup_ratio": 0.1,
                    "grad_accum_steps": 1,
                    "show_progress": True,
                }
        if isinstance(model, (TabPFNClassifier)):
            if finetune_mode == "sft":
                return {
                    "device": device,
                    "epochs": 25,
                    "learning_rate": 1e-5,
                    "show_progress": True,
                    "max_episode_size": None,   # you can set to len(X) only at fit-time
                    "query_set_ratio": 0.3,
                    "weight_decay": 1e-4,
                }
            else:
                return {
                    "device": device,
                    "epochs": 3,
                    "learning_rate": 1e-5,
                    "batch_size": 256,
                    "show_progress": True,
                }

        # ConTextTab full FT
        if isinstance(model, ConTextTabClassifier):
            return {
                "device": device,
                "epochs": 5,
                "learning_rate": 1e-4,
                "batch_size": 128,
                "show_progress": True,
            }

        # TabDPT
        if isinstance(model, TabDPTClassifier):
            if finetune_mode == "sft":
                return {
                    "device": device,
                    "epochs": 5,
                    "learning_rate": 2e-5,
                    "batch_size": 32,
                    "show_progress": True,
                    "weight_decay": 1e-4,
                    "warmup_epochs": 1,
                }
            else:
                return {
                    "device": device,
                    "epochs": 5,
                    "learning_rate": 1e-5,
                    "batch_size": 8,
                    "support_size": 512,
                    "query_size": 256,
                    "steps_per_epoch": 100,
                    "show_progress": True,
                }

        # Mitra / Tab2D
        if isinstance(model, Tab2D):
            if finetune_mode == "sft":
                return {
                    "device": device,
                    "epochs": 5,
                    "learning_rate": 1e-5,
                    "batch_size": 128,
                    "show_progress": True,
                    "weight_decay": 1e-4,
                    "warmup_epochs": 1,
                }
            else:
                return {
                    "device": device,
                    "epochs": 3,
                    "learning_rate": 1e-5,
                    "batch_size": 4,
                    "support_size": 128,
                    "query_size": 128,
                    "steps_per_epoch": 50,
                    "show_progress": True,
                }

        if isinstance(model, TabICLv2Regressor):
            return {
                "device": device,
                "epochs": 5,
                "learning_rate": 2e-6,
                "weight_decay": 0.01,
                "support_size": 128,
                "query_size": 64,
                "steps_per_epoch": 200,
                "grad_clip": 1.0,
                "show_progress": True,
            }

        # Limix
        # if isinstance(model, LimixClassifier):
        #     return {
        #         "device": device,
        #         "epochs": 5,
        #         "learning_rate": 1e-5,
        #         "show_progress": True,
        #         "support_size": 48,
        #         "query_size": 32,
        #         "n_episodes": 1000,
        #     }

        # fallback: no tuning defaults known
        return {"device": device}



    def _contexttab_regression_make_episode(self, model, X_all, y_all, context_size, query_size, seed=None):
        """
        Build one regression episode for ContextTab:
        - context rows = training context (targets known)
        - query rows   = test/query (targets masked in train_target; true labels returned separately)
        Returns: tokenized_data dict ready for model forward + loss.
        """
        if not isinstance(X_all, pd.DataFrame):
            X_all = pd.DataFrame(X_all)
        if isinstance(y_all, (pd.DataFrame, pd.Series)):
            y_all = np.array(y_all).reshape(-1)
        y_all = np.asarray(y_all).astype(float)

        n = len(X_all)
        if n < (context_size + 1):
            # fall back: tiny datasets
            context_size = max(1, n - 1)
            query_size = 1

        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)

        ctx_idx = idx[:context_size]
        remaining = idx[context_size:]
        if len(remaining) == 0:
            # if context consumes all rows, reuse some rows as query
            qry_idx = rng.choice(ctx_idx, size=query_size, replace=True)
        else:
            qry_idx = rng.choice(remaining, size=min(query_size, len(remaining)), replace=False)
            if len(qry_idx) < query_size:
                # top up with replacement from remaining/context
                pool = remaining if len(remaining) > 0 else ctx_idx
                extra = rng.choice(pool, size=(query_size - len(qry_idx)), replace=True)
                qry_idx = np.concatenate([qry_idx, extra])

        X_ctx = X_all.iloc[ctx_idx].copy()
        y_ctx = pd.DataFrame({'TARGET': y_all[ctx_idx]}, index=X_ctx.index)

        X_qry = X_all.iloc[qry_idx].copy()
        y_qry = pd.DataFrame({'TARGET': y_all[qry_idx]}, index=X_qry.index)

        # tokenizer returns:
        # - data: dict with 'target' where query rows are masked (<= -99 sentinel)
        # - labels: true labels for query rows (normalized for l2 if tokenizer does it)
        data, labels, label_classes = model.tokenizer(
            X_ctx, y_ctx,
            X_qry, y_qry,
            model.classification_or_regression
        )

        target_mean, target_std = 0, 0
        if model.classification_or_regression == 'regression' and getattr(model, 'regression_type', 'l2') == 'l2':
            _, target_mean, target_std = model.tokenizer.standard_scale_column(y_ctx, y_qry)

        tokenized = {
            'data': data,
            'num_rows': context_size + query_size,
            'num_cols': X_all.shape[1] + 1,  # incl target col
            'labels': labels,                # <-- IMPORTANT for training
            'is_regression': torch.tensor(True),
            'label_classes': np.asarray(label_classes),
            'target_mean': target_mean,
            'target_std': target_std
        }
        return tokenized


    def _finetune_contexttab_regression_turn_by_turn(self, model, X_train, y_train, params):
        """
        Turn-by-turn regression fine-tuning for ContextTab.
        Uses episodic (context, query) batches and optimizes regression loss on query rows.
        """
        logger = logging.getLogger(__name__)

        device = params.get('device', getattr(model, 'device', resolve_device('auto')))

        # loop params
        epochs = int(params.get('epochs', 1))
        steps_per_epoch = int(params.get('steps_per_epoch', 200))
        context_size = int(params.get('context_size', 256))
        query_size = int(params.get('query_size', 64))
        seed = int(params.get('seed', 42))

        # optim params
        lr = float(params.get('lr', 1e-5))
        weight_decay = float(params.get('weight_decay', 0.01))
        clip_grad_norm = float(params.get('clip_grad_norm', 1.0))

        # checkpoint (optional)
        save_checkpoint_path = params.get('save_checkpoint_path', None)

        # ensure context is stored (ConTextTab API expectation)
        model.fit(X_train, y_train)

        # train mode
        model.model.train()
        model.model.to(device)

        opt = AdamW(model.model.parameters(), lr=lr, weight_decay=weight_decay)

        global_step = 0
        for ep in range(epochs):
            running_loss = 0.0

            step_iter = range(steps_per_epoch)
            if params.get("show_progress", False):
                step_iter = tqdm(
                    step_iter,
                    desc=f"ContextTab-Reg TBT | Epoch {ep+1}/{epochs}",
                    leave=True
                )
                
            for s in step_iter:
                tokenized = self._contexttab_regression_make_episode(
                    model=model,
                    X_all=X_train,
                    y_all=y_train,
                    context_size=context_size,
                    query_size=query_size,
                    seed=seed + global_step
                )

                # move to device
                tokenized = to_device(tokenized, device, raise_on_unexpected=False)

                out = model.model(**tokenized)

                # ContextTab forward sometimes returns (logits, aux) or a dict-like output
                if isinstance(out, tuple):
                    logits = out[0]
                elif isinstance(out, dict):
                    logits = out.get("logits", out.get("preds", out.get("output", out)))
                else:
                    logits = out


                # labels are required for regression loss (query rows)
                labels = tokenized.get('labels', None)
                if labels is None:
                    raise RuntimeError(
                        "ContextTab regression finetune requires tokenized['labels'] from tokenizer(). "
                        "If you see this, your tokenization path dropped labels."
                    )

                # compute regression loss on query rows (where train_target <= -99)
                train_target = tokenized ['data']['target']
                ret = model.model.compute_regression_output_loss_and_metric(
                    logits=logits,
                    labels=labels,
                    train_target=train_target
                )

                # ContextTab regression may return (loss, metric) OR (loss, metric, ...)
                if isinstance(ret, tuple):
                    loss = ret[1]
                    metric = ret[2] if len(ret) > 1 else None
                else:
                    loss = ret
                    metric = None

                # loss must be a scalar for backward()
                if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                    loss = loss.mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()

                if clip_grad_norm is not None and clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.model.parameters(), clip_grad_norm)

                opt.step()

                running_loss += float(loss.detach().cpu().item())
                global_step += 1

            avg_loss = running_loss / max(1, steps_per_epoch)
            logger.info(
                f"[ContextTab-Regression-TBT] epoch={ep+1}/{epochs} "
                f"steps={steps_per_epoch} avg_loss={avg_loss:.5f}"
            )

            if save_checkpoint_path:
                os.makedirs(os.path.dirname(save_checkpoint_path), exist_ok=True)
                torch.save({'model_state_dict': model.model.state_dict()}, save_checkpoint_path)
                logger.info(f"[ContextTab-Regression-TBT] saved checkpoint -> {save_checkpoint_path}")

        model.model.eval()
        return model




    def safe_r2(y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)

        # finite + same length
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]

        # need at least 2 points
        if y_true.size < 2:
            return np.nan

        # undefined if variance is 0
        if np.allclose(y_true, y_true[0]):
            return np.nan

        return r2_score(y_true, y_pred)
    
    def _finetune_limix_regression(self, model, X_train, y_train, params):
        """
        Episodic fine-tuning for Limix regression.
        Builds (support, query) episodes and trains MSE on query predictions.
        """
        import numpy as np
        import torch
        from torch.optim import AdamW
        from tqdm import tqdm

        logger.info("[TuningManager] Starting Limix regression fine-tuning")

        config = {
            "device": params.get("device", resolve_device('auto')),
            "epochs": int(params.get("epochs", 3)),
            "steps_per_epoch": int(params.get("steps_per_epoch", 100)),
            "support_size": int(params.get("support_size", params.get("context_size", 256))),
            "query_size": int(params.get("query_size", 64)),
            "lr": float(params.get("lr", 1e-5)),
            "weight_decay": float(params.get("weight_decay", 0.01)),
            "clip_grad_norm": float(params.get("clip_grad_norm", 1.0)),
            "seed": int(params.get("seed", 42)),
            "show_progress": bool(params.get("show_progress", True)),
        }

        device = torch.device(config["device"])

        # Ensure dataframe/series -> numpy
        if hasattr(X_train, "to_numpy"):
            X_np = X_train.to_numpy()
        else:
            X_np = np.asarray(X_train)

        if hasattr(y_train, "to_numpy"):
            y_np = y_train.to_numpy()
        else:
            y_np = np.asarray(y_train)

        X_np = X_np.astype(np.float32)
        y_np = y_np.astype(np.float32).reshape(-1)

        n = X_np.shape[0]
        if n < (config["support_size"] + 2):
            raise ValueError(f"Not enough rows for episodic finetune: n={n}, support={config['support_size']}")

        # Normalize target like LimixRegressor.fit does
        y_mean = float(np.mean(y_np))
        y_std = float(np.std(y_np)) if float(np.std(y_np)) > 1e-12 else 1.0
        y_norm = (y_np - y_mean) / y_std

        # Under the wrapper, you have an ensemble
        # Each estimator has .model (torch module)
        estimators = getattr(model, "estimators", None)

        # Some older codepaths might call it "models"
        if estimators is None:
            estimators = getattr(model, "models", None)

        # Single-estimator fallback
        if estimators is None and hasattr(model, "model"):
            estimators = [model]

        if not estimators:
         # Important: wrapper creates estimators only after fit()
            if hasattr(model, "fit"):
                model.fit(X_train, y_train)
                estimators = getattr(model, "estimators", None) or getattr(model, "models", None)
            if not estimators:
                raise AttributeError(
                    f"Could not find Limix estimators on {type(model).__name__}. "
                    "Expected `.estimators` (wrapper/ensemble) or `.model` (single regressor)."
            )

        mse = torch.nn.MSELoss()

        rng = np.random.default_rng(config["seed"])

        for est_i, est in enumerate(estimators):
            torch_model = getattr(est, "model", None)
            if torch_model is None or not isinstance(torch_model, torch.nn.Module):
                raise RuntimeError(f"Estimator {est_i} has no torch model to finetune.")

            torch_model.to(device)
            torch_model.train()

            opt = AdamW(torch_model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

            for epoch in range(config["epochs"]):
                it = range(config["steps_per_epoch"])
                if config["show_progress"]:
                    it = tqdm(it, desc=f"Limix-Reg FT | est {est_i+1}/{len(estimators)} | epoch {epoch+1}/{config['epochs']}")

                for step in it:
                    # sample episode indices
                    total = config["support_size"] + config["query_size"]
                    idx = rng.choice(n, size=total, replace=False)

                    s = config["support_size"]
                    sup_idx = idx[:s]
                    qry_idx = idx[s:]

                    X_sup = X_np[sup_idx]
                    y_sup = y_norm[sup_idx]
                    X_qry = X_np[qry_idx]
                    y_qry = y_norm[qry_idx]

                    # Build [1, S+Q, F]
                    X_episode = np.concatenate([X_sup, X_qry], axis=0)
                    X_t = torch.from_numpy(X_episode).unsqueeze(0).to(device)

                    # y input: true for support, dummy for query (prevents leakage)
                    y_in = np.concatenate([y_sup, np.zeros_like(y_qry)], axis=0)
                    y_t = torch.from_numpy(y_in).unsqueeze(0).to(device)

                    opt.zero_grad(set_to_none=True)

                    out = torch_model(X_t, y_t, eval_pos=s, task_type="reg")

                    # Robustly extract reg output
                    if isinstance(out, dict):
                        pred = out.get("reg_output", None)
                    elif isinstance(out, (tuple, list)) and len(out) > 0:
                        pred = out[0]
                    else:
                        pred = out

                    if pred is None:
                        raise RuntimeError("Limix forward did not return reg_output.")

                    # pred is typically [1, Q, 1]
                    pred = pred.squeeze(0).squeeze(-1)  # -> [Q]
                    target = torch.from_numpy(y_qry).to(device)

                    loss = mse(pred, target)
                    loss.backward()

                    if config["clip_grad_norm"] and config["clip_grad_norm"] > 0:
                        torch.nn.utils.clip_grad_norm_(torch_model.parameters(), config["clip_grad_norm"])

                    opt.step()

                    if config["show_progress"]:
                        it.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

            #torch_model.eval()

        # After finetune, set context for inference
        # Refit context on the SAME estimators (no recreation)
        for est in estimators:
            if hasattr(est, "fit"):
                est.fit(X_train, y_train)
                if hasattr(est, "model") and isinstance(est.model, torch.nn.Module):
                    est.model.eval()

        logger.info("[TuningManager] Limix regression fine-tuning complete")
        return model

    def _finetune_tabdpt_regression_turn_by_turn(self, model, X_train, y_train, params: dict):
        """
        Key correctness points vs common broken implementations:
        - Fits the wrapper FIRST to ensure TabDPT preprocessing/caches (imputer/scaler/PCA V/X_train/y_train) exist.
          This avoids finetuning on raw/unprocessed data (distribution mismatch + dtype/categorical crashes).
        - Robustly resolves the underlying torch module for wrapper/estimator conventions.
        - Fixes pred/y shape mismatch to avoid mse broadcasting.
        - Keeps weights safe: if `fit()` recreates the torch module, we restore the pre-fit weights if possible.
        - Leaves the model in eval mode at the end.
        """
        import logging
        import numpy as np
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW
    
        logger = logging.getLogger(__name__)
        logger.info("[TuningManager] Starting TabDPT regression fine-tuning (turn-by-turn)")
    
        params = params or {}
    
        # ---------------------------
        # Device / reproducibility
        # ---------------------------
        device_str = params.get("device", resolve_device('auto'))
        device = torch.device(device_str)
    
        epochs = int(params.get("epochs", 5))
        steps_per_epoch = int(params.get("steps_per_epoch", 100))
        context_size = int(params.get("context_size", params.get("support_size", 512)))
        query_size = int(params.get("query_size", 128))
    
        lr = float(params.get("lr", params.get("learning_rate", 1e-5)))
        weight_decay = float(params.get("weight_decay", 0.0))
        clip_grad_norm = params.get("clip_grad_norm", None)
        show_progress = bool(params.get("show_progress", True))
        seed = params.get("seed", None)
    
        if seed is not None:
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
    
        # -------------------------------------------------------------------------
        # Resolve underlying torch module robustly for TabDPT wrappers
        # -------------------------------------------------------------------------
        def _resolve_torch_module_and_cfg_host(wrapper):
            """
            Returns (torch_model, cfg_host, inner_holder)
            - torch_model: the actual torch.nn.Module
            - cfg_host: object where config attrs live (max_features, feature_reduction, V, etc.)
            - inner_holder: the object that directly contains the torch module (for eval() at end)
            """
            est = getattr(wrapper, "model", None)
            est2 = getattr(wrapper, "model_", None)
    
            # Case 1: wrapper.model is a torch module
            if isinstance(est, torch.nn.Module):
                return est, wrapper, est
    
            # Case 2: wrapper.model is an estimator that has .model (torch module)
            if est is not None and hasattr(est, "model") and isinstance(est.model, torch.nn.Module):
                return est.model, est, est
    
            # Case 3: wrapper.model_ is a torch module
            if isinstance(est2, torch.nn.Module):
                return est2, wrapper, est2
    
            # Case 4: wrapper.model_ is an estimator that has .model (torch module)
            if est2 is not None and hasattr(est2, "model") and isinstance(est2.model, torch.nn.Module):
                return est2.model, est2, est2
    
            return None, None, None
    
        torch_model, cfg_host, inner_holder = _resolve_torch_module_and_cfg_host(model)
        if torch_model is None:
            raise AttributeError(
                "TabDPTRegressorWrapper: could not locate underlying torch module. "
                "Expected either (a) `wrapper.model` to be torch.nn.Module, or "
                "(b) `wrapper.model`/`wrapper.model_` to be an estimator with `.model` torch.nn.Module."
            )
    
        prefit_state = None
        try:
            prefit_state = {k: v.detach().cpu() for k, v in torch_model.state_dict().items()}
        except Exception:
            prefit_state = None
    
        # Keep model in eval while fitting preprocessing (fit should not train)
        try:
            torch_model.eval()
        except Exception:
            pass
    
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            logger.exception("[TuningManager] model.fit(X_train, y_train) failed before finetune.")
            raise
    
        # Re-resolve after fit (in case wrapper swapped internals)
        torch_model2, cfg_host2, inner_holder2 = _resolve_torch_module_and_cfg_host(model)
        if torch_model2 is None:
            raise AttributeError(
                "After model.fit, TabDPTRegressorWrapper: could not locate underlying torch module."
            )
    
        # If module object changed, attempt to restore previous weights
        if torch_model2 is not torch_model and prefit_state is not None:
            try:
                missing, unexpected = torch_model2.load_state_dict(prefit_state, strict=False)
                logger.info(
                    f"[TuningManager] Torch module changed after fit(); restored weights "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})."
                )
            except Exception:
                logger.warning(
                    "[TuningManager] Torch module changed after fit(); could not restore weights. "
                    "Continuing with post-fit weights."
                )
    
        torch_model, cfg_host, inner_holder = torch_model2, cfg_host2, inner_holder2
    
        # Keep device consistent with estimator/wrapper
        try:
            setattr(model, "device", str(device))
        except Exception:
            pass
    
        torch_model.to(device)
        torch_model.train()
    
        optimizer = AdamW(torch_model.parameters(), lr=lr, weight_decay=weight_decay)
    
        # -------------------------------------------------------------------------
        # Use PREPROCESSED cached training arrays from the fitted estimator/wrapper
        # -------------------------------------------------------------------------
        X_np = getattr(model, "X_train", None)
        y_np = getattr(model, "y_train", None)
    
        # Some implementations store caches on cfg_host (estimator). Fall back if needed.
        if X_np is None or y_np is None:
            X_np = getattr(cfg_host, "X_train", None)
            y_np = getattr(cfg_host, "y_train", None)
    
        if X_np is None or y_np is None:
            raise RuntimeError(
                "Could not find preprocessed caches `X_train` / `y_train` after fit(). "
                "Expected wrapper/estimator to expose them."
            )
    
        X_np = np.asarray(X_np, dtype=np.float32)
        y_np = np.asarray(y_np, dtype=np.float32).reshape(-1)
    
        n = len(X_np)
        if n < 2:
            raise ValueError("Need at least 2 training rows for episodic finetuning.")
    
        if n < (context_size + query_size):
            context_size = max(4, min(context_size, n // 2))
            query_size = max(1, min(query_size, n - context_size))
            logger.warning(f"[TuningManager] Shrunk episode sizes: context={context_size}, query={query_size}")
    
        # -------------------------------------------------------------------------
        # Feature reduction / padding (reuse fitted config)
        # pad_x must be importable in this module scope as in your codebase.
        # -------------------------------------------------------------------------
        max_features = getattr(cfg_host, "max_features", None)
        feature_reduction = getattr(cfg_host, "feature_reduction", None)
        V = getattr(cfg_host, "V", None)  # should be set by fit() when PCA enabled
    
        def _to_tensor(x: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(x, dtype=torch.float32, device=device)
    
        def _prep_x(x_chunk: np.ndarray) -> torch.Tensor:
            x_t = _to_tensor(x_chunk)  # (T, F)
    
            # If PCA reduction enabled, use the *fitted* V (do not recompute here)
            if feature_reduction == "pca" and max_features is not None and x_t.shape[1] > max_features:
                if V is None:
                    raise RuntimeError(
                        "feature_reduction='pca' but V is None after fit(). "
                        "Ensure estimator.fit computes/stores V."
                    )
                v_dev = V.to(device) if hasattr(V, "to") else torch.as_tensor(V, device=device)
                x_t = x_t @ v_dev
    
            x_t = x_t.unsqueeze(0)  # (1, T, F_reduced)
            if max_features is not None:
                x_t = pad_x(x_t, max_features)
            return x_t
    
        def _prep_y(y_chunk: np.ndarray) -> torch.Tensor:
            y_t = torch.as_tensor(y_chunk, dtype=torch.float32, device=device)
            return y_t.view(1, -1, 1)  # (1, S, 1)
    
        total_steps = max(1, epochs * steps_per_epoch)
        step_counter = 0
        log_every = max(1, total_steps // 20)
    
        # -------------------------------------------------------------------------
        # Episodic fine-tune loop
        # -------------------------------------------------------------------------
        for _ep in range(epochs):
            for _ in range(steps_per_epoch):
                step_counter += 1
    
                idx = np.random.permutation(n)
                s_idx = idx[:context_size]
                q_idx = idx[context_size: context_size + query_size]
    
                X_support, y_support = X_np[s_idx], y_np[s_idx]
                X_query, y_query = X_np[q_idx], y_np[q_idx]
    
                x_support_t = _prep_x(X_support)  # (1, S, F)
                x_query_t = _prep_x(X_query)      # (1, Q, F)
    
                x_src = torch.cat([x_support_t, x_query_t], dim=1)  # (1, S+Q, F)
                y_src = _prep_y(y_support)                           # (1, S, 1)
    
                optimizer.zero_grad(set_to_none=True)
    
                pred = torch_model(x_src, y_src, task="reg")
    
                # Force predictions to shape (Q,)
                pred_q = pred
                if pred_q.dim() == 3:
                    pred_q = pred_q.squeeze(0)          # (S+Q,1) or (Q,1) depending on model; common is (Q,1)
                if pred_q.dim() == 2 and pred_q.size(-1) == 1:
                    pred_q = pred_q.squeeze(-1)         # (Q,)
                pred_q = pred_q.reshape(-1)
    
                # Targets: (Q,)
                y_q = torch.as_tensor(y_query, dtype=torch.float32, device=device).reshape(-1)
    
                # Defensive: if model returned S+Q preds, keep only last Q
                if pred_q.numel() != y_q.numel():
                    if pred_q.numel() >= y_q.numel():
                        pred_q = pred_q[-y_q.numel():]
                    else:
                        raise RuntimeError(
                            f"Prediction length mismatch: pred={pred_q.numel()}, target={y_q.numel()}"
                        )
    
                loss = F.mse_loss(pred_q, y_q)
                loss.backward()
    
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(torch_model.parameters(), float(clip_grad_norm))
    
                optimizer.step()
    
                if show_progress and (step_counter % log_every == 0 or step_counter == total_steps):
                    logger.info(
                        f"[TuningManager] TabDPT reg FT: step {step_counter}/{total_steps} | loss={loss.item():.6f}"
                    )
    
        logger.info("[TuningManager] TabDPT regression fine-tuning complete")
        try:
            torch_model.eval()
        except Exception:
            pass
    
        # In case wrapper holds estimator and estimator holds torch module
        try:
            if isinstance(inner_holder, torch.nn.Module):
                inner_holder.eval()
            elif hasattr(inner_holder, "model") and isinstance(inner_holder.model, torch.nn.Module):
                inner_holder.model.eval()
        except Exception:
            pass
    
        return model



    def _finetune_mitra_regression_turn_by_turn(self, model, X_train, y_train, params: dict):
        """
        - Calls model.fit(...) FIRST to reuse the wrapper's exact preprocessing behavior:
          * converts categorical/object columns to numeric codes
          * normalizes y to [0,1] via min-max and stores y_min/y_max
          * stores X_train/y_train caches used by predict()
        - Uses Tab2D forward signature (including padding_obs_query__ kwarg).
        - Shape-safe MSE without broadcasting issues.
        - Leaves model in eval mode.
        """
        import torch
        import numpy as np
        import logging
        from torch.optim import AdamW
        import torch.nn.functional as F
    
        logger = logging.getLogger(__name__)
        logger.info("[TuningManager] Starting Mitra regression fine-tuning (turn-by-turn)")
    
        params = params or {}
        device_str = params.get("device", resolve_device('auto'))
        device = torch.device(device_str)
    
        epochs = int(params.get("epochs", 5))
        steps_per_epoch = int(params.get("steps_per_epoch", 100))
        context_size = int(params.get("context_size", params.get("support_size", 512)))
        query_size = int(params.get("query_size", 128))
    
        lr = float(params.get("lr", params.get("learning_rate", 1e-5)))
        weight_decay = float(params.get("weight_decay", 0.0))
        clip_grad_norm = params.get("clip_grad_norm", None)
        show_progress = bool(params.get("show_progress", True))
        seed = params.get("seed", None)
    
        if seed is not None:
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
    
        # -------------------------------------------------------------------------
        # Resolve underlying torch module (wrapper.model may be torch module, or estimator.model)
        # -------------------------------------------------------------------------
        def _resolve_torch_module(wrapper):
            est = getattr(wrapper, "model", None)
            est2 = getattr(wrapper, "model_", None)
    
            if isinstance(est, torch.nn.Module):
                return est
            if est is not None and hasattr(est, "model") and isinstance(est.model, torch.nn.Module):
                return est.model
            if isinstance(est2, torch.nn.Module):
                return est2
            if est2 is not None and hasattr(est2, "model") and isinstance(est2.model, torch.nn.Module):
                return est2.model
            return None
    
        torch_model = _resolve_torch_module(model)
        if torch_model is None:
            raise AttributeError(
                "MitraRegressorWrapper: could not locate underlying torch module. "
                "Expected either (a) `wrapper.model` to be torch.nn.Module, or "
                "(b) `wrapper.model`/`wrapper.model_` to be an estimator with `.model` torch.nn.Module."
            )

        try:
            # Ensure fit doesn't accidentally run with dropout enabled
            try:
                torch_model.eval()
            except Exception:
                pass
            model.fit(X_train, y_train)
        except Exception as e:
            logger.exception("[TuningManager] model.fit(...) failed before Mitra finetune.")
            raise
    
        X_np = getattr(model, "X_train", None)
        y_norm = getattr(model, "y_train", None)
        if X_np is None or y_norm is None:
            raise RuntimeError(
                "Expected Mitra wrapper to expose caches `X_train` and `y_train` after fit()."
            )
    
        X_np = np.asarray(X_np, dtype=np.float32)
        y_norm = np.asarray(y_norm, dtype=np.float32).reshape(-1)
    
        n = len(X_np)
        if n < 2:
            raise ValueError("Need at least 2 training rows for episodic finetuning.")
    
        if n < (context_size + query_size):
            context_size = max(4, min(context_size, n // 2))
            query_size = max(1, min(query_size, n - context_size))
            logger.warning(f"[TuningManager] Shrunk episode sizes: context={context_size}, query={query_size}")
    
        torch_model.to(device)
        torch_model.train()
    
        optimizer = AdamW(torch_model.parameters(), lr=lr, weight_decay=weight_decay)
    
        def _to_tensor(x: np.ndarray, dtype=torch.float32):
            return torch.as_tensor(x, dtype=dtype, device=device)
    
        total_steps = max(1, epochs * steps_per_epoch)
        step_counter = 0
        log_every = max(1, total_steps // 20)
    
        for _ep in range(epochs):
            for _ in range(steps_per_epoch):
                step_counter += 1
    
                idx = np.random.permutation(n)
                s_idx = idx[:context_size]
                q_idx = idx[context_size: context_size + query_size]
    
                X_support = X_np[s_idx]
                y_support = y_norm[s_idx]
                X_query = X_np[q_idx]
                y_query = y_norm[q_idx]
    
                x_support_t = _to_tensor(X_support).unsqueeze(0)   # (1, S, F)
                y_support_t = _to_tensor(y_support).unsqueeze(0)   # (1, S)
                x_query_t = _to_tensor(X_query).unsqueeze(0)       # (1, Q, F)
    
                b, n_s, f = x_support_t.shape
                n_q = x_query_t.shape[1]
    
                # In repo predict(), these are all-false masks (no padding)
                padding_features = torch.zeros(b, f, dtype=torch.bool, device=device)
                padding_obs_support = torch.zeros(b, n_s, dtype=torch.bool, device=device)
                padding_obs_query = torch.zeros(b, n_q, dtype=torch.bool, device=device)
    
                optimizer.zero_grad(set_to_none=True)
    
                pred = torch_model(
                    x_support=x_support_t,
                    y_support=y_support_t,
                    x_query=x_query_t,
                    padding_features=padding_features,
                    padding_obs_support=padding_obs_support,
                    padding_obs_query__=padding_obs_query,  # <-- double underscore matches Tab2D.forward
                )
    
                # Shape-safe: pred -> (Q,)
                pred_q = pred
                if pred_q.dim() == 3:
                    pred_q = pred_q.squeeze(0)
                if pred_q.dim() == 2 and pred_q.size(-1) == 1:
                    pred_q = pred_q.squeeze(-1)
                if pred_q.dim() == 2 and pred_q.size(0) == 1:
                    pred_q = pred_q.squeeze(0)
                pred_q = pred_q.reshape(-1)
    
                y_q = _to_tensor(y_query).reshape(-1)
    
                # Defensive: if pred returns (S+Q,) or something odd, take last Q
                if pred_q.numel() != y_q.numel():
                    if pred_q.numel() >= y_q.numel():
                        pred_q = pred_q[-y_q.numel():]
                    else:
                        raise RuntimeError(
                            f"Prediction length mismatch: pred={pred_q.numel()}, target={y_q.numel()}"
                        )
    
                loss = F.mse_loss(pred_q, y_q)
                loss.backward()
    
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(torch_model.parameters(), float(clip_grad_norm))
    
                optimizer.step()
    
                if show_progress and (step_counter % log_every == 0 or step_counter == total_steps):
                    logger.info(
                        f"[TuningManager] Mitra reg FT: step {step_counter}/{total_steps} | loss={loss.item():.6f}"
                    )
    
        torch_model.eval()
        logger.info("[TuningManager] Mitra regression fine-tuning complete")

        return model


    def _finetune_tabpfn_regression_turn_by_turn(self, model, X_train, y_train, params: dict):
        import torch
        import numpy as np
        import logging
        from torch.optim import AdamW
    
        from tabtune.models.tabpfn.utils import translate_probs_across_borders
    
        logger = logging.getLogger(__name__)
        logger.info("[TuningManager] Starting TabPFN regression fine-tuning (turn-by-turn)")
    
        params          = params or {}
        device_str      = params.get("device", resolve_device('auto'))
        device          = torch.device(device_str)
        epochs          = int(params.get("epochs", 5))
        steps_per_epoch = int(params.get("steps_per_epoch", 100))
        context_size    = int(params.get("context_size", params.get("support_size", 512)))
        query_size      = int(params.get("query_size", 128))
        lr              = float(params.get("lr", params.get("learning_rate", 1e-5)))
        weight_decay    = float(params.get("weight_decay", 0.0))
        clip_grad_norm  = params.get("clip_grad_norm", None)
        show_progress   = bool(params.get("show_progress", True))
        seed            = params.get("seed", None)
    
        if seed is not None:
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        X_np = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train)
        y_np = y_train.to_numpy() if hasattr(y_train, "to_numpy") else np.asarray(y_train)
        y_np = np.asarray(y_np).reshape(-1)
        n    = len(X_np)
    
        if n < (context_size + query_size):
            context_size = max(4, min(context_size, n // 2))
            query_size   = max(1, min(query_size, n - context_size))
            logger.warning(f"[TuningManager] Shrunk episode sizes: context={context_size}, query={query_size}")

        if not hasattr(model, "interface_config_"):
            model._initialize_model_variables()

        _orig_n_estimators = model.n_estimators
        model.n_estimators = 1
    
        optimizer    = None
        total_steps  = epochs * steps_per_epoch
        step_counter = 0
    
        try:
            for _ep in range(epochs):
                for _ in range(steps_per_epoch):
                    step_counter += 1
    
                    idx   = np.random.permutation(n)
                    s_idx = idx[:context_size]
                    q_idx = idx[context_size: context_size + query_size]
    
                    X_episode = np.concatenate([X_np[s_idx], X_np[q_idx]], axis=0)
                    y_episode = np.concatenate([y_np[s_idx], y_np[q_idx]], axis=0)
    
                    def _split_fn(X_full, y_full):
                        S = context_size
                        return X_full[:S], X_full[S:], y_full[:S], y_full[S:]
    
                    ds = model.get_preprocessed_datasets(
                        X_raw=X_episode,
                        y_raw=y_episode,
                        split_fn=_split_fn,
                        max_data_size=None,
                    )

                    (
                        X_trains_pre,   # List[Tensor] len=1, each (context_size, n_feat) — 2D
                        X_tests_pre,    # List[Tensor] len=1, each (query_size,   n_feat) — 2D
                        y_trains_pre,   # List[Tensor] len=1, each (context_size,)        — 1D
                        y_test_std,     # Tensor (query_size,) — standardised query targets
                        cat_ixs,        # List[List[int]] len=1
                        confs,          # List[EnsembleConfig] len=1
                        raw_space_bardist,
                        znorm_space_bardist,
                        _x_test_raw,
                        _y_test_raw,
                    ) = ds[0]
    
                    X_trains_pre = [
                        x.unsqueeze(0) if (isinstance(x, torch.Tensor) and x.dim() == 2) else x
                        for x in X_trains_pre
                    ]

    
                    X_tests_pre = [
                        x.unsqueeze(0) if (isinstance(x, torch.Tensor) and x.dim() == 2) else x
                        for x in X_tests_pre
                    ]

                    y_trains_pre = [
                        yt.unsqueeze(0) if (isinstance(yt, torch.Tensor) and yt.dim() == 1) else yt
                        for yt in y_trains_pre
                    ]

                    cat_ix_batched = [cat_ixs]
    
                    model.fit_from_preprocessed(
                        X_preprocessed=X_trains_pre,
                        y_preprocessed=y_trains_pre,
                        cat_ix=cat_ix_batched,
                        configs=confs,
                        no_refit=True,
                    )

                    torch_model = model.model_
                    torch_model.to(device)
                    torch_model.train()
    
                    if optimizer is None:
                        optimizer = AdamW(torch_model.parameters(), lr=lr, weight_decay=weight_decay)
    
                    optimizer.zero_grad(set_to_none=True)

                    yq = y_test_std
                    if isinstance(yq, torch.Tensor):
                        if yq.dim() == 2 and yq.size(-1) == 1:
                            yq = yq.squeeze(-1)
                        yq = yq.to(device).reshape(-1)

    
                    _avg, outputs, borders = model.forward(
                        X_tests_pre, use_inference_mode=False
                    )
    

                    std_borders = model.znorm_space_bardist_.borders.to(device)
    
                    transformed_probs = []
                    for probs, b in zip(outputs, borders):
                        p = probs
                        if p.dim() == 3:
                            p = p.squeeze(1)
                        p = translate_probs_across_borders(
                            p,
                            frm=torch.as_tensor(b, device=device),
                            to=std_borders,
                        )
                        transformed_probs.append(p)
    
                    probs_mean  = torch.stack(transformed_probs, dim=0).mean(dim=0)
                    q_probs     = probs_mean[-len(yq):]
                    q_log_probs = (q_probs + 1e-12).log()
    
                    crit = model.znorm_space_bardist_.to(device)
                    loss = crit(q_log_probs, yq).mean()
    
                    loss.backward()
    
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(torch_model.parameters(), float(clip_grad_norm))
    
                    optimizer.step()
    
                    if show_progress and (step_counter % max(1, total_steps // 20) == 0):
                        logger.info(
                            f"[TuningManager] TabPFN reg FT: step {step_counter}/{total_steps}"
                            f" | loss={float(loss.item()):.6f}"
                        )
    
        finally:
            model.n_estimators = _orig_n_estimators
    
        torch_model.eval()
        logger.info("[TuningManager] TabPFN regression fine-tuning complete")
    
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            logger.warning(f"[TuningManager] Post-finetune model.fit failed (predict may break): {e}")
    
        return model