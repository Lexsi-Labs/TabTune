import pandas as pd
import numpy as np
import joblib
import torch
import logging
import json
import os

from ..Dataprocess.data_processor import DataProcessor
from ..TuningManager.tuning import TuningManager

from ..models.tabpfn.classifier import TabPFNClassifier
from ..models.tabicl.sklearn.classifier import TabICLClassifier
from ..models.contexttab.contexttab import ConTextTabClassifier
from ..models.mitra.tab2d import Tab2D
from ..models.orion_bix.sklearn.classifier import OrionBixClassifier
from ..models.tabdpt.classifier import TabDPTClassifier
from ..models.orion_msp.sklearn.classifier import OrionMSPClassifier
from ..models.orionmsp_v15.sklearn.classifier import OrionMSPv15Classifier
from ..models.limix.classifier import LimixClassifier

from ..models.regression.tabpfn.regressor import TabPFNRegressorWrapper
from ..models.regression.contexttab.regressor import ConTextTabRegressorWrapper
from ..models.regression.tabdpt.regressor import TabDPTRegressorWrapper
from ..models.regression.mitra.regressor import MitraRegressorWrapper
from ..models.regression.limix.regressor_wrapper import LimixRegressorWrapper
from ..Dataprocess.regression.base_processor import RegressionDataProcessor

from ..resampling.context_sampling import sample_context, normalize_sampling_strategy_name


# imported for ContextTab cleanup
try:
    from ..models.contexttab.scripts.start_embedding_server import stop_embedding_server
except ImportError:
    stop_embedding_server = None

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
import time 


from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    equal_opportunity_difference
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    log_loss,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score, 
    recall_score,
    recall_score,
    brier_score_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
    max_error,
    mean_squared_log_error
)

from sklearn.preprocessing import LabelEncoder


logger = logging.getLogger(__name__)



class TabularPipeline:
    """
    The complete TabularPipeline with a robust constructor that
    explicitly handles parameters for each component and uses late initialization
    for complex models like ContextTab and Mitra.
    """
    def __init__(self, model_name: str, 
                 task_type: str = 'classification', 
                 tuning_strategy: str = 'inference', 
                 tuning_params: dict | None = None,
                 processor_params: dict | None = None,
                 model_params: dict | None = None,
                 model_checkpoint_path: str | None = None,
                 finetune_mode: str = 'meta-learning'):

        print("\n" + "="*80)
        print(r"""
  ████████╗ █████╗ ██████╗  ████████╗██╗   ██╗███╗   ██╗███████╗
  ╚══██╔══╝██╔══██╗██╔══██╗ ╚══██╔══╝██║   ██║████╗  ██║██╔════╝
     ██║   ███████║██████╔╝    ██║   ██║   ██║██╔██╗ ██║█████╗  
     ██║   ██╔══██║██╔══██╗    ██║   ██║   ██║██║╚██╗██║██╔══╝  
     ██║   ██║  ██║██████╔╝    ██║   ╚██████╔╝██║ ╚████║███████╗
     ╚═╝   ╚═╝  ╚═╝╚═════╝     ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚══════╝
        """)
        print("Unified Library for Fine-Tuning and Inference of Foundational Tabular Models")
        print("="*80 + "\n")
        
        self.model_name = model_name
        self.task_type = task_type
        self.tuning_strategy = tuning_strategy
        self.tuning_params = tuning_params or {}
        self.model_params = model_params or {}
        self.processor_params = processor_params or {}
        
        self.model = None 
        self.model_checkpoint_path = model_checkpoint_path
        self.finetune_mode = finetune_mode

        # -------------------------
        # NEW: Context sampling / resampling integration (pipeline-level)
        # These params are popped out of processor_params so DataProcessor doesn't see them.
        # -------------------------
        proc_params = dict(self.processor_params)  # copy so we don't mutate user dict

        self.context_sampling_params = {
            # allow either key name:
            "context_sampling_strategy": normalize_sampling_strategy_name(
                proc_params.pop("context_sampling_strategy", None)
                or proc_params.pop("context_resampling_strategy", None)
            ),
            "context_size": (
                proc_params.pop("context_size", None)
                or proc_params.pop("sampling_context_size", None)
            ),
            "strat_set": proc_params.pop("strat_set", 10),
            "hybrid_ratio": proc_params.pop("hybrid_ratio", 0.7),
            "sampling_seed": proc_params.pop("sampling_seed", 42),
            "allow_replacement": proc_params.pop("allow_replacement", True),
            # knobs
            "kmeans_centers": proc_params.pop("kmeans_centers", 2000),
            "min_pos": proc_params.pop("min_pos", 50),
            "oversample_weight": proc_params.pop("oversample_weight", 5.0),
        }

        # Pass task_type to DataProcessor
        proc_params['task_type'] = self.task_type
        self.processor = DataProcessor(model_name=self.model_name, **proc_params)

        self.tuner = TuningManager()

        # Validate regression mode: only inference is supported
        # Allow regression finetune for ContextTab (others still inference-only for now)
        if self.task_type == 'regression' and self.tuning_strategy != 'inference':
            allowed = {"ContextTab", "Limix", "TabDPT","Mitra","TabPFN"}
            if not (self.model_name in allowed and self.tuning_strategy == "finetune"):
                raise ValueError(
                    f"Regression finetuning is not enabled for model '{self.model_name}'. "
                    f"Enabled: {sorted(allowed)}. Got task_type='{self.task_type}', tuning_strategy='{self.tuning_strategy}'."
        )


        if self.tuning_strategy in ('finetune', 'peft'):
            self.tuning_params['finetune_mode'] = self.finetune_mode
        
        # Model initialization based on task_type
        if self.task_type == 'regression':
            # Regression models (inference-only)
            if self.model_name == 'TabPFN':
                device = self.tuning_params.get('device', self.model_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
                config = {'device': device, 'ignore_pretraining_limits': True, 'tuning_strategy': 'inference'}
                config.update(self.model_params)
                logger.info(f"[Pipeline] Config: {config}")
                self.model = TabPFNRegressorWrapper(**config)
            elif self.model_name == 'ContextTab':
                config = {'tuning_strategy': self.tuning_strategy}
                config.update(self.model_params)
                self.model = ConTextTabRegressorWrapper(**config)
            elif self.model_name == 'TabDPT':
                device = self.tuning_params.get('device', self.model_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
                config = {
                    'device': device,
                    'compile': True,
                    'use_flash': True,
                    'normalizer': 'standard',
                    'missing_indicators': False,
                    'clip_sigma': 4.0,
                    'feature_reduction': 'pca',
                    'faiss_metric': 'l2',
                    'verbose': True,
                    'tuning_strategy': self.tuning_strategy
                }
                config.update(self.model_params)
                self.model = TabDPTRegressorWrapper(**config)
            elif self.model_name == 'Limix':
                device = self.tuning_params.get('device', self.model_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
                config = {
                    'device': device,
                    'repo_id': 'stableai-org/LimiX-16M',
                    'filename': 'LimiX-16M.ckpt',
                    'tuning_strategy': self.tuning_strategy
                }
                config.update(self.model_params)
                self.model = LimixRegressorWrapper(**config)
            elif self.model_name == 'Mitra':
                device = self.tuning_params.get('device', self.model_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
                config = {
                    'dim': 512,  # Match AutoGluon's default architecture
                    'n_layers': 12,  # Match AutoGluon's default architecture
                    'n_heads': 4,  # Match AutoGluon's default architecture
                    'use_pretrained_weights': 'auto',  # Auto-download from HuggingFace
                    'path_to_weights': '',
                    'device': device,
                    'tuning_strategy': 'inference',
                    'cache_dir': self.model_params.get('cache_dir', None)
                }
                config.update(self.model_params)
                self.model = MitraRegressorWrapper(**config)
            else:
                raise ValueError(f"Model '{self.model_name}' does not support regression. Supported models: TabPFN, ContextTab, TabDPT, Mitra, Limix")
        else:
            # Classification models (existing code - unchanged)
            if self.model_name in ['TabPFN']:
                device = self.tuning_params.get('device', self.model_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
                config = {'device': device, 'ignore_pretraining_limits': True}
                config.update(self.model_params)
                logger.info(f"[Pipeline] Config: {config}")
                self.model = TabPFNClassifier(**config)
                if self.tuning_strategy in ['finetune', 'peft'] and hasattr(self.model, '_initialize_model_variables'):
                    self.model._initialize_model_variables()

            elif self.model_name == 'ContextTab':
                self.model = ConTextTabClassifier(**self.model_params)
    
            elif self.model_name in ['TabICL', 'OrionBix','OrionMSP', 'OrionMSPv1.5']:
                device = self.tuning_params.get('device', self.model_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
                config = {'n_jobs': 1, 'device': device}
                config.update(self.model_params)
                if self.model_name == 'TabICL':
                    self.model = TabICLClassifier(**config)
                    if self.tuning_strategy == 'finetune':
                        self.model._load_model()
                elif self.model_name == 'OrionMSP':
                    self.model = OrionMSPClassifier(**config)
                    if self.tuning_strategy == 'finetune':
                        self.model._load_model()
                elif self.model_name == 'OrionMSPv1.5':
                    self.model = OrionMSPv15Classifier(**config)
                    if self.tuning_strategy == 'finetune':
                        self.model._load_model()
                else:
                    self.model = OrionBixClassifier(**config)
                    if self.tuning_strategy == 'finetune':
                        self.model._load_model()

            elif self.model_name == 'TabDPT':
                # Use GPU if available, otherwise fall back to CPU
                device = self.tuning_params.get('device', self.model_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
                config = {
                    'device': device,
                    'compile': True,  # Disable compilation to avoid GPU issues
                    'use_flash': True,  # Disable flash attention to avoid kernel issues
                    'normalizer': 'standard',
                    'missing_indicators': False,
                    'clip_sigma': 4.0,
                    'feature_reduction': 'pca',
                    'faiss_metric': 'l2',
                    # Inference parameters with GPU-friendly defaults
                    'n_ensembles': 8,
                    'temperature': 0.8,
                    'context_size': 512,
                    'permute_classes': True,
                    'seed': None,
                }
                config.update(self.model_params)  # All parameters now valid
                self.model = TabDPTClassifier(**config)

            elif self.model_name == 'Limix':
                device = self.tuning_params.get('device', self.model_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
                config = {'device': device}
                config.update(self.model_params)
                self.model = LimixClassifier(**config)

            # Handle models that require late initialization (processor needs to be fit first)
            elif self.model_name == 'Mitra':
                # Mitra classification will be initialized later in fit() method
                pass
            elif self.model_name == 'APT':
                # APT will be initialized later if needed
                pass
            else:
                raise ValueError(f"Model '{self.model_name}' not supported for classification.")


        if self.model is not None and self.model_checkpoint_path:
            logger.info(f"[Pipeline] Attempting to load model state from checkpoint: {self.model_checkpoint_path}")
            try:
                # Determine the underlying torch model attribute
                torch_model = None
                if hasattr(self.model, 'model_'): # For TabPFN, TabICL, OrionMSP, OrionBix
                    torch_model = self.model.model_
                elif hasattr(self.model, 'model'): # For ContextTab, TabDPT
                    torch_model = self.model.model
                elif isinstance(self.model, torch.nn.Module): # For Mitra (Tab2D)
                    torch_model = self.model

                if torch_model:
                    torch_model.load_state_dict(torch.load(self.model_checkpoint_path, map_location=torch.device('cpu')))
                    logger.info(f"[Pipeline] Successfully loaded checkpoint for {type(self.model)._name_}.")
                else:
                    logger.warning(f"[Pipeline] Could not determine the underlying torch model for {type(self.model)._name_} to load checkpoint.")
            except Exception as e:
                logger.error(f"[Pipeline] Failed to load checkpoint: {e}")
            
        self._is_fitted = False
        self.X_train_processed_ = None
        self.y_train_processed_ = None
        
        # Store the sampled context used during fit (for debugging / reproducibility)
        self.X_context_train_ = None
        self.y_context_train_ = None
        
        logger.info(f"[Pipeline] TabularPipeline initialized for model '{self.model_name}', task '{self.task_type}', with strategy '{self.tuning_strategy}'")
        ("TabTune - Unified Library for fine-tuning and inference of Foundational Tabular Models")

    def __del__(self):
        """Cleanup method to properly shut down resources when pipeline is destroyed."""
        # ContextTab ZMQ server cleanup is handled automatically by atexit.register()
        # in the start_embedding_server function, so no manual cleanup needed
        pass

    # -------------------------
    # NEW: context sampling helper
    # -------------------------
    def _apply_context_sampling_if_configured(self, X: pd.DataFrame, y: pd.Series):
        cfg = self.context_sampling_params or {}
        context_size = cfg.get("context_size", None)

        # Not configured -> no-op
        if context_size is None:
            return X, y

        try:
            context_size_int = int(context_size)
        except Exception:
            logger.warning(f"[Pipeline] Invalid context_size={context_size}; skipping context sampling.")
            return X, y

        if context_size_int <= 0:
            return X, y

        strategy = normalize_sampling_strategy_name(cfg.get("context_sampling_strategy", "uniform"))
        logger.info(
            f"[Pipeline] Applying context sampling: strategy='{strategy}', context_size={context_size_int}, task_type={self.task_type}"
        )

        Xs, ys = sample_context(
            X=X,
            y=y,
            task_type=self.task_type,
            strategy=strategy,
            context_size=context_size_int,
            strat_set=int(cfg.get("strat_set", 10)),
            hybrid_ratio=float(cfg.get("hybrid_ratio", 0.7)),
            seed=int(cfg.get("sampling_seed", 42)),
            allow_replacement=bool(cfg.get("allow_replacement", True)),
            kmeans_centers=int(cfg.get("kmeans_centers", 2000)),
            min_pos=int(cfg.get("min_pos", 50)),
            oversample_weight=float(cfg.get("oversample_weight", 5.0)),
        )

        logger.info(f"[Pipeline] Context sampling complete: {len(X)} -> {len(Xs)} rows")
        return Xs, ys


    def fit(self, X: pd.DataFrame, y: pd.Series):

        # Store full raw training data (always)
        self.X_raw_train = X.copy()
        self.y_raw_train = y.copy()
        
        # NEW: Apply context sampling BEFORE any processor/model fit
        X_fit, y_fit = self._apply_context_sampling_if_configured(X, y)
        self.X_context_train_ = X_fit.copy()
        self.y_context_train_ = y_fit.copy()
        
        logger.info("[Pipeline] Starting fit process")

        # Handle regression models (inference-only)
        if self.task_type == 'regression':
            logger.info("[Pipeline] Fitting regression model (inference-only mode)")
            
            # ContextTab handles all preprocessing internally - skip TabTune preprocessing
            if isinstance(self.model, ConTextTabRegressorWrapper):
                if self.tuning_strategy == "inference":
                    logger.info(f"[Pipeline] Fitting {self.model_name} regressor in inference mode (raw data)")
                    self.model.fit(X_fit, y_fit)
                    self._is_fitted = True
                    logger.info("[Pipeline] Fit process complete")
                    return self
                if self.tuning_strategy == "finetune":
                    logger.info(f"[Pipeline] Fine-tuning {self.model_name} regressor (raw data -> TuningManager)")
                    self.model = self.tuner.tune(
                        self.model,
                        X_fit,
                        y_fit,
                        strategy=self.tuning_strategy,
                        params=self.tuning_params,
                        processor=None,   # ContextTab uses raw data; processor not needed
                    )
                    self._is_fitted = True
                    logger.info("[Pipeline] Fit process complete")
                    return self

                raise ValueError(f"Unsupported tuning_strategy for ContextTab regression: {self.tuning_strategy}")

            
            # Limix handles all preprocessing internally - skip TabTune preprocessing
            if isinstance(self.model, LimixRegressorWrapper):
                if self.tuning_strategy == "inference":
                    logger.info(f"[Pipeline] Fitting {self.model_name} regressor in inference mode (raw data, no preprocessing)")
                    self.model.fit(X_fit, y_fit)
                    self._is_fitted = True
                    logger.info("[Pipeline] Fit process complete")
                    return self

                if self.tuning_strategy == "finetune":
                    logger.info(f"[Pipeline] Fine-tuning {self.model_name} regressor (raw data -> TuningManager)")
                    self.model = self.tuner.tune(
                        self.model,
                        X_fit,
                        y_fit,
                        strategy="finetune",
                        params=self.tuning_params,
                        processor=None,  # Limix uses raw data
                    )
                    self._is_fitted = True
                    logger.info("[Pipeline] Fit process complete")
                    return self

                raise ValueError(f"Unsupported tuning_strategy for Limix regression: {self.tuning_strategy}")

            
            # For other regression models, use preprocessing (fit on sampled context)
            logger.info("[Pipeline] Fitting regression processor...")
            self.processor.fit(X_fit, y_fit)

            processed = self.processor.transform(X_fit, y_fit)
            if isinstance(processed, tuple):
                X_to_tune, y_to_tune = processed
            else:
                X_to_tune = processed
                y_to_tune = y_fit
 
            # Cache processed train for consistency / debugging
            self.X_train_processed_, self.y_train_processed_ = X_to_tune, y_to_tune
 
              # ---- TabDPT regression ----
            if isinstance(self.model, TabDPTRegressorWrapper):
                if self.tuning_strategy == "inference":
                    logger.info(f"[Pipeline] Fitting {self.model_name} regressor in inference mode (processed data)")
                    self.model.fit(X_to_tune, y_to_tune)
                    self._is_fitted = True
                    logger.info("[Pipeline] Fit process complete")
                    return self
 
                if self.tuning_strategy == "finetune":
                    logger.info(f"[Pipeline] Fine-tuning {self.model_name} regressor (processed data -> TuningManager)")
                    self.model = self.tuner.tune(
                        self.model,
                        X_to_tune,
                        y_to_tune,
                        strategy="finetune",
                        params=self.tuning_params,
                        processor=self.processor,
                     )
                    self._is_fitted = True
                    logger.info("[Pipeline] Fit process complete")
                    return self
                raise ValueError(f"Unsupported tuning_strategy for TabDPT regression: {self.tuning_strategy}")

            if isinstance(self.model, MitraRegressorWrapper):
                if self.tuning_strategy == "inference":
                    logger.info(f"[Pipeline] Fitting {self.model_name} regressor in inference mode (processed data)")
                    # IMPORTANT: use processed data so train cache matches what predict will see
                    self.model.fit(X_to_tune, y_to_tune)
                    self._is_fitted = True
                    logger.info("[Pipeline] Fit process complete")
                    return self
            
                if self.tuning_strategy == "finetune":
                    logger.info(f"[Pipeline] Fine-tuning {self.model_name} regressor (processed data -> TuningManager)")
                    self.model = self.tuner.tune(
                        self.model,
                        X_to_tune,
                        y_to_tune,
                        strategy="finetune",
                        params=self.tuning_params,
                        processor=self.processor,
                    )
                    self._is_fitted = True
                    logger.info("[Pipeline] Fit process complete")
                    return self
            
                raise ValueError(f"Unsupported tuning_strategy for Mitra regression: {self.tuning_strategy}")

            
            # Regression models use inference mode only
            # ---- TabPFN regression ----
            if isinstance(self.model, TabPFNRegressorWrapper):
                if self.tuning_strategy == "inference":
                    logger.info(f"[Pipeline] Fitting {self.model_name} regressor in inference mode (processed data)")
                    self.model.fit(X_to_tune, y_to_tune)
                    self._is_fitted = True
                    logger.info("[Pipeline] Fit process complete")
                    return self
            
                if self.tuning_strategy == "finetune":
                    logger.info(f"[Pipeline] Fine-tuning {self.model_name} regressor (processed data -> TuningManager)")
                    self.model = self.tuner.tune(
                        self.model,
                        X_to_tune,
                        y_to_tune,
                        strategy="finetune",
                        params=self.tuning_params,
                        processor=self.processor,
                    )
                    self._is_fitted = True
                    logger.info("[Pipeline] Fit process complete")
                    return self
            
                raise ValueError(f"Unsupported tuning_strategy for TabPFN regression: {self.tuning_strategy}")


        # Special handling for models that are TRULY self-contained and do not need the pipeline's processor for inference
        if self.tuning_strategy == 'inference' and isinstance(self.model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, LimixClassifier, OrionMSPv15Classifier)):
            logger.info("[Pipeline] Handing off to TuningManager for inference setup.")
            self.processor.fit(X_fit, y_fit)
            self.model = self.tuner.tune(self.model, X_fit, y_fit, strategy=self.tuning_strategy)
            self._is_fitted = True
            logger.info("[Pipeline] Fit process complete")
            return self

        # For ALL other models and strategies (including ConTextTab), we must fit the DataProcessor first.
        logger.info("[Pipeline] Fitting data processor...")
        self.processor.fit(X_fit, y_fit) 

        # Handle ConTextTab inference AFTER the processor has been fitted
        if self.tuning_strategy == 'inference' and isinstance(self.model, ConTextTabClassifier):
            logger.info(f"[Pipeline] Handing off to TuningManager for inference setup for {self.model_name}")
            self.model = self.tuner.tune(self.model, X_fit, y_fit, strategy=self.tuning_strategy)
            self._is_fitted = True
            logger.info("[Pipeline] Fit process complete")
            return self

        # Late initialization for models that need info from the fitted processor (classification only)
        if self.model is None and self.task_type == 'classification':
            logger.info("[Pipeline] Performing late initialization of the model...")
            if self.model_name == 'Mitra':
                n_classes = len(self.processor.custom_preprocessor_.label_encoder_.classes_)
                device = self.tuning_params.get('device', self.model_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
                config = {'dim': 256, 'n_layers': 6, 'n_heads': 8, 'task': 'CLASSIFICATION', 'dim_output': n_classes, 'use_pretrained_weights': False, 'path_to_weights': '', 'device': device}
                config.update(self.model_params)
                self.model = Tab2D(**config)

                if self.model_checkpoint_path:
                    logger.info(f"[Pipeline] Attempting to load model state from checkpoint for late-initialized model: {self.model_checkpoint_path}")
                    try:
                        self.model.load_state_dict(torch.load(self.model_checkpoint_path, map_location=torch.device()))
                        logger.info(f"[Pipeline] Successfully loaded checkpoint for {type(self.model)._name_}.")
                    except Exception as e:
                        logger.error(f"[Pipeline] Failed to load checkpoint: {e}")

        if hasattr(self.model, 'to'):
            device_str = self.tuning_params.get('device', self.model_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            device = torch.device(device_str)
            self.model.to(device)
            if self.model_name == 'Mitra':
                try:
                    setattr(self.model, 'device_type', device_str)
                except Exception:
                    pass
            if isinstance(self.model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, OrionMSPv15Classifier)):
                self.model.device = device


        if isinstance(self.model, ConTextTabClassifier) and self.tuning_strategy in ['finetune']:
            logger.info("[Pipeline] Preparing raw data for ConTextTab fine-tuning")
            if not isinstance(X_fit, pd.DataFrame):
                X_to_tune = pd.DataFrame(X_fit)
            else:
                X_to_tune = X_fit.copy()
            if not isinstance(y_fit, pd.Series):
                y_to_tune = pd.Series(y_fit)
            else:
                y_to_tune = y_fit.copy()
        else:
            logger.info("[Pipeline] Transforming data for model tuning...")
            processed_data = self.processor.transform(X_fit, y_fit)
            if isinstance(processed_data, tuple):
                self.X_train_processed_, self.y_train_processed_ = processed_data
            else:
                self.X_train_processed_ = processed_data
                if hasattr(self.processor, 'custom_preprocessor_') and hasattr(self.processor.custom_preprocessor_, 'label_encoder_') and self.processor.custom_preprocessor_.label_encoder_ is not None:
                    self.y_train_processed_ = self.processor.custom_preprocessor_.label_encoder_.transform(y_fit)
                else:
                    self.y_train_processed_ = y_fit 
                    
            if self.task_type == 'regression':
                if hasattr(self.processor, 'regression_processor_') and self.processor.regression_processor_:
                    self.y_train_processed_ = self.processor.regression_processor_.transform(y_fit)
                else:
                    self.y_train_processed_ = y_fit

            X_to_tune, y_to_tune = self.X_train_processed_, self.y_train_processed_


        logger.info("[Pipeline] Handing off to Tuning Manager")

        if self.tuning_strategy == "peft":
            logger.info("[Pipeline] PEFT MODE: Attempting Parameter-Efficient Fine-Tuning")
            logger.info("[Pipeline] NOTE: PEFT may have compatibility limitations with tabular models")
            logger.info("[Pipeline] FALLBACK: Base fine-tuning will be used if PEFT fails")
            
        self.model = self.tuner.tune(
            self.model, 
            X_to_tune, 
            y_to_tune, 
            strategy=self.tuning_strategy, 
            params=self.tuning_params, 
            processor=self.processor
        )

        if isinstance(self.model, TabDPTClassifier) and self.tuning_strategy in ['finetune', 'peft']:
            logger.info("[Pipeline] Finalizing TabDPT setup after fine-tuning")
            self.model.num_classes = len(np.unique(y_to_tune))
            self.model.fit(X_to_tune, y_to_tune)

        self._is_fitted = True
        logger.info("[Pipeline] Fit process complete")
        if self.tuning_strategy == "peft":
            logger.info("[Pipeline] PEFT STATUS SUMMARY")
            logger.info("[Pipeline] LoRA adapters were applied to the model")
            logger.warning("[Pipeline] Note: PEFT compatibility with tabular models is experimental")
            logger.info("[Pipeline] If you encounter issues, try inference strategy for full compatibility")
            logger.info("[Pipeline] See documentation for more details on PEFT limitations")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("You must call fit() on the pipeline before calling predict().")
        
        logger.info("[Pipeline] Starting prediction")
        
        # Handle regression models
        if self.task_type == 'regression':
            if isinstance(self.model, (ConTextTabRegressorWrapper, LimixRegressorWrapper)):
                predictions = self.model.predict(X)
                return predictions
            else:
                X_processed = self.processor.transform(X)
                return self.model.predict(X_processed)
                
            raise ValueError(f"Unsupported regression model: {self.model_name}")

        if hasattr(self.model, 'model') and isinstance(self.model.model, torch.nn.Module):
            self.model.model.eval()
        elif hasattr(self.model, 'model_') and isinstance(self.model.model_, torch.nn.Module):
            self.model.model_.eval()

        if isinstance(self.model, TabPFNClassifier):
            if self.tuning_strategy in ['finetune', 'peft']:
                logger.debug("[Pipeline] Setting TabPFN inference context (without refitting weights)...")
                saved_weights = self.model.model_.state_dict()
                self.model.model_.load_state_dict(saved_weights)
                self.model.fit(self.X_train_processed_, self.y_train_processed_)
                logger.debug("[Pipeline] Restored fine-tuned weights after context setup")
        
            X_processed = self.processor.transform(X)
            return self.model.predict(X_processed)
        

        if isinstance(self.model, TabDPTClassifier):
            X_processed = self.processor.transform(X)
            predictions_raw = self.model.predict(X_processed)
            predictions = self.processor.custom_preprocessor_.label_encoder_.inverse_transform(predictions_raw)
            return predictions
            

        if isinstance(self.model, (ConTextTabClassifier)):
            logger.debug(f"[Pipeline] Using model's native in-context prediction for {type(self.model).__name__}")
            predictions = self.model.predict(X)
            
        elif isinstance(self.model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, LimixClassifier, OrionMSPv15Classifier)):
            logger.debug(f"[Pipeline] Using model's native in-context prediction for {type(self.model).__name__}")  
            X_processed = self.processor.transform(X)
            
            if self.tuning_strategy == 'inference':
                predictions = self.model.predict(X)
            else:
                label_encoder = self.processor.custom_preprocessor_.label_encoder_
                known_class = label_encoder.classes_[0]
                y_dummy = pd.Series([known_class] * len(X))
                X_query, _ = self.processor.transform(X, y_dummy)
                if not isinstance(X_query, pd.DataFrame):
                    cols = None
                    if hasattr(self.processor, "feature_names_") and self.processor.feature_names_ is not None:
                        cols = list(self.processor.feature_names_)
                    elif hasattr(X, "columns"):
                        cols = list(X.columns)
                    if cols is not None and hasattr(X_query, "shape") and X_query.shape[1] != len(cols):
                        cols = None
                    X_query = pd.DataFrame(X_query, columns=cols)
                predictions = self.model.predict(X_query)
            
            if self.tuning_strategy in ['finetune', 'peft'] and hasattr(self.processor, 'custom_preprocessor_') and hasattr(self.processor.custom_preprocessor_, 'label_encoder_'):
                predictions = self.processor.custom_preprocessor_.label_encoder_.inverse_transform(predictions)

        
        elif self.model_name == 'Mitra':
            logger.debug("[Pipeline] Using in-context prediction for Mitra (Tab2D)")
            label_encoder = self.processor.custom_preprocessor_.label_encoder_
            known_class = label_encoder.classes_[0]
            y_dummy = pd.Series([known_class] * len(X))

            X_query, _ = self.processor.transform(X, y_dummy)
            
            X_support, y_support = self.X_train_processed_, self.y_train_processed_
            
            device_str = self.tuning_params.get('device', self.model_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
            device = device_str
            
            X_support_t = torch.tensor(X_support, dtype=torch.float32).unsqueeze(0).to(device)
            y_support_t = torch.tensor(y_support, dtype=torch.long).unsqueeze(0).to(device)
            X_query_t = torch.tensor(X_query, dtype=torch.float32).unsqueeze(0).to(device)
            
            b, f = X_support_t.shape[0], X_support_t.shape[2]
            padding_features = torch.zeros(b, f, dtype=torch.bool, device=device)
            padding_obs_support = torch.zeros_like(y_support_t, dtype=torch.bool, device=device)
            padding_obs_query = torch.zeros(b, X_query_t.shape[1], dtype=torch.bool, device=device)
            
            self.model.eval()
            with torch.no_grad():
                logits = self.model(
                    x_support=X_support_t, y_support=y_support_t, x_query=X_query_t,
                    padding_features=padding_features, padding_obs_support=padding_obs_support,
                    padding_obs_query__=padding_obs_query
                )
            
            predictions_raw = logits.squeeze(0).cpu().numpy().argmax(axis=-1)
            predictions = self.processor.custom_preprocessor_.label_encoder_.inverse_transform(predictions_raw)
            
        else: 
            logger.debug("[Pipeline] Applying learned transformations to new data")
            X_processed = self.processor.transform(X)
            logger.debug("[Pipeline] Getting predictions from the model")
            predictions = self.model.predict(X_processed)
        return predictions

    # NOTE:
    # Everything below this point is unchanged from your file.
    # (I left it intact to avoid accidental behavioral regressions.)
    # -----------------------------------------------------------------

    def predict_quantiles(self, X: pd.DataFrame, quantiles: list[float] = None) -> dict:
        """
        Predict quantiles for regression tasks.
        
        Currently only supported for TabPFN regression models.
        
        Args:
            X: Input data for prediction
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
                     Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        Returns:
            dict: Dictionary mapping quantile values to prediction arrays
                 e.g., {0.1: array([...]), 0.5: array([...]), ...}
        """
        if not self._is_fitted:
            raise RuntimeError("You must call fit() on the pipeline before calling predict_quantiles().")
        
        if self.task_type != 'regression':
            raise ValueError("predict_quantiles() is only available for regression tasks.")
        
        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Validate quantiles
        if not all(0 <= q <= 1 for q in quantiles):
            raise ValueError("All quantiles must be between 0 and 1.")
        
        logger.info(f"[Pipeline] Starting quantile prediction for quantiles: {quantiles}")
        
        # Only TabPFN supports quantiles currently
        if isinstance(self.model, TabPFNRegressorWrapper):
            quantile_predictions = self.model.predict(X, output_type="quantiles", quantiles=quantiles)
            result = {q: pred for q, pred in zip(quantiles, quantile_predictions)}
            logger.info("[Pipeline] Quantile prediction complete")
            return result
        else:
            raise NotImplementedError(
                f"Quantile prediction is not yet supported for {self.model_name}. "
                "Currently only TabPFN supports quantile predictions."
            )

    def predict_intervals(self, X: pd.DataFrame, confidence: float = 0.95) -> dict:
        """
        Predict confidence intervals for regression tasks.
        
        Currently only supported for TabPFN regression models.
        """
        if not self._is_fitted:
            raise RuntimeError("You must call fit() on the pipeline before calling predict_intervals().")
        
        if self.task_type != 'regression':
            raise ValueError("predict_intervals() is only available for regression tasks.")
        
        if not 0 < confidence < 1:
            raise ValueError("Confidence must be between 0 and 1.")
        
        logger.info(f"[Pipeline] Starting interval prediction with {confidence*100:.1f}% confidence")
        
        alpha = 1 - confidence
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        if isinstance(self.model, TabPFNRegressorWrapper):
            mean_pred = self.predict(X)
            quantiles = [lower_quantile, upper_quantile]
            quantile_preds = self.model.predict(X, output_type="quantiles", quantiles=quantiles)
            lower = quantile_preds[0]
            upper = quantile_preds[1]
            
            result = {
                'mean': mean_pred,
                'lower': lower,
                'upper': upper,
                'confidence': confidence
            }
            
            logger.info("[Pipeline] Interval prediction complete")
            return result
        else:
            raise NotImplementedError(
                f"Prediction intervals are not yet supported for {self.model_name}. "
                "Currently only TabPFN supports prediction intervals."
            )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts class probabilities for the input data.
        Required for calculating AUC score.
        """
        if not self._is_fitted:
            raise RuntimeError("You must call fit() on the pipeline before calling predict_proba().")
        
        logger.info("[Pipeline] Starting probability prediction")

        if hasattr(self.model, 'model') and isinstance(self.model.model, torch.nn.Module):
            self.model.model.eval()
        elif hasattr(self.model, 'model_') and isinstance(self.model.model_, torch.nn.Module):
            self.model.model_.eval()

        if isinstance(self.model, TabDPTClassifier):
            logger.debug("[Pipeline] Using TabDPT's internal predict_proba")
            X_processed = self.processor.transform(X)
            return self.model.ensemble_predict_proba(X_processed)

        elif isinstance(self.model, TabPFNClassifier):
            if self.tuning_strategy in ['finetune', 'peft']:
                logger.debug("[Pipeline] Setting TabPFN inference context for proba...")
                self.model.fit(self.X_train_processed_, self.y_train_processed_)
            
            X_processed = self.processor.transform(X)
            return self.model.predict_proba(X_processed)
            
        
        if isinstance(self.model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, ConTextTabClassifier, LimixClassifier, OrionMSPv15Classifier)):
            logger.debug("[Pipeline] Using model's native predict_proba method")
            
            X_processed = self.processor.transform(X)
            if isinstance(self.model, (ConTextTabClassifier)):
                 return self.model.predict_proba(X)

            if isinstance(self.model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, LimixClassifier, OrionMSPv15Classifier)):
                if self.tuning_strategy == 'inference':
                    return self.model.predict_proba(X)
                else:
                    label_encoder = self.processor.custom_preprocessor_.label_encoder_
                    known_class = label_encoder.classes_[0]
                    y_dummy = pd.Series([known_class] * len(X))
                    
                    X_query, _ = self.processor.transform(X, y_dummy)
                    if not isinstance(X_query, pd.DataFrame):
                        cols = None
                        if hasattr(self.processor, "feature_names_") and self.processor.feature_names_ is not None:
                            cols = list(self.processor.feature_names_)
                        elif hasattr(X, "columns"):
                            cols = list(X.columns)
                        if cols is not None and hasattr(X_query, "shape") and X_query.shape[1] != len(cols):
                            cols = None
                        X_query = pd.DataFrame(X_query, columns=cols)
                    return self.model.predict_proba(X_query)
           
            return self.model.predict_proba(X_processed)

        label_encoder = self.processor.custom_preprocessor_.label_encoder_
        known_class = label_encoder.classes_[0]
        y_dummy = pd.Series([known_class] * len(X))
        X_query, _ = self.processor.transform(X, y_dummy)
        X_support = self.X_train_processed_
        y_support = self.y_train_processed_
        
        device = next(self.model.parameters()).device

        X_support_t = torch.tensor(X_support, dtype=torch.float32).unsqueeze(0).to(device)
        y_support_t = torch.tensor(y_support, dtype=torch.long).unsqueeze(0).to(device)
        X_query_t = torch.tensor(X_query, dtype=torch.float32).unsqueeze(0).to(device)

        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, Tab2D):
                logger.debug("[Pipeline] Generating probabilities for Mitra (Tab2D)")
                b, f = X_support_t.shape[0], X_support_t.shape[2]
                padding_features = torch.zeros(b, f, dtype=torch.bool, device=device)
                padding_obs_support = torch.zeros_like(y_support_t, dtype=torch.bool, device=device)
                padding_obs_query = torch.zeros(b, X_query_t.shape[1], dtype=torch.bool, device=device)
                logits = self.model(
                    x_support=X_support_t, y_support=y_support_t, x_query=X_query_t,
                    padding_features=padding_features, padding_obs_support=padding_obs_support,
                    padding_obs_query__=padding_obs_query
                )
                probabilities = torch.softmax(logits.squeeze(0), dim=-1).cpu().numpy()
            else:
                 if self.model_name == 'Mitra':
                    raise NotImplementedError("predict_proba is not implemented for Mitra (Tab2D)")
                    raise NotImplementedError(f"predict_proba is not implemented for model type {type(self.model).__name__}")
        
        logger.info("[Pipeline] Probability prediction complete")
        return probabilities

    ############### Helpers #############################
    def _get_model_class_labels(self):
        """
        Best-effort to recover the class label order that predict_proba columns use.
        """
        if hasattr(self.model, "classes_"):
            return list(self.model.classes_)
        if hasattr(self.model, "y_encoder_") and hasattr(self.model.y_encoder_, "classes_"):
            return list(self.model.y_encoder_.classes_)
        if hasattr(self.model, "classes_"):
            return list(self.model.classes_)
        return None

    def _align_proba_to_encoder(self, probabilities, label_encoder):
        """
        Ensure the columns of `probabilities` line up with label_encoder.classes_.
        Returns a 2D array with shape (n_samples, K) where K==len(label_encoder.classes_).
        If the model returns only the positive-class column for binary, we upcast it
        to two columns [P(class0), P(class1)] assuming classes_ are [0,1] after encoding.
        """
        import numpy as np

        if probabilities is None:
            logger.warning("[Pipeline] Probabilities are None in _align_proba_to_encoder")
            return None
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)
        
        if probabilities.size == 0:
            logger.warning("[Pipeline] Empty probabilities array in _align_proba_to_encoder")
            return None

        encoder_classes = list(label_encoder.classes_)
        K = len(encoder_classes)

        if K == 2:
            if probabilities.shape[1] == 1:
                p_pos = probabilities[:, 0]
                if np.any(p_pos < 0) or np.any(p_pos > 1):
                    logger.warning(f"[Pipeline] Single-column probabilities outside [0,1] range (min: {p_pos.min():.6f}, max: {p_pos.max():.6f})")
                p_neg = 1.0 - p_pos
                return np.column_stack([p_neg, p_pos])
            elif probabilities.shape[1] == 2:
                if np.any(probabilities < 0) or np.any(probabilities > 1):
                    logger.warning(f"[Pipeline] Two-column probabilities outside [0,1] range (min: {probabilities.min():.6f}, max: {probabilities.max():.6f})")
                return probabilities
            else:
                logger.warning(f"[Pipeline] Unexpected number of probability columns ({probabilities.shape[1]}) for binary classification")
                return None

        model_labels = self._get_model_class_labels()
        if not model_labels or probabilities.shape[1] == K and set(model_labels) == set(encoder_classes):
            if probabilities.shape[1] == K:
                if np.any(probabilities < 0) or np.any(probabilities > 1):
                    logger.warning(f"[Pipeline] Multiclass probabilities outside [0,1] range (min: {probabilities.min():.6f}, max: {probabilities.max():.6f})")
                return probabilities
            else:
                logger.warning(f"[Pipeline] Shape mismatch: expected {K} columns, got {probabilities.shape[1]}")
                return None

        aligned = np.zeros((probabilities.shape[0], K), dtype=float)

        try:
            model_to_encoder_idx = {
                lbl: int(label_encoder.transform([lbl])[0]) for lbl in model_labels
            }
        except Exception:
            model_to_encoder_idx = {}
            for j, lbl in enumerate(model_labels):
                try:
                    enc_idx = int(lbl)
                except Exception:
                    enc_idx = j
                model_to_encoder_idx[lbl] = enc_idx

        for j_model, lbl in enumerate(model_labels):
            if j_model >= probabilities.shape[1]:
                break
            enc_j = model_to_encoder_idx.get(lbl, None)
            if enc_j is not None and 0 <= enc_j < K:
                aligned[:, enc_j] = probabilities[:, j_model]

        if np.any(aligned < 0) or np.any(aligned > 1):
            logger.warning(f"[Pipeline] Aligned probabilities outside [0,1] range (min: {aligned.min():.6f}, max: {aligned.max():.6f})")
        
        zero_rows = np.all(aligned == 0, axis=1)
        if np.any(zero_rows):
            logger.warning(f"[Pipeline] {np.sum(zero_rows)} samples have all-zero probabilities (missing class predictions)")

        return aligned

    # -----------------------------------------------------------------
    # Rest of your methods remain unchanged.
    # (evaluate, fairness, baseline, cross_validate, get_params, etc.)
    # -----------------------------------------------------------------

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, output_format: str = 'rich'):
        """
        Makes predictions on the test set and prints a report with
        Accuracy, F1 Score, and ROC AUC Score.
        """
        if not self._is_fitted:
            raise RuntimeError("You must call fit() on the pipeline before evaluating.")
        
        logger.info("\n" + "="*60)
        logger.info("[Pipeline] Running Evaluation")
        
        predictions = self.predict(X_test)
        
        if self.task_type == 'classification':
            probabilities = self.predict_proba(X_test)
            
            y_test_encoded = None
            if hasattr(self.processor, 'custom_preprocessor_') and hasattr(self.processor.custom_preprocessor_, 'label_encoder_'):
                y_test_encoded = self.processor.custom_preprocessor_.label_encoder_.transform(y_test)
            elif isinstance(self.model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, OrionMSPv15Classifier)):
                y_test_encoded = self.model.y_encoder_.transform(y_test)
            elif isinstance(self.model, TabPFNClassifier):
                le = LabelEncoder()
                le.classes_ = self.model.classes_
                y_test_encoded = le.transform(y_test)
            elif isinstance(self.model, ConTextTabClassifier):
                if hasattr(self.processor_, 'label_encoder_'):
                    if y_test.dtype == object or y_test.dtype.kind in {'U','S'}:
                        y_test = self.processor_.label_encoder_.transform(y_test)
            elif hasattr(self.processor, 'label_encoder_') and self.processor.label_encoder_ is not None:
                y_test_encoded = self.processor.label_encoder_.transform(y_test)

            if y_test_encoded is None:
                 raise RuntimeError("Could not find a fitted label encoder to evaluate metrics.")

            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')
            mcc = matthews_corrcoef(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')
            
            unique_test = np.unique(y_test_encoded)
            if len(unique_test) < 2:
                auc = float("nan")
            else:
                if hasattr(self.processor, 'custom_preprocessor_') and hasattr(self.processor.custom_preprocessor_, 'label_encoder_'):
                    le = self.processor.custom_preprocessor_.label_encoder_
                elif isinstance(self.model, (TabICLClassifier, OrionBixClassifier, OrionMSPClassifier, OrionMSPv15Classifier)):
                    le = self.model.y_encoder_
                elif isinstance(self.model, TabPFNClassifier):
                    le = LabelEncoder(); le.classes_ = self.model.classes_
                elif hasattr(self.processor, 'label_encoder_') and self.processor.label_encoder_ is not None:
                    le = self.processor.label_encoder_
                else:
                    raise RuntimeError("Could not find a fitted label encoder to align probabilities.")

                probs_aligned = self._align_proba_to_encoder(probabilities, le)

                K = len(le.classes_)
                if K == 2:
                    auc = roc_auc_score(y_test_encoded, probs_aligned[:, 1])
                else:
                    auc = roc_auc_score(
                        y_test_encoded,
                        probs_aligned,
                        labels=list(range(K)),
                        multi_class="ovr",
                        average="weighted",
                    )

            results = {
                "accuracy": accuracy,
                "roc_auc_score": auc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "mcc": mcc
            }

        elif self.task_type == 'regression':
             y_true = y_test
             y_pred = predictions
             results = self._calculate_regression_metrics(y_true, y_pred)

             if output_format == 'rich':
                logger.info("\n" + "="*60)
                logger.info("[Pipeline] Running Evaluation")
                logger.info("\n[Pipeline] Regression Evaluation Report")
                logger.info(f"[Pipeline] MSE: {results['mse']:.4f}")
                logger.info(f"[Pipeline] RMSE: {results['rmse']:.4f}")
                logger.info(f"[Pipeline] MAE: {results['mae']:.4f}")
                logger.info(f"[Pipeline] R2 Score: {results['r2_score']:.4f}")
                if 'mape' in results and results['mape'] is not None:
                    logger.info(f"[Pipeline] MAPE: {results['mape']:.4f}%")
                if 'medae' in results:
                    logger.info(f"[Pipeline] MedAE: {results['medae']:.4f}")
                if 'explained_variance' in results:
                    logger.info(f"[Pipeline] Explained Variance: {results['explained_variance']:.4f}")
                if 'max_error' in results:
                    logger.info(f"[Pipeline] Max Error: {results['max_error']:.4f}")
                if 'msle' in results and results['msle'] is not None:
                    logger.info(f"[Pipeline] MSLE: {results['msle']:.4f}")
                logger.info("="*60)
             elif output_format == 'json':
                print(json.dumps(results, indent=4))

        if output_format == 'json' and self.task_type == 'classification':
                print(json.dumps(results, indent=4))
        elif output_format == 'rich' and self.task_type == 'classification':
                logger.info("\n" + "="*60)
                logger.info("[Pipeline] Running Evaluation")
                logger.info("\n[Pipeline] Evaluation Report")
                logger.info(f"[Pipeline] Accuracy: {accuracy:.4f}")
                logger.info(f"[Pipeline] Weighted F1-Score: {f1:.4f}")
                logger.info(f"[Pipeline] Weighted Precision: {precision:.4f}")
                logger.info(f"[Pipeline] Weighted Recall: {recall:.4f}")
                logger.info(f"[Pipeline] MCC: {mcc:.4f}")
                logger.info(f"[Pipeline] ROC AUC Score: {auc:.4f}")
                logger.info("\n[Pipeline] Classification Report")
                logger.info(classification_report(y_test, predictions, zero_division=0))
                logger.info("="*60)
        else:
            logger.warning(f"[Pipeline] Unknown output_format: '{output_format}'. No output printed.")

        return results

    def get_residuals(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Calculate residuals (y_true - y_pred) for regression tasks.
        
        Args:
            X: Input features
            y: True target values
        
        Returns:
            numpy.ndarray: Array of residuals
        """
        if not self._is_fitted:
            raise RuntimeError("You must call fit() on the pipeline before calling get_residuals().")
        
        if self.task_type != 'regression':
            raise ValueError("get_residuals() is only available for regression tasks.")
        
        predictions = self.predict(X)
        residuals = np.array(y).flatten() - np.array(predictions).flatten()
        
        return residuals

    def analyze_residuals(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Analyze residuals for regression diagnostics.
        
        Args:
            X: Input features
            y: True target values
        
        Returns:
            dict: Dictionary containing residual statistics:
                 - 'mean': Mean of residuals
                 - 'median': Median of residuals
                 - 'std': Standard deviation of residuals
                 - 'min': Minimum residual
                 - 'max': Maximum residual
                 - 'skewness': Skewness of residual distribution
                 - 'kurtosis': Kurtosis of residual distribution
        """
        if not self._is_fitted:
            raise RuntimeError("You must call fit() on the pipeline before calling analyze_residuals().")
        
        if self.task_type != 'regression':
            raise ValueError("analyze_residuals() is only available for regression tasks.")
        
        residuals = self.get_residuals(X, y)
        
        from scipy import stats
        
        results = {
            'mean': np.mean(residuals),
            'median': np.median(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }
        
        # Normality test (Shapiro-Wilk for small samples, otherwise D'Agostino)
        if len(residuals) <= 5000:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                results['normality_test'] = {
                    'test': 'shapiro-wilk',
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
            except Exception as e:
                logger.warning(f"[Pipeline] Could not perform Shapiro-Wilk test: {e}")
        else:
            try:
                # D'Agostino's test for larger samples
                k2_stat, k2_p = stats.normaltest(residuals)
                results['normality_test'] = {
                    'test': 'd_agostino',
                    'statistic': k2_stat,
                    'p_value': k2_p,
                    'is_normal': k2_p > 0.05
                }
            except Exception as e:
                logger.warning(f"[Pipeline] Could not perform D'Agostino test: {e}")
        
        return results

    def plot_residuals(self, X: pd.DataFrame, y: pd.Series, save_path: str = None):
        """
        Create diagnostic plots for residuals.
        
        Args:
            X: Input features
            y: True target values
            save_path: Optional path to save the plot. If None, plot is displayed.
        
        Returns:
            matplotlib.figure.Figure: The figure object (if matplotlib is available)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_residuals(). "
                "Install it with: pip install matplotlib"
            )
        
        if not self._is_fitted:
            raise RuntimeError("You must call fit() on the pipeline before calling plot_residuals().")
        
        if self.task_type != 'regression':
            raise ValueError("plot_residuals() is only available for regression tasks.")
        
        predictions = self.predict(X)
        residuals = np.array(y).flatten() - np.array(predictions).flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Residual Analysis Plots', fontsize=14)
        
        # 1. Residuals vs Predicted
        axes[0, 0].scatter(predictions, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normality Check)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals Histogram
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Residuals vs Index (for detecting patterns)
        axes[1, 1].plot(residuals, alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Index')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"[Pipeline] Residual plots saved to {save_path}")
        else:
            plt.show()
        
        return fig

    def _calculate_regression_metrics(self, y_true, y_pred):
        """
        Helper method to calculate comprehensive regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            dict: Dictionary containing all regression metrics
        """
        # Convert to numpy arrays if needed
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        results = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2
        }
        
        # Additional metrics
        # Median Absolute Error
        medae = median_absolute_error(y_true, y_pred)
        results["medae"] = medae
        
        # Explained Variance Score
        explained_variance = explained_variance_score(y_true, y_pred)
        results["explained_variance"] = explained_variance
        
        # Max Error
        max_err = max_error(y_true, y_pred)
        results["max_error"] = max_err
        
        # Mean Absolute Percentage Error (MAPE)
        # Handle zero division: if any y_true is zero, MAPE is undefined
        if np.any(y_true == 0):
            mape = None
            logger.debug("[Pipeline] MAPE not calculated: some true values are zero")
        else:
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        results["mape"] = mape
        
        # Mean Squared Log Error (MSLE)
        # Only calculate if all values are positive
        if np.all(y_true > 0) and np.all(y_pred > 0):
            msle = mean_squared_log_error(y_true, y_pred)
            results["msle"] = msle
        else:
            msle = None
            logger.debug("[Pipeline] MSLE not calculated: some values are non-positive")
            results["msle"] = msle
        
        return results

    def save(self, file_path: str):
        if not self._is_fitted:
            raise RuntimeError("You can only save a pipeline after it has been fitted.")
        logger.info(f"[Pipeline] Saving pipeline to {file_path}")
        joblib.dump(self, file_path)
        logger.info("[Pipeline] Pipeline saved successfully")

    @classmethod
    def load(cls, file_path: str):
        logger.info(f"[Pipeline] Loading pipeline from {file_path}")
        pipeline = joblib.load(file_path)
        logger.info("[Pipeline] Pipeline loaded successfully")
        return pipeline

    def show_processing_summary(self):
        """
        Retrieves and logs the data processing summary from the DataProcessor.
        """
        logger.info("\n" + "="*60)
        summary = self.processor.get_processing_summary()
        # Log the multi-line summary as a single message
        summary_lines = summary.split('\n')
        
        for line in summary_lines:
            logger.info(line)


    def _calculate_calibration_errors(self, y_true, y_prob, n_bins=10):
        """Helper to calculate ECE and MCE."""
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true)

        ece = 0.0
        mce = 0.0
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                bin_abs_err = np.abs(accuracy_in_bin - avg_confidence_in_bin)
                
                ece += prop_in_bin * bin_abs_err
                mce = max(mce, bin_abs_err)
                
        return ece, mce

    def evaluate_calibration(self, X_test: pd.DataFrame, y_test: pd.Series, n_bins: int = 15, output_format: str = 'rich'):
        """
        Calculates and provides a detailed report on model calibration metrics.
        This version supports both binary and multiclass classification.
        """
        if not self._is_fitted:
            raise RuntimeError("You must call fit() on the pipeline before evaluating calibration.")

        # --- Metric Calculation (common for all formats) ---
        probabilities = self.predict_proba(X_test)

        # 1. Find the correct label encoder (same logic as in evaluate())
        le = None
        if hasattr(self.processor, 'custom_preprocessor_') and hasattr(self.processor.custom_preprocessor_, 'label_encoder_'):
            le = self.processor.custom_preprocessor_.label_encoder_
        elif isinstance(self.model, (TabICLClassifier, OrionBixClassifier, OrionMSPClassifier, OrionMSPv15Classifier)):
            # Use model's internal encoder if in inference mode
            if hasattr(self.model, 'y_encoder_'):
                le = self.model.y_encoder_
            # Use processor's encoder if in finetune mode
            elif hasattr(self.processor, 'custom_preprocessor_') and hasattr(self.processor.custom_preprocessor_, 'label_encoder_'):
                 le = self.processor.custom_preprocessor_.label_encoder_
        elif isinstance(self.model, TabPFNClassifier):
            if hasattr(self.model, 'classes_'):
                le = LabelEncoder()
                le.classes_ = self.model.classes_
        elif hasattr(self.processor, 'label_encoder_') and self.processor.label_encoder_ is not None:
            le = self.processor.label_encoder_
        
        if le is None:
             raise RuntimeError("Could not find a fitted label encoder to evaluate calibration.")

        # 2. Encode y_test using the found encoder
        y_test_encoded = le.transform(y_test)
        
        # 3. Align probability columns to match the encoder's class order
        probs_aligned = self._align_proba_to_encoder(probabilities, le)

        # 4. Calculate metrics using the aligned probabilities
        # brier_score_loss handles (n_samples, n_classes) for multiclass
        # when y_true is (n_samples,) with integer labels [0, K-1].
        
        # Validate inputs before calculating Brier score
        if probs_aligned is None:
            logger.warning("[Pipeline] Probabilities are None, skipping Brier score calculation")
            brier_score = float('nan')
        else:
            # Check for NaN or infinite values
            if np.any(np.isnan(probs_aligned)) or np.any(np.isinf(probs_aligned)):
                logger.warning("[Pipeline] Probabilities contain NaN or infinite values, skipping Brier score calculation")
                brier_score = float('nan')
            else:
                # Validate that probabilities sum to 1.0 (within tolerance)
                prob_sums = np.sum(probs_aligned, axis=1)
                if not np.allclose(prob_sums, 1.0, rtol=1e-6):
                    logger.warning(f"[Pipeline] Probabilities don't sum to 1.0 (range: {prob_sums.min():.6f} to {prob_sums.max():.6f})")
                    logger.warning("[Pipeline] This may indicate model calibration issues")
                
                # Validate that y_test_encoded contains valid class indices
                max_class_idx = len(le.classes_) - 1
                if np.any(y_test_encoded < 0) or np.any(y_test_encoded > max_class_idx):
                    logger.warning(f"[Pipeline] Invalid class indices in y_test_encoded (range: {y_test_encoded.min()} to {y_test_encoded.max()})")
                    logger.warning(f"[Pipeline] Expected range: 0 to {max_class_idx}")
                    brier_score = float('nan')
                else:
                    try:
                        brier_score = brier_score_loss(y_test_encoded, probs_aligned)
                    except Exception as e:
                        logger.error(f"[Pipeline] Error calculating Brier score: {e}")
                        brier_score = float('nan')
        
        # _calculate_calibration_errors also works with (n, K) probability matrix
        if probs_aligned is None:
            logger.warning("[Pipeline] Probabilities are None, skipping ECE and MCE calculation")
            ece, mce = float('nan'), float('nan')
        else:
            ece, mce = self._calculate_calibration_errors(y_test_encoded, probs_aligned, n_bins=n_bins)

        results = {
            "brier_score_loss": brier_score,
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce
        }

        if output_format == 'rich':
            logger.info("\n" + "="*80)
            logger.info("[Pipeline] Running Detailed Calibration Evaluation")
            logger.info("="*80)
            logger.info("[Pipeline] Calibration measures how well a model's predicted probabilities match the true likelihood of outcomes.")
            logger.info("[Pipeline] A well-calibrated model is trustworthy: if it predicts a 70% probability, it should be correct 70% of the time.\n")
            
            logger.info("[Pipeline] Brier Score Loss")
            logger.info("[Pipeline] Measures the mean squared difference between predicted probabilities and actual outcomes.")
            if np.isnan(brier_score):
                logger.info(f"[Pipeline] Your Score: NaN (calculation skipped due to validation issues)")
                logger.info("[Pipeline] Interpretation: Check warnings above for details on why Brier score could not be calculated.")
            else:
                logger.info(f"[Pipeline] Your Score: {brier_score:.4f}")
                logger.info("[Pipeline] Interpretation: Scores range from 0.0 to 1.0, where lower is better. A score near 0.0 indicates excellent calibration.")
                logger.info("[Pipeline] Note: For multiclass problems, this is the average Brier score across all classes.")
                logger.info("[Pipeline] Note: For imbalanced datasets, consider class-specific Brier scores for better insights.")
            logger.info("")

            logger.info("[Pipeline] Expected & Maximum Calibration Error (ECE / MCE)")
            logger.info("[Pipeline] These metrics group predictions into bins by confidence (e.g., 80-90%) and measure the gap between the average confidence and the actual accuracy in each bin.")
            
            if np.isnan(ece) or np.isnan(mce):
                logger.info(f"[Pipeline] Expected Calibration Error (ECE): NaN (calculation skipped due to validation issues)")
                logger.info(f"[Pipeline] Maximum Calibration Error (MCE): NaN (calculation skipped due to validation issues)")
                logger.info("[Pipeline] Interpretation: Check warnings above for details on why ECE/MCE could not be calculated.")
            else:
                logger.info(f"[Pipeline] Expected Calibration Error (ECE): {ece:.4f}")
                logger.info(f"[Pipeline] Interpretation: ECE represents the average gap between confidence and accuracy across all bins. Your score indicates the model's confidence is off by an average of {ece*100:.2f}%. An ECE below 0.05 (5%) is generally considered good.")

                logger.info(f"[Pipeline] Maximum Calibration Error (MCE): {mce:.4f}")
                logger.info("[Pipeline] Interpretation: MCE identifies the single worst-performing bin, representing the 'worst-case scenario' for your model's calibration. A high MCE reveals specific confidence ranges where the model is particularly unreliable.")
            logger.info("")
            logger.info("="*80)
            
        elif output_format == 'json':
            print(json.dumps(results, indent=4))
            
        else:
            logger.warning(f"[Pipeline] Unknown output_format: '{output_format}'. No console output printed.")

        # The method still returns the dictionary for programmatic use
        return results

    def evaluate_interval_calibration(self, X_test: pd.DataFrame, y_test: pd.Series, 
                                     confidence: float = 0.95, n_bins: int = 10, 
                                     output_format: str = 'rich'):
        """
        Evaluate calibration of prediction intervals for regression tasks.
        
        Measures how well the prediction intervals match their nominal coverage.
        For example, a 95% confidence interval should contain the true value 95% of the time.
        
        Currently only supported for TabPFN regression models.
        
        Args:
            X_test: Test features
            y_test: True target values
            confidence: Confidence level used for intervals (default: 0.95)
            n_bins: Number of bins for reliability diagram
            output_format: 'rich' for detailed output, 'json' for JSON output
        
        Returns:
            dict: Dictionary containing calibration metrics:
                 - 'coverage_probability': Actual coverage of intervals
                 - 'nominal_coverage': Nominal coverage (confidence level)
                 - 'coverage_error': Difference between actual and nominal coverage
                 - 'average_interval_width': Mean width of prediction intervals
                 - 'reliability_data': Binned coverage data for reliability diagram
        """
        if not self._is_fitted:
            raise RuntimeError("You must call fit() on the pipeline before evaluating interval calibration.")
        
        if self.task_type != 'regression':
            raise ValueError("evaluate_interval_calibration() is only available for regression tasks.")
        
        # Check if model supports prediction intervals
        if not isinstance(self.model, TabPFNRegressorWrapper):
            raise NotImplementedError(
                f"Interval calibration is not yet supported for {self.model_name}. "
                "Currently only TabPFN supports prediction intervals."
            )
        
        logger.info(f"[Pipeline] Evaluating interval calibration for {confidence*100:.1f}% confidence intervals")
        
        # Get prediction intervals
        intervals = self.predict_intervals(X_test, confidence=confidence)
        y_true = np.array(y_test).flatten()
        lower = intervals['lower']
        upper = intervals['upper']
        
        # Calculate coverage: fraction of true values within intervals
        in_interval = (y_true >= lower) & (y_true <= upper)
        coverage_probability = np.mean(in_interval)
        coverage_error = abs(coverage_probability - confidence)
        
        # Calculate average interval width
        interval_widths = upper - lower
        average_interval_width = np.mean(interval_widths)
        
        # Reliability diagram: bin by predicted value and calculate coverage per bin
        predictions = intervals['mean']
        reliability_data = []
        
        # Create bins based on predicted values
        pred_min, pred_max = np.min(predictions), np.max(predictions)
        bin_edges = np.linspace(pred_min, pred_max, n_bins + 1)
        
        for i in range(n_bins):
            bin_mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i+1])
            if i == n_bins - 1:  # Include right edge for last bin
                bin_mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i+1])
            
            if np.sum(bin_mask) > 0:
                bin_coverage = np.mean(in_interval[bin_mask])
                bin_count = np.sum(bin_mask)
                bin_avg_pred = np.mean(predictions[bin_mask])
                reliability_data.append({
                    'bin_index': i,
                    'bin_lower': bin_edges[i],
                    'bin_upper': bin_edges[i+1],
                    'bin_center': (bin_edges[i] + bin_edges[i+1]) / 2,
                    'coverage': bin_coverage,
                    'count': bin_count,
                    'avg_prediction': bin_avg_pred
                })
        
        results = {
            "coverage_probability": coverage_probability,
            "nominal_coverage": confidence,
            "coverage_error": coverage_error,
            "average_interval_width": average_interval_width,
            "reliability_data": reliability_data
        }
        
        if output_format == 'rich':
            logger.info("\n" + "="*80)
            logger.info("[Pipeline] Running Prediction Interval Calibration Evaluation")
            logger.info("="*80)
            logger.info("[Pipeline] Interval calibration measures how well prediction intervals match their nominal coverage.")
            logger.info(f"[Pipeline] For {confidence*100:.1f}% confidence intervals, we expect {confidence*100:.1f}% of true values to fall within the intervals.\n")
            
            logger.info(f"[Pipeline] Coverage Probability: {coverage_probability:.4f} ({coverage_probability*100:.2f}%)")
            logger.info(f"[Pipeline] Nominal Coverage: {confidence:.4f} ({confidence*100:.2f}%)")
            logger.info(f"[Pipeline] Coverage Error: {coverage_error:.4f} ({coverage_error*100:.2f}%)")
            logger.info("[Pipeline] Interpretation: Coverage error measures the difference between actual and expected coverage.")
            logger.info("[Pipeline] A well-calibrated model should have coverage error close to 0.\n")
            
            logger.info(f"[Pipeline] Average Interval Width: {average_interval_width:.4f}")
            logger.info("[Pipeline] Interpretation: Narrower intervals are more informative, but must maintain good coverage.\n")
            
            logger.info(f"[Pipeline] Reliability Diagram: {len(reliability_data)} bins")
            logger.info("[Pipeline] The reliability diagram shows coverage across different prediction ranges.")
            logger.info("="*80)
            
        elif output_format == 'json':
            # Convert reliability_data to JSON-serializable format
            json_results = {
                "coverage_probability": float(coverage_probability),
                "nominal_coverage": float(confidence),
                "coverage_error": float(coverage_error),
                "average_interval_width": float(average_interval_width),
                "reliability_data": [
                    {k: float(v) if isinstance(v, (np.integer, np.floating, np.float32, np.float64)) 
                      else int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else v
                     for k, v in item.items()}
                    for item in reliability_data
                ]
            }
            print(json.dumps(json_results, indent=4))
        else:
            logger.warning(f"[Pipeline] Unknown output_format: '{output_format}'. No console output printed.")
        
        return results
        
    def evaluate_fairness(self, X_test: pd.DataFrame, y_test: pd.Series, sensitive_features: pd.Series, output_format: str = 'rich'):
        """
        Calculates and provides a detailed report on group fairness metrics.
        Supports both classification and regression tasks.
        """
        if not self._is_fitted:
            raise RuntimeError("You must call fit() on the pipeline before evaluating fairness.")

        predictions = self.predict(X_test)
        
        if self.task_type == 'regression':
            # Regression fairness metrics
            y_true = np.array(y_test).flatten()
            y_pred = np.array(predictions).flatten()
            
            # Calculate group-wise statistics
            groups = sensitive_features.unique()
            group_stats = {}
            
            for group in groups:
                group_mask = sensitive_features == group
                group_y_true = y_true[group_mask]
                group_y_pred = y_pred[group_mask]
                
                if len(group_y_true) > 0:
                    group_stats[group] = {
                        'mean_prediction': np.mean(group_y_pred),
                        'mean_true': np.mean(group_y_true),
                        'mse': mean_squared_error(group_y_true, group_y_pred),
                        'mae': mean_absolute_error(group_y_true, group_y_pred),
                        'r2': r2_score(group_y_true, group_y_pred),
                        'mean_residual': np.mean(group_y_true - group_y_pred),
                        'count': len(group_y_true)
                    }
            
            # Calculate fairness metrics
            if len(groups) >= 2:
                group_list = list(groups)
                # Mean prediction difference (statistical parity for regression)
                mean_pred_diff = abs(group_stats[group_list[0]]['mean_prediction'] - 
                                    group_stats[group_list[1]]['mean_prediction'])
                
                # Residual difference (equalized residuals)
                residual_diff = abs(group_stats[group_list[0]]['mean_residual'] - 
                                   group_stats[group_list[1]]['mean_residual'])
                
                # MSE difference
                mse_diff = abs(group_stats[group_list[0]]['mse'] - 
                              group_stats[group_list[1]]['mse'])
                
                # MAE difference
                mae_diff = abs(group_stats[group_list[0]]['mae'] - 
                              group_stats[group_list[1]]['mae'])
            else:
                mean_pred_diff = residual_diff = mse_diff = mae_diff = 0.0
            
            results = {
                "mean_prediction_difference": mean_pred_diff,
                "residual_difference": residual_diff,
                "mse_difference": mse_diff,
                "mae_difference": mae_diff,
                "group_statistics": group_stats
            }
            
            if output_format == 'rich':
                logger.info("\n" + "="*80)
                logger.info("[Pipeline] Running Detailed Fairness Evaluation (Regression)")
                logger.info("="*80)
                logger.info(f"[Pipeline] Fairness is evaluated with respect to the '{sensitive_features.name}' attribute.")
                logger.info("[Pipeline] These metrics measure disparities in model behavior between different groups.\n")
                
                logger.info("[Pipeline] Mean Prediction Difference")
                logger.info("[Pipeline] Measures the difference in average predicted values between groups.")
                logger.info(f"[Pipeline] Your Score: {mean_pred_diff:.4f}")
                logger.info("[Pipeline] Interpretation: A value of 0 indicates groups receive the same average predictions. Large differences may indicate bias.\n")
                
                logger.info("[Pipeline] Residual Difference")
                logger.info("[Pipeline] Measures the difference in average residuals (errors) between groups.")
                logger.info(f"[Pipeline] Your Score: {residual_diff:.4f}")
                logger.info("[Pipeline] Interpretation: A value of 0 indicates groups have similar prediction errors. Large differences suggest unequal model performance.\n")
                
                logger.info("[Pipeline] MSE Difference")
                logger.info(f"[Pipeline] Measures the difference in Mean Squared Error between groups: {mse_diff:.4f}")
                logger.info("[Pipeline] Interpretation: Large differences indicate one group has significantly worse predictions.\n")
                
                logger.info("[Pipeline] MAE Difference")
                logger.info(f"[Pipeline] Measures the difference in Mean Absolute Error between groups: {mae_diff:.4f}")
                logger.info("[Pipeline] Interpretation: Large differences indicate one group has significantly worse predictions.\n")
                
                logger.info("[Pipeline] Group Statistics:")
                for group, stats in group_stats.items():
                    logger.info(f"[Pipeline]   Group '{group}':")
                    logger.info(f"[Pipeline]     Count: {stats['count']}")
                    logger.info(f"[Pipeline]     Mean Prediction: {stats['mean_prediction']:.4f}")
                    logger.info(f"[Pipeline]     Mean True Value: {stats['mean_true']:.4f}")
                    logger.info(f"[Pipeline]     MSE: {stats['mse']:.4f}")
                    logger.info(f"[Pipeline]     MAE: {stats['mae']:.4f}")
                    logger.info(f"[Pipeline]     R²: {stats['r2']:.4f}")
                logger.info("="*80)
                
            elif output_format == 'json':
                # Convert group_stats to JSON-serializable format
                json_results = {
                    "mean_prediction_difference": float(mean_pred_diff),
                    "residual_difference": float(residual_diff),
                    "mse_difference": float(mse_diff),
                    "mae_difference": float(mae_diff),
                    "group_statistics": {
                        str(k): {kk: float(vv) if isinstance(vv, (np.integer, np.floating, np.float32, np.float64)) 
                                else int(vv) if isinstance(vv, (np.integer, np.int32, np.int64)) else vv
                                for kk, vv in v.items()}
                        for k, v in group_stats.items()
                    }
                }
                print(json.dumps(json_results, indent=4))
            else:
                logger.warning(f"[Pipeline] Unknown output_format: '{output_format}'. No console output printed.")
            
        else:
            # Classification fairness metrics (existing code)
            y_test_encoded, predictions_encoded = self._get_encoded_labels(y_test, predictions)

            spd = demographic_parity_difference(
                y_true=y_test_encoded, y_pred=predictions_encoded, sensitive_features=sensitive_features
            )
            eod = equal_opportunity_difference(
                y_true=y_test_encoded, y_pred=predictions_encoded, sensitive_features=sensitive_features
            )
            aod = equalized_odds_difference(
                y_true=y_test_encoded, y_pred=predictions_encoded, sensitive_features=sensitive_features
            )
            
            results = {
                "statistical_parity_difference": spd,
                "equal_opportunity_difference": eod,
                "equalized_odds_difference": aod
            }

            if output_format == 'rich':
                logger.info("\n" + "="*80)
                logger.info("[Pipeline] Running Detailed Fairness Evaluation")
                logger.info("="*80)
                logger.info(f"[Pipeline] Fairness is evaluated with respect to the '{sensitive_features.name}' attribute.")
                logger.info("[Pipeline] These metrics measure disparities in model behavior between different groups. For these difference-based metrics, a value of 0 indicates perfect fairness.\n")

                logger.info("[Pipeline] Statistical Parity Difference (Selection Rate)")
                logger.info("[Pipeline] Measures the difference in the rate of positive predictions (e.g., 'Churn') between groups.")
                logger.info(f"[Pipeline] Your Score: {spd:.4f}")
                logger.info(f"[Pipeline] Interpretation: Your score means there is a {abs(spd*100):.2f}% difference in the selection rate between groups. Values close to 0 are ideal. Disparities above 10-20% are often considered significant.\n")

                logger.info("[Pipeline] Equal Opportunity Difference (True Positive Rate)")
                logger.info("[Pipeline] Measures the difference in the true positive rate—the rate at which the model correctly identifies positive outcomes—between groups.")
                logger.info(f"[Pipeline] Your Score: {eod:.4f}")
                logger.info(f"[Pipeline] Interpretation: For cases that are genuinely positive, your score means the model's ability to correctly identify them differs by {abs(eod*100):.2f}% between groups. High values indicate the model's benefits are not being applied equally.\n")
                
                logger.info("[Pipeline] Equalized Odds Difference (Overall Error Rate)")
                logger.info("[Pipeline] Measures the larger of the true positive rate difference and the false positive rate difference between groups.")
                logger.info(f"[Pipeline] Your Score: {aod:.4f}")
                logger.info(f"[Pipeline] Interpretation: This score represents the 'worst-case' error rate disparity. A score of {abs(aod*100):.2f}% indicates the largest gap in performance. If this value is close to the Equal Opportunity Difference, the main issue is with true positives.\n")
                logger.info("="*80)

            elif output_format == 'json':
                print(json.dumps(results, indent=4))
                
            else:
                logger.warning(f"[Pipeline] Unknown output_format: '{output_format}'. No console output printed.")
            
        return results

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series = None, 
                               method: str = 'permutation', n_repeats: int = 10,
                               random_state: int = None) -> dict:
        """
        Calculate feature importance for regression or classification models.
        
        Args:
            X: Input features (can be training or test data)
            y: Target values (optional, used for permutation importance)
            method: Method to use ('permutation' or 'shap')
                   - 'permutation': Permutation importance (model-agnostic, default)
                   - 'shap': SHAP values (if model supports it)
            n_repeats: Number of times to permute each feature (for permutation importance)
            random_state: Random state for reproducibility
        
        Returns:
            dict: Dictionary mapping feature names to importance scores
                 Features are sorted by importance (descending)
        """
        if not self._is_fitted:
            raise RuntimeError("You must call fit() on the pipeline before getting feature importance.")
        
        if method == 'shap':
            try:
                import shap
            except ImportError:
                raise ImportError(
                    "SHAP is required for method='shap'. "
                    "Install it with: pip install shap"
                )
            
            # SHAP support is model-specific and may not be available for all models
            logger.warning("[Pipeline] SHAP importance is not yet fully implemented for all models.")
            logger.info("[Pipeline] Falling back to permutation importance.")
            method = 'permutation'
        
        if method == 'permutation':
            # Use manual permutation importance implementation
            # (sklearn's permutation_importance may not work with TabularPipeline due to sklearn compatibility)
            if y is None:
                # Use predictions as baseline
                y = self.predict(X)
            
            logger.info(f"[Pipeline] Calculating feature importance using permutation method ({n_repeats} repeats)")
            return self._calculate_permutation_importance_manual(X, y, n_repeats, random_state)
        
        else:
            raise ValueError(f"Unknown method: {method}. Supported methods: 'permutation', 'shap'")
    
    def _calculate_permutation_importance_manual(self, X: pd.DataFrame, y: pd.Series, 
                                                 n_repeats: int, random_state: int) -> dict:
        """
        Manual implementation of permutation importance (fallback).
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Get baseline score
        if self.task_type == 'regression':
            baseline_pred = self.predict(X)
            baseline_score = r2_score(y, baseline_pred)
        else:
            baseline_pred = self.predict(X)
            baseline_score = accuracy_score(y, baseline_pred)
        
        # Get feature names
        if hasattr(X, 'columns'):
            feature_names = list(X.columns)
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        importances = {}
        X_permuted = X.copy()
        
        for feature_name in feature_names:
            feature_importances = []
            
            for _ in range(n_repeats):
                # Permute the feature
                X_permuted[feature_name] = np.random.permutation(X_permuted[feature_name].values)
                
                # Calculate score with permuted feature
                if self.task_type == 'regression':
                    permuted_pred = self.predict(X_permuted)
                    permuted_score = r2_score(y, permuted_pred)
                else:
                    permuted_pred = self.predict(X_permuted)
                    permuted_score = accuracy_score(y, permuted_pred)
                
                # Importance is the decrease in score
                importance = baseline_score - permuted_score
                feature_importances.append(importance)
                
                # Restore original feature
                X_permuted[feature_name] = X[feature_name]
            
            importances[feature_name] = np.mean(feature_importances)
        
        # Sort by importance (descending)
        importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        
        return importances

    def _get_encoded_labels(self, y_true, y_pred):
        """Helper to consistently encode true and predicted labels."""
        y_true_encoded = None
        y_pred_encoded = None

        # Find the correct LabelEncoder
        le = None
        if hasattr(self.processor, 'custom_preprocessor_') and hasattr(self.processor.custom_preprocessor_, 'label_encoder_'):
            le = self.processor.custom_preprocessor_.label_encoder_
        elif isinstance(self.model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, TabPFNClassifier, OrionMSPv15Classifier)):
             # Fit a temporary encoder on the training labels seen during .fit()
            le = LabelEncoder().fit(self.y_train_processed_ if self.y_train_processed_ is not None else y_true)
        elif isinstance(self.model, LimixClassifier) and hasattr(self.model, 'le_'):
             le = self.model.le_
        else:
            raise RuntimeError("Could not find a fitted label encoder to evaluate metrics.")

        y_true_encoded = le.transform(y_true)
        # Handle cases where y_pred might be different (e.g., raw y_test for fairness)
        if y_pred is not None:
            y_pred_encoded = le.transform(y_pred)
            
        return y_true_encoded, y_pred_encoded

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5, 
                      scoring: list = None, random_state: int = None) -> dict:
        """
        Perform cross-validation for regression or classification tasks.
        
        Args:
            X: Full dataset features
            y: Full dataset target values
            cv: Number of folds (default: 5)
            scoring: List of metrics to calculate. If None, uses default metrics:
                    - Regression: ['mse', 'mae', 'r2_score']
                    - Classification: ['accuracy', 'f1_score', 'roc_auc_score']
            random_state: Random state for reproducibility
        
        Returns:
            dict: Dictionary containing:
                 - 'mean_scores': Mean score for each metric across folds
                 - 'std_scores': Standard deviation for each metric across folds
                 - 'fold_scores': Per-fold scores for each metric
                 - 'fit_times': Time to fit each fold
                 - 'score_times': Time to score each fold
        """
        from sklearn.model_selection import KFold, StratifiedKFold
        
        if scoring is None:
            if self.task_type == 'regression':
                scoring = ['mse', 'mae', 'r2_score']
            else:
                scoring = ['accuracy', 'f1_score', 'roc_auc_score']
        
        # Prepare CV splitter
        if self.task_type == 'classification':
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        # Initialize results storage
        fold_scores = {metric: [] for metric in scoring}
        fit_times = []
        score_times = []
        
        logger.info(f"[Pipeline] Starting {cv}-fold cross-validation")
        logger.info(f"[Pipeline] Metrics: {', '.join(scoring)}")
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            logger.info(f"[Pipeline] Fold {fold_idx + 1}/{cv}")
            
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
            
            # Create a new pipeline instance for this fold
            fold_pipeline = TabularPipeline(
                model_name=self.model_name,
                task_type=self.task_type,
                tuning_strategy=self.tuning_strategy,
                tuning_params=self.tuning_params.copy() if self.tuning_params else {},
                model_params=self.model_params.copy() if self.model_params else {},
                processor_params=self.processor_params.copy() if self.processor_params else {}
            )
            
            # Fit and evaluate
            import time
            start_fit = time.time()
            fold_pipeline.fit(X_train_fold, y_train_fold)
            fit_time = time.time() - start_fit
            fit_times.append(fit_time)
            
            start_score = time.time()
            # evaluate() returns a dict regardless of output_format
            metrics = fold_pipeline.evaluate(X_test_fold, y_test_fold, output_format='json')
            score_time = time.time() - start_score
            score_times.append(score_time)
            
            # Store scores for each metric
            for metric in scoring:
                if metric in metrics:
                    fold_scores[metric].append(metrics[metric])
                else:
                    logger.warning(f"[Pipeline] Metric '{metric}' not found in evaluation results for fold {fold_idx + 1}")
                    fold_scores[metric].append(np.nan)
        
        # Calculate mean and std for each metric
        mean_scores = {}
        std_scores = {}
        
        for metric in scoring:
            scores = np.array(fold_scores[metric])
            # Filter out NaN values
            valid_scores = scores[~np.isnan(scores)]
            if len(valid_scores) > 0:
                mean_scores[metric] = np.mean(valid_scores)
                std_scores[metric] = np.std(valid_scores)
            else:
                mean_scores[metric] = np.nan
                std_scores[metric] = np.nan
        
        results = {
            'mean_scores': mean_scores,
            'std_scores': std_scores,
            'fold_scores': fold_scores,
            'fit_times': fit_times,
            'score_times': score_times,
            'mean_fit_time': np.mean(fit_times),
            'mean_score_time': np.mean(score_times)
        }
        
        logger.info("\n[Pipeline] Cross-Validation Results:")
        logger.info("="*60)
        for metric in scoring:
            mean_val = mean_scores[metric]
            std_val = std_scores[metric]
            if not np.isnan(mean_val):
                logger.info(f"[Pipeline] {metric}: {mean_val:.4f} (+/- {std_val:.4f})")
            else:
                logger.info(f"[Pipeline] {metric}: NaN")
        logger.info(f"[Pipeline] Mean fit time: {np.mean(fit_times):.2f}s")
        logger.info(f"[Pipeline] Mean score time: {np.mean(score_times):.2f}s")
        logger.info("="*60)
        
        return results

    def baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        models: list | str | None = None,
        time_limit: int = 60
        ):
        """
        Trains and evaluates baseline models using AutoGluon on the provided train/test split.
        Supports both classification and regression tasks.
        """

        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            raise ImportError("AutoGluon is not installed. Install it with: pip install autogluon")

        logger.info("Preparing data for AutoGluon...")

        # Prepare data with target column
        X_train_with_label = X_train.copy()
        X_train_with_label['__target__'] = y_train.values if hasattr(y_train, 'values') else y_train
        X_test_with_label = X_test.copy()
        X_test_with_label['__target__'] = y_test.values if hasattr(y_test, 'values') else y_test

        # Configure model hyperparameters
        hyperparameters = None
        if models is not None:
            models_to_run = [models] if isinstance(models, str) else models
            model_map = {
                'xgboost': 'XGB', 'catboost': 'CAT', 'randomforest': 'RF', 'lightgbm': 'GBM',
                'extratrees': 'XT', 'knn': 'KNN', 'linear': 'LR', 'neuralnet': 'NN_TORCH'
            }
            ag_models = [model_map.get(m.lower(), m.upper()) for m in models_to_run]
            hyperparameters = {model: {} for model in ag_models}

        # Determine problem type and evaluation metric
        if self.task_type == 'regression':
            problem_type = 'regression'
            eval_metric = 'root_mean_squared_error'
        else:
            problem_type = 'multiclass' if len(y_train.unique()) > 2 else 'binary'
            eval_metric = 'accuracy'

        logger.info(f"Training AutoGluon predictor ({problem_type}) with time_limit={time_limit}s...")
        start_time = time.time()
        predictor = TabularPredictor(
            label='__target__',
            problem_type=problem_type,
            eval_metric=eval_metric,
            verbosity=2
         ).fit(
            train_data=X_train_with_label,
            time_limit=time_limit,
            hyperparameters=hyperparameters,
            presets='medium_quality'
        )
        total_train_time = time.time() - start_time

        logger.info("Generating test predictions using best model ensemble...")
        predictions = predictor.predict(X_test)

        # Calculate metrics based on task type
        if self.task_type == 'regression':
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            overall_mse = mean_squared_error(y_test, predictions)
            overall_mae = mean_absolute_error(y_test, predictions)
            overall_r2 = r2_score(y_test, predictions)
            overall_rmse = np.sqrt(overall_mse)
        else:
            from sklearn.metrics import accuracy_score, f1_score
            overall_accuracy = accuracy_score(y_test, predictions)
            overall_f1 = f1_score(y_test, predictions, average='weighted')

        leaderboard = predictor.leaderboard(X_test_with_label, silent=True)
        baseline_results = []

        logger.info("Calculating per-model scores...")
        for _, row in leaderboard.iterrows():
            model_name = row['model']

            # Individual model predictions
            model_pred = predictor.predict(X_test, model=model_name)

            if self.task_type == 'regression':
                # Model-specific regression metrics
                model_mse = mean_squared_error(y_test, model_pred)
                model_mae = mean_absolute_error(y_test, model_pred)
                model_r2 = r2_score(y_test, model_pred)

                baseline_results.append({
                    "Model": model_name,
                    "Validation Score": row['score_val'],
                    "MSE": model_mse,
                    "MAE": model_mae,
                    "R2": model_r2,
                    "Training Time": row['fit_time']
                })
            else:
                # Model-specific classification metrics
                model_f1 = f1_score(y_test, model_pred, average='weighted')
                model_accuracy = accuracy_score(y_test, model_pred)

                baseline_results.append({
                    "Model": model_name,
                    "Validation Score": row['score_val'],
                    "Accuracy": model_accuracy,
                    "F1 Score": model_f1,
                    "Training Time": row['fit_time']
                })

        # Log results
        if self.task_type == 'regression':
            logger.info("\nAutoGluon Baseline Evaluation Report (Regression)")
            logger.info(f"Overall RMSE: {overall_rmse:.4f}")
            logger.info(f"Overall MSE: {overall_mse:.4f}")
            logger.info(f"Overall MAE: {overall_mae:.4f}")
            logger.info(f"Overall R²: {overall_r2:.4f}")
            logger.info(f"Total Training Time: {total_train_time:.2f}s\n")

            header = f"{'Model':<30} {'Val Score':<15} {'MSE':<15} {'MAE':<15} {'R2':<15} {'Train Time (s)':<15}"
            logger.info(header)
            for result in baseline_results:
                logger.info(
                    f"{result['Model']:<30} {result['Validation Score']:<15.4f} "
                    f"{result['MSE']:<15.4f} {result['MAE']:<15.4f} {result['R2']:<15.4f} "
                    f"{result['Training Time']:<15.2f}"
                )
        else:
            logger.info("\nAutoGluon Baseline Evaluation Report (Classification)")
            logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
            logger.info(f"Overall Weighted F1-Score: {overall_f1:.4f}")
            logger.info(f"Total Training Time: {total_train_time:.2f}s\n")

            header = f"{'Model':<30} {'Val Score':<15} {'Accuracy':<15} {'F1 Score':<15} {'Train Time (s)':<15}"
            logger.info(header)
            for result in baseline_results:
                logger.info(
                    f"{result['Model']:<30} {result['Validation Score']:<15.4f} "
                    f"{result['Accuracy']:<15.4f} {result['F1 Score']:<15.4f} "
                    f"{result['Training Time']:<15.2f}"
                )
        logger.info("=" * 80)

        # Return appropriate results
        if self.task_type == 'regression':
            return {
                "overall_rmse": overall_rmse,
                "overall_mse": overall_mse,
                "overall_mae": overall_mae,
                "overall_r2": overall_r2,
                "total_training_time": total_train_time,
                "individual_models": baseline_results,
                "predictor": predictor,
                "leaderboard": leaderboard
            }
        else:
            return {
                "overall_accuracy": overall_accuracy,
                "overall_f1": overall_f1,
                "total_training_time": total_train_time,
                "individual_models": baseline_results,
                "predictor": predictor,
                "leaderboard": leaderboard
            }

    

    def evaluate_checkpoints(self, X_test, y_test, checkpoint_dir, epochs, map_location: str | None = None):
        results = {}
        for ep in epochs:
            ckpt_name = f"{type(self.model).__name__}_epoch{ep}.pt"
            ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                logger.warning(f" - Missing checkpoint for epoch {ep}, skipping")
                continue
    
            logger.info(f"\n🔁 Evaluating checkpoint at epoch {ep}")
            self.model = self.tuner.load_checkpoint(self.model, ckpt_path, map_location or 'cpu')
    
            for name, param in self.model.model.named_parameters():
                logger.info(f"   {name} mean: {torch.mean(param).item():.6f}")
                break
    
            # then evaluate normally
            metrics = self.evaluate(X_test, y_test)
            results[ep] = metrics
    
        return results



    def get_params(self, deep: bool = True) -> dict:
        """
         Get parameters for this estimator.

         Parameters
         ----------
         deep : bool, default=True
        If True, will return the parameters for this estimator and
        contained subobjects that are estimators (like the processor or the underlying model).

        Returns
        -------
        params : dict
        Parameter names mapped to their values.
        """
 
        user_tuning_params = self.tuning_params if isinstance(self.tuning_params, dict) else (self.tuning_params or {})
        model_params = self.model_params if isinstance(self.model_params, dict) else (self.model_params or {})
        processor_params = (
            self.processor_params
            if isinstance(self.processor_params, dict)
            else (self.processor_params or {})
        )

        # --- NEW: compute "effective" tuning params = defaults + user overrides ---
        finetune_mode = user_tuning_params.get("finetune_mode", getattr(self, "finetune_mode", "meta-learning"))
        strategy = getattr(self, "tuning_strategy", "inference")

        # Match your TuningManager logic
        finetune_method = user_tuning_params.get("finetune_method", None)
        selected_strategy = strategy
        if strategy == "finetune" and finetune_method == "peft":
            selected_strategy = "peft"
        elif strategy == "finetune":
            selected_strategy = "finetune"

        # Defaults resolver that DOES NOT depend on isinstance()
        def _default_tuning_config(model_name: str, finetune_mode: str) -> dict:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # TabICL / Orion defaults (meta-learning)
            if model_name in {"TabICL", "OrionMSP", "OrionBix", "OrionMSPv1.5"}:
                if finetune_mode == "meta-learning":
                    return {
                    "device": device,
                    "epochs": 5,
                    "learning_rate": 2e-6,
                    "show_progress": True,
                    "support_size": 48,
                    "query_size": 32,
                    "n_episodes": 1000,
                    }
                # TabICL simple SFT defaults
                return {
                "device": device,
                "epochs": 5,
                "learning_rate": 1e-5,
                "batch_size": 16,
                "show_progress": True,
                }

            if model_name == "TabPFN":
                if finetune_mode == "sft":
                    return {
                    "device": device,
                    "epochs": 25,
                    "learning_rate": 1e-5,
                    "show_progress": True,
                    "query_set_ratio": 0.3,
                    "weight_decay": 1e-4,
                    # max_episode_size is data-dependent; leave it out here
                    }
                return {
                "device": device,
                "epochs": 3,
                "learning_rate": 1e-5,
                "batch_size": 256,
                "show_progress": True,
                }

            if model_name == "ConTextTab":
                return {
                "device": device,
                "epochs": 5,
                "learning_rate": 1e-4,
                "batch_size": 128,
                "show_progress": True,
                }

            if model_name == "TabDPT":
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

            if model_name in {"Mitra", "Tab2D"}:
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

            if model_name == "Limix":
                return {
                "device": device,
                "epochs": 5,
                "learning_rate": 1e-5,
                "show_progress": True,
                "support_size": 48,
                "query_size": 32,
                "n_episodes": 1000,
                }

            return {"device": device}

        defaults = _default_tuning_config(self.model_name, finetune_mode)

        # Always include finetune_mode in tuning_params (even if defaults also include it)
        effective_tuning_params = dict(defaults)
        effective_tuning_params["finetune_mode"] = finetune_mode

        # User overrides win (even if empty dict -> no changes)
        effective_tuning_params.update(user_tuning_params or {})

        # Base params (always include keys)
        params = {
        "model_name": self.model_name,
        "task_type": self.task_type,
        "tuning_strategy": self.tuning_strategy,
        "tuning_params": effective_tuning_params,  # <-- this is what you want
        "processor_params": processor_params,
        "model_params": model_params,
        "model_checkpoint_path": self.model_checkpoint_path,
        "finetune_mode": self.finetune_mode,
        }

        if not deep:
            return params

        # Deep: Processor params
        if hasattr(self.processor, "get_params"):
            try:
                proc_params = self.processor.get_params(deep=True)
                for key, value in proc_params.items():
                    params[f"processor__{key}"] = value
            except Exception as e:
                logger.debug(f"[Pipeline] Could not get params from processor: {e}")
    
        # Deep: Model params
        if self.model is not None and hasattr(self.model, "get_params"):
            try:
                model_inner_params = self.model.get_params(deep=True)
                for key, value in model_inner_params.items():
                    params[f"model__{key}"] = value
            except Exception as e:
                logger.debug(f"[Pipeline] Could not get params from model: {e}")
        elif self.model is not None:
            if hasattr(self.model, "config"):
                params["model__config"] = self.model.config
            elif hasattr(self.model, "args"):
                params["model__args"] = self.model.args
    
        # Optional: expose what strategy resolution decided
        params["tuning__selected_strategy"] = selected_strategy

        return params


