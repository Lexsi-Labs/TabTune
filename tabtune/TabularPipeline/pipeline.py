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
    brier_score_loss
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
        
        self.processor = DataProcessor(model_name=self.model_name, **(processor_params or {}))
        self.tuner = TuningManager()
        self.model = None 
        self.model_checkpoint_path = model_checkpoint_path
        self.finetune_mode = finetune_mode

        if self.tuning_strategy in ('finetune', 'base-ft', 'peft'):
            self.tuning_params['finetune_mode'] = self.finetune_mode
        

        if self.model_name in ['TabPFN']:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            config = {'device': device, 'ignore_pretraining_limits': True}
            config.update(self.model_params)
            logger.info(f"[Pipeline] Config: {config}")
            self.model = TabPFNClassifier(**config)
            if self.tuning_strategy in ['finetune', 'base-ft', 'peft'] and hasattr(self.model, '_initialize_model_variables'):
                self.model._initialize_model_variables()


        elif self.model_name == 'ContextTab':
            self.model = ConTextTabClassifier(**self.model_params)
    
        elif self.model_name in ['TabICL', 'OrionBix','OrionMSP']:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            else:
                self.model = OrionBixClassifier(**config)
                if self.tuning_strategy == 'finetune':
                    self.model._load_model()

        elif self.model_name == 'TabDPT':
            # Use GPU if available, otherwise fall back to CPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

        # Handle models that require late initialization (processor needs to be fit first)
        elif self.model_name not in ['Mitra', 'APT']:
            raise ValueError(f"Model '{self.model_name}' not supported.")


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
        
        logger.info(f"[Pipeline] TabularPipeline initialized for model '{self.model_name}', task '{self.task_type}', with strategy '{self.tuning_strategy}'")
        ("TabTune - Unified Library for fine-tuning and inference of Foundational Tabular Models")

    def __del__(self):
        """Cleanup method to properly shut down resources when pipeline is destroyed."""
        # ContextTab ZMQ server cleanup is handled automatically by atexit.register()
        # in the start_embedding_server function, so no manual cleanup needed
        pass


    def fit(self, X: pd.DataFrame, y: pd.Series):

        self.X_raw_train = X.copy()
        self.y_raw_train = y.copy()
        
        logger.info("[Pipeline] Starting fit process")

    # Special handling for models that are TRULY self-contained and do not need the pipeline's processor for inference
        if self.tuning_strategy == 'inference' and isinstance(self.model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier)):
            logger.info("[Pipeline] Handing off to TuningManager for inference setup.")
            self.processor.fit(X, y)
            self.model = self.tuner.tune(self.model, X, y, strategy=self.tuning_strategy)
            self._is_fitted = True
            logger.info("[Pipeline] Fit process complete")
            return self

    #For ALL other models and strategies (including ConTextTab), we must fit the DataProcessor first.
        logger.info("[Pipeline] Fitting data processor...")
        self.processor.fit(X, y) 

    # Handle ConTextTab inference AFTER the processor has been fitted
        if self.tuning_strategy == 'inference' and isinstance(self.model, ConTextTabClassifier):
            logger.info(f"[Pipeline] Handing off to TuningManager for inference setup for {self.model_name}")
            # The tuner calls the model's native .fit() method with the raw data
            self.model = self.tuner.tune(self.model, X, y, strategy=self.tuning_strategy)
            self._is_fitted = True
            logger.info("[Pipeline] Fit process complete")
            return self

    # Late initialization for models that need info from the fitted processor
        if self.model is None:
            logger.info("[Pipeline] Performing late initialization of the model...")
            if self.model_name == 'Mitra':
                n_classes = len(self.processor.custom_preprocessor_.label_encoder_.classes_)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            device_str = self.tuning_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            device = torch.device(device_str)
            self.model.to(device)
            if self.model_name == 'Mitra':
                # Set device type on model or wrapper
                try:
                    setattr(self.model, 'device_type', device_str)
                except Exception:
                    pass
            if isinstance(self.model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier)):
                self.model.device = device


        if isinstance(self.model, ConTextTabClassifier) and self.tuning_strategy in ['finetune', 'base-ft']:
            logger.info("[Pipeline] Preparing raw data for ConTextTab fine-tuning")
            if not isinstance(X, pd.DataFrame):
                X_to_tune = pd.DataFrame(X)
            else:
                X_to_tune = X.copy()
            if not isinstance(y, pd.Series):
                y_to_tune = pd.Series(y)
            else:
                y_to_tune = y.copy()
        else:
            logger.info("[Pipeline] Transforming data for model tuning...")
            processed_data = self.processor.transform(X, y)
            if isinstance(processed_data, tuple):
                self.X_train_processed_, self.y_train_processed_ = processed_data
            else:
                self.X_train_processed_ = processed_data
                if hasattr(self.processor, 'custom_preprocessor_') and hasattr(self.processor.custom_preprocessor_, 'label_encoder_') and self.processor.custom_preprocessor_.label_encoder_ is not None:
                    self.y_train_processed_ = self.processor.custom_preprocessor_.label_encoder_.transform(y)
                else: # Fallback for models without a main processor label encoder
                    self.y_train_processed_ = y 
        
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

        if isinstance(self.model, TabDPTClassifier) and self.tuning_strategy in ['finetune', 'base-ft', 'peft']:
            logger.info("[Pipeline] Finalizing TabDPT setup after fine-tuning")
            self.model.num_classes = len(np.unique(y_to_tune))
            # Fit the model for inference after fine-tuning
            self.model.fit(X_to_tune, y_to_tune)

        self._is_fitted = True
        logger.info("[Pipeline] Fit process complete")
        if self.tuning_strategy == "peft":
            logger.info("[Pipeline] PEFT STATUS SUMMARY")
            logger.info("[Pipeline] LoRA adapters were applied to the model")
            logger.warning("[Pipeline] Note: PEFT compatibility with tabular models is experimental")
            logger.info("[Pipeline] If you encounter issues, try 'base-ft' strategy for full compatibility")
            logger.info("[Pipeline] See documentation for more details on PEFT limitations")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("You must call fit() on the pipeline before calling predict().")
        
        logger.info("[Pipeline] Starting prediction")

        if hasattr(self.model, 'model') and isinstance(self.model.model, torch.nn.Module):
            self.model.model.eval()
        elif hasattr(self.model, 'model_') and isinstance(self.model.model_, torch.nn.Module):
            self.model.model_.eval()

        if isinstance(self.model, TabPFNClassifier):
            if self.tuning_strategy in ['finetune', 'base-ft', 'peft']:
                logger.debug("[Pipeline] Setting TabPFN inference context (without refitting weights)...")
            
            # Store current model weights
                saved_weights = self.model.model_.state_dict()
                self.model.model_.load_state_dict(saved_weights)
            
            # Call fit to set up inference context
                self.model.fit(self.X_train_processed_, self.y_train_processed_)
            
            # Restore fine-tuned weights immediately
                #self.model.model_.load_state_dict(saved_weights)
                logger.debug("[Pipeline] Restored fine-tuned weights after context setup")
        
            X_processed = self.processor.transform(X)
            return self.model.predict(X_processed)
        

        if isinstance(self.model, TabDPTClassifier):
            # Apply the same preprocessing as during fit()
            X_processed = self.processor.transform(X)
            
            # Get integer predictions from model
            predictions_raw = self.model.predict(X_processed)
            
            # Convert integer predictions back to original string labels (same as TabICL/OrionMSP/OrionBix)
            predictions = self.processor.custom_preprocessor_.label_encoder_.inverse_transform(predictions_raw)
            return predictions
            

        if isinstance(self.model, (ConTextTabClassifier)):
            logger.debug(f"[Pipeline] Using model's native in-context prediction for {type(self.model).__name__}")
            predictions = self.model.predict(X)
            
        elif isinstance(self.model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier)):
            logger.debug(f"[Pipeline] Using model's native in-context prediction for {type(self.model).__name__}")  
            X_processed = self.processor.transform(X)
            #predictions = self.model.predict(X)
            
            if self.tuning_strategy == 'inference':
                # For inference mode, pass raw data directly to the model
                # The model's internal encoders will handle the preprocessing
                predictions = self.model.predict(X)
            else:
                # For fine-tuning mode, use preprocessed data to match training
                label_encoder = self.processor.custom_preprocessor_.label_encoder_
                known_class = label_encoder.classes_[0]
                y_dummy = pd.Series([known_class] * len(X))
                X_query, _ = self.processor.transform(X, y_dummy)
                # Convert to DataFrame to maintain feature names for sklearn compatibility
                if not isinstance(X_query, pd.DataFrame):
                    # Prefer processor feature names if available; else fall back to input X
                    cols = None
                    if hasattr(self.processor, "feature_names_") and self.processor.feature_names_ is not None:
                        cols = list(self.processor.feature_names_)
                    elif hasattr(X, "columns"):
                        cols = list(X.columns)
                    # Avoid shape/columns mismatch
                    if cols is not None and hasattr(X_query, "shape") and X_query.shape[1] != len(cols):
                        cols = None
                    X_query = pd.DataFrame(X_query, columns=cols)
                predictions = self.model.predict(X_query)
            
            # Convert numerical predictions back to string format for evaluation
            if self.tuning_strategy in ['finetune', 'base-ft', 'peft'] and hasattr(self.processor, 'custom_preprocessor_') and hasattr(self.processor.custom_preprocessor_, 'label_encoder_'):
                predictions = self.processor.custom_preprocessor_.label_encoder_.inverse_transform(predictions)

        
        elif self.model_name == 'Mitra':
            logger.debug("[Pipeline] Using in-context prediction for Mitra (Tab2D)")
            label_encoder = self.processor.custom_preprocessor_.label_encoder_
            known_class = label_encoder.classes_[0]
            y_dummy = pd.Series([known_class] * len(X))

            X_query, _ = self.processor.transform(X, y_dummy)
            
            X_support, y_support = self.X_train_processed_, self.y_train_processed_
            
            device_str = self.tuning_params.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
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
            # TabPFN and other standard models do not need y_dummy for prediction transforms
            logger.debug("[Pipeline] Applying learned transformations to new data")
            X_processed = self.processor.transform(X) # Pass only X
            logger.debug("[Pipeline] Getting predictions from the model")
            predictions = self.model.predict(X_processed)
        return predictions


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
            # Apply the same preprocessing as during fit()
            X_processed = self.processor.transform(X)
            # Use stored defaults from model initialization
            return self.model.ensemble_predict_proba(X_processed)

        elif isinstance(self.model, TabPFNClassifier):
            # Special handling for fine-tuned TabPFN to set inference context
            if self.tuning_strategy in ['finetune', 'base-ft', 'peft']:
                logger.debug("[Pipeline] Setting TabPFN inference context for proba...")
                self.model.fit(self.X_train_processed_, self.y_train_processed_)
            
            X_processed = self.processor.transform(X)
            return self.model.predict_proba(X_processed)
            
        
        if isinstance(self.model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, ConTextTabClassifier)):
            logger.debug("[Pipeline] Using model's native predict_proba method")
            
            X_processed = self.processor.transform(X)
            if isinstance(self.model, (ConTextTabClassifier)):
                 return self.model.predict_proba(X)

            if isinstance(self.model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier)):
                if self.tuning_strategy == 'inference':
                    return self.model.predict_proba(X)
                else:
                    label_encoder = self.processor.custom_preprocessor_.label_encoder_
                    known_class = label_encoder.classes_[0]
                    y_dummy = pd.Series([known_class] * len(X))
                    
                    X_query, _ = self.processor.transform(X, y_dummy)
                    # Convert to DataFrame to maintain feature names for sklearn compatibility
                    if not isinstance(X_query, pd.DataFrame):
                        # Prefer processor feature names if available; else fall back to input X
                        cols = None
                        if hasattr(self.processor, "feature_names_") and self.processor.feature_names_ is not None:
                            cols = list(self.processor.feature_names_)
                        elif hasattr(X, "columns"):
                            cols = list(X.columns)

                        # Avoid shape/columns mismatch
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
                    # Not implemented for Mitra
                    raise NotImplementedError("predict_proba is not implemented for Mitra (Tab2D)")
                    raise NotImplementedError(f"predict_proba is not implemented for model type {type(self.model).__name__}")
        
        logger.info("[Pipeline] Probability prediction complete")
        return probabilities

    ############### Helpers #############################
    def _get_model_class_labels(self):
        """
        Best-effort to recover the class label order that predict_proba columns use.
        """
        # sklearn-style estimators
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

        # Force 2D and validate input
        if probabilities is None:
            logger.warning("[Pipeline] Probabilities are None in _align_proba_to_encoder")
            return None
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)
        
        # Check for empty probabilities
        if probabilities.size == 0:
            logger.warning("[Pipeline] Empty probabilities array in _align_proba_to_encoder")
            return None

        encoder_classes = list(label_encoder.classes_)
        K = len(encoder_classes)

        # Binary convenience cases
        if K == 2:
            if probabilities.shape[1] == 1:
                # Validate that single column probabilities are in [0, 1]
                p_pos = probabilities[:, 0]
                if np.any(p_pos < 0) or np.any(p_pos > 1):
                    logger.warning(f"[Pipeline] Single-column probabilities outside [0,1] range (min: {p_pos.min():.6f}, max: {p_pos.max():.6f})")
                # assume encoder maps positives to label 1 (LabelEncoder does 0..K-1)
                p_neg = 1.0 - p_pos
                return np.column_stack([p_neg, p_pos])
            # or two columns already — validate and return
            elif probabilities.shape[1] == 2:
                # Validate that probabilities are in [0, 1]
                if np.any(probabilities < 0) or np.any(probabilities > 1):
                    logger.warning(f"[Pipeline] Two-column probabilities outside [0,1] range (min: {probabilities.min():.6f}, max: {probabilities.max():.6f})")
                return probabilities
            else:
                logger.warning(f"[Pipeline] Unexpected number of probability columns ({probabilities.shape[1]}) for binary classification")
                return None

        # Multiclass: align by class labels
        model_labels = self._get_model_class_labels()
        # If we can't recover model labels, assume current order already matches encoder
        if not model_labels or probabilities.shape[1] == K and set(model_labels) == set(encoder_classes):
            # Still ensure shape matches
            if probabilities.shape[1] == K:
                # Validate that probabilities are in [0, 1]
                if np.any(probabilities < 0) or np.any(probabilities > 1):
                    logger.warning(f"[Pipeline] Multiclass probabilities outside [0,1] range (min: {probabilities.min():.6f}, max: {probabilities.max():.6f})")
                return probabilities
            else:
                logger.warning(f"[Pipeline] Shape mismatch: expected {K} columns, got {probabilities.shape[1]}")
                return None

        # Build aligned matrix (zeros for any missing classes)
        aligned = np.zeros((probabilities.shape[0], K), dtype=float)

        # Map model label -> encoder index
        try:
            model_to_encoder_idx = {
                lbl: int(label_encoder.transform([lbl])[0]) for lbl in model_labels
            }
        except Exception:
            # If transform fails (types differ), fall back to identity numeric mapping
            model_to_encoder_idx = {}
            for j, lbl in enumerate(model_labels):
                try:
                    enc_idx = int(lbl)  # numeric labels already 0..K-1
                except Exception:
                    enc_idx = j
                model_to_encoder_idx[lbl] = enc_idx

        for j_model, lbl in enumerate(model_labels):
            if j_model >= probabilities.shape[1]:
                break
            enc_j = model_to_encoder_idx.get(lbl, None)
            if enc_j is not None and 0 <= enc_j < K:
                aligned[:, enc_j] = probabilities[:, j_model]

        # Final validation of aligned probabilities
        if np.any(aligned < 0) or np.any(aligned > 1):
            logger.warning(f"[Pipeline] Aligned probabilities outside [0,1] range (min: {aligned.min():.6f}, max: {aligned.max():.6f})")
        
        # Check if any samples have all-zero probabilities (missing class predictions)
        zero_rows = np.all(aligned == 0, axis=1)
        if np.any(zero_rows):
            logger.warning(f"[Pipeline] {np.sum(zero_rows)} samples have all-zero probabilities (missing class predictions)")

        return aligned
    
    
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
            # Case 1: Custom preprocessor has a label encoder (Mitra, TabICL, APT, OrionMSP, OrionBix)
            if hasattr(self.processor, 'custom_preprocessor_') and hasattr(self.processor.custom_preprocessor_, 'label_encoder_'):
                y_test_encoded = self.processor.custom_preprocessor_.label_encoder_.transform(y_test)
            elif isinstance(self.model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier)):
                y_test_encoded = self.model.y_encoder_.transform(y_test)
            elif isinstance(self.model, TabPFNClassifier):
                le = LabelEncoder()
                le.classes_ = self.model.classes_ # Use the classes the model learned during .fit()
                y_test_encoded = le.transform(y_test)
            # Case 3: Standard pipeline with a main label encoder
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
            
            # Guard: AUC is undefined if the test fold has < 2 classes
            unique_test = np.unique(y_test_encoded)
            if len(unique_test) < 2:
                auc = float("nan")
            else:
                # Align probability columns to the SAME label order used by y_test_encoded
                # Choose the same encoder you used above when computing y_test_encoded
                if hasattr(self.processor, 'custom_preprocessor_') and hasattr(self.processor.custom_preprocessor_, 'label_encoder_'):
                    le = self.processor.custom_preprocessor_.label_encoder_
                elif isinstance(self.model, (TabICLClassifier, OrionBixClassifier, OrionMSPClassifier)):
                    le = self.model.y_encoder_
                elif isinstance(self.model, TabPFNClassifier):
                    le = LabelEncoder(); le.classes_ = self.model.classes_
                elif hasattr(self.processor, 'label_encoder_') and self.processor.label_encoder_ is not None:
                    le = self.processor.label_encoder_
                else:
                    raise RuntimeError("Could not find a fitted label encoder to align probabilities.")

                probs_aligned = self._align_proba_to_encoder(probabilities, le)

                # Binary vs multiclass handling with explicit labels to match encoded y
                K = len(le.classes_)
                if K == 2:
                    # probs_aligned has 2 columns by construction: [:, 1] is positive class
                    auc = roc_auc_score(y_test_encoded, probs_aligned[:, 1])
                else:
                    auc = roc_auc_score(
                        y_test_encoded,
                        probs_aligned,
                        labels=list(range(K)),   # encoded labels are 0..K-1
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

            if output_format == 'json':
                print(json.dumps(results, indent=4))
            elif output_format == 'rich':
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
        elif isinstance(self.model, (TabICLClassifier, OrionBixClassifier, OrionMSPClassifier)):
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
        
    def evaluate_fairness(self, X_test: pd.DataFrame, y_test: pd.Series, sensitive_features: pd.Series, output_format: str = 'rich'):
        """
        Calculates and provides a detailed report on group fairness metrics.
        """
        if not self._is_fitted:
            raise RuntimeError("You must call fit() on the pipeline before evaluating fairness.")

        predictions = self.predict(X_test)
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

    def _get_encoded_labels(self, y_true, y_pred):
        """Helper to consistently encode true and predicted labels."""
        y_true_encoded = None
        y_pred_encoded = None

        # Find the correct LabelEncoder
        le = None
        if hasattr(self.processor, 'custom_preprocessor_') and hasattr(self.processor.custom_preprocessor_, 'label_encoder_'):
            le = self.processor.custom_preprocessor_.label_encoder_
        elif isinstance(self.model, (TabICLClassifier, OrionMSPClassifier, OrionBixClassifier, TabPFNClassifier)):
             # Fit a temporary encoder on the training labels seen during .fit()
            le = LabelEncoder().fit(self.y_train_processed_ if self.y_train_processed_ is not None else y_true)
        else:
            raise RuntimeError("Could not find a fitted label encoder to evaluate metrics.")

        y_true_encoded = le.transform(y_true)
        # Handle cases where y_pred might be different (e.g., raw y_test for fairness)
        if y_pred is not None:
            y_pred_encoded = le.transform(y_pred)
            
        return y_true_encoded, y_pred_encoded

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
        Now returns per-model F1 scores along with validation scores and training time.
        """

        try:
            from autogluon.tabular import TabularPredictor
            from sklearn.metrics import accuracy_score, f1_score
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

        logger.info(f"Training AutoGluon predictor with time_limit={time_limit}s...")
        start_time = time.time()
        predictor = TabularPredictor(
            label='__target__',
            eval_metric='accuracy',
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

        overall_accuracy = accuracy_score(y_test, predictions)
        overall_f1 = f1_score(y_test, predictions, average='weighted')

        leaderboard = predictor.leaderboard(X_test_with_label, silent=True)
        baseline_results = []

        logger.info("Calculating per-model F1 scores...")
        for _, row in leaderboard.iterrows():
            model_name = row['model']

        # Individual model predictions
            model_pred = predictor.predict(X_test, model=model_name)

        # Model-specific F1 score
            model_f1 = f1_score(y_test, model_pred, average='weighted')

            baseline_results.append({
                "Model": model_name,
                "Validation Score": row['score_val'],
                "F1 Score": model_f1,
                "Training Time": row['fit_time']
            })

        logger.info("\nAutoGluon Baseline Evaluation Report")
        logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")
        logger.info(f"Overall Weighted F1-Score: {overall_f1:.4f}")
        logger.info(f"Total Training Time: {total_train_time:.2f}s\n")

        header = f"{'Model':<30} {'Val Score':<15} {'F1 Score':<15} {'Train Time (s)':<15}"
        logger.info(header)
        for result in baseline_results:
            logger.info(
                f"{result['Model']:<30} {result['Validation Score']:<15.4f} "
                f"{result['F1 Score']:<15.4f} {result['Training Time']:<15.2f}"
            )
        logger.info("=" * 80)

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