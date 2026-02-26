"""
Ensemble LimiX Regression Model for TabTune Pipeline.

This module provides an ensemble version of LimixRegressor that uses multiple
model instances with different configurations to match official LimiX performance.
Each estimator handles its own preprocessing internally using the new preprocess.py module.
"""
import numpy as np
import pandas as pd
import torch
import logging
import json
import os
from sklearn.base import RegressorMixin
from tabtune.models.regression.limix.regressor import LimixRegressor

logger = logging.getLogger(__name__)


class LimixRegressorEnsemble(RegressorMixin):
    """
    Ensemble LimiX Regressor using multiple FeaturesTransformer instances.
    
    This regressor creates multiple model instances (similar to official LimiX's
    ensemble approach) and averages their predictions for better performance.
    """
    
    def __init__(self, 
                 n_estimators=8,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 repo_id="stableai-org/LimiX-16M", 
                 filename="LimiX-16M.ckpt",
                 nlayers=6,
                 nhead=6,
                 embed_dim=192,
                 hid_dim=768,
                 dropout=0.1,
                 seed=0,
                 **kwargs):
        """
        Initialize LimixRegressorEnsemble.
        
        Args:
            n_estimators: Number of model instances in ensemble (default: 8, matching official)
            device: Device to run on ('cuda' or 'cpu')
            repo_id: HuggingFace repository ID for model weights
            filename: Model checkpoint filename
            nlayers: Number of transformer layers
            nhead: Number of attention heads
            embed_dim: Embedding dimension
            hid_dim: Hidden dimension
            dropout: Dropout rate
            seed: Random seed (each estimator gets seed + estimator_index)
            **kwargs: Additional model parameters
        """
        self.n_estimators = n_estimators
        self.device = device
        self.repo_id = repo_id
        self.filename = filename
        self.nlayers = nlayers
        self.nhead = nhead
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.seed = seed
        self.model_params = kwargs
        
        self.estimators = []
        self.y_mean_ = None
        self.y_std_ = None
        
        # Load inference config (default or from file)
        self.inference_config = self._load_inference_config(kwargs.get('inference_config', None))
        
        logger.info(f"[LimixRegressorEnsemble] Initializing {n_estimators} model instances with official LimiX preprocessing pipelines")
    
    def _load_inference_config(self, inference_config):
        """Load inference configuration from file, dict, or use default."""
        if inference_config is None:
            # Try to load from LimiX config directory
            limix_config_path = "/home/jovyan/LimiX/config/reg_default_noretrieval.json"
            if os.path.exists(limix_config_path):
                try:
                    with open(limix_config_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load config from {limix_config_path}: {e}")
            
            # Fallback: minimal 4-pipeline config (will be repeated for 8 estimators)
            return [
                {
                    "RebalanceFeatureDistribution": {
                        "worker_tags": ["quantile_uniform_all_data"],
                        "discrete_flag": False,
                        "original_flag": True,
                        "svd_tag": "svd"
                    },
                    "CategoricalFeatureEncoder": {
                        "encoding_strategy": "ordinal_strict_feature_shuffled"
                    },
                    "FeatureShuffler": {
                        "mode": "shuffle"
                    },
                    "retrieval_config": {
                        "use_retrieval": False
                    }
                },
                {
                    "RebalanceFeatureDistribution": {
                        "worker_tags": ["power"],
                        "discrete_flag": False,
                        "original_flag": False,
                        "svd_tag": None
                    },
                    "CategoricalFeatureEncoder": {
                        "encoding_strategy": "onehot"
                    },
                    "FeatureShuffler": {
                        "mode": "shuffle"
                    },
                    "retrieval_config": {
                        "use_retrieval": False
                    }
                }
            ]
        
        if isinstance(inference_config, str):
            if os.path.exists(inference_config):
                with open(inference_config, 'r') as f:
                    return json.load(f)
            else:
                raise FileNotFoundError(f"Inference config file not found: {inference_config}")
        
        return inference_config
    
    def fit(self, X, y):
        """
        Fit all ensemble members on training data.
        
        Args:
            X: Training features (pd.DataFrame or np.ndarray)
            y: Training targets (pd.Series or np.ndarray)
        
        Returns:
            self
        """
        # Normalize target (same for all estimators)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        if isinstance(y, pd.Series):
            y = y.values
        
        y = np.array(y).flatten().astype(float)
        
        # Store normalization parameters
        self.y_mean_ = np.mean(y)
        self.y_std_ = np.std(y)
        if self.y_std_ == 0:
            self.y_std_ = 1.0
        
        # Detect categorical features
        if isinstance(X, pd.DataFrame):
            categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
            categorical_indices = [X.columns.get_loc(col) for col in categorical_features]
            X_array = X.values
        else:
            # For numpy arrays, assume all numeric (no categorical detection)
            categorical_indices = []
            X_array = np.array(X)
        
        # Create and fit each estimator with different preprocessing pipelines (matching official LimiX)
        self.estimators = []
        
        for i in range(self.n_estimators):
            logger.info(f"[LimixRegressorEnsemble] Fitting estimator {i+1}/{self.n_estimators}")
            
            # Each estimator gets a different seed for variation
            estimator_seed = self.seed + i * 1000
            
            # Get inference config for this estimator (cycle through available configs)
            estimator_config = self.inference_config[i % len(self.inference_config)]
            
            # Create estimator with its own inference config
            # Each estimator will handle its own preprocessing internally
            estimator_kwargs = {k: v for k, v in self.model_params.items() if k != 'features_per_group'}
            
            estimator = LimixRegressor(
                device=self.device,
                repo_id=self.repo_id,
                filename=self.filename,
                nlayers=self.nlayers,
                nhead=self.nhead,
                embed_dim=self.embed_dim,
                hid_dim=self.hid_dim,
                dropout=self.dropout,
                seed=estimator_seed,
                inference_config=[estimator_config],  # Single config per estimator
                use_ensemble=True,  # Enable ensemble preprocessing within each estimator
                **estimator_kwargs
            )
            
            # Fit the estimator on raw training data (it will do preprocessing internally)
            estimator.fit(X_array, y)
            self.estimators.append(estimator)
        
        logger.info(f"[LimixRegressorEnsemble] All {self.n_estimators} estimators fitted")
        return self
    
    def predict(self, X):
        """
        Predict target values by averaging predictions from all ensemble members.
        
        Args:
            X: Input features (pd.DataFrame or np.ndarray)
        
        Returns:
            np.ndarray: Averaged predicted target values (denormalized)
        """
        if len(self.estimators) == 0:
            raise ValueError("Model must be fitted before prediction")
        
        # Get predictions from all estimators
        # Each estimator handles its own preprocessing internally
        predictions_list = []
        
        # Convert X to array if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        for i, estimator in enumerate(self.estimators):
            logger.debug(f"[LimixRegressorEnsemble] Getting prediction from estimator {i+1}/{self.n_estimators}")
            
            # Each estimator handles its own preprocessing and prediction
            # It will concatenate train+test internally before preprocessing (matching official LimiX)
            pred = estimator.predict(X_array)
            predictions_list.append(pred)
        
        # Stack predictions and average
        predictions_array = np.stack(predictions_list, axis=0)  # [n_estimators, n_samples]
        predictions_ensemble = np.mean(predictions_array, axis=0)  # [n_samples]
        
        logger.info(f"[LimixRegressorEnsemble] Averaged predictions from {self.n_estimators} estimators")
        
        return predictions_ensemble
