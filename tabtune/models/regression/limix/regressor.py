"""
LimiX Regression Model Wrapper for TabTune Pipeline.

This module provides a wrapper around LimiX FeaturesTransformer for regression
to ensure compatibility with the TabTune pipeline and enforce inference-only mode.
"""
import numpy as np
import pandas as pd
import torch
import logging
import json
import os
from typing import Literal
from sklearn.base import RegressorMixin
from tabtune.models.limix.transformer import FeaturesTransformer
from tabtune.models.limix.preprocess import (
    FilterValidFeatures,
    FeatureShuffler,
    CategoricalFeatureEncoder,
    RebalanceFeatureDistribution,
)
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

DEFAULT_WEIGHT_PATH = "LimiX-16M.ckpt"


class LimixRegressor(RegressorMixin):
    """
    LimiX Regressor using FeaturesTransformer for regression tasks.
    
    This regressor uses the same FeaturesTransformer architecture as LimixClassifier
    but configured for regression (task_type='reg').
    """
    
    def __init__(self, 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 repo_id="stableai-org/LimiX-16M", 
                 filename="LimiX-16M.ckpt",
                 nlayers=6,
                 nhead=6,
                 embed_dim=192,
                 hid_dim=768,
                 dropout=0.1,
                 seed=0,
                 inference_config=None,
                 use_ensemble=True,
                 **kwargs):
        """
        Initialize LimixRegressor.
        
        Args:
            device: Device to run on ('cuda' or 'cpu')
            repo_id: HuggingFace repository ID for model weights
            filename: Model checkpoint filename
            nlayers: Number of transformer layers
            nhead: Number of attention heads
            embed_dim: Embedding dimension
            hid_dim: Hidden dimension
            dropout: Dropout rate
            seed: Random seed
            inference_config: Inference config (JSON file path, dict, or list of dicts)
            use_ensemble: Whether to use ensemble inference (default: True)
            **kwargs: Additional model parameters
        """
        self.device = device
        self.nlayers = nlayers
        self.repo_id = repo_id
        self.filename = filename
        self.nhead = nhead
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.seed = seed
        self.model_params = kwargs
        self.use_ensemble = use_ensemble
        
        # Load inference config
        self.inference_config = self._load_inference_config(inference_config)
        self.n_estimators = len(self.inference_config) if isinstance(self.inference_config, list) else 1
        
        self.model = None
        self.X_train_ = None
        self.y_train_ = None
        self.y_train_numpy_ = None  # Store normalized y as numpy for preprocessing
        self.enc_ = None  # Internal encoder for X features
        self.num_features_ = None
        
        # For target normalization (LimiX expects normalized targets)
        self.y_mean_ = None
        self.y_std_ = None
        
        # Ensemble preprocessing pipelines
        self.preprocess_pipelines = []
        self.pipeline_seeds = []
        self.feature_shifts = []
        
        # Set random seed for reproducibility
        import random
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    
    def _init_model(self):
        """
        Initializes the FeaturesTransformer for regression.
        CRITICAL: Load model config from checkpoint first (matching official LimiX).
        """
        # CRITICAL FIX: Load checkpoint to get model config BEFORE building model
        # Official LimiX loads config from checkpoint and uses it to build model
        try:
            logger.info(f"Loading checkpoint to extract model config: {self.repo_id}/{self.filename}...")
            cached_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)
            checkpoint = torch.load(cached_path, map_location=self.device, weights_only=False)
            
            if 'config' in checkpoint:
                checkpoint_config = checkpoint['config']
                logger.info("Found config in checkpoint, using it to initialize model...")
                
                # Override parameters with checkpoint config (matching official LimiX)
                self.nlayers = checkpoint_config.get('nlayers', self.nlayers or 6)
                self.nhead = checkpoint_config.get('nhead', self.nhead or 6)
                self.embed_dim = checkpoint_config.get('embed_dim', self.embed_dim or 192)
                self.hid_dim = checkpoint_config.get('hid_dim', self.hid_dim or 768)
                self.dropout = checkpoint_config.get('dropout', self.dropout if self.dropout is not None else 0.0)
                
                # Update model_params with checkpoint config
                self.model_params.update({
                    'pre_norm': checkpoint_config.get('pre_norm', True),
                    'activation': checkpoint_config.get('activation', 'gelu'),
                    'layer_norm_eps': checkpoint_config.get('layer_norm_eps', 1e-5),
                    'recompute_attn': checkpoint_config.get('recompute_attn', False),
                    'layer_arch': checkpoint_config.get('layer_arch', 'fmfmsm'),
                    'self_share_all_kv_heads': checkpoint_config.get('self_share_all_kv_heads', False),
                    'cross_share_all_kv_heads': checkpoint_config.get('cross_share_all_kv_heads', True),
                    'seq_attn_isolated': checkpoint_config.get('seq_attn_isolated', False),
                    'seq_attn_serial': checkpoint_config.get('seq_attn_serial', False),
                })
                
                logger.info(f"Model config from checkpoint: nlayers={self.nlayers}, nhead={self.nhead}, "
                          f"embed_dim={self.embed_dim}, hid_dim={self.hid_dim}, dropout={self.dropout}")
                
                # Store checkpoint for later weight loading
                self._checkpoint = checkpoint
            else:
                logger.warning("No config found in checkpoint, using provided/default parameters")
                self._checkpoint = checkpoint
                # Use defaults if None
                self.nlayers = self.nlayers or 6
                self.nhead = self.nhead or 6
                self.embed_dim = self.embed_dim or 192
                self.hid_dim = self.hid_dim or 768
                self.dropout = self.dropout if self.dropout is not None else 0.1
        except Exception as e:
            logger.warning(f"Failed to load checkpoint for config: {e}. Using provided/default parameters.")
            self._checkpoint = None
            # Use defaults if None
            self.nlayers = self.nlayers or 6
            self.nhead = self.nhead or 6
            self.embed_dim = self.embed_dim or 192
            self.hid_dim = self.hid_dim or 768
            self.dropout = self.dropout if self.dropout is not None else 0.1
        
        # Get the grouping size (default is 2 in FeaturesTransformer)
        features_per_group = self.model_params.get("features_per_group", 2)
        
        # Get preprocessing variation (for ensemble diversity)
        preprocess_variant = self.model_params.get("preprocess_variant", "default")
        
        # 1. Configure X Preprocessing
        # Note: Official LimiX applies preprocessing AND still uses FeaturesTransformer's normalization
        # So we should keep normalize_x=True to match official behavior
        preprocess_config_x = {
            "num_features": features_per_group, 
            "nan_handling_enabled": True,
            "normalize_on_train_only": True,
            "normalize_x": True,  # Re-enable to match official (preprocessing + normalization)
            "remove_outliers": True,  # Re-enable to match official
            "normalize_by_used_features": True
        }
        
        # Vary preprocessing parameters for ensemble (if needed)
        if preprocess_variant == "no_outliers":
            preprocess_config_x["remove_outliers"] = False
        elif preprocess_variant == "strong_outliers":
            preprocess_config_x["remove_outliers"] = True
            # Could vary std_sigma if available
        
        # 2. Configure X Encoder
        encoder_config_x = {
            "num_features": features_per_group,
            "embedding_size": self.embed_dim,
            "mask_embedding_size": self.embed_dim,
            "encoder_use_bias": True,
            "numeric_embed_type": "linear",
            "RBF_config": None,
            "in_keys": ['data']
        }
        
        # 3. Configure Y Encoder (for regression, max_num_classes is still required by encoder)
        # The regression encoder uses LinearEncoder which requires nan_encoding even if nan_handling_y_encoder=False
        # We need to enable nan_handling to ensure nan_encoding is created
        encoder_config_y = {
            "num_inputs": 1,
            "embedding_size": self.embed_dim,
            "nan_handling_y_encoder": True,  # Required for LinearEncoder to work properly
            "max_num_classes": 2  # Not used for regression, but required parameter
        }
        
        # 4. Configure Decoder (for regression, num_classes is still required by decoder)
        # The decoder handles both cls and reg outputs, so we need to provide num_classes
        decoder_config = {
            "num_classes": 2  # Not used for regression output, but required by decoder
        }
        
        # Filter out internal parameters before passing to FeaturesTransformer
        filtered_model_params = {
            k: v for k, v in self.model_params.items() 
            if k not in ['preprocess_variant']  # Internal parameter, not for FeaturesTransformer
        }
        
        # Initialize the PyTorch Module
        self.model = FeaturesTransformer(
            preprocess_config_x=preprocess_config_x,
            encoder_config_x=encoder_config_x,
            encoder_config_y=encoder_config_y,
            decoder_config=decoder_config,
            nlayers=self.nlayers,
            nhead=self.nhead,
            embed_dim=self.embed_dim,
            hid_dim=self.hid_dim,
            features_per_group=features_per_group,
            feature_positional_embedding_type='subortho',
            dropout=self.dropout,
            device=torch.device(self.device),
            dtype=torch.float32,
            **filtered_model_params
        )
        
        try:
            # Use checkpoint already loaded in _init_model, or load it now
            if hasattr(self, '_checkpoint') and self._checkpoint is not None:
                checkpoint = self._checkpoint
                logger.info("Using checkpoint already loaded for config extraction...")
            else:
                logger.info(f"Retrieving weights from Hugging Face: {self.repo_id}/{self.filename}...")
                cached_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)
                logger.info(f"Loading weights from {cached_path}...")
                checkpoint = torch.load(cached_path, map_location=self.device, weights_only=False)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # For regression, we skip cls_y_encoder and cls_y_decoder to avoid size mismatches
            # The checkpoint may have been trained with different num_classes for classification
            # We only need reg_y_encoder and reg_y_decoder for regression tasks
            new_state_dict = {}
            ignored_keys = []
            reg_keys_found = []
            
            for k, v in state_dict.items():
                # Remove 'model.' prefix if present (some checkpoints have this)
                name = k[6:] if k.startswith('model.') else k
                
                # Skip classification-specific components for regression
                # These can have size mismatches (checkpoint trained with different num_classes)
                if name.startswith("cls_y_encoder") or name.startswith("cls_y_decoder"):
                    ignored_keys.append(name)
                    continue
                
                # Track regression-specific keys
                if name.startswith("reg_y_encoder") or name.startswith("reg_y_decoder"):
                    reg_keys_found.append(name)
                
                new_state_dict[name] = v
            
            if ignored_keys:
                logger.info(f"Ignored {len(ignored_keys)} cls_y_encoder/cls_y_decoder keys (not needed for regression)")
            
            logger.info(f"Found {len(reg_keys_found)} regression-specific keys: {reg_keys_found[:5]}..." if len(reg_keys_found) > 5 else f"Found {len(reg_keys_found)} regression-specific keys: {reg_keys_found}")
            
            # Load with strict=False to allow missing classification components
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            
            # Log missing/unexpected keys for debugging
            if missing:
                # Check if critical regression components are missing
                critical_missing = [k for k in missing if 'reg_y_decoder' in k or 'reg_y_encoder' in k]
                if critical_missing:
                    logger.error(f"CRITICAL: Missing regression decoder/encoder keys: {critical_missing}")
                else:
                    logger.info(f"Missing keys (may be expected): {len(missing)} keys")
                    if len(missing) <= 10:
                        logger.info(f"Missing keys: {missing}")
            
            if unexpected:
                logger.info(f"Unexpected keys in state dict: {len(unexpected)} keys")
                if len(unexpected) <= 10:
                    logger.info(f"Unexpected keys: {unexpected}")
            
            logger.info(f"Successfully loaded model weights. Missing layers: {len(missing)} (expected for regression-only setup)")
            
        except Exception as e:
            logger.error(f"Failed to load weights from Hugging Face: {e}")
            raise e
        
        self.model.to(self.device)
    
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
            
            # Fallback: minimal 4-pipeline config
            return self._get_minimal_config()
        
        if isinstance(inference_config, str):
            # Assume it's a file path
            if os.path.exists(inference_config):
                with open(inference_config, 'r') as f:
                    return json.load(f)
            else:
                raise FileNotFoundError(f"Inference config file not found: {inference_config}")
        
        # Already a dict or list
        return inference_config
    
    def _get_minimal_config(self):
        """Get minimal default config for regression (4 pipelines)."""
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
    
    def _build_preprocess_pipelines(self):
        """Build multiple preprocessing pipelines for ensemble inference."""
        self.preprocess_pipelines = []
        
        # Generate random seeds for each pipeline step
        np_rng = np.random.default_rng(self.seed)
        self.pipeline_seeds = [
            np_rng.integers(0, 10000) 
            for _ in range(self.n_estimators * 10)
        ]
        
        # Generate feature shuffler offsets
        start_idx = np_rng.integers(0, 1000)
        all_shifts = list(range(start_idx, start_idx + self.n_estimators))
        self.feature_shifts = np_rng.choice(
            all_shifts, 
            size=self.n_estimators, 
            replace=False
        )
        
        # Build each pipeline
        for idx in range(self.n_estimators):
            pipeline = []
            config = self.inference_config[idx % len(self.inference_config)]
            
            # Add preprocessing steps in order (matching official LimiX)
            pipeline.append(FilterValidFeatures())
            
            if 'RebalanceFeatureDistribution' in config:
                pipeline.append(RebalanceFeatureDistribution(**config['RebalanceFeatureDistribution']))
            
            if 'CategoricalFeatureEncoder' in config:
                pipeline.append(CategoricalFeatureEncoder(**config['CategoricalFeatureEncoder']))
            
            if 'FeatureShuffler' in config:
                shuffler = FeatureShuffler(**config['FeatureShuffler'])
                shuffler.offset = self.feature_shifts[idx]
                pipeline.append(shuffler)
            
            self.preprocess_pipelines.append(pipeline)
        
        logger.info(f"Built {len(self.preprocess_pipelines)} preprocessing pipelines for ensemble")
    
    def _get_categorical_indices(self, x):
        """Infer categorical feature indices from data."""
        if x.shape[0] < 100:
            return []
        
        categorical_idx = []
        for idx, col in enumerate(x.T):
            unique_vals = np.unique(col[~np.isnan(col)])
            if len(unique_vals) < 4:  # Few unique values suggests categorical
                categorical_idx.append(idx)
        
        return categorical_idx
    
    def convert_x_dtypes(self, x: np.ndarray, dtypes: str = "float32"):
        """
        Convert x to DataFrame format and ensure proper dtypes.
        Matches official LimiX convert_x_dtypes method.
        """
        NUMERIC_DTYPE_KINDS = "?bBiufm"
        OBJECT_DTYPE_KINDS = "OV"
        STRING_DTYPE_KINDS = "SaU"
        
        if x.dtype.kind in NUMERIC_DTYPE_KINDS:
            x = pd.DataFrame(x, copy=False, dtype=dtypes)
        elif x.dtype.kind in OBJECT_DTYPE_KINDS:
            x = pd.DataFrame(x, copy=True)
            x = x.convert_dtypes()
        else:
            raise ValueError(f"Unsupported string dtypes! {x.dtype}")
        
        integer_columns = x.select_dtypes(include=["number"]).columns
        if len(integer_columns) > 0:
            x[integer_columns] = x[integer_columns].astype(dtypes)
        return x
    
    def convert_category2num(self, x, dtype: np.floating = np.float64, placeholder: str = "__MISSING__"):
        """
        Convert categorical features to numeric.
        Matches official LimiX convert_category2num method.
        """
        from sklearn.compose import ColumnTransformer, make_column_selector
        from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
        
        ordinal_encoder = OrdinalEncoder(
            categories="auto",
            dtype=dtype,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=np.nan
        )
        col_encoder = ColumnTransformer(
            transformers=[
                ("encoder", ordinal_encoder, make_column_selector(dtype_include=["category", "string", "bool"]))
            ],
            remainder=FunctionTransformer(),
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        )
        
        string_cols = x.select_dtypes(include=["string", "object"]).columns
        if len(string_cols) > 0:
            x[string_cols] = x[string_cols].fillna(placeholder)
        
        X_encoded = col_encoder.fit_transform(x)
        
        # Handle placeholder values
        if len(string_cols) > 0:
            string_cols_ix = [x.columns.get_loc(col) for col in string_cols]
            placeholder_mask = x[string_cols] == placeholder
            string_cols_ix_2 = list(range(len(string_cols_ix)))
            X_encoded[:, string_cols_ix_2] = np.where(
                placeholder_mask.values,
                np.nan,
                X_encoded[:, string_cols_ix_2],
            )
        
        return X_encoded
    
    def _encode_X(self, X, fit=False):
        """
        Handles encoding of categorical (string/object) columns into float32.
        """
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OrdinalEncoder
        
        # If input is already a numeric tensor/array, assume it's processed
        if isinstance(X, torch.Tensor):
            return X.cpu().numpy() if X.is_cuda else X.numpy()
        
        # If input is DataFrame, we can smartly detect types
        if isinstance(X, pd.DataFrame):
            if fit:
                # Detect categorical columns
                cat_cols = X.select_dtypes(include=['object', 'category', 'string', 'bool']).columns.tolist()
                num_cols = X.select_dtypes(exclude=['object', 'category', 'string', 'bool']).columns.tolist()
                
                # If we have categoricals, set up the encoder
                if cat_cols:
                    logger.debug(f"[LimixRegressor] Encoding {len(cat_cols)} categorical columns.")
                    self.enc_ = ColumnTransformer(
                        transformers=[
                            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), cat_cols),
                            ('num', 'passthrough', num_cols)
                        ],
                        verbose_feature_names_out=False
                    )
                    X = self.enc_.fit_transform(X)
                else:
                    self.enc_ = None
                    X = X.values
            else:
                # Transform using existing encoder
                if self.enc_ is not None:
                    X = self.enc_.transform(X)
                else:
                    X = X.values
        
        # Fallback for numpy arrays
        elif isinstance(X, np.ndarray):
            if X.dtype == object:
                # Try to convert to float
                try:
                    X = X.astype(np.float32)
                except ValueError:
                    logger.warning("Could not convert object array to float32. Assuming already numeric.")
            pass
        
        # Final cast to float32
        return np.array(X, dtype=np.float32)
    
    def fit(self, X, y):
        """
        Fit the regressor on training data.
        
        Args:
            X: Training features (pd.DataFrame or np.ndarray)
            y: Training targets (pd.Series or np.ndarray)
        
        Returns:
            self
        """
        # 1. Normalize target (LimiX expects normalized targets)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        if isinstance(y, pd.Series):
            y = y.values
        
        y = np.array(y).flatten().astype(float)
        
        # Store normalization parameters
        self.y_mean_ = np.mean(y)
        self.y_std_ = np.std(y)
        if self.y_std_ == 0:
            self.y_std_ = 1.0  # Avoid division by zero
        
        y_normalized = (y - self.y_mean_) / self.y_std_
        
        # 2. Store raw training data for later use in predict
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        self.X_train_raw_ = X_array.copy()
        
        # 3. Encode Features (Handle Strings -> Numbers)
        # Apply convert_x_dtypes and convert_category2num (matching official LimiX)
        X_df = self.convert_x_dtypes(X_array, dtypes="float32")
        X_encoded = self.convert_category2num(X_df, dtype=np.float32)
        X_encoded = X_encoded.astype(np.float32)
        num_features = X_encoded.shape[1]
        
        # CRITICAL: If num_features changed, we need to reinitialize the model
        if self.num_features_ is not None and self.num_features_ != num_features:
            logger.warning(f"[LimixRegressor] Feature count changed from {self.num_features_} to {num_features}, reinitializing model")
            self.model = None  # Force reinitialization
        
        self.num_features_ = num_features
        
        # 3. Store preprocessed training data as numpy (for ensemble to use in predict)
        # Also store as torch tensor for model use
        self.X_train_numpy_ = X_encoded  # Store as numpy array for ensemble use
        self.X_train_ = torch.tensor(X_encoded, dtype=torch.float32).to(self.device)  # Store as tensor for model
        
        # 4. Move to Device for model
        self.y_train_ = torch.tensor(y_normalized, dtype=torch.float32).to(self.device)
        self.y_train_numpy_ = y_normalized  # Store as numpy for preprocessing
        
        # 5. Build preprocessing pipelines for ensemble
        if self.use_ensemble:
            self._build_preprocess_pipelines()
        
        # 6. Init Model (with correct num_features)
        if self.model is None:
            self._init_model()
        
        return self
    
    def _prepare_batch(self, X_query):
        """
        Concatenates Support (Train) and Query (Test) sets for In-Context Learning.
        """
        # Encode Query Features using the same encoder fitted in fit()
        X_query_encoded = self._encode_X(X_query, fit=False)
        X_query_t = torch.tensor(X_query_encoded, dtype=torch.float32).to(self.device)
        
        # 1. Concatenate X: [Support; Query] -> [1, Seq, F]
        # Official LimiX concatenates x_train and x_test before preprocessing
        logger.debug(f"[LimixRegressor] Concatenating X_train_ shape {self.X_train_.shape} with X_query_t shape {X_query_t.shape}")
        X_full = torch.cat([self.X_train_, X_query_t], dim=0).unsqueeze(0)
        logger.debug(f"[LimixRegressor] X_full shape after concat: {X_full.shape}")
        
        # 2. Y: Pass only training y - model will pad internally (matches official LimiX)
        # Official LimiX: y_ = y_train.copy(), then y_.unsqueeze(0), eval_pos=y_.shape[1]
        # The model internally pads y to match x length and sets test positions to NaN
        y_full = self.y_train_.unsqueeze(0)  # [1, n_train] - model will pad to match x
        
        # 3. Define Split Point
        # Official LimiX: eval_pos=y_.shape[1] where y_ is y_train only (after unsqueeze)
        eval_pos = self.y_train_.shape[0]  # Length of training data - this is the split point
        
        # Verify: eval_pos should match the length of training data
        assert eval_pos == self.X_train_.shape[0], \
            f"eval_pos ({eval_pos}) should match X_train length ({self.X_train_.shape[0]})"
        
        # Verify X_full shape
        assert X_full.shape[1] == self.X_train_.shape[0] + X_query_t.shape[0], \
            f"X_full shape mismatch: {X_full.shape[1]} != {self.X_train_.shape[0]} + {X_query_t.shape[0]}"
        # y_full is [1, n_train] - model will pad it internally, so no need to check length match
        
        return X_full, y_full, eval_pos
    
    def predict(self, X):
        """
        Predict target values for input data using ensemble inference.
        
        Args:
            X: Input features (pd.DataFrame or np.ndarray)
        
        Returns:
            np.ndarray: Predicted target values (denormalized)
        """
        if not self.use_ensemble or len(self.preprocess_pipelines) == 0:
            # Fallback to single model inference
            return self._predict_single(X)
        
        # Ensemble inference: apply multiple preprocessing pipelines and average
        # 1. Convert X to proper format (matching official LimiX)
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        # 2. Concatenate train and test BEFORE preprocessing (CRITICAL - matches official LimiX)
        # Store raw training data if not already stored
        if not hasattr(self, 'X_train_raw_'):
            self.X_train_raw_ = self.X_train_numpy_.copy()
        
        X_full_raw = np.concatenate([
            self.X_train_raw_,
            X_array
        ], axis=0)
        
        # 3. Apply convert_x_dtypes and convert_category2num (matching official LimiX)
        X_full_df = self.convert_x_dtypes(X_full_raw, dtypes="float32")
        X_full_numpy = self.convert_category2num(X_full_df, dtype=np.float32)
        X_full_numpy = X_full_numpy.astype(np.float32)
        
        # 3. Run ensemble inference
        outputs = []
        
        for pipe_idx, pipeline in enumerate(self.preprocess_pipelines):
            # Apply preprocessing pipeline
            x_processed = X_full_numpy.copy()
            categorical_idx = self._get_categorical_indices(x_processed)
            
            # Apply each preprocessing step (matching official LimiX seed pattern)
            # Official uses: self.seeds[id_pipe*self.preprocess_num+id_step] where preprocess_num=10
            for step_idx, step in enumerate(pipeline):
                # Match official seed calculation: id_pipe * preprocess_num + id_step
                seed = self.pipeline_seeds[pipe_idx * 10 + step_idx]
                try:
                    x_processed, categorical_idx = step.fit_transform(
                        x_processed,
                        categorical_idx,
                        seed,
                        y=self.y_train_numpy_  # Pass y for steps that need it
                    )
                except Exception as e:
                    logger.warning(f"Error in preprocessing step {step_idx} of pipeline {pipe_idx}: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
                    # Continue with unprocessed data
                    break
            
            # Convert to tensor (matching official LimiX pattern exactly)
            # Official: x_ = torch.from_numpy(x_[:, :]).float().to(self.device)
            x_tensor = torch.from_numpy(x_processed[:, :]).float().to(self.device)
            
            # Prepare y (matching official LimiX: use numpy array, convert to tensor, then unsqueeze)
            # Official: y_ = torch.from_numpy(y_).float().to(self.device), then y_ = y_.unsqueeze(0)
            # We use y_train_numpy_ (normalized numpy array) to match official pattern
            y_tensor = torch.from_numpy(self.y_train_numpy_).float().to(self.device)
            y_tensor = y_tensor.unsqueeze(0)  # [1, n_train]
            
            # eval_pos should be y_.shape[1] after unsqueeze (matching official line 590)
            eval_pos = y_tensor.shape[1]
            
            # Debug logging for first pipeline (always log for debugging)
            if pipe_idx == 0:
                logger.info(f"[LimixRegressor] Pipeline {pipe_idx} - Model Input Analysis:")
                logger.info(f"  x_tensor shape: {x_tensor.shape} (after unsqueeze: {x_tensor.unsqueeze(0).shape})")
                logger.info(f"  y_tensor shape: {y_tensor.shape}")
                logger.info(f"  eval_pos: {eval_pos}")
                logger.info(f"  x_train portion: shape={x_processed[:eval_pos].shape}, mean={x_processed[:eval_pos].mean():.4f}, std={x_processed[:eval_pos].std():.4f}, range=[{x_processed[:eval_pos].min():.4f}, {x_processed[:eval_pos].max():.4f}]")
                logger.info(f"  x_test portion: shape={x_processed[eval_pos:].shape}, mean={x_processed[eval_pos:].mean():.4f}, std={x_processed[eval_pos:].std():.4f}, range=[{x_processed[eval_pos:].min():.4f}, {x_processed[eval_pos:].max():.4f}]")
                logger.info(f"  y_train: shape={self.y_train_numpy_.shape}, mean={self.y_train_numpy_.mean():.4f}, std={self.y_train_numpy_.std():.4f}, range=[{self.y_train_numpy_.min():.4f}, {self.y_train_numpy_.max():.4f}]")
                
                # Check for NaN/Inf
                x_nan_count = np.isnan(x_processed).sum()
                x_inf_count = np.isinf(x_processed).sum()
                y_nan_count = np.isnan(self.y_train_numpy_).sum()
                y_inf_count = np.isinf(self.y_train_numpy_).sum()
                if x_nan_count > 0 or x_inf_count > 0:
                    logger.warning(f"  x_processed has {x_nan_count} NaN and {x_inf_count} Inf values")
                if y_nan_count > 0 or y_inf_count > 0:
                    logger.warning(f"  y_train has {y_nan_count} NaN and {y_inf_count} Inf values")
            
            # Model inference (matching official LimiX exactly)
            # Official sets random seed before inference (lines 553-554)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            
            self.model.to(self.device)
            with torch.autocast(device_type=self.device.type if isinstance(self.device, torch.device) else self.device, enabled=True), torch.inference_mode():
                # Official: x_=x_.unsqueeze(0), y_ = y_.unsqueeze(0)
                x_tensor_batch = x_tensor.unsqueeze(0)  # [1, seq, feat]
                # y_tensor is already [1, n_train] from above
                
                output = self.model(
                    x=x_tensor_batch,
                    y=y_tensor,
                    eval_pos=eval_pos,
                    task_type='reg'
                )
            
            # Debug: Log output before extraction (always log for first pipeline)
            if pipe_idx == 0:
                logger.info(f"[LimixRegressor] Pipeline {pipe_idx} - Model Output Analysis:")
                logger.info(f"  Raw output type: {type(output)}")
                if isinstance(output, dict):
                    logger.info(f"  Output dict keys: {list(output.keys())}")
                    for k, v in output.items():
                        if isinstance(v, torch.Tensor):
                            logger.info(f"    {k}: shape {v.shape}, dtype {v.dtype}, mean={v.mean().item():.4f}, std={v.std().item():.4f}, range=[{v.min().item():.4f}, {v.max().item():.4f}]")
                            if v.numel() > 1 and v.std().item() < 0.1:
                                logger.warning(f"    WARNING: {k} has very low std ({v.std().item():.6f}), predictions are nearly constant!")
                elif isinstance(output, torch.Tensor):
                    logger.info(f"  Output tensor: shape {output.shape}, dtype {output.dtype}, mean={output.mean().item():.4f}, std={output.std().item():.4f}, range=[{output.min().item():.4f}, {output.max().item():.4f}]")
                    if output.numel() > 1:
                        output_std = output.std().item()
                        if output_std < 0.1:
                            logger.warning(f"  WARNING: Output has very low std ({output_std:.6f}), predictions are nearly constant!")
                        else:
                            logger.info(f"  Output has reasonable variance (std={output_std:.4f})")
            
            # Extract regression output (matching official LimiX pattern)
            if isinstance(output, dict):
                output = output.get('reg_output', output)
            
            # Calculate number of test samples
            num_test = x_tensor.shape[0] - eval_pos
            
            # Official LimiX pattern: output.squeeze(0) then stack, then squeeze(2).mean(dim=0)
            # Model returns [batch, test_seq, 1] where batch=1
            if isinstance(output, torch.Tensor):
                # Match official: squeeze(0) to remove batch dimension
                # This converts [1, n_test, 1] -> [n_test, 1] or [1, n_test] -> [n_test]
                if output.ndim == 3:  # [batch, test_seq, 1]
                    output = output.squeeze(0)  # [test_seq, 1]
                elif output.ndim == 2:
                    if output.shape[0] == 1:  # [1, test_seq]
                        output = output.squeeze(0)  # [test_seq]
                    # else: already [test_seq, 1] or [test_seq]
                
                # Ensure output is [n_test, 1] format (matching official after squeeze(0))
                if output.ndim == 1:
                    # [n_test] -> [n_test, 1]
                    output = output.unsqueeze(-1)
                elif output.ndim == 2 and output.shape[1] != 1:
                    # [n_test, something] -> [n_test, 1] (take first column or reshape)
                    if output.shape[1] > 1:
                        output = output[:, 0:1]  # Take first column
                    # else already [n_test, 1]
                
                # Ensure correct size
                if output.shape[0] != num_test:
                    logger.warning(f"Output size mismatch: got {output.shape[0]}, expected {num_test}. Truncating/padding.")
                    if output.shape[0] > num_test:
                        output = output[:num_test, :]
                    else:
                        padding = torch.zeros(num_test - output.shape[0], output.shape[1], 
                                            device=output.device, dtype=output.dtype)
                        output = torch.cat([output, padding], dim=0)
                
                # Debug logging
                if logger.isEnabledFor(logging.DEBUG) and pipe_idx == 0:
                    logger.debug(f"[LimixRegressor] Pipeline {pipe_idx} - Output shape after extraction: {output.shape}")
                
                outputs.append(output.cpu())
            else:
                logger.warning(f"Non-tensor output from pipeline {pipe_idx}: {type(output)}")
                outputs.append(torch.zeros(num_test, 1))
        
        # 4. Ensemble averaging (matching official LimiX: stack, squeeze(2), mean)
        # Official code: output = torch.stack(outputs).squeeze(2).mean(dim=0)
        # After squeeze(0) above, each output is [n_test, 1]
        # Stack: [n_estimators, n_test, 1]
        # Squeeze(2): [n_estimators, n_test] (removes last dim if size=1)
        # Mean(dim=0): [n_test]
        if len(outputs) > 0:
            outputs_tensor = torch.stack(outputs)  # [n_estimators, n_test, 1]
            
            # Debug logging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[LimixRegressor] Stacked outputs shape: {outputs_tensor.shape}")
            
            # Match official pattern: squeeze(2) then mean
            outputs_tensor = outputs_tensor.squeeze(2)  # [n_estimators, n_test]
            
            # Check for NaN/inf in tensor before converting to numpy
            if torch.any(~torch.isfinite(outputs_tensor)):
                nan_inf_count = torch.sum(~torch.isfinite(outputs_tensor)).item()
                logger.warning(
                    f"[LimixRegressor] Found {nan_inf_count} non-finite values in stacked outputs "
                    f"({nan_inf_count}/{outputs_tensor.numel()} = {100*nan_inf_count/outputs_tensor.numel():.1f}%). "
                    f"Replacing with 0 (mean of normalized targets)."
                )
                # Replace NaN/inf with 0 (mean of normalized targets)
                outputs_tensor = torch.where(torch.isfinite(outputs_tensor), outputs_tensor, torch.zeros_like(outputs_tensor))
            
            predictions_normalized = outputs_tensor.mean(dim=0).numpy()  # [n_test]
            
            # Check for NaN/inf after mean (shouldn't happen, but just in case)
            if np.any(~np.isfinite(predictions_normalized)):
                nan_inf_count = np.sum(~np.isfinite(predictions_normalized))
                logger.warning(
                    f"[LimixRegressor] Found {nan_inf_count} non-finite values after ensemble averaging. "
                    f"Replacing with 0."
                )
                predictions_normalized = np.where(np.isfinite(predictions_normalized), predictions_normalized, 0.0)
            
            # Debug logging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[LimixRegressor] Final predictions shape: {predictions_normalized.shape}, "
                           f"mean={predictions_normalized.mean():.4f}, std={predictions_normalized.std():.4f}")
        else:
            logger.error("No valid predictions from ensemble!")
            predictions_normalized = np.zeros(X_query_encoded.shape[0])
        
        # 5. Denormalize
        # Check for NaN/inf in normalized predictions before denormalization
        if np.any(~np.isfinite(predictions_normalized)):
            nan_inf_count = np.sum(~np.isfinite(predictions_normalized))
            logger.warning(
                f"[LimixRegressor] Found {nan_inf_count} non-finite values in normalized predictions "
                f"({nan_inf_count}/{len(predictions_normalized)} = {100*nan_inf_count/len(predictions_normalized):.1f}%). "
                f"Replacing with 0 (mean of normalized targets)."
            )
            nan_inf_mask = ~np.isfinite(predictions_normalized)
            predictions_normalized[nan_inf_mask] = 0.0  # Mean of normalized targets is 0
        
        # Check normalization parameters
        if not np.isfinite(self.y_mean_) or not np.isfinite(self.y_std_):
            logger.error(
                f"[LimixRegressor] Invalid normalization parameters: y_mean_={self.y_mean_}, y_std_={self.y_std_}"
            )
            # Fallback: use simple mean prediction
            predictions = np.full(len(predictions_normalized), self.y_mean_ if np.isfinite(self.y_mean_) else 0.0)
            return predictions
        
        predictions = predictions_normalized * self.y_std_ + self.y_mean_
        
        # Handle any NaN or Inf values after denormalization
        if np.any(~np.isfinite(predictions)):
            nan_inf_count = np.sum(~np.isfinite(predictions))
            logger.warning(
                f"[LimixRegressor] Found {nan_inf_count} non-finite predictions after denormalization. "
                f"Replacing with mean target value (y_mean_={self.y_mean_:.4f})."
            )
            nan_inf_mask = ~np.isfinite(predictions)
            predictions[nan_inf_mask] = self.y_mean_
        
        # Final validation
        if np.any(~np.isfinite(predictions)):
            logger.error(
                f"[LimixRegressor] CRITICAL: Still have {np.sum(~np.isfinite(predictions))} non-finite predictions after cleanup!"
            )
            # Last resort: replace all invalid with mean
            predictions = np.where(np.isfinite(predictions), predictions, self.y_mean_)
        
        logger.debug(
            f"[LimixRegressor] Final predictions: shape={predictions.shape}, "
            f"mean={np.mean(predictions):.4f}, std={np.std(predictions):.4f}, "
            f"min={np.min(predictions):.4f}, max={np.max(predictions):.4f}, "
            f"finite={np.sum(np.isfinite(predictions))}/{len(predictions)}"
        )
        
        return predictions
    
    def _predict_single(self, X):
        """Single model inference (fallback when ensemble is disabled)."""
        X_full, y_full, eval_pos = self._prepare_batch(X)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                x=X_full,
                y=y_full,
                eval_pos=eval_pos,
                task_type='reg'
            )
        
        # Extract predictions
        if isinstance(output, dict):
            output = output.get('reg_output', output)
        
        if isinstance(output, torch.Tensor):
            output = output.squeeze(0)
            if output.ndim > 1 and output.shape[-1] == 1:
                output = output.squeeze(-1)
            predictions = output.cpu().numpy()
        else:
            predictions = np.array(output)
            if predictions.ndim > 1:
                predictions = predictions.squeeze()
        
        predictions = predictions.flatten()
        
        # Denormalize
        predictions_denorm = predictions * self.y_std_ + self.y_mean_
        
        # Handle NaN/Inf
        if np.any(~np.isfinite(predictions_denorm)):
            nan_inf_mask = ~np.isfinite(predictions_denorm)
            predictions_denorm[nan_inf_mask] = self.y_mean_
        
        return predictions_denorm
