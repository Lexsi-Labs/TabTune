"""
Mitra (Tab2D) Regression Model Wrapper for TabTune Pipeline.

This module provides a wrapper around Tab2D model configured for regression
to ensure compatibility with the TabTune pipeline and enforce inference-only mode.

The wrapper supports loading pretrained weights from HuggingFace:
https://huggingface.co/autogluon/mitra-regressor
"""
import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path
from tabtune.models.mitra.tab2d import Tab2D

logger = logging.getLogger(__name__)

try:
    from tabtune.models.mitra.model_loading import (
        download_mitra_regressor,
        load_mitra_regressor_from_hf,
    )
    MITRA_LOADING_AVAILABLE = True
except ImportError:
    MITRA_LOADING_AVAILABLE = False
    logger.warning(
        "[MitraRegressorWrapper] HuggingFace model loading not available. "
        "Install huggingface_hub for automatic model downloads."
    )


class MitraRegressorWrapper:
    """
    Wrapper for Tab2D model configured for regression.
    
    This wrapper ensures:
    - Only inference mode is supported (no fine-tuning)
    - Proper integration with TabTune's DataProcessor
    - Consistent API with other regression models
    - Tab2D is initialized with task='REGRESSION' and dim_output=1
    """
    
    def __init__(self, tuning_strategy='inference', dim=512, n_layers=12, n_heads=4,
                 use_pretrained_weights='auto', path_to_weights='', device=None,
                 cache_dir=None, **kwargs):
        """
        Initialize MitraRegressorWrapper.
        
        Args:
            tuning_strategy: Must be 'inference' for regression models
            dim: Model dimension (default: 512, ignored if using pretrained weights)
            n_layers: Number of transformer layers (default: 12, ignored if using pretrained weights)
            n_heads: Number of attention heads (default: 4, ignored if using pretrained weights)
            use_pretrained_weights: 
                - 'auto' (default): Automatically download and use pretrained weights from HuggingFace
                - True: Use pretrained weights from path_to_weights
                - False: Use randomly initialized weights
            path_to_weights: Path to pretrained weights file (local path or HuggingFace repo ID)
            device: Device to use ('cuda' or 'cpu')
            cache_dir: Directory to cache downloaded models (default: ~/.cache/mitra)
            **kwargs: Additional arguments (filtered for Tab2D compatibility)
        """
        # Validate that only inference mode is used for regression
        if tuning_strategy not in ("inference", "finetune"):
            raise ValueError(
                f"Regression models only support 'inference' mode. "
                f"Received: '{tuning_strategy}'. Fine-tuning for regression is not supported."
            )
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Handle pretrained weights loading
        if use_pretrained_weights == 'auto':
            # Try to load from HuggingFace using Tab2D.from_pretrained directly
            try:
                logger.info("[MitraRegressorWrapper] Loading pretrained weights from HuggingFace...")
                # Use Tab2D.from_pretrained which now supports HuggingFace repo IDs
                self.model = Tab2D.from_pretrained("autogluon/mitra-regressor", device=device)
                self.tuning_strategy = tuning_strategy
                self.device = device
                self._is_fitted = False
                logger.info("[MitraRegressorWrapper] Successfully loaded pretrained Mitra regressor from HuggingFace")
                return
            except Exception as e:
                logger.warning(
                    f"[MitraRegressorWrapper] Failed to load from HuggingFace: {e}. "
                    "Falling back to randomly initialized weights."
                )
                use_pretrained_weights = False
        elif use_pretrained_weights is True and path_to_weights:
            # Load from local path or HuggingFace repo ID
            try:
                logger.info(f"[MitraRegressorWrapper] Loading pretrained weights from {path_to_weights}...")
                self.model = Tab2D.from_pretrained(path_to_weights, device=device)
                self.tuning_strategy = tuning_strategy
                self.device = device
                self._is_fitted = False
                logger.info(f"[MitraRegressorWrapper] Successfully loaded pretrained model from {path_to_weights}")
                return
            except Exception as e:
                logger.warning(
                    f"[MitraRegressorWrapper] Failed to load from {path_to_weights}: {e}. "
                    "Falling back to randomly initialized weights."
                )
                use_pretrained_weights = False
        
        # Filter out pipeline-specific parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() 
            if k not in ['task_type', 'tuning_strategy', 'dim_output', 'task', 'cache_dir']
        }
        
        # Initialize Tab2D for regression
        # For regression, dim_output should be 1
        self.model = Tab2D(
            dim=dim,
            dim_output=1,  # Regression output dimension
            n_layers=n_layers,
            n_heads=n_heads,
            task='REGRESSION',
            use_pretrained_weights=bool(use_pretrained_weights),
            path_to_weights=path_to_weights,
            device=device,
            **filtered_kwargs
        )
        
        self.tuning_strategy = tuning_strategy
        self.device = device
        self._is_fitted = False
        
        # Ensure model is on the correct device
        if self.device is not None:
            self.model = self.model.to(self.device)
        
        if use_pretrained_weights:
            logger.info(f"[MitraRegressorWrapper] Initialized with pretrained weights from {path_to_weights}")
        else:
            logger.info(f"[MitraRegressorWrapper] Initialized with randomly initialized weights (inference-only mode)")
    
    def fit(self, X, y):
        """
        Fit the regressor on training data.
        For inference-only mode, this just stores the data for in-context learning.
        
        CRITICAL: Normalizes targets using min-max scaling (matching AutoGluon's Preprocessor.normalize_y)
        This is essential for the model to work correctly, as it expects targets in [0, 1] range.
        
        Args:
            X: Training features (pd.DataFrame or np.ndarray)
            y: Training targets (pd.Series or np.ndarray)
        
        Returns:
            self
        """
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            # Check for non-numeric columns and convert them
            X = X.copy()
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    # Convert categorical/string columns to numeric codes
                    X[col] = pd.Categorical(X[col]).codes
            X = X.values
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0].values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Ensure y is numeric and 1D
        y = np.array(y).flatten().astype(float)
        X = np.array(X)
        
        # Handle object/string arrays - convert to numeric
        if X.dtype == object:
            # Try to convert object arrays to float, handling errors
            try:
                X = X.astype(float)
            except (ValueError, TypeError):
                # If direct conversion fails, use pandas to handle mixed types
                X_df = pd.DataFrame(X)
                for col in X_df.columns:
                    if X_df[col].dtype == 'object':
                        X_df[col] = pd.Categorical(X_df[col]).codes
                X = X_df.values.astype(float)
        
        # CRITICAL FIX: Normalize targets using min-max scaling (matching AutoGluon)
        # AutoGluon's Preprocessor.normalize_y: y = (y - y_min) / (y_max - y_min)
        self.y_min = float(np.min(y))
        self.y_max = float(np.max(y))
        
        # Handle constant targets (y_max == y_min)
        if self.y_max == self.y_min:
            logger.warning(f"[MitraRegressorWrapper] Constant target detected (all values = {self.y_min}). Using identity normalization.")
            y_normalized = y - self.y_min  # Results in all zeros
        else:
            y_normalized = (y - self.y_min) / (self.y_max - self.y_min)
        
        # Store training data for in-context learning (with normalized targets)
        self.X_train = X
        self.y_train = y_normalized  # Store normalized targets
        self._is_fitted = True
        
        logger.info(f"[MitraRegressorWrapper] Stored training data for inference: {X.shape[0]} samples")
        logger.info(f"[MitraRegressorWrapper] Target normalization: min={self.y_min:.4f}, max={self.y_max:.4f}")
        return self
    
    def predict(self, X):
        """
        Predict target values for input data using in-context learning.
        
        Args:
            X: Input features (pd.DataFrame or np.ndarray)
        
        Returns:
            np.ndarray: Predicted target values
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            # Check for non-numeric columns and convert them
            X = X.copy()
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    # Convert categorical/string columns to numeric codes
                    X[col] = pd.Categorical(X[col]).codes
            X = X.values
        X = np.array(X)
        
        # Handle object/string arrays - convert to numeric
        if X.dtype == object:
            # Try to convert object arrays to float, handling errors
            try:
                X = X.astype(float)
            except (ValueError, TypeError):
                # If direct conversion fails, use pandas to handle mixed types
                X_df = pd.DataFrame(X)
                for col in X_df.columns:
                    if X_df[col].dtype == 'object':
                        X_df[col] = pd.Categorical(X_df[col]).codes
                X = X_df.values.astype(float)
        
        # Convert to torch tensors for training data (reused for all batches)
        X_train_tensor = torch.FloatTensor(self.X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(self.y_train).to(self.device)
        
        # Process in batches to avoid memory issues with large test sets
        batch_size = 1
        n_support = X_train_tensor.shape[0]
        n_features = X_train_tensor.shape[1]
        max_query_batch_size = 1024  # Process query samples in batches
        
        # Prepare training data tensors (same for all batches)
        X_train_batch = X_train_tensor.unsqueeze(0)  # (1, n_support, n_features)
        y_train_batch = y_train_tensor.unsqueeze(0)  # (1, n_support)
        
        # Padding tensors for support (same for all batches)
        padding_features = torch.zeros(batch_size, n_features, dtype=torch.bool, device=self.device)
        padding_obs_support = torch.zeros(batch_size, n_support, dtype=torch.bool, device=self.device)
        
        # Ensure model is on the correct device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        all_predictions = []
        
        # Process query samples in batches
        with torch.no_grad():
            for i in range(0, len(X), max_query_batch_size):
                X_test_batch_data = X[i:i+max_query_batch_size]
                n_query_batch = len(X_test_batch_data)
                
                X_test_tensor = torch.FloatTensor(X_test_batch_data).to(self.device)
                X_test_batch = X_test_tensor.unsqueeze(0)  # (1, n_query_batch, n_features)
                
                # Padding for query batch
                padding_obs_query = torch.zeros(batch_size, n_query_batch, dtype=torch.bool, device=self.device)
                
                # Predict
                predictions = self.model(
                    x_support=X_train_batch,
                    y_support=y_train_batch,
                    x_query=X_test_batch,
                    padding_features=padding_features,
                    padding_obs_support=padding_obs_support,
                    padding_obs_query__=padding_obs_query
                )
                
                # Model returns shape (batch_size, n_query) for regression with dim_output=1
                # Extract predictions: (batch_size, n_query) -> (n_query,)
                predictions = predictions.cpu().numpy()
                
                # Handle different output shapes
                if predictions.ndim == 0:
                    # Single scalar prediction
                    predictions = np.array([predictions])
                elif predictions.ndim == 1:
                    # Already 1D: (n_query,)
                    pass
                elif predictions.ndim == 2:
                    # Shape (batch_size, n_query) - remove batch dimension
                    if predictions.shape[0] == 1:
                        predictions = predictions[0]  # (n_query,)
                    else:
                        # Multiple batches - flatten (shouldn't happen with batch_size=1)
                        predictions = predictions.flatten()
                else:
                    # Unexpected shape - flatten
                    logger.warning(f"[MitraRegressorWrapper] Unexpected prediction shape: {predictions.shape}, flattening")
                    predictions = predictions.flatten()
                
                # Ensure correct length for this batch
                if len(predictions) != n_query_batch:
                    logger.warning(
                        f"[MitraRegressorWrapper] Prediction length mismatch in batch {i}: "
                        f"got {len(predictions)}, expected {n_query_batch}"
                    )
                    # Take first n_query_batch predictions or pad if needed
                    if len(predictions) > n_query_batch:
                        predictions = predictions[:n_query_batch]
                    elif len(predictions) < n_query_batch:
                        # Pad with mean of training targets
                        padding = np.full(n_query_batch - len(predictions), float(np.mean(self.y_train)), dtype=predictions.dtype)
                        predictions = np.concatenate([predictions, padding])
                
                all_predictions.append(predictions)
        
        # Concatenate all batch predictions
        predictions_normalized = np.concatenate(all_predictions)
        
        # CRITICAL FIX: Inverse transform predictions back to original scale
        # AutoGluon's Preprocessor.undo_normalize_y: y = y * (y_max - y_min) + y_min
        if not hasattr(self, 'y_min') or not hasattr(self, 'y_max'):
            logger.warning("[MitraRegressorWrapper] Target normalization parameters not found. Predictions may be in wrong scale.")
            return predictions_normalized
        
        # Handle constant targets
        if self.y_max == self.y_min:
            predictions = predictions_normalized + self.y_min
        else:
            predictions = predictions_normalized * (self.y_max - self.y_min) + self.y_min
        
        return predictions
