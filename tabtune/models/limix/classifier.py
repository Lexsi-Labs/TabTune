import torch
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import logging
import os

# --- NEW IMPORT ---
from huggingface_hub import hf_hub_download
    
from .transformer import FeaturesTransformer

logger = logging.getLogger(__name__)

DEFAULT_WEIGHT_PATH = "LimiX-16M.ckpt"

class LimixClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 repo_id="stable-ai/LimiX-16M", 
                 filename="LimiX-16M.ckpt",
                 nlayers=6,
                 nhead=6,
                 embed_dim=192,
                 hid_dim=768,
                 dropout=0.1,
                 **kwargs):
        """
        Wrapper for LimiX FeaturesTransformer compatible with TabTune.
        """
        self.device = device
        self.nlayers = nlayers
        self.repo_id = repo_id
        self.filename = filename
        self.nhead = nhead
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.model_params = kwargs
        
        self.model = None
        self.X_train_ = None
        self.y_train_ = None
        self.le_ = None
        self.enc_ = None # Internal encoder for X features
        self.num_classes_ = None
        self.num_features_ = None

    def _init_model(self):
        """
        Initializes the FeaturesTransformer using the config structure.
        """
        # Get the grouping size (default is 2 in FeaturesTransformer)
        features_per_group = self.model_params.get("features_per_group", 2)

        # 1. Configure X Preprocessing
        # FIXED: Set num_features to features_per_group (2), not total features (18)
        preprocess_config_x = {
            "num_features": features_per_group, 
            "nan_handling_enabled": True,
            "normalize_on_train_only": True,
            "normalize_x": True,
            "remove_outliers": True,
            "normalize_by_used_features": True
        }

        # 2. Configure X Encoder
        # FIXED: Set num_features to features_per_group (2)
        encoder_config_x = {
            "num_features": features_per_group,
            "embedding_size": self.embed_dim,
            "mask_embedding_size": self.embed_dim,
            "encoder_use_bias": True,
            "numeric_embed_type": "linear",
            "RBF_config": None,
            "in_keys": ['data']
        }

        # 3. Configure Y Encoder
        encoder_config_y = {
            "num_inputs": 1,
            "embedding_size": self.embed_dim,
            "nan_handling_y_encoder": False,
            "max_num_classes": self.num_classes_
        }

        # 4. Configure Decoder
        decoder_config = {
            "num_classes": self.num_classes_
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
            features_per_group=features_per_group, # Pass this explicitly
            feature_positional_embedding_type='subortho',
            dropout=self.dropout,
            device=torch.device(self.device),
            dtype=torch.float32,
            **self.model_params
        )

        try:
            print(f"Retrieving weights from Hugging Face: {self.repo_id}/{self.filename}...")
            # FIXED LINE BELOW:
            cached_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)
            
            logger.info(f"Loading weights from {cached_path}...")
            checkpoint = torch.load(cached_path, map_location=self.device)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # --- IMPROVED FILTERING LOGIC ---
            new_state_dict = {}
            ignored_keys = []
            
            for k, v in state_dict.items():
                # 1. Remove 'model.' prefix
                name = k[6:] if k.startswith('model.') else k
                
                # 2. STRICTLY EXCLUDE DECODER HEADS
                if name.startswith("cls_y_decoder") or name.startswith("reg_y_decoder"):
                    ignored_keys.append(name)
                    continue
                        
                new_state_dict[name] = v
            
            if ignored_keys:
                print(f"Ignored {len(ignored_keys)} mismatched decoder keys.")

            # Load with strict=False to allow missing decoder heads
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            
            print(f"Successfully loaded backbone weights. Re-initialized layers: {len(missing)}")
            
        except Exception as e:
            logger.error(f"Failed to load weights from Hugging Face: {e}")
            raise e
        
        self.model.to(self.device)

    def _encode_X(self, X, fit=False):
        """
        Handles encoding of categorical (string/object) columns into float32.
        """
        # If input is already a numeric tensor/array, assume it's processed
        if isinstance(X, torch.Tensor):
            return X
        
        # If input is DataFrame, we can smartly detect types
        if isinstance(X, pd.DataFrame):
            if fit:
                # Detect categorical columns
                cat_cols = X.select_dtypes(include=['object', 'category', 'string', 'bool']).columns.tolist()
                num_cols = X.select_dtypes(exclude=['object', 'category', 'string', 'bool']).columns.tolist()
                
                # If we have categoricals, set up the encoder
                if cat_cols:
                    logger.debug(f"[LimixClassifier] Encoding {len(cat_cols)} categorical columns.")
                    self.enc_ = ColumnTransformer(
                        transformers=[
                            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), cat_cols),
                            ('num', 'passthrough', num_cols)
                        ],
                        verbose_feature_names_out=False
                    )
                    # Use set_output(transform="pandas") if sklearn version >= 1.2 for easier debugging, 
                    # but numpy is safer for compatibility.
                    X = self.enc_.fit_transform(X)
                else:
                    self.enc_ = None # No encoding needed
                    X = X.values
            else:
                # Transform using existing encoder
                if self.enc_ is not None:
                    X = self.enc_.transform(X)
                else:
                    X = X.values

        # Fallback for numpy object arrays if not DataFrame
        elif isinstance(X, np.ndarray) and X.dtype == object:
            # Simple fallback: try casting to float, if fail, good luck (or use pandas wrap)
            try:
                X = X.astype(np.float32)
            except ValueError:
                # If we are here, we have strings but no DataFrame metadata. 
                # Wrap in DF and recurse to use dtypes logic? 
                # For safety, let's assume the user passes DataFrames in TabTune.
                pass

        # Final cast to float32
        return np.array(X, dtype=np.float32)

    def fit(self, X, y):
        """
        Store training data to be used as 'Support Set' during inference.
        """
        # 1. Encode Labels
        if isinstance(y, pd.Series):
            y = y.values
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.num_classes_ = len(self.le_.classes_)

        # 2. Encode Features (Handle Strings -> Numbers)
        X_encoded = self._encode_X(X, fit=True)
        self.num_features_ = X_encoded.shape[1]

        # 3. Move to Device
        self.X_train_ = torch.tensor(X_encoded, dtype=torch.float32).to(self.device)
        self.y_train_ = torch.tensor(y_encoded, dtype=torch.float32).to(self.device)

        # 4. Init Model
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
        X_full = torch.cat([self.X_train_, X_query_t], dim=0).unsqueeze(0)
        
        # 2. Concatenate Y: [Support; Dummy] -> [1, Seq]
        y_dummy = torch.zeros(X_query_t.shape[0], device=self.device)
        y_full = torch.cat([self.y_train_, y_dummy], dim=0).unsqueeze(0)
        
        # 3. Define Split Point
        eval_pos = self.X_train_.shape[0]
        
        return X_full, y_full, eval_pos

    def predict(self, X):
        X_full, y_full, eval_pos = self._prepare_batch(X)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                x=X_full, 
                y=y_full, 
                eval_pos=eval_pos,
                task_type='cls'
            )
            
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        return self.le_.inverse_transform(preds.squeeze())

    def predict_proba(self, X):
        X_full, y_full, eval_pos = self._prepare_batch(X)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                x=X_full, 
                y=y_full, 
                eval_pos=eval_pos,
                task_type='cls'
            )
            
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        # Handle batch dim squeeze properly
        if len(probs.shape) == 3 and probs.shape[0] == 1:
            return probs.squeeze(0)
        return probs