import torch
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, QuantileTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
import logging
import os
import random
from itertools import chain, repeat

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
                 n_ensemble=4,
                 seed=42,
                 softmax_temperature=0.9,
                 **kwargs):
        """
        Wrapper for LimiX FeaturesTransformer compatible with TabTune.
        Matches Original LimiX preprocessing exactly.
        """
        self.device = device
        self.nlayers = nlayers
        self.repo_id = repo_id
        self.filename = filename
        self.nhead = nhead
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.n_ensemble = n_ensemble
        self.seed = seed
        self.softmax_temperature = softmax_temperature
        self.model_params = kwargs
        
        self.model = None
        self.X_train_base_ = None
        self.y_train_ = None
        self.le_ = None
        self.enc_ = None
        self.num_classes_ = None
        self.num_features_ = None
        # Note: Original uses self.seed directly, not per-ensemble seeds
        self.preprocess_pipelines = None
        self.preprocess_seeds = None
        self.all_shifts = None
        self.class_permutations = None
        self.categorical_indices = None
        self.rebalance_workers = None  # Store fitted RebalanceFeatureDistribution workers

    def _init_model(self):
        """
        Initializes the FeaturesTransformer using the config from checkpoint (matching Original LimiX).
        """
        try:
            print(f"Retrieving weights from Hugging Face: {self.repo_id}/{self.filename}...")
            cached_path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)
            
            logger.info(f"Loading weights from {cached_path}...")
            checkpoint = torch.load(cached_path, map_location=self.device)
            
            # Load config from checkpoint (matching Original LimiX load_model)
            if 'config' in checkpoint:
                config = checkpoint['config'].copy()
            else:
                # Fallback to default config if not in checkpoint
                features_per_group = self.model_params.get("features_per_group", 2)
                config = {
                    'preprocess_config_x': {
                        "num_features": features_per_group, 
                        "nan_handling_enabled": True,
                        "normalize_on_train_only": True,
                        "normalize_x": True,
                        "remove_outliers": True,
                        "normalize_by_used_features": True
                    },
                    'encoder_config_x': {
                        "num_features": features_per_group,
                        "embedding_size": self.embed_dim,
                        "mask_embedding_size": self.embed_dim,
                        "encoder_use_bias": True,
                        "numeric_embed_type": "linear",
                        "RBF_config": None,
                        "in_keys": ['data']
                    },
                    'encoder_config_y': {
                        "num_inputs": 1,
                        "embedding_size": self.embed_dim,
                        "nan_handling_y_encoder": False,
                        "max_num_classes": self.num_classes_
                    },
                    'decoder_config': {
                        "num_classes": self.num_classes_
                    },
                    'nlayers': self.nlayers,
                    'nhead': self.nhead,
                    'embed_dim': self.embed_dim,
                    'hid_dim': self.hid_dim,
                    'features_per_group': features_per_group,
                    'dropout': self.dropout,
                }
            
            # CRITICAL: Match Original LimiX exactly - use checkpoint config as-is
            # Original LimiX uses checkpoint's config (max_num_classes=10, num_classes=10)
            # and slices output during prediction: output[:, :self.n_classes]
            # Do NOT override decoder_config or max_num_classes - this ensures all weights load
            config['mask_prediction'] = False  # No mask prediction for classification
            
            # Build model using config from checkpoint (matching Original)
            self.model = FeaturesTransformer(
                preprocess_config_x=config['preprocess_config_x'],
                encoder_config_x=config['encoder_config_x'],
                encoder_config_y=config['encoder_config_y'],
                decoder_config=config['decoder_config'],
                feature_positional_embedding_type=config.get('feature_positional_embedding_type', 'subortho'),
                nlayers=config['nlayers'],
                nhead=config['nhead'],
                embed_dim=config['embed_dim'],
                hid_dim=config['hid_dim'],
                mask_prediction=config.get('mask_prediction', False),
                features_per_group=config['features_per_group'],
                dropout=config['dropout'],
                pre_norm=config.get('pre_norm', True),
                activation=config.get('activation', 'gelu'),
                layer_norm_eps=config.get('layer_norm_eps', 1e-5),
                device=torch.device(self.device),
                dtype=torch.float32,
                recompute_attn=config.get('recompute_attn', False),
                layer_arch=config.get('layer_arch', 'fmfmsm'),
                self_share_all_kv_heads=config.get('self_share_all_kv_heads', False),
                cross_share_all_kv_heads=config.get('cross_share_all_kv_heads', True),
                seq_attn_isolated=config.get('seq_attn_isolated', False),
                seq_attn_serial=config.get('seq_attn_serial', False),
                **self.model_params
            )
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            new_state_dict = {}
            ignored_keys = []
            
            for k, v in state_dict.items():
                name = k[6:] if k.startswith('model.') else k
                
                # All weights should load now since we use checkpoint config as-is
                # No need to skip decoder or y_encoder weights
                
                new_state_dict[name] = v
            
            if ignored_keys:
                print(f"Ignored {len(ignored_keys)} mismatched decoder keys.")

            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            
            if missing:
                print(f"\n⚠️  Missing keys ({len(missing)}): Layers that will be re-initialized")
                for key in missing[:10]:  # Show first 10
                    print(f"   - {key}")
                if len(missing) > 10:
                    print(f"   ... and {len(missing) - 10} more")
            
            if unexpected:
                print(f"\n⚠️  Unexpected keys ({len(unexpected)}): Keys in checkpoint but not in model")
                for key in unexpected[:10]:  # Show first 10
                    print(f"   - {key}")
                if len(unexpected) > 10:
                    print(f"   ... and {len(unexpected) - 10} more")
            
            print(f"\n✅ Successfully loaded backbone weights. Re-initialized layers: {len(missing)}")
            
        except Exception as e:
            logger.error(f"Failed to load weights from Hugging Face: {e}")
            raise e
        
        self.model.to(self.device)

    def _build_preprocess_pipelines(self):
        """
        Build preprocessing pipelines matching Original LimiX config exactly.
        """
        self.preprocess_pipelines = []
        random.seed(self.seed)  # Use Python's random to match Original (line 120)
        np_rng = np.random.default_rng(self.seed)
        
        # Generate seeds for each preprocessing step (matching Original line 122)
        # Original uses random.randint, not np_rng.integers
        preprocess_num = 3  # FilterValidFeatures, RebalanceFeatureDistribution, CategoricalFeatureEncoder, FeatureShuffler
        self.preprocess_seeds = [random.randint(0, 10000) for _ in range(self.n_ensemble * preprocess_num)]
        
        # Generate feature shuffle offsets (matching Original line 124-125)
        start_idx = np_rng.integers(0, 1000)
        all_shifts = list(range(start_idx, start_idx + self.n_ensemble))
        self.all_shifts = np_rng.choice(all_shifts, size=self.n_ensemble, replace=False)
        
        for ensemble_idx in range(self.n_ensemble):
            pipeline = {}
            
            # Match config: members 0-1 use quantile+SVD, members 2-3 use different config
            if ensemble_idx < 2:
                # Members 0-1: quantile_uniform_10 + SVD + ordinal_strict_feature_shuffled
                pipeline['rebalance'] = {
                    'worker_tags': ['quantile_uniform_10'],
                    'discrete_flag': False,
                    'original_flag': True,
                    'svd_tag': 'svd'
                }
                pipeline['categorical'] = {
                    'encoding_strategy': 'ordinal_strict_feature_shuffled'
                }
            else:
                # Members 2-3: no quantile + discrete_flag + numeric
                pipeline['rebalance'] = {
                    'worker_tags': [None],
                    'discrete_flag': True,
                    'original_flag': False,
                    'svd_tag': None
                }
                pipeline['categorical'] = {
                    'encoding_strategy': 'numeric'
                }
            
            pipeline['feature_shuffle'] = {
                'mode': 'shuffle',
                'offset': self.all_shifts[ensemble_idx]
            }
            
            self.preprocess_pipelines.append(pipeline)

    def _apply_rebalance_feature_distribution(self, x_concat, y_train, ensemble_idx, categorical_indices):
        """
        Apply RebalanceFeatureDistribution matching Original LimiX.
        Fits on train only, then transforms both train and test.
        CRITICAL: SVD augmentation doubles feature count for members 0-1.
        """
        pipeline_config = self.preprocess_pipelines[ensemble_idx]['rebalance']
        seed = self.preprocess_seeds[ensemble_idx * 3 + 1]  # RebalanceFeatureDistribution seed
        
        # Split concatenated data (matching Original line 475-476)
        n_train = len(y_train)
        x_train = x_concat[:n_train]
        x_test = x_concat[n_train:]
        
        if x_train.shape[1] != x_test.shape[1]:
            x_test = x_test[:, :x_train.shape[1]]
        
        # Get categorical and continuous indices
        all_indices = list(range(x_concat.shape[1]))
        cont_indices = [i for i in all_indices if i not in categorical_indices]
        
        # Build ColumnTransformer based on config (matching Original lines 494-637)
        workers = []
        if pipeline_config['original_flag']:
            # Add original features passthrough
            workers.append(("original", "passthrough", all_indices))
        
        # Apply quantile transform if specified (matching Original lines 507-637)
        if pipeline_config['worker_tags']:
            worker_tag = pipeline_config['worker_tags'][0]
            # Determine trans_indices based on discrete_flag (matching Original lines 494-503)
            if pipeline_config['original_flag']:
                trans_indices = categorical_indices + cont_indices if pipeline_config['discrete_flag'] else cont_indices
            elif pipeline_config['discrete_flag']:
                trans_indices = categorical_indices + cont_indices
            else:
                trans_indices = cont_indices
            
            # Matching Original lines 534-637: handle different worker_tags
            if worker_tag == "quantile_uniform_10":
                qt = QuantileTransformer(
                    output_distribution="uniform",
                    n_quantiles=max(n_train // 10, 2),
                    random_state=seed
                )
                workers.append((f"feat_transform_{worker_tag}", qt, trans_indices))
            elif worker_tag is None:
                # Matching Original line 623-624: even None adds a worker with FunctionTransformer
                workers.append((f"feat_transform_{worker_tag}", FunctionTransformer(lambda x: x), trans_indices))
            else:
                # Matching Original line 634: default to FunctionTransformer
                workers.append((f"feat_transform_{worker_tag}", FunctionTransformer(lambda x: x), trans_indices))
        
        # Build ColumnTransformer
        if workers:
            ct_worker = ColumnTransformer(workers, remainder="drop", sparse_threshold=0.0)
        else:
            # No transformation, just passthrough
            ct_worker = FunctionTransformer(lambda x: x)
        
        # Apply SVD augmentation if specified (CRITICAL - doubles features!)
        # Matching Original lines 640-653
        if pipeline_config['svd_tag'] == "svd" and x_concat.shape[1] >= 2:
            n_components = max(1, min(n_train // 10 + 1, x_concat.shape[1] // 2))
            
            # Matching Original line 643-650: nested Pipeline structure
            save_standard_pipeline = Pipeline([
                ("i2n_pre", FunctionTransformer(
                    func=lambda x: np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan),
                    inverse_func=lambda x: x,
                    check_inverse=False
                )),
                ("fill_missing_pre", SimpleImputer(missing_values=np.nan, strategy="mean", keep_empty_features=True)),
                ("standard", StandardScaler(with_mean=False)),
                ("i2n_post", FunctionTransformer(
                    func=lambda x: np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan),
                    inverse_func=lambda x: x,
                    check_inverse=False
                )),
                ("fill_missing_post", SimpleImputer(missing_values=np.nan, strategy="mean", keep_empty_features=True))
            ])
            
            svd_pipeline = Pipeline([
                ("save_standard", save_standard_pipeline),
                ("svd", TruncatedSVD(algorithm="arpack", n_components=n_components, random_state=seed))
            ])
            
            # FeatureUnion: original + SVD features (DOUBLES feature count!)
            svd_worker = FeatureUnion([
                ("default", FunctionTransformer(func=lambda x: x)),
                ("svd", svd_pipeline)
            ])
            
            full_worker = Pipeline([("worker", ct_worker), ("svd_worker", svd_worker)])
        else:
            full_worker = ct_worker
        
        # Fit on train only, transform both (matching Original lines 479-482)
        x_train_transformed = full_worker.fit_transform(x_train)
        x_test_transformed = full_worker.transform(x_test)
        
        # Concatenate back
        x_transformed = np.concatenate([x_train_transformed, x_test_transformed], axis=0)
        
        return x_transformed

    def _apply_categorical_encoder(self, x_concat, y_train, ensemble_idx, categorical_indices):
        """
        Apply CategoricalFeatureEncoder matching Original LimiX.
        Re-encodes categoricals with optional shuffling.
        Note: After SVD augmentation, feature count may have doubled, but categorical_indices
        still refer to original positions (first half of features).
        """
        pipeline_config = self.preprocess_pipelines[ensemble_idx]['categorical']
        seed = self.preprocess_seeds[ensemble_idx * 3 + 2]  # CategoricalFeatureEncoder seed
        
        if pipeline_config['encoding_strategy'] == 'numeric':
            # No re-encoding, just return as is
            return x_concat, categorical_indices
        
        # Split for processing
        n_train = len(y_train)
        x_train = x_concat[:n_train]
        x_test = x_concat[n_train:]
        
        # After SVD, features may have doubled. Categorical indices refer to original features.
        # If SVD was applied, categorical features are in the first half (original features).
        n_original_features = self.num_features_  # Original feature count before SVD
        n_current_features = x_concat.shape[1]
        
        # Filter categorical indices to only those that exist in current feature set
        # If SVD doubled features, categorical features are in first n_original_features
        valid_cat_indices = [idx for idx in categorical_indices if idx < n_current_features]
        
        if pipeline_config['encoding_strategy'].startswith('ordinal'):
            # Re-encode categoricals (matching Original lines 273-288)
            if valid_cat_indices:
                ct = ColumnTransformer(
                    [("ordinal_encoder", OrdinalEncoder(
                        handle_unknown="use_encoded_value", 
                        unknown_value=np.nan
                    ), valid_cat_indices)],
                    remainder="passthrough"
                )
                
                x_train_encoded = ct.fit_transform(x_train)
                x_test_encoded = ct.transform(x_test)
                
                # Apply category shuffling if ordinal_strict_feature_shuffled
                if pipeline_config['encoding_strategy'].endswith('_shuffled'):
                    np_rng = np.random.default_rng(seed)
                    category_mappings = {}
                    
                    # Get the transformer index for each categorical feature
                    # ColumnTransformer uses the position in the list, not the actual column index
                    for cat_idx_pos, cat_idx in enumerate(valid_cat_indices):
                        # Access categories using the position in the transformer list
                        ordinal_transformer = ct.named_transformers_["ordinal_encoder"]
                        col_cats = len(ordinal_transformer.categories_[cat_idx_pos])
                        perm = np_rng.permutation(col_cats)
                        category_mappings[cat_idx] = perm
                        
                        # Apply permutation to train - use the actual column index in the transformed data
                        col_data = x_train_encoded[:, cat_idx].copy()
                        valid_mask = ~np.isnan(col_data)
                        if np.any(valid_mask):
                            col_vals = col_data[valid_mask].astype(int)
                            # Ensure values are within bounds
                            col_vals = np.clip(col_vals, 0, len(perm) - 1)
                            col_data[valid_mask] = perm[col_vals]
                            x_train_encoded[:, cat_idx] = col_data
                        
                        # Apply permutation to test
                        col_data_test = x_test_encoded[:, cat_idx].copy()
                        valid_mask_test = ~np.isnan(col_data_test)
                        if np.any(valid_mask_test):
                            col_vals_test = col_data_test[valid_mask_test].astype(int)
                            # Ensure values are within bounds
                            col_vals_test = np.clip(col_vals_test, 0, len(perm) - 1)
                            col_data_test[valid_mask_test] = perm[col_vals_test]
                            x_test_encoded[:, cat_idx] = col_data_test
                
                x_concat_encoded = np.concatenate([x_train_encoded, x_test_encoded], axis=0)
                # Categorical indices remain the same after ordinal encoding
                return x_concat_encoded, valid_cat_indices
            else:
                return x_concat, categorical_indices
        else:
            return x_concat, categorical_indices
    def _apply_feature_shuffler(self, x_concat, ensemble_idx):
        """
        Apply FeatureShuffler matching Original LimiX.
        """
        pipeline_config = self.preprocess_pipelines[ensemble_idx]['feature_shuffle']
        # Use base seed + offset (matching Original LimiX line 168)
        seed = self.seed + pipeline_config['offset']
        
        if pipeline_config['mode'] == 'shuffle':
            np_rng = np.random.default_rng(seed)
            indices = np.arange(x_concat.shape[1])
            shuffled_indices = np_rng.permutation(indices)
            return x_concat[:, shuffled_indices]
        else:
            return x_concat

    def fit(self, X, y):
        """
        Store training data. Preprocessing will be applied during predict.
        """
        # Note: Original LimiX uses same seed (self.seed) for all ensemble members in model call
        # No need to initialize ensemble_seeds since we use self.seed directly
        
        # 1. Encode Labels
        if isinstance(y, pd.Series):
            y = y.values
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.num_classes_ = len(self.le_.classes_)

        # 2. Prepare class permutations (matching Original LimiX)
        np_rng = np.random.default_rng(self.seed)  # Initialize for class permutations
        class_shuffle_factor = 2
        noise = np_rng.random((self.n_ensemble * class_shuffle_factor, self.num_classes_))
        shufflings = np.argsort(noise, axis=1)
        uniqs = np.unique(shufflings, axis=0)
        balance_count = self.n_ensemble // len(uniqs)
        self.class_permutations = list(chain.from_iterable(repeat(elem, balance_count) for elem in uniqs))
        cout = self.n_ensemble % len(uniqs)
        if cout > 0:
            self.class_permutations += [uniqs[i] for i in np_rng.choice(len(uniqs), size=cout)]

        # 3. Basic encoding (matching convert_category2num)
        if isinstance(X, pd.DataFrame):
            cat_cols = X.select_dtypes(include=['object', 'category', 'string', 'bool']).columns.tolist()
            num_cols = X.select_dtypes(exclude=['object', 'category', 'string', 'bool']).columns.tolist()
            
            if cat_cols:
                self.enc_ = ColumnTransformer(
                    transformers=[
                        ('cat', OrdinalEncoder(
                            handle_unknown='use_encoded_value', 
                            unknown_value=np.nan,
                            encoded_missing_value=np.nan
                        ), cat_cols),
                        ('num', 'passthrough', num_cols)
                    ],
                    verbose_feature_names_out=False
                )
                X_base = self.enc_.fit_transform(X)
            else:
                self.enc_ = None
                X_base = X.values
        else:
            X_base = np.array(X, dtype=np.float32)
            self.enc_ = None
        
        X_base = np.array(X_base, dtype=np.float32)
        self.num_features_ = X_base.shape[1]
        
        # 4. Get categorical indices (matching get_categorical_features_indices)
        self.categorical_indices = []
        for idx, col in enumerate(X_base.T):
            if len(np.unique(col[~np.isnan(col)])) < 10:  # min_unique_num_for_numerical_infer
                self.categorical_indices.append(idx)
        
        # 5. Build preprocessing pipelines
        self._build_preprocess_pipelines()
        
        # 6. Store base training data
        self.X_train_base_ = X_base
        self.y_train_ = torch.tensor(y_encoded, dtype=torch.float32).to(self.device)

        # 7. Init Model
        if self.model is None:
            self._init_model()
            
        return self

    def _prepare_batch(self, X_query, ensemble_idx=0, class_permutation=None):
        """
        Matches Original LimiX preprocessing exactly.
        """
        # 1. Encode query (base encoding only)
        if self.enc_ is not None:
            X_query_base = self.enc_.transform(X_query)
        else:
            X_query_base = np.array(X_query, dtype=np.float32)
        X_query_base = np.array(X_query_base, dtype=np.float32)
        
        # 2. Concatenate train and test BEFORE preprocessing (matching Original line 298)
        x_concat = np.concatenate([self.X_train_base_, X_query_base], axis=0)
        
        # 3. Get permuted y_train labels (CRITICAL: must be permuted BEFORE passing to fit_transform)
        # Matching Original line 325: y_ = self.class_permutations[id_pipe][y.copy()]
        y_train_labels = self.y_train_.cpu().numpy().astype(int)
        if class_permutation is not None:
            # FIX: Apply permutation to class labels, not sample indices!
            # Original does: class_permutations[id_pipe][y.copy()]
            # This means: for each label value, look up its permuted value
            # Ensure class_permutation is a numpy array for indexing
            if isinstance(class_permutation, list):
                class_permutation = np.array(class_permutation)
            y_train_permuted = class_permutation[y_train_labels]
        else:
            y_train_permuted = y_train_labels
        
        # 4. Apply preprocessing pipeline (matching Original line 327-345)
        # Step 1: FilterValidFeatures (matching Original lines 158-189)
        # Check for constant features
        valid_features = ((x_concat[0:1, :] == x_concat).mean(axis=0) < 1.0)
        # Also check for all-NaN features in train vs test (matching Original lines 163-172)
        n_train = len(y_train_permuted)
        nan_train = np.isnan(x_concat[:n_train, :])
        all_nan_train = np.all(nan_train, axis=0)
        nan_test = np.isnan(x_concat[n_train:, :])
        all_nan_test = np.all(nan_test, axis=0)
        features_nan = all_nan_train | all_nan_test
        # Convert to boolean array if needed
        if isinstance(valid_features, np.ndarray):
            valid_features = valid_features & ~features_nan
        else:
            valid_features = valid_features & ~np.array(features_nan)
        
        if not np.any(valid_features):
            raise ValueError("All features are constant! Please check your data.")
        
        if not np.all(valid_features):
            x_concat = x_concat[:, valid_features]
            # Update categorical indices (matching Original lines 177-181)
            valid_cat_indices = [
                index
                for index, idx in enumerate(np.where(valid_features)[0])
                if idx in self.categorical_indices
            ]
            categorical_indices = valid_cat_indices
        else:
            categorical_indices = self.categorical_indices
        
        # Step 2: RebalanceFeatureDistribution (fits on train, transforms both)
        # CRITICAL: Pass permuted y_train_labels, not original! (matching Original line 345)
        x_concat = self._apply_rebalance_feature_distribution(
            x_concat, y_train_permuted, ensemble_idx, categorical_indices
        )
        
        # Step 3: CategoricalFeatureEncoder (re-encode with shuffling)
        x_concat, categorical_indices = self._apply_categorical_encoder(
            x_concat, y_train_permuted, ensemble_idx, categorical_indices
        )
        
        # Step 4: FeatureShuffler
        x_concat = self._apply_feature_shuffler(x_concat, ensemble_idx)
        
        # 5. Convert to tensor and split
        x_full = torch.tensor(x_concat, dtype=torch.float32).to(self.device)
        n_train = self.X_train_base_.shape[0]
        x_train_t = x_full[:n_train]
        x_query_t = x_full[n_train:]
        
        # 6. Prepare labels with class permutation (matching Original line 349)
        y_train_tensor = torch.from_numpy(y_train_permuted).float().to(self.device)
        
        # 7. Concatenate for model input (matching Original line 392)
        X_full = torch.cat([x_train_t, x_query_t], dim=0).unsqueeze(0)
        
        # 8. Concatenate Y (matching Original line 393)
        # Original LimiX: y_ contains only training labels [N_train], then unsqueezes to [1, N_train]
        # But the model expects y to match x's sequence length, so we pad with zeros for test
        y_dummy = torch.zeros(x_query_t.shape[0], device=self.device)
        y_full = torch.cat([y_train_tensor, y_dummy], dim=0).unsqueeze(0)
        
        # 9. eval_pos is the number of training samples (matching Original line 394)
        # Original: eval_pos=y_.shape[1] where y_ is [1, N_train]
        # Since we pad y_full, we use n_train directly
        eval_pos = n_train  # Number of training samples
        
        return X_full, y_full, eval_pos

    def predict(self, X):
        """Predict with ensemble averaging."""
        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        return self.le_.inverse_transform(preds)

    def predict_proba(self, X):
        """Predict probabilities with ensemble averaging (matching Original LimiX exactly)."""
        all_outputs = []  # Store logits, not probabilities (matching Original)
        
        for ensemble_idx in range(self.n_ensemble):
            class_perm = self.class_permutations[ensemble_idx] if self.class_permutations is not None else None
            
            X_full, y_full, eval_pos = self._prepare_batch(X, ensemble_idx=ensemble_idx, class_permutation=class_perm)
            
            # Set seed right before model call (matching Original lines 350-351)
            # CRITICAL: Original uses same seed (self.seed) for ALL ensemble members, not per-member seeds
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            
            self.model.eval()
            # Use autocast and inference_mode for mixed precision (matching Original line 391)
            with torch.autocast(device_type=self.device.type if isinstance(self.device, torch.device) else self.device, enabled=True), torch.inference_mode():
                logits = self.model(
                    x=X_full, 
                    y=y_full, 
                    eval_pos=eval_pos,
                    task_type='cls'
                )
            
            # Handle output shape (matching Original line 403)
            # Original: output = output if isinstance(output, dict) else output.squeeze(0)
            if isinstance(logits, dict):
                logits = logits['cls_output']
            # After extracting from dict, logits is a tensor, so we need to squeeze if needed
            if len(logits.shape) == 3:
                logits = logits.squeeze(0)
            
            # Convert to tensor if needed
            if isinstance(logits, np.ndarray):
                logits = torch.from_numpy(logits).float()
            else:
                logits = logits.float()
            
            # Model should already return only test predictions
            n_test_expected = len(X) if hasattr(X, '__len__') else X.shape[0] if hasattr(X, 'shape') else None
            if n_test_expected is not None and len(logits.shape) == 2:
                if logits.shape[0] > n_test_expected:
                    logits = logits[-n_test_expected:, :]
            
            # Apply softmax temperature BEFORE class permutation (matching Original lines 405-406)
            # CRITICAL: Slice to actual num_classes (model outputs 10, but we only use first n_classes)
            # Matching Original: output = (output[:, :self.n_classes].float() / self.softmax_temperature)
            if self.softmax_temperature != 1:
                logits = (logits[:, :self.num_classes_].float() / self.softmax_temperature)
            else:
                logits = logits[:, :self.num_classes_].float()
            
            # Apply FORWARD class permutation to output (matching Original line 408)
            if class_perm is not None:
                logits = logits[..., class_perm]
            
            all_outputs.append(logits)
        
        # Apply softmax to each output, then average (matching Original lines 411-412)
        outputs = [torch.nn.functional.softmax(o, dim=1) for o in all_outputs]
        output = torch.stack(outputs).mean(dim=0)
        
        # Convert to numpy before final normalization (matching Original line 414)
        output = output.float().cpu().numpy()
        
        # Final normalization (matching Original lines 417-419)
        output = output / output.sum(axis=1, keepdims=True)
        
        return output