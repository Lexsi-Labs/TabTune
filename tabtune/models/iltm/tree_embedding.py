import logging
import gc
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomTreesEmbedding, GradientBoostingClassifier
from .utils import preprocess_cat_features_for_catboost, get_gpu_memory_info, pick_gpu_ram_part, clear_cuda_cache


# NOTE (TabTune vendoring): xgboost and catboost are OPTIONAL dependencies.
# They are only required for the tree-embedding checkpoints (e.g. 'xgbrconcat',
# 'cbrconcat', 'xgb', 'catb', 'rtrcb'). Import them lazily so that the core
# torch/sklearn/numpy path (e.g. 'r128bn', 'rnobn', 'rtr' or local checkpoints
# without tree embeddings) works without them.
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - depends on environment
    xgb = None
try:
    from catboost import CatBoostRegressor, Pool, CatBoostClassifier
except ImportError:  # pragma: no cover - depends on environment
    CatBoostRegressor = Pool = CatBoostClassifier = None
logger = logging.getLogger(__name__)


class TreeEmbedding:
    def __init__(
        self,
        tree_model: str,
        cat_features: list,
        task_type: str,
        seed: int = 42,
        n_estimators: int = 100,
        lr: float = None,
        max_depth: int = None,
        min_samples_leaf: int = None,
        subsample: float = None,
        feature_fraction: float = None,
        device: str = 'gpu',
        select_best_model: bool = True,
        eval_size: float = 0.2,
        use_default_params: bool = False,
        max_leaves: int = None,
        gamma: float = None,
        l2_leaf_reg: float = None,
        bagging_temperature: float = None,
        onehot_max_features: bool = False
    ):
        """
        Initializes the TreeEmbedding class.

        Parameters:
            tree_model (str): The tree-based model to use ('GB', 'RT', 'XGBoost_approx', 'XGBoost_hist', 'CatBoost').
            cat_features (list): List of categorical features.
            task_type (str): The type of task ('classification' or 'regression').
            seed (int): Random seed for reproducibility.
            n_estimators (int): Number of estimators for the tree-based model. Default is 100.
            lr (float): Learning rate for the tree-based model. Default is None (use default parameters in specific tree model).
            max_depth (int): Maximum depth of the tree-based model. Default is None (use default parameters in specific tree model).
            min_samples_leaf (int): Minimum number of samples per leaf (for CatBoost: min_data_in_leaf, for XGBoost: min_child_weight).
                For XGBoost, this is the minimum sum of instance weights in a child node. If sample weights are all 1 (the default), this is equivalent to the minimum number of samples per leaf.
                Range: [1, 50] (recommended for both CatBoost and XGBoost).
            subsample (float): Fraction of samples to be used for fitting the tree-based model. Default is None (use default parameters in specific tree model).
            feature_fraction (float): Fraction of features to be used for fitting the tree-based model. Default is None (use default parameters in specific tree model).
            device (str): Compute device: 'cpu', or 'cuda'/'cuda:N'/'gpu'/'gpu:N' for GPU (XGBoost/CatBoost only).
            select_best_model (bool): Whether to select the best model based on the validation set. Default is True.
            eval_size (float): Fraction of data to use for validation.
            use_default_params (bool): Whether to use default parameters for the tree-based model. Will override any other parameters that are set.
            max_leaves (int): Maximum number of leaves. Used by XGBoost (if grow_policy='lossguide') and CatBoost.
            gamma (float): Minimum loss reduction required to make a further partition on a leaf node of the tree. Used by XGBoost.
            l2_leaf_reg (float): L2 regularization term on weights. Used by CatBoost (l2_leaf_reg) and XGBoost (reg_lambda).
            bagging_temperature (float): Controls intensity of Bayesian bagging. Used by CatBoost.
            onehot_max_features (bool): If True, one-hot encoded embeddings will be subset to the top 8192 most frequent features.
        """
        self.tree_model = tree_model
        if self.tree_model in ('XGBoost_approx', 'XGBoost_hist') and xgb is None:
            raise ImportError(
                f"Tree embedding model '{self.tree_model}' requires the optional "
                "'xgboost' package, which is not installed. Install it with "
                "`pip install xgboost`, or use an iLTM checkpoint that does not "
                "rely on XGBoost tree embeddings (e.g. 'r128bn', 'rnobn', 'rtr')."
            )
        if self.tree_model == 'CatBoost' and CatBoostRegressor is None:
            raise ImportError(
                "Tree embedding model 'CatBoost' requires the optional "
                "'catboost' package, which is not installed. Install it with "
                "`pip install catboost`, or use an iLTM checkpoint that does not "
                "rely on CatBoost tree embeddings (e.g. 'r128bn', 'rnobn', 'rtr')."
            )
        self.cat_features = cat_features
        self.task_type = task_type
        self.seed = seed
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.feature_fraction = feature_fraction
        self.device = device
        self.select_best_model = select_best_model
        self.eval_size = eval_size
        self.max_leaves = max_leaves
        self.gamma = gamma
        self.l2_leaf_reg = l2_leaf_reg
        self.bagging_temperature = bagging_temperature
        self.onehot_max_features = onehot_max_features
        self.n_orig_features_to_keep_ = None

        if use_default_params: # override any other parameters that are set
            self.lr = None
            self.max_depth = None
            self.min_samples_leaf = None
            self.subsample = None
            self.feature_fraction = None
            self.max_leaves = None
            self.gamma = None
            self.l2_leaf_reg = None
            self.bagging_temperature = None

        self.encoders = {}
        self.onehot_encoder = None
        self.model = None
        self.onehot_top_features_idx_ = None 

        # OOM guards
        if self.max_depth is not None:
            # If both are large, drop estimators straight to 300
            if self.max_depth >= 8 and self.n_estimators > 300:
                logger.warning(
                    "n_estimators is %d and max_depth is %d. Limiting n_estimators to 300 to avoid out of memory issues.",
                    self.n_estimators, self.max_depth
                )
                self.n_estimators = 300

            # If estimators are 300 or more, cap depth at 7
            if self.n_estimators >= 300 and self.max_depth > 7:
                logger.warning(
                    "n_estimators is %d and max_depth is %d. Limiting max_depth to 7 to avoid out of memory issues.",
                    self.n_estimators, self.max_depth
                )
                self.max_depth = min(self.max_depth, 7)

    def _is_gpu_device(self) -> bool:
        device = str(self.device).lower()
        return device == 'gpu' or device.startswith('gpu:') or device == 'cuda' or device.startswith('cuda:')

    def _xgboost_device(self) -> str:
        if not self._is_gpu_device():
            return 'cpu'

        device = str(self.device).lower()
        if device.startswith('cuda'):
            return device
        if device.startswith('gpu:'):
            return f"cuda:{device.split(':', 1)[1]}"
        return 'cuda'

    def _catboost_task_type(self) -> str:
        return 'GPU' if self._is_gpu_device() else 'CPU'

    def _catboost_devices(self) -> str:
        if not self._is_gpu_device():
            return ''

        device = str(self.device).lower()
        if ':' in device:
            return device.split(':', 1)[1]
        return '0'


    def _handle_categorical_features(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """
        Encodes categorical features based on the tree model requirements.

        Parameters:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: Transformed features with categorical variables encoded.
        """
        X_encoded = X.copy()
        X_encoded = pd.DataFrame(X_encoded)

        # If the model is CatBoost, do nothing (it handles categorical features natively).
        if self.tree_model == 'CatBoost':
            return X_encoded

        # For all other supported tree-based models, perform encoding if categorical features are specified.
        if self.tree_model in ['GB', 'RT', 'XGBoost_approx', 'XGBoost_hist']:
            if self.cat_features:
                # Convert indices in self.cat_features to column names
                cat_feature_cols = X_encoded.columns[self.cat_features]

                for col in cat_feature_cols:
                    # force categorical columns to string so there's a single uniform type (encoder crashes for mixed types)
                    X_encoded[col] = X_encoded[col].astype(str)

                    encoder = OrdinalEncoder(
                        handle_unknown='use_encoded_value',
                        unknown_value=-1
                    )
                    X_encoded[[col]] = encoder.fit_transform(X_encoded[[col]])
                    self.encoders[col] = encoder

            return X_encoded

        # If none of the above conditions match, the tree_model is unknown.
        raise ValueError(f"Unknown tree-based model: {self.tree_model}")

    def _handle_inf_and_value_too_large(self, X: pd.DataFrame, cap_threshold: float = 1e10) -> pd.DataFrame:
        """
        1. Replaces ±inf with NaN
        2. Capping very large (and very negative) values at `cap_threshold`

        Parameters:
            X (pd.DataFrame): The input DataFrame to sanitize.
            cap_threshold (float): The max absolute value to allow before capping.
                                    Defaults to 1e10.

        Returns:
            pd.DataFrame: A copy of X with values sanitized.
        """
        # 1. Replace ±inf with NaN
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting behavior.*')
            X = X.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)

        # 2. Cap large numeric values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].clip(-cap_threshold, cap_threshold)

        return X

    def _fit_model(self, X: pd.DataFrame, y: np.ndarray | pd.Series, eval_set: tuple = None) -> None:
        """
        Fits the tree-based model based on the specified type.

        Parameters:
            X (pd.DataFrame): Training features for model.
            y (np.ndarray | pd.Series): Training labels for model.
        """
        X_val, y_val = None, None

        # Models that don't use eval_set
        if self.tree_model == 'GB':
            params = {
                "random_state": self.seed,
                "validation_fraction": self.eval_size if self.select_best_model else 0.0,
                "n_iter_no_change": 50 if self.select_best_model else None,
            }
            if self.n_estimators is not None:
                params["n_estimators"] = self.n_estimators
            if self.lr is not None:
                params["learning_rate"] = self.lr
            if self.max_depth is not None:
                params["max_depth"] = self.max_depth
            if self.min_samples_leaf is not None:
                params["min_samples_leaf"] = self.min_samples_leaf
            if self.subsample is not None:
                params["subsample"] = self.subsample
            if self.feature_fraction is not None:
                params["max_features"] = self.feature_fraction
            if self.task_type == 'classification':
                self.model = GradientBoostingClassifier(**params)
            else:
                self.model = GradientBoostingRegressor(**params)
            self.model.fit(X, y)  # Doesn't support eval_set
            return

        elif self.tree_model == 'RT':
            params = {
                "random_state": self.seed,
                "sparse_output": False,
            }
            if self.n_estimators is not None:
                params["n_estimators"] = self.n_estimators
            if self.max_depth is not None:
                params["max_depth"] = self.max_depth
            if self.min_samples_leaf is not None:
                params["min_samples_leaf"] = self.min_samples_leaf
            if self.lr is not None:
                logger.warning("RT does not support lr parameter.")
            if self.subsample is not None:
                logger.warning("RT does not support subsample parameter.")
            if self.feature_fraction is not None:
                logger.warning("RT does not support feature_fraction parameter.")
            self.model = RandomTreesEmbedding(**params)
            self.model.fit(X, y)  # Doesn't support eval_set
            return

        # Prepare eval_set if requested
        if self.select_best_model:
            # If select_best_model is True, use eval_size from X for validation if eval_set is None
            if eval_set is None:
                assert 0 < self.eval_size < 1, "eval_size must be between 0 and 1."

                if self.task_type == 'classification':
                    # Convert y to a pandas Series with alignment to X
                    if not isinstance(y, pd.Series):
                        y_series = pd.Series(y, index=X.index)
                    else:
                        y_series = y.reindex(X.index)

                    # Identify rare classes
                    class_counts = y_series.value_counts()
                    rare_classes = class_counts[class_counts < 2].index.tolist()

                    # Separate rare instances to always include in training
                    is_rare = y_series.isin(rare_classes)
                    X_rare = X[is_rare]
                    y_rare = y_series[is_rare]

                    # Data without the rare classes for stratified split
                    X_rest = X[~is_rare]
                    y_rest = y_series[~is_rare]

                    # Check if stratification is possible on the remaining data
                    if not y_rest.empty and y_rest.nunique() > 0:
                        # Calculate the fraction required to have at least one sample per class
                        required_fraction = y_rest.nunique() / len(y_rest)
                        # Adjust eval_size if necessary
                        adjusted_eval_size = max(self.eval_size, required_fraction)
                        if adjusted_eval_size != self.eval_size:
                            logger.warning(f"eval_size {self.eval_size} is too low for stratification; adjusting to {adjusted_eval_size}.")
                        X_train_rest, X_val, y_train_rest, y_val = train_test_split(
                            X_rest, y_rest,
                            test_size=adjusted_eval_size,  # self.eval_size,
                            random_state=self.seed,
                            stratify=y_rest
                        )
                    else:
                        # If no data remains for stratification, fall back to a simple split
                        X_train_rest, X_val, y_train_rest, y_val = train_test_split(
                            X, y_series,
                            test_size=self.eval_size,
                            random_state=self.seed
                        )
                        X_val, y_val = X_val.copy(), y_val.copy()  # Ensure independence

                    # Combine the rare class instances with the stratified training set
                    X_train = pd.concat([X_train_rest, X_rare])
                    y_train = pd.concat([y_train_rest, y_rare])
                else:
                    # Regression: simple random split without stratification
                    if not isinstance(y, pd.Series):
                        y_series = pd.Series(y, index=X.index)
                    else:
                        y_series = y.reindex(X.index)

                    X_train, X_val, y_train, y_val = train_test_split(
                        X,
                        y_series,
                        test_size=self.eval_size,
                        random_state=self.seed,
                    )

                # Create eval_set
                eval_set = (X_val, y_val)
                X, y = X_train, y_train
            else:
                X_val, y_val = eval_set
                X_val = self._handle_categorical_features(X_val)

        # Now fit the chosen model
        if self.tree_model in ['XGBoost_approx', 'XGBoost_hist']:
            params = {
                'tree_method': 'hist' if self.tree_model == 'XGBoost_hist' else 'approx',
                'seed': self.seed,
                'device': self._xgboost_device(),
                'n_jobs': -1
            }
            if self.task_type == 'regression':
                params['objective'] = 'reg:squarederror'
            if self.lr is not None:
                params['learning_rate'] = self.lr
            if self.max_depth is not None:
                params['max_depth'] = self.max_depth
            if self.min_samples_leaf is not None:
                params['min_child_weight'] = self.min_samples_leaf
            if self.subsample is not None:
                params['subsample'] = self.subsample
            if self.feature_fraction is not None:
                params['colsample_bytree'] = self.feature_fraction
            if self.max_leaves is not None:
                params['max_leaves'] = self.max_leaves
                params['grow_policy'] = 'lossguide'
                if 'max_depth' not in params or params['max_depth'] in (None, 0):
                    params['max_depth'] = 0
            if self.gamma is not None:
                params['gamma'] = self.gamma
            if self.l2_leaf_reg is not None:
                params['reg_lambda'] = self.l2_leaf_reg

            logger.debug(
                "XGBoost %s: requested device=%s, resolved device=%s",
                self.tree_model, self.device, params['device'],
            )

            # conservative max_bin when VRAM is tight
            info = get_gpu_memory_info()
            if info and info["free_mb"] < 1024:
                params['max_bin'] = max(64, int(params.get('max_bin', 256) // 2))

            X = self._handle_inf_and_value_too_large(X)
            dtrain = xgb.DMatrix(X, y, enable_categorical=True)

            if X_val is not None:
                X_val = self._handle_inf_and_value_too_large(X_val)
                dval = xgb.DMatrix(X_val, y_val, enable_categorical=True)
                evals = [(dtrain, 'train'), (dval, 'valid')]
            else:
                dval = None
                evals = None

            num_rounds = int(self.n_estimators or 100)
            max_attempts = 4

            for attempt in range(1, max_attempts + 1):
                try:
                    logger.debug("XGBoost %s: training attempt %d on device=%s", self.tree_model, attempt, params['device'])
                    self.model = xgb.train(
                        params=params,
                        dtrain=dtrain,
                        num_boost_round=num_rounds,
                        evals=evals,
                        early_stopping_rounds=100 if self.select_best_model else None,
                        verbose_eval=0
                    )
                    break

                except Exception as e:
                    msg = str(e).lower()
                    if "out of memory" not in msg and "cuda" not in msg:
                        raise
                    logger.warning("XGBoost OOM on attempt %d. Adapting and retrying.", attempt)
                    clear_cuda_cache()

                    # Backoff order: max_bin -> max_leaves -> max_depth -> rounds
                    if params.get('max_bin', None) is None:
                        params['max_bin'] = 256
                        continue
                    if params['max_bin'] > 64:
                        params['max_bin'] = max(64, params['max_bin'] // 2)
                        continue

                    if params.get('max_leaves', None):
                        params['max_leaves'] = max(31, int(params['max_leaves'] // 2))
                        continue

                    if int(params.get('max_depth', 0) or 0) > 3:
                        md = int(params.get('max_depth', 0) or 0)
                        params['max_depth'] = max(3, md - 1)
                        continue

                    if num_rounds > 50:
                        num_rounds = max(50, num_rounds // 2)
                        continue

                    if str(params['device']).startswith('cuda'):
                        logger.warning("XGBoost OOM persists. Falling back to CPU.")
                        params['device'] = 'cpu'
                        logger.debug("XGBoost %s: falling back to device=cpu", self.tree_model)
                        continue

                    raise

            del dtrain
            if dval is not None:
                del dval
            gc.collect()


        elif self.tree_model == 'CatBoost':
            catboost_params = {
                'task_type': self._catboost_task_type(),
                'devices': self._catboost_devices(),
                'random_seed': self.seed,
                'verbose': 0,
                'thread_count': -1
            }
            if self.n_estimators is not None:
                catboost_params['iterations'] = self.n_estimators
            if self.lr is not None:
                catboost_params['learning_rate'] = self.lr
            if self.max_depth is not None:
                catboost_params['max_depth'] = self.max_depth
            if self.min_samples_leaf is not None:
                catboost_params['min_data_in_leaf'] = self.min_samples_leaf
            if self.max_leaves is not None:
                catboost_params['max_leaves'] = self.max_leaves
                catboost_params['grow_policy'] = 'Lossguide'
            if self.l2_leaf_reg is not None:
                catboost_params['l2_leaf_reg'] = self.l2_leaf_reg

            # Memory-aware sampling settings
            if self.bagging_temperature is not None:
                # 1) If user passed bagging_temperature, force Bayesian sampling
                catboost_params['bootstrap_type'] = 'Bayesian'
                catboost_params['bagging_temperature'] = self.bagging_temperature
                if self.subsample is not None and self.subsample < 1.0 :
                    logger.debug("CatBoost: bagging_temperature is set, so 'subsample' parameter will be ignored by CatBoost in favor of Bayesian bagging.")
            elif self.subsample is not None and 0.0 < self.subsample < 1.0:
                # 2) Else if user passed subsample < 1.0, force Bernoulli sampling
                catboost_params['bootstrap_type'] = 'Bernoulli'
                catboost_params['subsample'] = self.subsample

            if self.feature_fraction is not None and self.feature_fraction < 1.0:
                if self._is_gpu_device():
                    logger.debug("CatBoost: feature_fraction (rsm) < 1.0 is not supported on GPU. "
                                   "Ignoring feature_fraction to stay on GPU. Effective feature_fraction will be 1.0.")
                else: # device is 'cpu'
                    catboost_params['rsm'] = self.feature_fraction

            # Cap GPU memory use fraction based on what is currently free
            if catboost_params['task_type'] == 'GPU':
                catboost_params['gpu_ram_part'] = pick_gpu_ram_part(self.device)
                info = get_gpu_memory_info(self.device)
                if info and info["free_mb"] < 2048:
                    logger.warning("Low VRAM detected, training CatBoost on CPU to avoid tree-ctr OOM.")
                    catboost_params['task_type'] = 'CPU'
                    catboost_params.pop('devices', None)
                    catboost_params.pop('gpu_ram_part', None)
                    logger.debug("CatBoost: low VRAM fallback to task_type=CPU")

            logger.debug(
                "CatBoost: requested device=%s, resolved task_type=%s, devices=%s",
                self.device,
                catboost_params['task_type'],
                catboost_params.get('devices', ''),
            )

            if self.task_type == 'regression':
                ctor = CatBoostRegressor
            else:
                ctor = CatBoostClassifier

            X_prep = preprocess_cat_features_for_catboost(X, self.cat_features)
            if X_val is not None:
                X_val_prep = preprocess_cat_features_for_catboost(X_val, self.cat_features)
                train_pool = Pool(X_prep, label=y, cat_features=self.cat_features)
                eval_pool = Pool(X_val_prep, label=y_val, cat_features=self.cat_features)
            else:
                train_pool = Pool(X_prep, label=y, cat_features=self.cat_features)
                eval_pool = None

            # NEW: OOM-safe training loop with backoff and CPU fallback
            max_attempts = 4
            iters = int(catboost_params.get('iterations', 100))
            max_depth_local = catboost_params.get('max_depth', None)
            max_leaves_local = catboost_params.get('max_leaves', None)

            for attempt in range(1, max_attempts + 1):
                try:
                    logger.debug(
                        "CatBoost: training attempt %d on task_type=%s, devices=%s",
                        attempt,
                        catboost_params['task_type'],
                        catboost_params.get('devices', ''),
                    )
                    self.model = ctor(**catboost_params)
                    if eval_pool is not None:
                        self.model.fit(train_pool, eval_set=eval_pool, use_best_model=True, early_stopping_rounds=100)
                    else:
                        self.model.fit(train_pool)
                    break  # success

                except Exception as e:
                    msg = str(e).lower()
                    # Catch the variety of CatBoost GPU OOM wordings
                    oom_like = (
                        "out of memory" in msg
                        or "not enough memory" in msg
                        or ("cuda" in msg and "memory" in msg)
                        or ("tree-ctr" in msg and "memory" in msg)
                        or ("insufficient memory" in msg)
                    )
                    if not oom_like:
                        raise
                    logger.warning("CatBoost OOM on attempt %d. Adapting and retrying.", attempt)
                    clear_cuda_cache()

                    # First backoff: reduce iterations
                    if iters > 100:
                        iters = max(50, iters // 2)
                        catboost_params['iterations'] = iters
                        continue

                    # Second backoff: reduce depth or leaves
                    if max_leaves_local is not None and max_leaves_local > 31:
                        max_leaves_local = max(31, max_leaves_local // 2)
                        catboost_params['max_leaves'] = max_leaves_local
                        continue
                    if max_depth_local is not None and max_depth_local > 4:
                        max_depth_local = max(4, max_depth_local - 1)
                        catboost_params['max_depth'] = max_depth_local
                        continue

                    # Last fallback: switch to CPU
                    if catboost_params['task_type'] == 'GPU':
                        logger.warning("CatBoost OOM persists. Falling back to CPU.")
                        catboost_params['task_type'] = 'CPU'
                        catboost_params.pop('devices', None)
                        catboost_params.pop('gpu_ram_part', None)
                        logger.debug("CatBoost: OOM fallback to task_type=CPU")
                        continue

                    # If we are already on CPU and still OOM, rethrow
                    raise

            # cleanup
            del train_pool
            if eval_pool is not None:
                del eval_pool
            gc.collect()
        else:
            raise ValueError(f"Unknown tree-based model: {self.tree_model}")

    def _get_embeddings(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms the data into embeddings using the fitted tree-based model.

        Parameters:
            X (pd.DataFrame): Input features to transform.

        Returns:
            np.ndarray: Embeddings generated by the tree-based model.
        """
        if self.model is None:
            raise RuntimeError("TreeEmbedding model is not fitted. _fit_model did not set self.model.")
            
        if self.tree_model == 'GB':
            emb = self.model.apply(X)

        elif self.tree_model == 'RT':
            emb = self.model.transform(X)

        elif self.tree_model in ['XGBoost_approx', 'XGBoost_hist']:
            # Clean infs and values too large (crashes)
            X = self._handle_inf_and_value_too_large(X)
            dtest = xgb.DMatrix(X, enable_categorical=True)
            emb = self.model.predict(dtest, pred_leaf=True)

        elif self.tree_model == 'CatBoost':
            X = preprocess_cat_features_for_catboost(X, self.cat_features)
            test_pool = Pool(X, cat_features=self.cat_features) 
            emb = self.model.calc_leaf_indexes(test_pool)
            del test_pool
            gc.collect()

        else:
            raise ValueError(f"Unknown tree-based model: {self.tree_model}")

        emb = emb.reshape(emb.shape[0], -1)

        return emb

    def fit_tree(self, X: pd.DataFrame | np.ndarray, y: np.ndarray, eval_set: tuple = None, concat_with_orig_features: bool = True) -> None:
        """
        Fits a tree-based model on the input data.

        Parameters:
            X (pd.DataFrame | np.ndarray): Input features for model training.
            y (np.ndarray): Target variable for model training.
        """
        logger.info(f"Fitting tree model: {self.tree_model}")

        # Handle categorical features
        X_encoded = self._handle_categorical_features(X)

        # Fit the tree-based model
        self._fit_model(X_encoded, y, eval_set)

        # Get embeddings
        emb = self._get_embeddings(X_encoded)

        if self.tree_model != 'RT':
            # Fit OneHotEncoder
            self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.int8).fit(emb)

            if self.onehot_max_features:
                emb_oh = self.onehot_encoder.transform(emb)
                # identify top features
                feature_counts = emb_oh.sum(axis=0)
                # Sort features by frequency and get their indices
                sorted_feature_indices = np.argsort(feature_counts)[::-1]

                # Select the top features
                if concat_with_orig_features:
                    orig_feature_budget = 6144  # 75% for original features
                    n_orig_features = X.shape[1]

                    if n_orig_features > orig_feature_budget:
                        # If more original features than budget, select first `orig_feature_budget`
                        self.n_orig_features_to_keep_ = orig_feature_budget
                    else:
                        # Otherwise, keep all original features
                        self.n_orig_features_to_keep_ = n_orig_features
                    
                    # The rest of the 8192 budget goes to embeddings
                    top_k_embeddings = 8192 - self.n_orig_features_to_keep_
                    
                    logger.debug(f"Original features to keep: {self.n_orig_features_to_keep_}, GBDT embedding features to keep: {top_k_embeddings}")

                else:  # if not concatenating
                    self.n_orig_features_to_keep_ = None  # Not applicable
                    top_k_embeddings = 8192

                self.onehot_top_features_idx_ = sorted_feature_indices[:top_k_embeddings]

        logger.info(f"Tree model {self.tree_model} fitted successfully")

    def transform(self, X: pd.DataFrame | np.ndarray, onehot_encode: bool = True) -> np.ndarray:
        """
        Transforms the input data using the fitted tree-based model and encoders.

        Parameters:
            X (pd.DataFrame): Input features to transform.

        Returns:
            np.ndarray: Transformed embeddings.
        """
        assert self.model is not None, "Model not fitted. Call fit_tree() first."
        logger.info(f"Transforming data using tree model: {self.tree_model}")
        
        # Handle categorical features
        X_encoded = X.copy()
        X_encoded = pd.DataFrame(X_encoded)
        orig_data_shape = X_encoded.shape
        if self.tree_model in ['GB', 'RT', 'XGBoost_approx', 'XGBoost_hist']:
            for col, encoder in self.encoders.items():
                X_encoded[col] = X_encoded[col].astype(str)
                X_encoded[[col]] = encoder.transform(X_encoded[[col]])
        elif self.tree_model == 'CatBoost':
            # CatBoost handles categorical features natively
            pass
        else:
            raise ValueError(f"Unknown tree-based model: {self.tree_model}")

        # Get embeddings
        emb = self._get_embeddings(X_encoded)
        orig_emb_shape = emb.shape

        if onehot_encode and self.tree_model != 'RT':
            # Transform embeddings using OneHotEncoder
            assert self.onehot_encoder is not None, "Model not fitted. Call fit_tree() first."
            logger.debug(f"One-hot transforming leaves: emb.shape={emb.shape}, dtype={emb.dtype}")
            emb_oh = self.onehot_encoder.transform(emb)
            logger.debug(f"Original data shape: {orig_data_shape}, embedding shape: {orig_emb_shape}, after one-hot encoding: {emb_oh.shape}")
            
            if self.onehot_max_features:
                if self.onehot_top_features_idx_ is None:
                    raise Exception("onehot_max_features is True, but the top features were not identified. Make sure to call fit_tree first.")
                emb_oh = emb_oh[:, self.onehot_top_features_idx_]
                logger.debug(f"Subsetting to the top {emb_oh.shape[1]} features. New shape: {emb_oh.shape}")
            
            return emb_oh
        else:
            logger.debug(f"Original data shape: {orig_data_shape}, embedding shape: {orig_emb_shape} (no one-hot encoding applied)")
            return emb

    def get_tree_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Prepares data for the fitted tree model and returns integer predictions.
        Args:
            X (np.ndarray): The input data (shape [n_samples, n_features]).
        Returns:
            np.ndarray: Predicted class labels as integers (shape [n_samples]).
        """
        logger.info(f"Getting predictions using tree model: {self.tree_model}")
        
        X_prepared = self._handle_categorical_features(X)
        X_prepared = self._handle_inf_and_value_too_large(X_prepared)

        # Doesn't work with RT
        assert self.tree_model != 'RT', "RT does not support predict method."

        if self.tree_model in ['GB']:
            raw_preds = self.model.predict(X_prepared)
        elif self.tree_model in ['XGBoost_approx', 'XGBoost_hist']:
            dmat = xgb.DMatrix(X_prepared, enable_categorical=True)
            raw_preds = self.model.predict(dmat)
        elif self.tree_model == 'CatBoost':
            X_prepared = preprocess_cat_features_for_catboost(X_prepared, self.cat_features)
            raw_preds = self.model.predict(X_prepared)
            if len(raw_preds.shape) > 1:
                assert raw_preds.shape[1] == 1, "CatBoost predictions should have shape [n_samples, 1]."
                raw_preds = raw_preds.squeeze()
        else:
            raise ValueError(f"Unsupported tree_model: {self.tree_model}")
        if self.task_type == 'regression':
            return raw_preds.astype(float)
        else:
            return raw_preds.astype(int)
