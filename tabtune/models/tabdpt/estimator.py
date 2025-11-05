import json
from typing import Literal

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from safetensors import safe_open
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.utils.validation import check_is_fitted

from .model import TabDPTModel
from .utils import FAISS, convert_to_torch_tensor, Log1pScaler, generate_random_permutation

# Constants for model caching and download
_VERSION = "1_1"
_MODEL_NAME = f"tabdpt{_VERSION}.safetensors"
_HF_REPO_ID = "Layer6/TabDPT"
CPU_INF_BATCH = 16


class TabDPTEstimator(BaseEstimator):
    @staticmethod
    def download_weights() -> str:
        path = hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=_MODEL_NAME,
        )
        return path

    def __init__(
        self,
        mode: Literal["cls", "reg"],
        inf_batch_size: int = 512,
        normalizer: Literal["standard", "minmax", "robust", "power", "quantile-uniform", "quantile-normal", "log1p"] | None
            = "standard",
        missing_indicators: bool = False,
        clip_sigma: float = 4.,
        feature_reduction: Literal["pca", "subsample"] = "pca",
        faiss_metric: Literal["l2", "ip"] = "l2",
        device: str = None,
        use_flash: bool = True,
        compile: bool = True,
        model_weight_path: str | None = None,
    ):
        """
        Initializes the TabDPT Estimator
        Args:
            mode: Defines what mode the estimator is
                "cls" is classification, "reg" is regression
            inf_batch_size: The batch size for inferencing
            normalizer: Specifies normalization used for preprocessing before retrieval. Note that
                the model performs additional normalization in its forward function. By default the
                scikit-learn StandardScaler is used, which matches model training. Other options are:
                - "minmax": scikit-learn MinMaxScaler(feature_range=(-1,1))
                - "robust": scikit-learn RobustScaler()
                - "power": scikit-learn PowerTransformer()
                - "quantile-uniform": scikit-learn QuantileTransformer(output_distribution="uniform"), rescaled to (-1,1)
                - "quantile-normal": scikit-learn QuantileTransformer(output_distribution="normal")
                - "log1p": sign(X) * log(1 + abs(X))
                - None: no normalization
            missing_indicators: If True, adds an additional binary column for each feature with
                missing values indicating their position.
            clip_sigma: n*sigma used for outlier clipping
            feature_reduction: Method used to reduce the number of features when over the model's
                limit, either "pca" or "subsample"
            faiss_metric: Distance used for retrieval, either "l2" or "ip"
            device: Specifies the computational device (e.g., CPU, GPU)
                Identical to https://docs.pytorch.org/docs/stable/generated/torch.cuda.device.html
            use_flash: Specifies whether to use flash attention or not
            compile: Specifies whether to compile the model with torch before inference
            model_weight_path: path on file system specifying the model weights
                If no path is specified, then the model weights are downloaded from HuggingFace

        """
        self.mode = mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.inf_batch_size = inf_batch_size if self.device == "cuda" else min(inf_batch_size, CPU_INF_BATCH)
        self.use_flash = use_flash and self.device == "cuda"
        self.missing_indicators = missing_indicators

        if model_weight_path:
            self.path = model_weight_path
        else:
            self.path = self.download_weights()

        with safe_open(self.path, framework="pt", device=self.device) as f:
            meta = f.metadata()
            cfg_dict = json.loads(meta["cfg"])
            cfg = OmegaConf.create(cfg_dict)
            model_state = {k: f.get_tensor(k) for k in f.keys()}

        cfg.env.device = self.device
        self.model = TabDPTModel.load(model_state=model_state, config=cfg, use_flash=self.use_flash, clip_sigma=clip_sigma)
        self.model.eval()

        self.max_features = self.model.num_features
        self.max_num_classes = self.model.n_out
        self.compile = compile and self.device == "cuda"
        self.feature_reduction = feature_reduction
        self.faiss_metric = faiss_metric
        assert self.mode in ["cls", "reg"], "mode must be 'cls' or 'reg'"
        assert self.feature_reduction in ["pca", "subsample"], \
                "feature_reduction must be 'pca' or 'subsample'"
        assert self.faiss_metric in ["l2", "ip"], 'faiss_metric must be "l2" or "ip"'

        self.normalizer = normalizer
        match normalizer:
            case "standard":
                self.scaler = StandardScaler()
            case "minmax":
                self.scaler = MinMaxScaler(feature_range=(-1,1))
            case "robust":
                self.scaler = RobustScaler()
            case "power":
                self.scaler = PowerTransformer()
            case "quantile-uniform":
                self.scaler = QuantileTransformer(output_distribution="uniform")
            case "quantile-normal":
                self.scaler = QuantileTransformer(output_distribution="normal")
            case "log1p":
                self.scaler = Log1pScaler()
            case None:
                self.scaler = None
            case _:
                raise ValueError(
                    'normalizer must be one of '
                    '["standard", "minmax", "robust", "power", "quantile-uniform", "quantile-normal", "log1p", None]'
                )

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
        assert X.ndim == 2, "X must be a 2D array"
        assert y.ndim == 1, "y must be a 1D array"


        if self.missing_indicators:
            inds = np.isnan(X)
            self.has_missing_indicator = inds.any(axis=0)
            inds = inds[:, self.has_missing_indicator].astype(float)
            X = np.hstack((X, inds))

        self.imputer = SimpleImputer(strategy="mean")
        X = self.imputer.fit_transform(X)
        
        if self.scaler:
            X = self.scaler.fit_transform(X)
            
        if self.normalizer == 'quantile-uniform':
            X = 2*X - 1

        self.faiss_knn = FAISS(X, metric=self.faiss_metric)
        self.n_instances, self.n_features = X.shape
        
        self.X_train = X
        self.y_train = y
        
        if self.n_features > self.max_features and self.feature_reduction == "pca":
            train_x = convert_to_torch_tensor(self.X_train).to(self.device).float()
            q = min(train_x.shape[0], self.max_features)
            _, _, self.V = torch.pca_lowrank(train_x, q=q)

        self.is_fitted_ = True
        if self.compile:
            self.model = torch.compile(self.model)

    def _prepare_prediction(self, X: np.ndarray, class_perm: np.ndarray | None = None, seed: int | None = None):
        check_is_fitted(self)
        

        if self.missing_indicators:
            inds = np.isnan(X)[:, self.has_missing_indicator].astype(float)
            X = np.hstack((X, inds))
        self.X_test = self.imputer.transform(X)
        if self.scaler:
            self.X_test = self.scaler.transform(self.X_test)
            if self.normalizer == 'quantile-uniform':
                self.X_test = 2*self.X_test - 1

        train_x, train_y, test_x = (
            convert_to_torch_tensor(self.X_train).to(self.device).float(),
            convert_to_torch_tensor(self.y_train).to(self.device).float(),
            convert_to_torch_tensor(self.X_test).to(self.device).float(),
        )

        # Apply PCA/subsampling to reduce the number of features if necessary
        if self.n_features > self.max_features:
            if self.feature_reduction == "pca":
                train_x = train_x @ self.V
                test_x = test_x @ self.V
            elif self.feature_reduction == "subsample":
                feat_perm = generate_random_permutation(train_x.shape[1], seed)
                train_x = train_x[:, feat_perm][:, :self.max_features]
                test_x = test_x[:, feat_perm][:, :self.max_features]

        if class_perm is not None:
            assert self.mode == "cls", "class_perm only makes sense for classification"
            inv_perm = np.argsort(class_perm)
            train_y = train_y.to(torch.long)
            inv_perm = torch.as_tensor(inv_perm, device=train_y.device)
            train_y = inv_perm[train_y].to(torch.float)

        return train_x, train_y, test_x