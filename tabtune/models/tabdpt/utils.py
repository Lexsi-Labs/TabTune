from functools import wraps
from typing import Literal

import faiss
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from sklearn.base import BaseEstimator, TransformerMixin


def convert_to_torch_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def generate_random_permutation(n, seed=None):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return torch.randperm(n, generator=generator)


def flash_context(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "use_flash", False):
            assert torch.cuda.is_available(), "FlashAttention requires CUDA support"
            bf_support = torch.cuda.get_device_capability()[0] >= 8
            dtype = torch.bfloat16 if bf_support else torch.float16
            with (
                torch.autocast(device_type="cuda", dtype=dtype),
                sdpa_kernel(SDPBackend.FLASH_ATTENTION),
            ):
                return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)
    return wrapper


def normalize_data(x, eval_pos, return_mean_std=False):
    if eval_pos == -1:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
    else:
        mean = x[:eval_pos].mean(dim=0, keepdim=True)
        std = x[:eval_pos].std(dim=0, keepdim=True)
    x = (x - mean) / (std + 1e-6)
    if return_mean_std:
        return x, mean, std
    return x


def clip_outliers(x, eval_pos, n_sigma=4):
    if eval_pos == -1:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
    else:
        mean = x[:eval_pos].mean(dim=0, keepdim=True)
        std = x[:eval_pos].std(dim=0, keepdim=True)
    x = torch.clamp(x, mean - n_sigma * std, mean + n_sigma * std)
    return x


def pad_x(x, max_features):
    if x.shape[-1] < max_features:
        pad = torch.zeros(*x.shape[:-1], max_features - x.shape[-1], device=x.device)
        x = torch.cat([x, pad], dim=-1)
    return x


class Log1pScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.sign(X) * np.log1p(np.abs(X))

    def inverse_transform(self, X):
        return np.sign(X) * (np.exp(np.abs(X)) - 1)


class FAISS:
    def __init__(self, X, metric: Literal["l2", "ip"] = "l2"):
        self.metric = metric
        self.index = faiss.IndexFlatL2(X.shape[1]) if metric == "l2" else faiss.IndexFlatIP(X.shape[1])
        self.index.add(X.astype(np.float32))

    def get_knn_indices(self, X, k):
        if self.metric == "l2":
            _, indices = self.index.search(X.astype(np.float32), k)
        else:
            _, indices = self.index.search(X.astype(np.float32), k)
        return indices