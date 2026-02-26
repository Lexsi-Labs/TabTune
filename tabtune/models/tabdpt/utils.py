from functools import wraps
from typing import Literal

import faiss
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
def convert_to_torch_tensor(input):
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif torch.is_tensor(input):
        return input
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")


def generate_random_permutation(N, seed=None):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    return torch.randperm(N, generator=generator)


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


def maskmean(x, mask, dim):
    x = torch.where(mask, x, 0)
    return x.sum(dim=dim, keepdim=True) / mask.sum(dim=dim, keepdim=True)


def maskstd(x, mask, dim=0):
    num = mask.sum(dim=dim, keepdim=True)
    mean = maskmean(x, mask, dim=0)
    diffs = torch.where(mask, mean - x, 0)
    return ((diffs**2).sum(dim=0, keepdim=True) / (num - 1)) ** 0.5


def normalize_data(data, eval_pos=-1, dim=0, return_mean_std: bool = False):
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=dim)
    std = maskstd(X, mask, dim=dim) + 1e-6
    data = (data - mean) / std
    if return_mean_std:
        return data, mean, std
    return data


def clip_outliers(data, eval_pos=-1, n_sigma=4, dim=0):
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=dim)
    cutoff = n_sigma * maskstd(X, mask, dim=dim)
    mask &= cutoff >= torch.abs(X - mean)
    cutoff = n_sigma * maskstd(X, mask, dim=dim)
    return torch.clip(data, mean - cutoff, mean + cutoff)


def pad_x(X: torch.Tensor, num_features: int = 100):
    if num_features is None:
        return X
    n_features = X.shape[-1]
    zero_feature_padding = torch.zeros((*X.shape[:-1], num_features - n_features), device=X.device)
    return torch.cat([X, zero_feature_padding], dim=-1)


class Log1pScaler:
    def fit(self, X: np.ndarray):
        pass

    def fit_transform(self, X: np.ndarray):
        return np.log1p(np.abs(X)) * np.sign(X)

    def transform(self, X: np.ndarray):
        return np.log1p(np.abs(X)) * np.sign(X)


class FAISS:
    def __init__(self, X, metric: Literal["l2", "ip"] = "l2"):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        X = np.ascontiguousarray(X)
        X = X.astype(np.float32)
        if metric == "l2":
            self.index = faiss.IndexFlatL2(X.shape[1])
        elif metric == "ip":
            self.index = faiss.IndexFlatIP(X.shape[1])
        else:
            raise ValueError('metric must be "l2" or "ip"')
        self.index.add(X)

    def get_knn_indices(self, queries, k):
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        queries = np.ascontiguousarray(queries)
        assert isinstance(k, int)

        knns = self.index.search(queries, k)
        indices_Xs = knns[1]
        return indices_Xs