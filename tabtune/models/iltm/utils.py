import os
import math
import time
import random
import torch
import copy
import contextlib
from collections import deque
from typing import Tuple, Callable, Sequence
import logging

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Configure logging
logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class Linear:
    # Intentionally not inheriting from nn.Module as it decreases accuracy
    # (this possibly happened because the layer would be initialized after passing
    # the model.parameters() iterator to the optimizer, thus not being optimized properly;
    # that is, as if m and b were detached from the computation graph using .detach() here)
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def __call__(self, x):
        return torch.mm(x, self.m) + self.b

    def to(self, device):
        self.m = self.m.to(device)
        self.b = self.b.to(device)
        return self


def forward_linear_layer(x, w, hs):
    w = w.view(-1, hs)
    m = w[:-1, :]
    b = w[-1, :]
    layer = Linear(m, b)
    x = layer(x)
    return x, layer


def svd_flip(u, v, u_based_decision=True):
    """
    Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    """
    if u_based_decision:
        max_abs_cols = torch.argmax(torch.abs(u), dim=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs.unsqueeze(1)
    else:
        max_abs_rows = torch.argmax(torch.abs(v), dim=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs.unsqueeze(1)
    return u, v


class TorchPCA:
    def __init__(self, n_components=None, fit="reduced", svd_driver=None):
        self.n_components = n_components
        self.fit = fit
        if fit == "lowrank" and svd_driver is not None:
            logger.warning("svd_driver is not used in 'lowrank' fit mode.")
        self.svd_driver = svd_driver

    def _fit(self, X):
        n_samples, n_features = X.shape
        self.n_components_ = min(self.n_components or min(X.shape), min(X.shape))

        if self.n_components_ > min(X.shape):
            raise ValueError(f"n_components should be <= min(n_samples: {n_samples}, n_features: {n_features})")

        self.mean_ = torch.mean(X, dim=0)

        if self.fit == "full":
            U, S, Vt = torch.linalg.svd(X - self.mean_, full_matrices=True, driver=self.svd_driver)
        elif self.fit == "reduced":
            U, S, Vt = torch.linalg.svd(X - self.mean_, full_matrices=False, driver=self.svd_driver)
        elif self.fit == "lowrank":
            U, S, V = torch.svd_lowrank(X, q=self.n_components_, M=self.mean_)
            Vt = V.mT  # Transpose V to get Vt
        else:
            raise ValueError(f"Unrecognized fit method: {self.fit}")

        U = U[:, :self.n_components_]
        S = S[:self.n_components_]
        Vt = Vt[:self.n_components_]

        U, Vt = svd_flip(U, Vt)
        self.components_ = Vt

        return U, S, Vt

    def fit(self, X):
        self._fit(X)
        return self

    def transform(self, X):
        return torch.matmul(X - self.mean_, self.components_.T)

    def fit_transform(self, X):
        U, S, _ = self._fit(X)
        return U * S.unsqueeze(0)

    def to(self, device):
        self.mean_ = self.mean_.to(device)
        self.components_ = self.components_.to(device)
        return self


def sample_data(X: Tensor | dict, y: Tensor,
                to_meta_model: bool = True,
                stratify_sampling: bool = False,
                batch_size: int = 2048,
                min_samples: int = 784,
                pca_sampling: str = 'repeat',
                return_rest: bool = False) -> Tuple[Tensor, Tensor]:
    # For regression, stratification is not applicable; always use random sampling:
    sampled_indices = torch.randperm(len(y))[:batch_size]

    if isinstance(X, dict):
        X_sampled = {
            'x_num': X['x_num'][sampled_indices],
            'x_cat': X['x_cat'][sampled_indices]
        }
    else:
        X_sampled = X[sampled_indices]
    y_sampled = y[sampled_indices]

    if return_rest:
        assert not to_meta_model, "to_meta_model must be False when return_rest is True."
        not_sampled_mask = torch.ones(len(y), dtype=torch.bool)
        not_sampled_mask[sampled_indices] = False
        if isinstance(X, dict):
            return X_sampled, y_sampled, {
                'x_num': X['x_num'][not_sampled_mask].flatten(start_dim=1),
                'x_cat': X['x_cat'][not_sampled_mask].flatten(start_dim=1)
            }, y[not_sampled_mask]
        else:
            return X_sampled, y_sampled, X[not_sampled_mask].flatten(start_dim=1), y[not_sampled_mask]

    if y_sampled.shape[0] < min_samples and to_meta_model:
        logger.debug(f"X_ has less samples than min_samples: {y_sampled.shape[0]} < {min_samples}")
        if pca_sampling == 'repeat':
            logger.debug(f"Repeat datapoints for PCA: at least {min_samples}")
            n_repeats = math.ceil(min_samples / X_sampled.shape[0])
            if isinstance(X_sampled, dict):
                X_sampled = {
                    'x_num': torch.repeat_interleave(X_sampled['x_num'], n_repeats, axis=0),
                    'x_cat': torch.repeat_interleave(X_sampled['x_cat'], n_repeats, axis=0)
                }
            else:
                X_sampled = torch.repeat_interleave(X_sampled, n_repeats, axis=0)
            y_sampled = torch.repeat_interleave(y_sampled, n_repeats, axis=0)
        elif pca_sampling == 'bootstrap':
            logger.debug("Using bootstrap sampling to reach minimum samples for PCA.")
            additional_samples_needed = min_samples - y_sampled.shape[0]
            bootstrap_indices = torch.randint(0, y_sampled.shape[0], (additional_samples_needed,))
            if isinstance(X_sampled, dict):
                X_sampled = {
                    'x_num': torch.cat([X_sampled['x_num'], X_sampled['x_num'][bootstrap_indices]], dim=0),
                    'x_cat': torch.cat([X_sampled['x_cat'], X_sampled['x_cat'][bootstrap_indices]], dim=0)
                }
            else:
                X_sampled = torch.cat([X_sampled, X_sampled[bootstrap_indices]], dim=0)
            y_sampled = torch.cat([y_sampled, y_sampled[bootstrap_indices]], dim=0)
        elif pca_sampling == 'zeropad':
            # Do nothing now, PCA with less components will be performed, and the rest will be zero-padded.
            pass
        else:
            raise ValueError("Invalid PCA sampling method.")

    return X_sampled, y_sampled


def transform_data_for_main_network(X, cfg, device, rf, pca, norm,
                                    training_finetuning: bool = False):
    with torch.set_grad_enabled(training_finetuning):
        if X.dtype != torch.float32:
            X = X.to(torch.float32)
        X = X.to(device)
        X = rf(X)
        X = pca.transform(X)
        if cfg['pca_sampling'] == 'zeropad' and X.shape[1] < cfg['n_dims']:
            X = F.pad(X, (0, cfg['n_dims'] - X.shape[1]), value=0)
        X = torch.clamp(X, -cfg['clip_data_value'], cfg['clip_data_value'])
        if norm is not None:
            X = norm(X)
    return X


def forward_main_network(
    x: Tensor,
    main_network: nn.ModuleList,
    training_finetuning: bool = False,
    finetuning_dropout: float = 0.0
) -> Tuple[Tensor, Tensor]:
    """
    Forwards through the main network.

    Args:
        x: input tensor
        main_network: list of Linear layers
        cfg: model config dict
        training_finetuning: whether we're in the finetuning stage, so that dropout is activated
        finetuning_dropout:  dropout probability in finetuning stage

    Returns:
        outputs, intermediate_state
    """
    for n, layer in enumerate(main_network):
        if n % 2 == 0:
            residual_connection = x

        x = layer(x)

        # add residual connection
        if n % 2 == 1 and n != len(main_network) - 1:
            x = x + residual_connection

        # capture the penultimate hidden
        if n == len(main_network) - 2:
            intermediate_state = x

        # activation + dropout on all but the last layer
        if n != len(main_network) - 1:
            x = F.relu(x)
            x = F.dropout(x, p=finetuning_dropout, training=training_finetuning)
    return x, intermediate_state


def forward_main_network_with_preprocessing(X, model_cfg, rf, pca, norm, main_network, device, use_amp=False,
                                            training_finetuning: bool = False, finetuning_dropout: float = 0.0
                                            ) -> Tuple[Tensor, Tensor]:

    X_transformed = transform_data_for_main_network(X=X, cfg=model_cfg, rf=rf, pca=pca, norm=norm, device=device,
                                                    training_finetuning=training_finetuning)

    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        outputs, intermediate_state = forward_main_network(
            X_transformed,
            main_network,
            training_finetuning=training_finetuning,
            finetuning_dropout=finetuning_dropout,
        )
    return outputs, intermediate_state


def retrieval(X_ctxt_superset, y_ctxt_superset, intermediate_state, n_classes, batch_size, distance_type, temperature,
              model_cfg, rf, pca, norm, main_network, device, use_amp=False,
              training_finetuning: bool = False, finetuning_dropout: float = 0.0) -> Tensor:
    # Based on ModernNCA code: https://github.com/qile2000/LAMDA-TALENT/blob/main/LAMDA_TALENT/model/methods/modernNCA.py
    X_ctxt, y_ctxt = sample_data(X_ctxt_superset, y_ctxt_superset, to_meta_model=False, batch_size=batch_size)  # stochastic neighbor sampling
    if isinstance(X_ctxt, dict):
        X_ctxt = {
            'x_num': X_ctxt['x_num'].to(device),
            'x_cat': X_ctxt['x_cat'].to(device)
        }
    else:
        X_ctxt = X_ctxt.to(device)
    y_ctxt = y_ctxt.to(device)
    if n_classes > 1:  # n_classes == 1 for regression
        y_ctxt = F.one_hot(y_ctxt, num_classes=n_classes).float()
    else:  # n_classes == 1 for regression, but y_ctxt is a 1D tensor
        y_ctxt = y_ctxt.unsqueeze(-1)
    _, X_ctxt = forward_main_network_with_preprocessing(X_ctxt, model_cfg, rf, pca, norm, main_network, device, use_amp,
                                                        training_finetuning=training_finetuning, finetuning_dropout=finetuning_dropout)

    if distance_type == 'euclidean':
        distances = torch.cdist(intermediate_state, X_ctxt, p=2)
    elif distance_type == 'cosine':  # We use this one
        distances = -torch.mm(
            F.normalize(intermediate_state, p=2, dim=-1),
            F.normalize(X_ctxt, p=2, dim=-1).T
        )
    else:
        raise ValueError(f"Invalid distance type: {distance_type}")

    distances = distances / temperature
    distances = F.softmax(-distances, dim=-1)
    outputs = torch.mm(distances, y_ctxt)
    if n_classes > 1:
        outputs = torch.log(outputs.clamp_min(1e-12))
    return outputs


def full_main_forward(X_grad, n_classes, batch_size, model_cfg,
                      rf, pca, norm, main_network, device, use_amp,
                      do_retrieval: bool = False, X_ctxt_superset: Tensor | None = None, y_ctxt_superset: Tensor | None = None,
                      retrieval_alpha: float = 0.5, retrieval_temperature: float = 1.0, retrieval_distance: str = 'cosine',
                      training_finetuning: bool = False, finetuning_dropout: float = 0.0) -> Tensor:

    if do_retrieval:
        use_amp = False

    outputs, intermediate_state = forward_main_network_with_preprocessing(
        X_grad, model_cfg, rf, pca, norm, main_network, device, use_amp=use_amp,
        training_finetuning=training_finetuning, finetuning_dropout=finetuning_dropout,
    )

    if do_retrieval:
        retrieval_outputs = retrieval(
            X_ctxt_superset, y_ctxt_superset, intermediate_state, n_classes,
            batch_size, retrieval_distance, retrieval_temperature,
            model_cfg, rf, pca, norm, main_network, device, use_amp=use_amp,
            training_finetuning=training_finetuning, finetuning_dropout=finetuning_dropout,
        )
        outputs = (1 - retrieval_alpha) * outputs + retrieval_alpha * retrieval_outputs

    if n_classes == 1:
        outputs = outputs.squeeze(1)

    return outputs



class SimpleBatchNorm1dFixed(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.fitted = False

        # Parameters for scale and shift
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Parameters to store running mean and variance
        self.running_mean = None
        self.running_var = None

    def fit(self, x):
        """ Fit the normalization layer with training data: calculate and store running mean and variance """
        self.running_mean = x.mean(dim=0)
        self.running_var = x.var(dim=0, unbiased=False)
        self.fitted = True

    def forward(self, x):
        if not self.fitted:
            raise RuntimeError("The layer has not been fitted with training data.")

        # Normalize using stored mean and variance
        x_normalized = (x - self.running_mean) / (torch.sqrt(self.running_var + self.eps))

        # Scale and shift
        out = self.gamma * x_normalized + self.beta

        return out


def reconstruct_main_network(main_network):
    main_network_reconstructed = []
    for layer in main_network:
        weight_matrix = layer.weight.data.T
        bias_vector = layer.bias.data
        main_network_reconstructed.append(Linear(weight_matrix, bias_vector))
    return main_network_reconstructed


class PCAModule(nn.Module):
    def __init__(self, pca):
        super().__init__()
        self.pca_mean = (nn.Parameter(pca.mean_))
        self.input_features, self.output_features = pca.components_.shape
        self.pca_components = nn.Linear(
            self.input_features, self.output_features, bias=False
        )
        self.pca_components.weight = (nn.Parameter(pca.components_))

    def transform(self, X):
        with torch.amp.autocast(device_type=X.device.type, enabled=False):
            X = X.to(torch.float32)
            X = X - self.pca_mean
            X = self.pca_components(X)
        return X


class MainNetworkTrainable(nn.Module):
    def __init__(self, cfg, n_classes, batch_size, rf, pca, main_network, norm, finetuning_dropout: float = 0.0, device: str = 'cuda',
                 do_retrieval: bool = False, retrieval_alpha: float = 0.5, retrieval_temperature: float = 1.0, retrieval_distance_type: str = 'cosine',
                 retrieval_alpha_finetuning: bool = False, retrieval_temperature_finetuning: bool = False,
                 initial_transformations_finetuning: bool = False,
                 use_amp_finetuning: bool = False):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.clip_data_value = cfg['clip_data_value']
        self.device = device
        self.finetuning_dropout = finetuning_dropout
        self.do_retrieval = do_retrieval
        if retrieval_alpha == 0.0:
            self.do_retrieval = False
        if not self.do_retrieval:
            retrieval_alpha_finetuning = False
            retrieval_temperature_finetuning = False

        # Allow exact 0/1 for alpha; temperature must be > 0
        if not (0.0 <= retrieval_alpha <= 1.0):
            raise ValueError(f"retrieval_alpha must be in [0, 1], got {retrieval_alpha}")
        # logit with eps handles alpha==0 or 1 without infs
        alpha_init_logit = torch.logit(torch.tensor(retrieval_alpha, dtype=torch.float32), eps=1e-6)
        if retrieval_alpha_finetuning:
            self._alpha_raw = nn.Parameter(alpha_init_logit)
        else:
            self.register_buffer("_alpha_raw", alpha_init_logit)

        # Avoid -inf if retrieval_temperature == 0 by clamping before log
        T0 = torch.as_tensor(retrieval_temperature, dtype=torch.float32).clamp_min(1e-4)
        if retrieval_temperature_finetuning:
            self._log_T = nn.Parameter(T0.log())
        else:
            self.register_buffer("_log_T", T0.log())

        self.retrieval_distance_type = retrieval_distance_type
        self.retrieval_alpha_finetuning = retrieval_alpha_finetuning
        self.retrieval_temperature_finetuning = retrieval_temperature_finetuning
        self.initial_transformations_finetuning = initial_transformations_finetuning
        self.use_amp_finetuning = use_amp_finetuning
        self.rf = rf

        # Expansion
        if not isinstance(self.rf, nn.Identity):
            self.rf[0].weight.requires_grad = initial_transformations_finetuning

        # Reduction
        self.pca = PCAModule(pca)

        # Initialize a conventional BatchNorm1d layer with the parameters from self.norm
        bn = nn.BatchNorm1d(self.cfg['n_dims']).to(self.device)
        bn.running_mean = norm.running_mean
        bn.running_var = norm.running_var
        bn.weight = norm.gamma
        bn.bias = norm.beta
        self.norm = bn

        # Main network
        self.main_network = nn.ModuleList()
        for linear in main_network:  # linear is Linear object above (not nn.Linear)
            matrix, bias = linear.m, linear.b
            linear_layer = nn.Linear(matrix.shape[0], matrix.shape[1])
            linear_layer.weight = nn.Parameter(matrix.T)
            linear_layer.bias = nn.Parameter(bias)
            self.main_network.append(linear_layer)

    @property
    def retrieval_alpha(self):
        return self._alpha_raw.sigmoid()

    @property
    def retrieval_temperature(self):
        return self._log_T.exp().clamp_min(1e-4)

    def forward(self, X: Tensor, X_ctxt_superset: Tensor | None = None, y_ctxt_superset: Tensor | None = None, training: bool = False):
        return full_main_forward(X, self.n_classes, self.batch_size, self.cfg,
                                 self.rf, self.pca, self.norm, self.main_network, self.device, self.use_amp_finetuning,
                                 self.do_retrieval, X_ctxt_superset, y_ctxt_superset,
                                 self.retrieval_alpha, self.retrieval_temperature, self.retrieval_distance_type,
                                 training_finetuning=training, finetuning_dropout=self.finetuning_dropout)

    def get_main_network_parts(self):

        rf_reconstructed = self.rf
        # Handle both nn.Sequential and nn.Identity for rf_reconstructed

        if isinstance(rf_reconstructed, nn.Sequential):
            rf_reconstructed[0].weight.requires_grad = False
            rf_reconstructed[0].weight.grad = None
        elif not isinstance(rf_reconstructed, nn.Identity):
            raise TypeError(f"Unexpected type for rf_reconstructed: {type(rf_reconstructed)}")

        pca_reconstructed = self.pca
        pca_reconstructed.pca_mean.requires_grad = False
        pca_reconstructed.pca_mean.grad = None
        pca_reconstructed.pca_components.weight.requires_grad = False
        pca_reconstructed.pca_components.weight.grad = None

        main_network_reconstructed = reconstruct_main_network(self.main_network)

        return {
            "rf": rf_reconstructed,
            "pca": pca_reconstructed,
            "main_network": main_network_reconstructed,
            "norm": self.norm,
            "retrieval_alpha": self.retrieval_alpha if not self.retrieval_alpha_finetuning else self.retrieval_alpha.item(),
            "retrieval_temperature": self.retrieval_temperature if not self.retrieval_temperature_finetuning else self.retrieval_temperature.item(),
        }


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    p.data.add_(p.data, alpha=-wd*lr)
                state = self.state[p]
                if 'exp_avg' not in state:
                    state['exp_avg'] = torch.zeros_like(p)
                m = state['exp_avg']
                m.mul_(b1).add_(g, alpha=1-b1)
                p.add_(torch.sign(m), alpha=-lr)
                m.mul_(b2).add_(g, alpha=1-b2)


def fine_tune_main_network(
    cfg, X, y, n_classes,
    rf, pca, main_network, norm,
    device, max_epochs, batch_size,
    finetuning_optimizer: str = 'AdamW',  # options: 'adamw', 'lion', 'sgd'
    finetuning_data='entire_dataset', finetuning_lr=1e-4, finetuning_dropout=0.0, X_val=None, y_val=None,
    finetuning_val_frac: float = 0.1,
    val_check_interval_batches: int | None = None,
    do_retrieval: bool = False, retrieval_alpha: float = 0.5, retrieval_temperature: float = 1, retrieval_distance: str = 'cosine',
    retrieval_alpha_finetuning: bool = False, retrieval_temperature_finetuning: bool = False,
    initial_transformations_finetuning: bool = True,
    gradient_clip_norm: float = 1.0,
    scheduler_min_lr: float = 1e-6,
    use_amp_finetuning: bool = False,
    early_stopping_mode: str = "auto",
    patience_epochs: int = 50,
    patience_checks: int | None = None,
    val_checks_per_epoch_target: int = 4,
    max_train_batches_per_epoch: int | None = None,
    finetuning_subset_frac: float | None = None,
    finetuning_subset_max_samples: int | None = 100_000,
    val_max_samples: int | None = 25_000,
    min_epochs: int = 0,
    cooldown_checks: int = 0,
    classification_val_metric: str = 'auto',  # options: 'auto', 'logloss', 'auc' ('auto': 'auc' binary, 'logloss' multiclass)
    *,
    fit_deadline: float | None = None,
    fit_time_cushion_frac: float = 0.0,
):
    """
    Fine-tune the main network of the model with the given data.

    Attributes:
        cfg: Model configuration dictionary.
        X: Input data tensor.
        y: Target labels tensor.
        n_classes: Number of classes.
        rf: Random features model for expansion.
        pca: PCA model for reduction.
        main_network: Main network layers.
        norm: Normalization layer.
        device: Device to use for training.
        max_epochs: Maximum number of epochs for fine-tuning.
        batch_size: Batch size for training.
        Early stopping is controlled by:
          - early_stopping_mode: "epoch", "batch", or "auto"
          - patience_epochs: maximum number of validation checks without improvement in epoch mode
          - patience_checks: maximum number of validation checks without improvement in batch mode
          - val_checks_per_epoch_target: used in "auto" mode to derive validation cadence
        finetuning_data: Data to use for fine-tuning. Can be 'entire_dataset' or 'bootstrap'.
        finetuning_lr: Learning rate for fine-tuning.
        finetuning_dropout: Dropout rate for fine-tuning.
        X_val: Validation input data tensor.
        y_val: Validation target labels tensor.
        finetuning_val_frac: Fraction of data to use for validation split when X_val and y_val are not provided.
        do_retrieval: Whether to use retrieval for fine-tuning.
        retrieval_alpha: Alpha parameter for retrieval.
        retrieval_temperature: Temperature parameter for retrieval.
        retrieval_distance: Distance metric for retrieval.
        retrieval_alpha_finetuning: Whether to make retrieval alpha trainable.
        retrieval_temperature_finetuning: Whether to make retrieval temperature trainable.
        initial_transformations_finetuning: Whether to make the initial transformations trainable.
        gradient_clip_norm: Gradient clip norm for finetuning.
        scheduler_min_lr: Minimum learning rate for scheduler.
        use_amp_finetuning: Whether to use AMP for finetuning.
        fit_deadline: Optional absolute time.time() deadline to stop training early if needed, based on average step time.
        fit_time_cushion_frac: Fraction of average step time to use as a cushion when estimating if the next step can complete before the deadline.
    """
    t0 = time.time()
    timeout_triggered = False

    step_times = deque(maxlen=50)
    avg_step_time = None

    def validate_model():
        main_model.eval()
        val_loss = 0
        val_targets = []
        val_outputs = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = main_model(inputs, X_ctxt_superset, y_ctxt_superset, training=False)

                # Verify shapes before computing loss
                if outputs.shape != targets.shape:
                    # For regression tasks where targets are floats, not integers (classification)
                    if n_classes == 1:
                        raise ValueError(f"Shape mismatch between outputs and targets: outputs {outputs.shape}, targets {targets.shape}")
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_targets.append(targets)
                val_outputs.append(outputs)

        val_targets = torch.cat(val_targets)
        val_outputs = torch.cat(val_outputs)

        # Get numpy arrays for metric calculation
        y_val = val_targets.cpu().numpy()
        if n_classes == 1:
            # For regression tasks
            y_pred = val_outputs.cpu().numpy()
            if y_val.shape != y_pred.shape:
                raise ValueError(f"Shape mismatch between y_val and y_pred: y_val {y_val.shape}, y_pred {y_pred.shape}")
            mse = ((y_val - y_pred) ** 2).mean()
            return val_loss, mse

        # For classification tasks
        if val_metric_name == 'AUC':
            y_proba = torch.softmax(val_outputs, dim=1).cpu().numpy()
            auc = robust_roc_auc_score(y_val, y_proba)
            return val_loss, auc

        # val_metric_name == 'LogLoss'
        return val_loss, val_loss

    if isinstance(X, dict):
        X_num = X['x_num']
        X_cat = X['x_cat']
        X = torch.cat([X_num, X_cat], dim=1)

        # assume X_val is also a dict if X_val is not None
        if X_val is not None:
            assert isinstance(X_val, dict)
            X_num_val = X_val['x_num']
            X_cat_val = X_val['x_cat']
            X_val = torch.cat([X_num_val, X_cat_val], dim=1)

    if X_val is None and y_val is None:
        # Split data into training and validation sets (different for each predictor) for early stopping
        if finetuning_val_frac > 0.0:
            # check if stratification is possible
            y_np = y.cpu().numpy()
            stratify_by = check_stratification(y_np, stratify=True, task_type='classification') if n_classes > 1 else None
            X_train, X_val, y_train, y_val = train_test_split(X.cpu().numpy(), y_np, test_size=finetuning_val_frac, random_state=None, stratify=stratify_by)
        else:
            X_train, y_train = X.cpu().numpy(), y.cpu().numpy()
    else:
        X_train, y_train = X.cpu().numpy(), y.cpu().numpy()
        X_val, y_val = X_val.cpu().numpy(), y_val.cpu().numpy()

    # If finetuning_data is 'entire_dataset', use all training data for fine-tuning.
    # If 'bootstrap', use a subset of same size as the original dataset, but sampling with replacement.
    if finetuning_data == 'bootstrap':
        X_train, y_train = resample(X_train, y_train, replace=True, n_samples=len(X_train), random_state=None)
        logger.debug("Using bootstrap sampling for fine-tuning.")

    X_train, y_train = torch.from_numpy(X_train), torch.from_numpy(y_train)
    if X_val is not None and y_val is not None:
        X_val, y_val = torch.from_numpy(X_val), torch.from_numpy(y_val)

    t0 = time.time()

    # Optionally cap validation set size
    if X_val is not None and y_val is not None and val_max_samples is not None and len(X_val) > val_max_samples:
        idx = torch.randperm(len(X_val))[:val_max_samples]
        X_val = X_val[idx]
        y_val = y_val[idx]

    # Subset training per epoch for huge datasets
    def make_train_indices_epoch(len_train: int) -> torch.Tensor:
        # choose how many samples this epoch
        if finetuning_subset_frac is not None:
            n_target = int(len_train * float(finetuning_subset_frac))
        else:
            n_target = len_train
        if finetuning_subset_max_samples is not None:
            n_target = min(n_target, int(finetuning_subset_max_samples))
        n_target = max(1, n_target)
        return torch.randperm(len_train)[:n_target]

    # Utility to create loaders for this epoch
    def build_epoch_loaders(X_train_full, y_train_full):
        idx = make_train_indices_epoch(len(X_train_full))
        X_train_e = X_train_full[idx]
        y_train_e = y_train_full[idx]
        train_dataset = TensorDataset(X_train_e, y_train_e)
        # Only use pin_memory for CUDA devices to avoid deprecation warnings on CPU
        use_pin_memory = device.type == 'cuda'
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_pin_memory)
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=use_pin_memory)
        else:
            val_loader = None
        return train_loader, val_loader, len(idx)

    # Prepare first epoch loaders with subset
    train_loader, val_loader, n_train_this_epoch = build_epoch_loaders(X_train, y_train)
    num_batches_full_epoch = math.ceil(n_train_this_epoch / batch_size)

    # Auto choose early stopping mode and validation cadence
    def compute_val_interval_and_mode():
        if early_stopping_mode == "epoch":
            return None, "epoch"
        if early_stopping_mode == "batch":
            # respect explicit interval if provided, else pick sane default
            if val_check_interval_batches and val_check_interval_batches > 0:
                return int(val_check_interval_batches), "batch"
            # default for batch mode: validate every 250 batches
            return max(50, min(250, num_batches_full_epoch // 10 or 1)), "batch"
        # auto
        small = num_batches_full_epoch <= max(4, val_checks_per_epoch_target)
        if small:
            return None, "epoch"
        # large: aim for a target number of checks per epoch
        interval = max(1, num_batches_full_epoch // max(1, val_checks_per_epoch_target))
        return interval, "batch"

    val_interval_batches, mode = compute_val_interval_and_mode()

    def _fmt_metric(v):
        try:
            return f"{float(v):.6f}"
        except Exception:
            return str(v)

    def _status_line(es, improved: bool, patience_in_checks: int | None, val_metric_name: str):
        if improved:
            return f"(new best {val_metric_name})"
        # cooldown shows progress if enabled
        cd = es.get("cooldown_left", 0)
        base = f"(no-improve {es['checks_no_improve']}"
        if patience_in_checks is not None:
            base += f"/{patience_in_checks}"
        base += " checks)"
        if cd and cd > 0:
            base += f" [cooldown {cd}]"
        return base

    # Translate early stopping tolerance into checks when in batch mode
    if mode == "batch":
        checks_per_epoch = max(1, num_batches_full_epoch // val_interval_batches)
        if patience_checks is not None:
            patience_in_checks = int(patience_checks)
        else:
            # default: patience_epochs worth of checks
            patience_in_checks = max(1, int(patience_epochs * checks_per_epoch))
    else:
        patience_in_checks = None  # not used

    if do_retrieval:
        max_ctxt = 65536
        if len(X_train) > max_ctxt:
            idx_ctxt = torch.randperm(len(X_train))[:max_ctxt]
            X_ctxt_superset = X_train[idx_ctxt].to(device)
            y_ctxt_superset = y_train[idx_ctxt].to(device)
        else:
            X_ctxt_superset = X_train.to(device)
            y_ctxt_superset = y_train.to(device)
    else:
        X_ctxt_superset = None
        y_ctxt_superset = None

    # Model setup
    main_model = MainNetworkTrainable(cfg, n_classes, batch_size, rf, pca, main_network, norm, finetuning_dropout=finetuning_dropout, device=device,
                                      do_retrieval=do_retrieval, retrieval_alpha=retrieval_alpha, retrieval_temperature=retrieval_temperature, retrieval_distance_type=retrieval_distance,
                                      retrieval_alpha_finetuning=retrieval_alpha_finetuning, retrieval_temperature_finetuning=retrieval_temperature_finetuning,
                                      initial_transformations_finetuning=initial_transformations_finetuning,
                                      use_amp_finetuning=use_amp_finetuning)

    main_model = main_model.to(device)

    # Freeze rf and pca layers
    if not main_model.initial_transformations_finetuning:
        for param in main_model.rf.parameters():
            param.requires_grad = False
        for param in main_model.pca.parameters():
            param.requires_grad = False

    if n_classes == 1:
        criterion = nn.MSELoss()
        val_metric_name = "MSE"
        lower_is_better = True
    else:
        criterion = nn.CrossEntropyLoss()
        if classification_val_metric == 'auto':
            if n_classes == 2:
                classification_val_metric = 'auc'
            else:
                classification_val_metric = 'logloss'
        if classification_val_metric.lower() == 'auc':
            val_metric_name = "AUC"
            lower_is_better = False
        else:
            val_metric_name = "LogLoss"
            lower_is_better = True

    bn_params, other_params, retrieval_params = [], [], []

    for name, p in main_model.named_parameters():
        if not p.requires_grad:
            continue

        # keep the retrieval params out of the other groups
        if name in ("_alpha_raw", "_log_T"):
            retrieval_params.append(p)
            continue

        # put BN affine params in their own group (no weight decay)
        if name.startswith("norm.") or ".norm." in name:
            bn_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": finetuning_lr, "weight_decay": 0.01})
    if bn_params:
        param_groups.append({"params": bn_params,  "lr": finetuning_lr, "weight_decay": 0.0})
    if retrieval_params:  # only present if *_finetuning flags made them nn.Parameters
        param_groups.append({"params": retrieval_params, "lr": finetuning_lr * 0.1, "weight_decay": 0.0})
    finetuning_optimizer = finetuning_optimizer.lower()
    if finetuning_optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups)
    elif finetuning_optimizer == 'lion':
        optimizer = Lion(param_groups)
    elif finetuning_optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_groups)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=scheduler_min_lr)

    # init
    best_model_wts = copy.deepcopy(main_model.state_dict())

    es = {
        "best_val_metric": float("inf") if lower_is_better else -float("inf"),
        "checks_no_improve": 0,
        "checks_since_improve": 0,
        "checks_done": 0,
        "cooldown_left": 0,
        "best_model_wts": best_model_wts,
    }

    scaler = torch.amp.GradScaler(enabled=use_amp_finetuning)
    log_check_freq = 10

    def is_better(curr, best):
        if isinstance(curr, float) and (math.isnan(curr) or math.isinf(curr)):
            return False
        return (curr < best) if lower_is_better else (curr > best)

    def validate_and_update(es, where: str):
        """Run validation, update early-stopping state, and log a compact line."""
        val_loss, val_metric = validate_model()
        es["checks_done"] += 1

        improved = is_better(val_metric, es["best_val_metric"])
        if improved:
            es["best_val_metric"] = val_metric
            es["best_model_wts"] = copy.deepcopy(main_model.state_dict())
            es["checks_no_improve"] = 0
            es["checks_since_improve"] = 0
            es["cooldown_left"] = cooldown_checks
        else:
            if es["cooldown_left"] > 0:
                es["cooldown_left"] -= 1
            else:
                es["checks_no_improve"] += 1
                es["checks_since_improve"] += 1

        # LOG the check result every log_check_freq
        if es["checks_done"] % log_check_freq == 0 or es["checks_done"] == 1 or improved:
            elapsed_s = time.time() - t0
            status = _status_line(es, improved, patience_in_checks if mode == "batch" else None, val_metric_name)
            logger.debug(
                "[val @ %-12s] check=%-5d loss=%s | %s=%s | best=%s | %s | t=%.1fs",
                where,
                es["checks_done"],
                _fmt_metric(val_loss),
                val_metric_name,
                _fmt_metric(val_metric),
                _fmt_metric(es["best_val_metric"]),
                status,
                elapsed_s,
            )
        return improved
    # Initial validation before training
    if val_loader is not None:
        _ = validate_and_update(es, where="init")

    stop_training = False
    global_batch_idx = 0

    def _enough_time_for_val():
        if fit_deadline is None or avg_step_time is None or val_loader is None:
            return True
        # crude estimate: assume a val pass ~ len(val_loader) * avg_step_time
        est_val = max(1, len(val_loader)) * avg_step_time
        remaining = fit_deadline - time.time()
        needed = est_val * (1.0 + fit_time_cushion_frac)  # no add here, as add is for whole fit
        return remaining > max(1e-3, needed)

    for epoch in range(max_epochs):
        main_model.train()
        # Rebuild epoch loaders with new subset
        train_loader, val_loader, n_train_this_epoch = build_epoch_loaders(X_train, y_train)
        num_batches_epoch = math.ceil(n_train_this_epoch / batch_size)

        # If "auto", recompute interval & checks to reflect this epoch's subset size
        if early_stopping_mode == "auto":
            if num_batches_epoch > max(4, val_checks_per_epoch_target):
                val_interval_batches = max(1, num_batches_epoch // max(1, val_checks_per_epoch_target))
                # also refresh patience_in_checks to match new checks/epoch
                checks_per_epoch = max(1, num_batches_epoch // val_interval_batches)
                if patience_checks is None:
                    patience_in_checks = max(1, int(patience_epochs * checks_per_epoch))

        batches_run_this_epoch = 0
        for it, (inputs, targets) in enumerate(train_loader):
            # before starting a new step, check deadline using the running average
            if fit_deadline is not None and avg_step_time:
                remaining = fit_deadline - time.time()
                needed = avg_step_time * (1.0 + fit_time_cushion_frac)
                if remaining <= max(1e-3, needed):
                    timeout_triggered = True
                    logger.warning("Early return: finetuning budget nearly exhausted.")
                    break

            step_t0 = time.time()                 # NEW

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device.type, enabled=(use_amp_finetuning and device.type == 'cuda')):
                outputs = main_model(inputs, X_ctxt_superset, y_ctxt_superset, training=True)
                if outputs.shape != targets.shape and n_classes == 1:
                    raise ValueError(f"Shape mismatch between outputs and targets: outputs {outputs.shape}, targets {targets.shape}")
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(main_model.parameters(), gradient_clip_norm, foreach=False)
            scaler.step(optimizer)
            scaler.update()

            global_batch_idx += 1
            batches_run_this_epoch += 1

            # Per batch validation for batch mode
            if val_loader is not None and mode == "batch" and val_interval_batches is not None and (global_batch_idx % val_interval_batches == 0):
                if not _enough_time_for_val():
                    timeout_triggered = True
                    break
                _ = validate_and_update(es, where=f"batch {global_batch_idx}")
                if patience_in_checks is not None and es["checks_no_improve"] >= patience_in_checks and epoch >= min_epochs:
                    logger.debug(f"Early stopping on checks at epoch {epoch+1}, check {es['checks_done']} without improvement")
                    stop_training = True

            # Update running average including validation time (if it happened)
            step_dur = time.time() - step_t0
            step_times.append(step_dur)
            avg_step_time = sum(step_times)/len(step_times)

            if stop_training:
                break

            # Optional cap number of training batches to bound epoch time
            if max_train_batches_per_epoch is not None and batches_run_this_epoch >= max_train_batches_per_epoch:
                break

        if timeout_triggered:
            break

        # epoch-mode validation
        if val_loader is not None and mode == "epoch":
            if not _enough_time_for_val():
                timeout_triggered = True
                break
            _ = validate_and_update(es, where=f"epoch {epoch+1}")
            scheduler.step()
            if epoch + 1 >= min_epochs and es["checks_no_improve"] >= patience_epochs:
                logger.debug(f"Early stopping at epoch {epoch+1} after {es['checks_no_improve']} checks without improvement")
                stop_training = True
                break
        else:
            scheduler.step()

        if stop_training:
            break

    # Load best
    logger.debug(
        "Training done | best %s=%s",
        val_metric_name,
        _fmt_metric(es["best_val_metric"]),
    )
    weights_to_load = es["best_model_wts"]
    main_model.load_state_dict(weights_to_load)
    out = main_model.get_main_network_parts()
    out["timed_out"] = timeout_triggered
    return out


def check_stratification(y, stratify, task_type):
    """
    Check if stratification is possible for the dataset.

    Parameters:
    y : array-like, shape (n_samples,)
        The target variable for supervised learning models.
    stratify : bool
        Indicates whether to stratify the split or not.
    task_type : str
        Specifies the task type ('classification' or 'regression').

    Returns:
    stratify_by : array-like or None
        The target variable if stratification is possible, None otherwise.
    """
    if stratify and task_type == 'classification':
        # Count the number of instances in each class
        class_counts = np.bincount(y)
        # Check if any class has fewer than 2 instances
        if np.any(class_counts < 2):
            logger.debug("Stratification not possible due to insufficient samples in one or more classes.")
            return None
        else:
            return y
    else:
        return None


def standardize_column_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        # Convert all entries to strings if the column contains mixed data types
        if df[column].apply(type).nunique() > 1:  # This checks if more than one type is present in the column
            df[column] = df[column].astype(str)  # Convert all entries to strings
        # bool to category
        if df[column].dtype == 'bool':
            df[column] = df[column].astype('category')

    return df


def robust_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray, multi_class: str = 'ovo') -> float:
    """
    Computes the ROC AUC score for binary or multi-class classification in a robust manner.

    Handles binary classification by calculating the AUC for class 1's predicted probabilities.
    For multi-class classification, it handles cases where not all classes are present in the test set.
    Specifically, it adjusts the predicted probabilities to ensure they sum to 1 after filtering
    out missing classes and avoids NaN values in the predictions.

    If there are unseen classes in the test set (i.e., classes that are not present in the training set),
    this function will catch the error and assign NaN to the ROC AUC score.

    Args:
        y_true (np.ndarray): Ground truth (correct) labels of shape (n_samples,).
        y_score (np.ndarray): Predicted probabilities for each class, of shape (n_samples, n_classes).
        multi_class (str): Specifies the strategy for handling multi-class ROC AUC.
                           - 'ovo' (One-vs-One, default)
                           - 'ovr' (One-vs-Rest)

    Returns:
        float: ROC AUC score for the given predictions and true labels, or NaN if there are unseen classes.

    Raises:
        ValueError: If there are NaN values in the filtered predicted probabilities.
    """

    # Determine if it's binary classification or multi-class
    if len(np.unique(y_true)) == 2:  # Binary classification
        roc_auc = roc_auc_score(y_true, y_score[:, 1])
    else:  # Multi-class classification, using One-vs-One strategy
        classes_in_test = np.unique(y_true)

        # Check if some classes in y_score are not present in y_true
        if len(classes_in_test) != y_score.shape[1]:
            logger.warning(f"Classes in test set: {classes_in_test}, classes in y_score: {y_score.shape[1]}")
            y_proba_filtered = y_score[:, classes_in_test]

            y_proba_sum = y_proba_filtered.sum(axis=1, keepdims=True)
            zero_sum_mask = (y_proba_sum == 0).flatten()
            nonzero_sum_mask = ~zero_sum_mask

            # Normalize samples where sum is not zero
            y_proba_filtered[nonzero_sum_mask] = y_proba_filtered[nonzero_sum_mask] / y_proba_sum[nonzero_sum_mask]

            # Assign uniform probabilities to samples where sum is zero
            y_proba_filtered[zero_sum_mask] = 1.0 / len(classes_in_test)

            # If there are still NaNs, raise an error
            if np.isnan(y_proba_filtered).any():
                logger.debug(f"y_proba_filtered has NaNs: {y_proba_filtered}")
                pass  # raise ValueError("y_proba_filtered STILL has NaNs")

            # Compute multi-class ROC AUC with the adjusted probabilities
            try:
                roc_auc = roc_auc_score(y_true, y_proba_filtered, multi_class=multi_class)
            except ValueError as e:
                logger.debug(f"Error computing ROC AUC: {e}")
                roc_auc = np.nan
        else:
            # If all classes are present in y_true, directly compute the ROC AUC
            roc_auc = roc_auc_score(y_true, y_score, multi_class=multi_class)

    return roc_auc


def preprocess_cat_features_for_catboost(df, cat_features, nan_replacement="missing"):
    """
    Preprocess categorical features for CatBoost by:
    1. Replacing NaNs in categorical columns with a specified replacement category.
    2. Converting float values in categorical columns to strings.
    """
    # First, fix NaN values
    df = catboost_fix_categorical_nan(df, cat_features, nan_replacement=nan_replacement)

    # Then, fix any floating categories
    df = catboost_fix_categorical_float(df, cat_features)

    return df


def catboost_fix_categorical_nan(df, cat_features, nan_replacement="missing"):
    """
    Fix NaN values in categorical features by adding a replacement category and filling NaNs.
    By default, we do not have "nan" appearing as a category in the data, so we replace it with "missing".
    """
    for col_idx in cat_features:
        # Get the column name using the index
        col_name = df.columns[col_idx]

        # Extract the column data
        col_data = df[col_name]

        # Proceed only if there are NaN values in the column
        if col_data.isnull().any():
            # Ensure the column is of Categorical dtype
            if not isinstance(col_data.dtype, pd.CategoricalDtype):
                df[col_name] = col_data.astype("category")
                col_data = df[col_name]

            # Add the replacement category if it's not already present
            if nan_replacement not in col_data.cat.categories:
                df[col_name] = col_data.cat.add_categories(nan_replacement)
                col_data = df[col_name]

            # Replace NaN values with the replacement category
            df[col_name] = col_data.fillna(nan_replacement)
    return df


def catboost_fix_categorical_float(df, cat_features):
    """
    Ensure that any categorical feature values are converted to string type if they are float.
    This avoids errors in CatBoost that occur when categorical features have float values.
    """
    for col_idx in cat_features:
        col_name = df.columns[col_idx]
        col_data = df[col_name]

        # If this is already categorical, we might need to convert categories that are numeric to strings.
        # If it's not categorical, we will convert to a categorical type with string categories.

        # First, ensure column is categorical
        if not pd.api.types.is_categorical_dtype(col_data):
            df[col_name] = col_data.astype("category")
            col_data = df[col_name]

        # Extract categories
        categories = col_data.cat.categories

        # Check if any category is numeric (float or int)
        # We'll convert all categories to strings to be safe.
        if any(isinstance(cat, float) for cat in categories):
            # Convert entire column to string categories
            string_col = col_data.astype(str).astype('category')
            df[col_name] = string_col

    return df

# Functions for computing permutation feature importance
def _ensure_dataframe(X, feature_names=None) -> pd.DataFrame:
    """Return a pandas DataFrame view of X with string column names."""
    if isinstance(X, pd.DataFrame):
        df = X.copy()
        # normalize names to strings
        df.columns = [str(c) for c in df.columns]
        return df
    X = np.asarray(X)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    else:
        feature_names = [str(c) for c in feature_names]
    return pd.DataFrame(X, columns=feature_names)


def _normal_sf(z: float) -> float:
    """Survival function for standard normal, used for one-sided p-values."""
    # sf(z) = 0.5 * erfc(z / sqrt(2))
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _make_scorer(task_type: str, metric: str, y_true: np.ndarray,
                 proba_available: bool) -> tuple[Callable[[np.ndarray, np.ndarray], float], bool, bool]:
    """
    Returns (score_fn, greater_is_better, needs_proba).
    score_fn(y_true, y_pred_or_proba) -> scalar
    """
    metric = (metric or "auto").lower()
    if task_type == "classification":
        # choose default
        if metric == "auto":
            metric = "roc_auc"  # robust multi-class support via robust_roc_auc_score

        if metric in ("roc_auc", "auc"):
            def _score(y_t, y_p):
                return float(robust_roc_auc_score(y_t, y_p))
            return _score, True, True

        if metric in ("log_loss", "neg_log_loss", "cross_entropy"):
            from sklearn.metrics import log_loss

            def _score(y_t, y_p):
                # higher is better, so negate log_loss
                return float(-log_loss(y_t, y_p, labels=np.unique(y_t)))
            return _score, True, True

        if metric in ("accuracy", "acc"):
            from sklearn.metrics import accuracy_score

            def _score(y_t, y_hat):
                return float(accuracy_score(y_t, y_hat))
            return _score, True, False

        raise ValueError(f"Unsupported classification metric: {metric}")

    # regression
    if metric == "auto":
        metric = "r2"
    if metric in ("r2", "r2_score"):
        from sklearn.metrics import r2_score

        def _score(y_t, y_p):
            return float(r2_score(y_t, y_p))
        return _score, True, False
    if metric in ("rmse",):
        from sklearn.metrics import mean_squared_error

        def _score(y_t, y_p):
            return float(-math.sqrt(mean_squared_error(y_t, y_p)))  # higher is better
        return _score, True, False
    if metric in ("mse",):
        from sklearn.metrics import mean_squared_error

        def _score(y_t, y_p):
            return float(-mean_squared_error(y_t, y_p))  # higher is better
        return _score, True, False
    if metric in ("mae",):
        from sklearn.metrics import mean_absolute_error

        def _score(y_t, y_p):
            return float(-mean_absolute_error(y_t, y_p))  # higher is better
        return _score, True, False

    raise ValueError(f"Unsupported regression metric: {metric}")


def compute_permutation_feature_importance(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray | pd.Series,
    *,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    predict_proba_fn: Callable[[pd.DataFrame], np.ndarray] | None = None,
    task_type: str = "regression",
    metric: str = "auto",
    features: Sequence[str] | None = None,
    groups: Sequence[tuple[str, Sequence[str]]] | None = None,
    n_repeats: int = 5,
    subsample_size: int | None = 5000,
    feature_names: Sequence[str] | None = None,
    random_state: int | None = 0,
    silent: bool = False,
) -> pd.DataFrame:
    """
    Permutation importance on ORIGINAL features or user-specified groups.
    - If `features` contains tuples like ('group_name', ['f1','f2']) they are treated as groups.
    - If `groups` is given, it is appended to the features list.
    """
    rng = np.random.default_rng(random_state)
    df = _ensure_dataframe(X, feature_names=feature_names)
    y = np.asarray(y)

    # select rows for speed
    if subsample_size is not None and len(df) > subsample_size:
        idx = rng.choice(len(df), size=subsample_size, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
        y = y[idx]

    # build worklist of items to permute
    work: list[tuple[str, list[str]]] = []
    if features is None:
        features = list(df.columns)
    for f in features:
        if isinstance(f, tuple) and len(f) == 2:
            name, cols = f
            cols = [str(c) for c in cols]
            work.append((str(name), cols))
        else:
            work.append((str(f), [str(f)]))
    if groups:
        for name, cols in groups:
            work.append((str(name), [str(c) for c in cols]))

    # choose score function and proba use
    proba_avail = predict_proba_fn is not None and task_type == "classification"
    score_fn, greater_is_better, needs_proba = _make_scorer(task_type, metric, y, proba_avail)
    use_proba = needs_proba and proba_avail

    # baseline
    with torch.no_grad():
        base_pred = predict_proba_fn(df) if use_proba else predict_fn(df)
    base_score = score_fn(y, base_pred)

    results = []
    for name, cols in work:
        scores = []
        for _ in range(int(n_repeats)):
            Xp = df.copy()
            perm = rng.permutation(len(Xp))
            # permute the set jointly to keep within-row relationships among the grouped columns
            for c in cols:
                Xp[c] = Xp[c].to_numpy()[perm]
            with torch.no_grad():
                pred_p = predict_proba_fn(Xp) if use_proba else predict_fn(Xp)
            s = score_fn(y, pred_p)
            # importance as performance drop
            imp = base_score - s if greater_is_better else s - base_score
            scores.append(float(imp))

        scores = np.asarray(scores, dtype=np.float64)
        mean_imp = float(np.mean(scores))
        std_imp = float(np.std(scores, ddof=1)) if len(scores) > 1 else float("nan")
        n = int(len(scores))

        # one-sided z test using normal approx
        if n > 1 and std_imp > 0 and math.isfinite(std_imp):
            z = mean_imp / (std_imp / math.sqrt(n))
            p_val = float(_normal_sf(max(0.0, z)))  # probability importance <= 0
        else:
            p_val = float("nan")

        results.append((name, mean_imp, std_imp, p_val, n))

        if not silent and len(results) % 10 == 0:
            logger.debug("[perm-imp] finished %d of %d", len(results), len(work))

    out = pd.DataFrame(results, columns=["feature", "importance", "stddev", "p_value", "n"]).set_index("feature")
    out = out.sort_values("importance", ascending=False)
    return out


def _to_torch_device(dev):
    # Accept common aliases
    if dev is None:
        return "cuda"
    if isinstance(dev, str) and dev.lower() == "gpu":
        return "cuda"
    return dev


def get_gpu_memory_info(device: torch.device | str | None = None) -> dict | None:
    """
    Returns a dict with GPU memory stats for the given device, or None if CUDA is not available.
    Keys: index, total_mb, free_mb, allocated_mb, reserved_mb
    """
    dev = torch.device(_to_torch_device(device))
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    index = dev.index if dev.index is not None else torch.cuda.current_device()

    with torch.cuda.device(index):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except Exception:
            total_bytes = torch.cuda.get_device_properties(index).total_memory
            free_bytes = total_bytes - torch.cuda.memory_reserved(index)

        allocated_bytes = torch.cuda.memory_allocated(index)
        reserved_bytes = torch.cuda.memory_reserved(index)

    def to_mb(b): return b / (1024 ** 2)
    return {
        "index": index,
        "total_mb": to_mb(total_bytes),
        "free_mb": to_mb(free_bytes),
        "allocated_mb": to_mb(allocated_bytes),
        "reserved_mb": to_mb(reserved_bytes),
    }


def pick_gpu_ram_part(device: torch.device | str | None = None,
                      cap: float = 0.95,
                      floor: float = 0.30) -> float:
    """
    For CatBoost: returns a safe gpu_ram_part in [floor, cap].
    Approximates 90% of the currently free fraction of total VRAM, capped in [floor, cap].
    """
    info = get_gpu_memory_info(device)
    if info is None:
        return cap
    frac_free_of_total = info["free_mb"] / max(1.0, info["total_mb"])
    return max(floor, min(cap, frac_free_of_total * 0.9))


def is_cuda_oom(err: BaseException) -> bool:
    """
    Heuristic OOM detector across common GPU stacks.
    """
    msg = str(err).lower()
    triggers = [
        "cuda out of memory",
        "cupy.cuda.memory",
        "hip out of memory",
        "cublas status alloc failed",
        "cudnn",
        "std::bad_alloc",
        "out of memory on device",
        "not enough memory",
        "insufficient memory",
        "tree-ctr",
        "cuda"
        "tried to allocate"
    ]
    return any(t in msg for t in triggers)


def clear_cuda_cache() -> None:
    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()



def compute_feature_target_correlations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation coefficient between each feature and the target.
    
    For each column in X, this computes the correlation with y using the
    Pearson correlation formula. Handles edge cases like zero variance features
    and returns cleaned correlations in the range [-1, 1].
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    
    Returns
    -------
    np.ndarray
        Pearson correlation coefficients for each feature, shape (n_features,).
        Values are in [-1, 1] with NaN/inf replaced by 0.0.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    
    X_centered = X - X.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()
    
    numerator = (X_centered * y_centered.reshape(-1, 1)).sum(axis=0)
    
    X_std = np.sqrt((X_centered**2).sum(axis=0))
    y_std = np.sqrt((y_centered**2).sum())
    denominator = X_std * y_std
    
    denominator[denominator == 0] = np.inf
    
    correlations = numerator / denominator
    correlations = np.clip(correlations, -1.0, 1.0)
    correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
    
    return correlations


def select_top_correlated_features(correlations: np.ndarray, num_features: int) -> np.ndarray:
    """
    Select top features using a balanced positive/negative correlation strategy.
    
    This function selects approximately num_features/2 features with the strongest
    positive correlations and num_features/2 with the strongest negative correlations.
    Only features with non-zero correlation are considered. If there are insufficient
    positive or negative correlations, the remaining slots are filled with features
    having the highest absolute correlations (still only from non-zero correlations).
    
    Parameters
    ----------
    correlations : np.ndarray
        Correlation coefficients for each feature, shape (n_features,)
    num_features : int
        Total number of features to select
    
    Returns
    -------
    np.ndarray
        Sorted indices of selected features, shape (num_features,) or fewer if
        there are fewer non-zero correlation features than requested.
    """
    epsilon = 1e-10
    
    num_total_features = correlations.shape[0]
    
    nonzero_mask = np.abs(correlations) > epsilon
    nonzero_indices = np.where(nonzero_mask)[0]
    
    if nonzero_indices.size == 0:
        raise ValueError("No features with non-zero correlation found.")
    
    num_features = min(int(num_features), nonzero_indices.size)
    
    if num_features <= 0:
        return np.array([], dtype=int)
    
    nonzero_correlations = correlations[nonzero_indices]
    
    num_positive = num_features // 2
    num_negative = num_features - num_positive
    
    positive_mask = nonzero_correlations > epsilon
    negative_mask = nonzero_correlations < -epsilon
    
    positive_indices_in_subset = np.where(positive_mask)[0]
    negative_indices_in_subset = np.where(negative_mask)[0]
    
    top_positive_in_subset = np.array([], dtype=int)
    top_negative_in_subset = np.array([], dtype=int)
    
    if positive_indices_in_subset.size > 0:
        sorted_positive = positive_indices_in_subset[np.argsort(nonzero_correlations[positive_indices_in_subset])]
        top_positive_in_subset = sorted_positive[-num_positive:]
        top_positive_indices = nonzero_indices[top_positive_in_subset]
    else:
        top_positive_indices = np.array([], dtype=int)
    
    if negative_indices_in_subset.size > 0:
        sorted_negative = negative_indices_in_subset[np.argsort(-nonzero_correlations[negative_indices_in_subset])]
        top_negative_in_subset = sorted_negative[-num_negative:]
        top_negative_indices = nonzero_indices[top_negative_in_subset]
    else:
        top_negative_indices = np.array([], dtype=int)
    
    selected_indices = np.concatenate([top_positive_indices, top_negative_indices])

    if selected_indices.size == 0:
        raise ValueError("No features selected: correlations array resulted in empty selection.")

    if selected_indices.size < num_features:
        remaining_needed = num_features - selected_indices.size
        already_selected_in_subset = np.concatenate([top_positive_in_subset, top_negative_in_subset])
        remaining_in_subset = np.setdiff1d(np.arange(len(nonzero_indices)), already_selected_in_subset, assume_unique=False)
        if remaining_in_subset.size > 0:
            sorted_remaining = remaining_in_subset[np.argsort(np.abs(nonzero_correlations[remaining_in_subset]))]
            top_remaining_in_subset = sorted_remaining[-remaining_needed:]
            remaining_indices = nonzero_indices[top_remaining_in_subset]
            selected_indices = np.concatenate([selected_indices, remaining_indices])[:num_features]
    
    return np.sort(selected_indices)
    

def detect_object_string_columns(X: np.ndarray | pd.DataFrame, cat_features: list[int] | None = None) -> list[int]:
    """
    Detect columns with dtype 'object' that contain string values and should be treated as categorical.
    
    This function identifies columns that have object dtype and contain string values,
    which should be treated as categorical but weren't explicitly marked as such.
    It avoids duplicates with already-identified categorical features.
    
    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Input feature matrix
    cat_features : list[int] or None
        Existing list of categorical feature indices (0-based column indices)
    
    Returns
    -------
    list[int]
        Combined list of categorical feature indices including both the original
        cat_features and newly detected object/string columns (sorted, no duplicates)
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'num': [1, 2, 3],
    ...     'cat': pd.Categorical(['a', 'b', 'c']),
    ...     'obj_str': ['x', 'y', 'z'],  # object dtype with strings
    ...     'obj_num': [1.0, 2.0, 3.0]   # object dtype but numeric
    ... })
    >>> detect_object_string_columns(df, cat_features=[1])
    [1, 2]  # column 1 was already categorical, column 2 (obj_str) detected
    """
    if cat_features is None:
        cat_features = []
    
    detected_cat_features = set(cat_features)

    if isinstance(X, pd.DataFrame):
        df = X
    else:
        df = pd.DataFrame(X)
    
    for col_idx, col_name in enumerate(df.columns):
        if col_idx in detected_cat_features:
            continue
        
        col_data = df.iloc[:, col_idx]
        
        if col_data.dtype == 'object':
            non_null_values = col_data.dropna()
            if len(non_null_values) == 0:
                continue
            
            sample_size = min(100, len(non_null_values))
            sample_values = non_null_values.iloc[:sample_size] if hasattr(non_null_values, 'iloc') else non_null_values[:sample_size]
            
            if all(isinstance(val, str) for val in sample_values):
                detected_cat_features.add(col_idx)
                logger.info(f"Detected object column with string values at index {col_idx} (column '{col_name}'). Adding to categorical features.")

    return sorted(list(detected_cat_features))
