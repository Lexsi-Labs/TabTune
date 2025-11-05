import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import TransformerBlock, PatchEmbedding, MixtureBlock
from .feedforward import FeedForward
from .utils import scatter_sum, auc_metric

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, log_loss,
    mean_squared_error, mean_absolute_error, r2_score
)


class APT(nn.Module):
    def __init__(self, n_blocks, d_patch=100, d_model=512, d_ff=2048, n_heads=4,
                 dropout=0.1, activation="gelu", norm_eps=1e-5, classification=True):
        super().__init__()
        self.n_blocks = n_blocks
        self.d_patch = d_patch
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = dropout
        self.activation = activation
        self.norm_eps = norm_eps
        self.classification = classification

        self._emb_x = PatchEmbedding(
            patch_size=d_patch, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            dropout=0.0, activation=activation, norm_eps=norm_eps
        )
        self._emb_y = FeedForward(
            d_model, in_dim=1, out_dim=d_model,
            activation=activation, bias=True
        )

        self._transformer = nn.ModuleList(
            [TransformerBlock(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                dropout=dropout, activation=activation, norm_eps=norm_eps
            ) for _ in range(n_blocks)]
        )

        if classification:
            self._out = MixtureBlock(
                d_model, n_heads=n_heads, d_ff=d_ff,
                dropout=dropout, activation=activation, temperature=0.2
            )
        else:
            self._out = FeedForward(
                d_model, in_dim=d_model, out_dim=2,
                activation=activation, bias=True
            )

    def forward(self, x, y_train, mask=None):
        """
        x: (batch_size, data_size, feature_size)
        y: (batch_size, data_size)
        mask: (batch_size, data_size)
        """
        split = y_train.shape[1]

        x = self._emb_x(x) # (batch_size, data_size, d_model)
        x_train, x_test = x[:, :split, ...], x[:, split:, ...] # (batch_size, n_train, d_model), (batch_size, n_test, d_model)
        y_train = self._emb_y(y_train.to(x.dtype).unsqueeze(-1)) # (batch_size, n_train, d_model)

        hidden = torch.cat([x_train + y_train, x_test], dim=1) # (batch_size, data_size, d_model)
        if mask is not None:
            mask = mask.to(hidden.dtype)
        mask = self.get_mask(split, hidden.shape[1] - split, mask=mask).to(hidden.device)

        for _, block in enumerate(self._transformer):
            hidden = block(hidden, mask=mask) # (batch_size, data_size, d_model)

        if self.classification:
            return self._out(hidden, split)
        return self._out(hidden[:, split:, ...])

    def loss(self, x, y, split=None, train_size=0.95):
        """
        x: (batch_size, data_size, feature_size)
        y: (batch_size, data_size)
        """
        if split is None:
            split = int(x.shape[1]*train_size)

        y_train, y_test = y[:, :split, ...], y[:, split:, ...] # (batch_size, n_train), (batch_size, n_test)

        out = self.forward(x, y_train)

        # prediction loss
        if self.classification:
            loss = self.classification_loss(out, y_train, y_test)
        else:
            loss = self.regression_loss(out, y_test)
        loss_dict = {"Prediction Loss": loss.item()}

        return loss, loss_dict

    def classification_loss(self, mixture_probs, y_train, y_test, eps=1e-4):
        """
        mixture_probs: (batch_size, test_size, train_size)
        y_train: (batch_size, train_size)
        y_test: (batch_size, test_size)
        """
        y = torch.cat((y_train, y_test), dim=1)

        offsets = torch.cumsum(y.max(1)[0] + 1, dim=0)[:-1].unsqueeze(-1)
        y[1:] = y[1:] + offsets
        y_train, y_test = y[:, :y_train.shape[1]], y[:, -y_test.shape[1]:]

        class_probs = scatter_sum(mixture_probs.transpose(0,1).reshape(mixture_probs.shape[1], -1),
            y_train.reshape(-1), dim=1, dim_size=y.max()+1
        )
        ce = -torch.log(torch.gather(class_probs, 1, y_test.transpose(0,1)) + eps)
        return ce.mean()

    def regression_loss(self, y_pred, y_test, eps=1e-6):
        """
        y_pred: (batch_size, test_size, 2)
        y_test: (batch_size, test_size)
        """
        return F.gaussian_nll_loss(
            y_pred[..., 0], y_test, F.softplus(y_pred[..., 1]), full=True, eps=eps
        )

    def get_mask(self, n_train, n_test, mask=None, as_bias=True, bias_scale=1.0):
        """
        attention mask:
            e.g. train - 4, test - 2
            [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            ]
        """
        if mask is None:
            attn_mask = torch.cat((
                torch.ones(n_train+n_test, n_train),
                torch.zeros(n_train+n_test, n_test)
            ), dim=1) # (n_train+n_test, n_train+n_test)
        else:
            """
            mask: (batch_size, n_train)
            """
            attn_mask = torch.cat((
                mask,
                torch.zeros(mask.shape[0], n_test, device=mask.device, dtype=mask.dtype)
            ), dim=1).unsqueeze(1).repeat(1, n_train+n_test, 1)  # (batch_size, n_train+n_test, n_train+n_test)

        if as_bias:
            # each test point attends to both train and test data
            # with an additive constant bias towards train data
            return bias_scale * attn_mask.float()
        return attn_mask.bool()

    @torch.no_grad()
    def predict_helper(self, x_train, y_train, x_test,
            max_train=3000, max_test=3000, subsample=True, n_classes=None, eps=1e-6):
        """
        x_train: (train_size, feature_size)
        y_train: (train_size,)
        x_test: (test_size, feature_size)
        """
        if self.classification:
            y_train = y_train.to(torch.long)
            if n_classes is None:
                n_classes = y_train.max() + 1
        device_m = next(self.parameters()).device
        device_t = y_train.device
        split = y_train.shape[0]

        if split > max_train and not subsample:
            batch_size = math.ceil(split / max_train)
            train_size = batch_size * max_train
            x_train = F.pad(
                x_train, (0, 0, 0, train_size - split), "constant", 0
            ).reshape(batch_size, max_train, -1)
            y_train = F.pad(
                y_train, (0, train_size - split), "constant", -1
            ).reshape(batch_size, max_train)
            mask = F.pad(
                torch.ones(split), (0, train_size - split), "constant", 0
            ).reshape(batch_size, max_train)
        else:
            batch_size = 1
            if split > max_train:
                inds = torch.randperm(split)
                x_train = x_train[inds[:max_train], :]
                y_train = y_train[inds[:max_train]]
            x_train = x_train.unsqueeze(0)
            y_train = y_train.unsqueeze(0)
            mask = None

        out = []
        for i in range(0, x_test.shape[0], max_test):
            x_test_batch = x_test[i:i+max_test, :].unsqueeze(0)
            x = torch.cat((x_train, x_test_batch.repeat(batch_size, 1, 1)), dim=1)

            x, y_train, mask = map(
                lambda t: t.to(device_m) if t is not None else None,
                (x, y_train, mask)
            )

            out_batch = self.forward(x, y_train, mask=mask)
            out.append(out_batch)
        out = torch.cat(out, dim=1)

        if self.classification:
            offsets = n_classes * torch.arange(1, batch_size, device=device_m).unsqueeze(-1)
            y_train[1:] = y_train[1:] + offsets

            res = scatter_sum(out.transpose(0,1).reshape(out.shape[1], -1),
                y_train.reshape(-1), dim=1, dim_size=(n_classes * batch_size))
            res = res.reshape(-1, batch_size, n_classes).transpose(0,1)
            if batch_size == 1:
                return res.squeeze(0).to(device_t)
            weights = mask.sum(dim=1, keepdim=True).unsqueeze(-1)
        else:
            res = out[..., 0]
            if batch_size == 1:
                return res.squeeze(0).to(device_t)
            weights = 1. / F.softplus(out[..., 1]).add(eps)
        weights = weights / weights.sum(0, keepdim=True)
        return (res * weights).sum(0).to(device_t)

    @torch.no_grad()
    def evaluate_helper(self, x_train, y_train, x_test, y_test,
            max_train=3000, max_test=3000, n_classes=None, metric=None):
        """
        x_train: (train_size, feature_size)
        y_train: (train_size)
        x_test: (test_size, feature_size)
        y_test: (test_size,)
        """
        if self.classification:
            y_train = y_train.to(torch.long)
            y_test = y_test.to(torch.long)
            if n_classes is None:
                n_classes = torch.cat((y_train, y_test)).max() + 1

        y_pred = self.predict_helper(
            x_train, y_train, x_test,
            max_train=max_train,
            max_test=max_test,
            n_classes=n_classes
        )

        target = y_test.cpu().numpy()
        if self.classification:
            proba = y_pred.cpu().numpy()
            pred = torch.argmax(y_pred, dim=-1).cpu().numpy()
            if metric == "acc":
                return accuracy_score(target, pred)
            elif metric == "bacc":
                return balanced_accuracy_score(target, pred)
            elif metric == "ce":
                return log_loss(target, proba)
            elif metric == "auc":
                return auc_metric(target, proba)
            else:
                return {
                    "Test ACC": accuracy_score(target, pred),
                    "Test BACC": balanced_accuracy_score(target, pred),
                    "Test CE": log_loss(target, proba),
                    "Test AUC": auc_metric(target, proba),
                }
        pred = y_pred.cpu().numpy()
        if metric == "mse":
            return mean_squared_error(target, pred)
        elif metric == "mae":
            return mean_absolute_error(target, pred)
        elif metric == "r2":
            return r2_score(target, pred)
        else:
            return {
                "Test MSE": mean_squared_error(target, pred),
                "Test MAE": mean_absolute_error(target, pred),
                "Test R2": r2_score(target, pred),
            }

    def save_checkpoint(self, path):
        init_args = {
            "n_blocks": self.n_blocks,
            "d_patch": self.d_patch,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "activation": self.activation,
            "norm_eps": self.norm_eps,
            "classification": self.classification
        }
        torch.save((self.state_dict(), init_args), path)
        print(f"Model saved to {path}")
