import math
from typing import Literal

import numpy as np
import torch
from scipy.special import softmax
from sklearn.base import ClassifierMixin
from tqdm import tqdm

from .estimator import TabDPTEstimator
from .utils import generate_random_permutation, pad_x


class TabDPTClassifier(TabDPTEstimator, ClassifierMixin):
    def __init__(
        self,
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
        # Inference parameters (with GPU-friendly defaults)
        n_ensembles: int = 6,
        temperature: float = 0.8,
        context_size: int = 512,
        permute_classes: bool = True,
        seed: int | None = None,
    ):
        super().__init__(
            mode="cls",
            inf_batch_size=inf_batch_size,
            normalizer=normalizer,
            missing_indicators=missing_indicators,
            clip_sigma=clip_sigma,
            feature_reduction=feature_reduction,
            faiss_metric=faiss_metric,
            device=device,
            use_flash=use_flash,
            compile=compile,
            model_weight_path=model_weight_path,
        )
        
        # Store inference parameters as instance attributes
        self.n_ensembles = n_ensembles
        self.temperature = temperature
        self.context_size = context_size
        self.permute_classes = permute_classes
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        super().fit(X, y)
        self.num_classes = len(np.unique(self.y_train))
        assert self.num_classes > 1, "Number of classes must be greater than 1"

    def _predict_large_cls(self, X_train, X_test, y_train):
        num_digits = math.ceil(math.log(self.num_classes, self.max_num_classes))

        digit_preds = []
        for i in range(num_digits):
            y_train_digit = (y_train // (self.max_num_classes**i)) % self.max_num_classes
            pred = self.model(
                x_src=torch.cat([X_train, X_test], dim=1),
                y_src=y_train_digit.unsqueeze(-1),
                task=self.mode,
            )
            digit_preds.append(pred.float())

        full_pred = torch.zeros((X_test.shape[0], X_test.shape[1], self.num_classes), device=X_train.device)
        for class_idx in range(self.num_classes):
            class_pred = torch.zeros_like(digit_preds[0][:, :, 0])
            for digit_idx, digit_pred in enumerate(digit_preds):
                digit_value = (class_idx // (self.max_num_classes**digit_idx)) % self.max_num_classes
                class_pred += digit_pred[:, :, digit_value]
            full_pred[:, :, class_idx] = class_pred.transpose(0, 1)

        return full_pred

    @torch.no_grad()
    def predict_proba(
        self,
        X: np.ndarray,
        temperature: float = 0.8,
        context_size: int = 2048,
        return_logits: bool = False,
        seed: int | None = None,
        class_perm: np.ndarray | None = None,
    ):
        train_x, train_y, test_x = self._prepare_prediction(X, class_perm=class_perm, seed=seed)

        if seed is not None:
            self.faiss_knn.index.seed = seed
            feat_perm = generate_random_permutation(train_x.shape[1], seed)
            train_x = train_x[:, feat_perm]
            test_x = test_x[:, feat_perm]

        if context_size >= self.n_instances:
            X_train = pad_x(train_x[None, :, :], self.max_features).to(self.device)
            X_test = pad_x(test_x[None, :, :], self.max_features).to(self.device)
            y_train = train_y[None, :].float()

            if self.num_classes <= self.max_num_classes:
                pred = self.model(
                    x_src=torch.cat([X_train, X_test], dim=1),
                    y_src=y_train.unsqueeze(-1),
                    task=self.mode,
                )
            else:
                pred = self._predict_large_cls(X_train, X_test, y_train)

            if not return_logits:
                pred = pred[..., :self.num_classes] / temperature
                pred = torch.nn.functional.softmax(pred.float(), dim=-1)
            pred_val = pred.float().squeeze().detach().cpu().numpy()
        else:
            pred_list = []
            for b in range(math.ceil(len(self.X_test) / self.inf_batch_size)):
                start = b * self.inf_batch_size
                end = min(len(self.X_test), (b + 1) * self.inf_batch_size)

                indices_nni = self.faiss_knn.get_knn_indices(self.X_test[start:end], k=context_size)
                X_nni = train_x[torch.tensor(indices_nni)]
                y_nni = train_y[torch.tensor(indices_nni)]

                X_nni, y_nni = (
                    pad_x(torch.Tensor(X_nni), self.max_features).to(self.device),
                    torch.Tensor(y_nni).to(self.device),
                )
                X_eval = test_x[start:end]
                X_eval = pad_x(X_eval.unsqueeze(1), self.max_features).to(self.device)

                if self.num_classes <= self.max_num_classes:
                    pred = self.model(
                        x_src=torch.cat([X_nni, X_eval], dim=1),
                        y_src=y_nni.unsqueeze(-1),
                        task=self.mode,
                    )
                else:
                    pred = self._predict_large_cls(X_nni, X_eval, y_nni)

                pred = pred.float()
                if not return_logits:
                    pred = pred[..., :self.num_classes] / temperature
                    pred = torch.nn.functional.softmax(pred, dim=-1)
                    pred /= pred.sum(axis=-1, keepdims=True)  # numerical stability

                pred_list.append(pred.squeeze(dim=0))
            pred_val = torch.cat(pred_list, dim=0).squeeze().detach().cpu().float().numpy()

        return pred_val

    def ensemble_predict_proba(
        self,
        X,
        n_ensembles: int | None = None,
        temperature: float | None = None,
        context_size: int | None = None,
        permute_classes: bool | None = None,
        seed: int | None = None,
    ):
        # Use instance defaults if not provided
        n_ensembles = n_ensembles if n_ensembles is not None else self.n_ensembles
        temperature = temperature if temperature is not None else self.temperature
        context_size = context_size if context_size is not None else self.context_size
        permute_classes = permute_classes if permute_classes is not None else self.permute_classes
        seed = seed if seed is not None else self.seed
        root_ss = np.random.SeedSequence(seed)
        inner_seeds = root_ss.generate_state(n_ensembles)
        logit_cumsum = None

        for inner_seed in tqdm(inner_seeds, desc="ensembles"):
            inner_seed = int(inner_seed)
            perm = torch.arange(self.num_classes)
            if permute_classes:
                perm = generate_random_permutation(self.num_classes, inner_seed)
            inv_perm = np.argsort(perm)

            logits = self.predict_proba(
                X,
                context_size=context_size,
                return_logits=True,
                seed=inner_seed,
                class_perm=perm,
            )
            logits = logits[..., inv_perm]
            if logit_cumsum is None:
                logit_cumsum = np.zeros_like(logits)
            logit_cumsum += logits

        logits = (logit_cumsum / n_ensembles)[..., :self.num_classes] / temperature
        pred = softmax(logits, axis=-1)
        pred /= pred.sum(axis=-1, keepdims=True)
        return pred

    def predict(
        self,
        X,
        n_ensembles: int | None = None,
        temperature: float | None = None,
        context_size: int | None = None,
        permute_classes: bool | None = None,
        seed: int | None = None,
    ):
        # Use instance defaults if not provided
        n_ensembles = n_ensembles if n_ensembles is not None else self.n_ensembles
        temperature = temperature if temperature is not None else self.temperature
        context_size = context_size if context_size is not None else self.context_size
        permute_classes = permute_classes if permute_classes is not None else self.permute_classes
        seed = seed if seed is not None else self.seed
        
        
        if n_ensembles == 1:
            return self.predict_proba(X, temperature=temperature, context_size=context_size, seed=seed).argmax(axis=-1)
        else:
            return self.ensemble_predict_proba(
                X,
                n_ensembles=n_ensembles,
                temperature=temperature,
                context_size=context_size,
                permute_classes=permute_classes,
                seed=seed,
            ).argmax(axis=-1)