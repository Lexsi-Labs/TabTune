import math

import numpy as np
import torch
from sklearn.base import RegressorMixin
from tqdm import tqdm

from .estimator import TabDPTEstimator
from .utils import generate_random_permutation, pad_x


class TabDPTRegressor(TabDPTEstimator, RegressorMixin):
    def __init__(
        self,
        inf_batch_size: int = 512,
        device: str = None,
        use_flash: bool = True,
        compile: bool = True,
        model_weight_path: str | None = None
    ):
        super().__init__(
            mode="reg",
            inf_batch_size=inf_batch_size,
            device=device,
            use_flash=use_flash,
            compile=compile,
            model_weight_path=model_weight_path
        )

    @torch.no_grad()
    def _predict(self, X: np.ndarray, context_size: int = 2048, seed: int | None = None):
        train_x, train_y, test_x = self._prepare_prediction(X)

        if seed is not None:
            self.faiss_knn.index.seed = seed
            feat_perm = generate_random_permutation(train_x.shape[1], seed)
            train_x = train_x[:, feat_perm]
            test_x = test_x[:, feat_perm]

        if context_size >= self.n_instances:
            X_train = pad_x(train_x[None, :, :], self.max_features).to(self.device)
            X_test = pad_x(test_x[None, :, :], self.max_features).to(self.device)
            y_train = train_y[None, :].float()
            pred = self.model(
                x_src=torch.cat([X_train, X_test], dim=1),
                y_src=y_train.unsqueeze(-1),
                task=self.mode,
            )

            return pred.float().squeeze().detach().cpu().float().numpy()
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
                pred = self.model(
                    x_src=torch.cat([X_nni, X_eval], dim=1),
                    y_src=y_nni.unsqueeze(-1),
                    task=self.mode,
                )

                pred_list.append(pred.squeeze(dim=0))

            return torch.cat(pred_list).squeeze().detach().cpu().float().numpy()

    def _ensemble_predict(
        self, X: np.ndarray, n_ensembles: int = 8, context_size: int = 2048, seed: int | None = None
    ):
        prediction_cumsum = 0
        generator = np.random.SeedSequence(seed)
        for _, inner_seed in tqdm(zip(range(n_ensembles), generator.generate_state(n_ensembles))):
            inner_seed = int(inner_seed)
            prediction_cumsum += self._predict(X, context_size=context_size, seed=inner_seed)
        return prediction_cumsum / n_ensembles

    def predict(
        self, X: np.ndarray, n_ensembles: int = 8, context_size: int = 2048, seed: int | None = None
    ):
        if n_ensembles == 1:
            return self._predict(X, context_size=context_size, seed=seed)
        else:
            return self._ensemble_predict(
                X, n_ensembles=n_ensembles, context_size=context_size, seed=seed
            )
