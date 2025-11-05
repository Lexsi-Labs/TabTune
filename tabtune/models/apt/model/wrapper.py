import os
import pathlib

import torch

from .model import APT

from apt.utils import NumericEncoder, CategoricalEncoder


class APTPredictor(APT):
    def __init__(self, n_blocks, d_patch=100, d_model=512, d_ff=2048, n_heads=4,
                 dropout=0.1, activation="gelu", norm_eps=1e-5, classification=True):
        super().__init__(n_blocks,
            d_patch=d_patch, d_model=d_model,d_ff=d_ff,n_heads=n_heads,
            dropout=dropout, activation=activation, norm_eps=norm_eps,
            classification=classification
        )
        self.x_train = None
        self.y_train = None

        self.x_encoder = None
        if classification:
            self.y_encoder = None
        else:
            self.y_encoder = None

        self.feature_perm = None

    def get_score(self, metric, value):
        if metric in ["ce", "mse", "mae"]:
            return -value
        else:
            return value

    @torch.no_grad()
    def fit(self, x_train, y_train, val_size=0.2, process_data=True,
            tune=True, metric=None, n_perms=32, max_train=3000, max_test=3000):
        """
        x_train: (train_size, feature_size)
        y_train: (train_size)
        x_test: (test_size, feature_size)
        y_test: (test_size,)
        """
        x_train = torch.as_tensor(x_train)
        y_train = torch.as_tensor(y_train)
        if process_data:
            self.x_encoder = NumericEncoder().fit(x_train)
            x_train = self.x_encoder.transform(x_train)
            if self.classification:
                self.y_encoder = CategoricalEncoder().fit(y_train)
                y_train = self.y_encoder.transform(y_train)
            else:
                self.y_encoder = NumericEncoder().fit(y_train)
                y_train = self.y_encoder.transform(y_train)
        else:
            self.x_encoder = None
            if self.classification:
                self.y_encoder = None
            else:
                self.y_encoder = None
        self.x_train = x_train
        self.y_train = y_train

        if tune:
            data_perm = torch.randperm(x_train.shape[0])
            val_size = int(val_size * x_train.shape[0])
            val_x_test = x_train[data_perm[:val_size]]
            val_x_train = x_train[data_perm[val_size:]]
            val_y_test = y_train[data_perm[:val_size]]
            val_y_train = y_train[data_perm[val_size:]]

            default_result = self.evaluate_helper(
                val_x_train, val_y_train, val_x_test, val_y_test,
                max_train=max_train, max_test=max_test, metric=metric
            )
            best_score = default_result["Test AUC"] if self.classification else -default_result["Test MSE"]
            best_perm = None
            for _ in range(n_perms):
                feature_perm = torch.randperm(x_train.shape[1])
                val_x_train_perm = val_x_train[:, feature_perm]
                val_x_test_perm = val_x_test[:, feature_perm]

                if metric is None:
                    metric = "auc" if self.classification else "mse"
                result = self.evaluate_helper(
                    val_x_train_perm, val_y_train, val_x_test_perm, val_y_test,
                    max_train=max_train, max_test=max_test, metric=metric
                )
                score = self.get_score(metric, result)

                if score > best_score:
                    best_perm = feature_perm
                    best_score = score
            self.feature_perm = best_perm
        else:
            self.feature_perm = None

        return self

    def get_data(self, x_test, y_test=None):
        x_train = self.x_train
        y_train = self.y_train
        x_test = torch.as_tensor(x_test)
        if self.x_encoder is not None:
            x_test = self.x_encoder.transform(x_test)
        if self.feature_perm is not None:
            x_train = x_train[:, self.feature_perm]
            x_test = x_test[:, self.feature_perm]

        if y_test is not None:
            y_test = torch.as_tensor(y_test)
            if self.y_encoder is not None:
                y_test = self.y_encoder.transform(y_test)
            return x_train, y_train, x_test, y_test
        return x_train, y_train, x_test

    def evaluate(self, x_test, y_test, max_train=3000, max_test=3000, metric=None):
        return self.evaluate_helper(
            *self.get_data(x_test, y_test),
            max_train=max_train, max_test=max_test, metric=metric
        )

    def predict_proba(self, x_test, max_train=3000, max_test=3000):
        if self.classification:
            y_pred = self.predict_helper(
                *self.get_data(x_test),
                max_train=max_train, max_test=max_test
            )
            return y_pred.cpu().numpy()
        return NotImplementedError

    def predict(self, x_test, max_train=3000, max_test=3000):
        y_pred = self.predict_helper(
            *self.get_data(x_test),
            max_train=max_train, max_test=max_test
        )
        if self.classification:
            y_pred = torch.argmax(y_pred, dim=-1)
        if self.y_encoder is not None:
            y_pred = self.y_encoder.inverse_transform(y_pred)
        return y_pred.cpu().numpy()


class APTClassifier(APTPredictor):
    def __init__(self, device="cpu", model_name="model_epoch=200_classification_2025.01.13_21:18:53.pt",
                 base_path=pathlib.Path(__file__).parent.parent.resolve(), model_dir="checkpoints",
                 url="https://osf.io/download/684c9eb0fdbd7bc7fab689be/"):
        model_path = os.path.join(base_path, model_dir, model_name)
        if not pathlib.Path(model_path).is_file():
            print(f"Model not found at {model_path}. Downloading from {url}...")
            import requests
            r = requests.get(url, allow_redirects=True)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            open(model_path, 'wb').write(r.content)

        state_dict, init_args = torch.load(model_path, map_location='cpu')
        super().__init__(**init_args)
        self.load_state_dict(state_dict)
        self.to(device)
        self.eval()


class APTRegressor(APTPredictor):
    def __init__(self):
        return NotImplementedError
