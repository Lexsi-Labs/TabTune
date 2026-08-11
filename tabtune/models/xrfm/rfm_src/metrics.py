from typing import List, Literal

import torch
import sklearn
import time

from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss


class Metric:
    name: str
    display_name: str
    should_maximize: bool
    task_types: List[Literal['reg', 'binclass', 'multiclass']]
    required_quantities: List[Literal['y_true', 'y_pred', 'y_pred_proba', 'agop', 'topk']]

    def __init__(self):
        assert self.__class__.compute == Metric.compute  # should not be overridden

    def compute(self, **kwargs) -> float:
        for q in self.required_quantities:
            if q not in kwargs:
                raise ValueError(f'Need to pass parameter {q} for metric {self.name}')

        return self._compute(**kwargs)

    def _compute(self, **kwargs) -> float:
        raise NotImplementedError()

    @staticmethod
    def from_name(name: str) -> 'Metric':
        all_metrics = [MSE, RMSE, MAE, Accuracy, Brier, AUC, Logloss, F1, TopAGOPVectorAUC, TopAGOPVectorPearsonR,
                       TopAGOPVectorsOLSAUC]
        all_metrics_dict = {m.name: m for m in all_metrics}
        return all_metrics_dict[name]()


class Metrics:
    def __init__(self, names: List[str]):
        self.names = names
        self.metrics = [Metric.from_name(name) for name in names]
        self.required_quantities = list(set.union(*[set(m.required_quantities) for m in self.metrics]))

    def compute(self, **kwargs):
        for q in self.required_quantities:
            if q not in kwargs:
                raise ValueError(f'Need to pass parameter {q} for metrics {self.names}')

        return {m.name: m.compute(**kwargs) for m in self.metrics}


# ----- specific metrics -----


class MSE(Metric):
    name = 'mse'
    display_name = 'MSE'
    should_maximize = False
    task_types = ['reg']
    required_quantities = ['y_true_reg', 'y_pred']

    def _compute(self, **kwargs) -> float:
        return (kwargs['y_true_reg'] - kwargs['y_pred']).square().mean().item()


class RMSE(Metric):
    name = 'rmse'
    display_name = 'RMSE'
    should_maximize = False
    task_types = ['reg']
    required_quantities = ['y_true_reg', 'y_pred']

    def _compute(self, **kwargs) -> float:
        return (kwargs['y_true_reg'] - kwargs['y_pred']).square().mean().sqrt().item()


class MAE(Metric):
    name = 'mae'
    display_name = 'MAE'
    should_maximize = False
    task_types = ['reg']
    required_quantities = ['y_true_reg', 'y_pred']

    def _compute(self, **kwargs) -> float:
        return (kwargs['y_true_reg'] - kwargs['y_pred']).abs().mean().item()


class Accuracy(Metric):
    name = 'accuracy'
    display_name = 'accuracy'
    should_maximize = True
    task_types = ['binclass', 'multiclass']
    required_quantities = ['y_true_class', 'y_pred_proba']

    def _compute(self, **kwargs) -> float:
        nz = torch.count_nonzero(kwargs['y_true_class'] == kwargs['y_pred_proba'].argmax(dim=-1))
        return (nz / kwargs['y_pred_proba'].shape[-2]).item()


class Brier(Metric):
    name = 'brier'
    display_name = 'Brier loss'
    should_maximize = False
    task_types = ['binclass', 'multiclass']
    required_quantities = ['y_true_class', 'y_pred_proba']

    def _compute(self, **kwargs) -> float:
        y_onehot = torch.nn.functional.one_hot(kwargs['y_true_class'],
                                               num_classes=kwargs['y_pred_proba'].shape[-1]).float()
        return (y_onehot - kwargs['y_pred_proba']).square().mean().item()


class AUC(Metric):
    name = 'auc'
    display_name = 'AUC'
    should_maximize = True
    task_types = ['binclass', 'multiclass']
    required_quantities = ['y_true_class', 'y_pred_proba']

    def _compute(self, **kwargs) -> float:
        # todo: this might fail in the case of missing classes
        probas = kwargs['y_pred_proba'].cpu().numpy()
        if probas.shape[1] == 2:
            probas = probas[:, 1]
        return roc_auc_score(kwargs['y_true_class'].cpu().numpy(), probas, multi_class='ovr')


class F1(Metric):
    name = 'f1'
    display_name = 'F1'
    should_maximize = True
    task_types = ['binclass', 'multiclass']
    required_quantities = ['y_true_class', 'y_pred_proba']

    def _compute(self, **kwargs) -> float:
        y_pred_proba = kwargs['y_pred_proba']
        n_classes = y_pred_proba.shape[-1]
        # I think macro matches the implementation in utils.py
        return sklearn.metrics.f1_score(kwargs['y_true_class'].cpu().numpy(),
                                        y_pred_proba.argmax(dim=-1).cpu().numpy(),
                                        average='binary' if n_classes == 2 else 'macro')


class Logloss(Metric):
    name = 'logloss'
    display_name = 'log-loss'
    should_maximize = False
    task_types = ['binclass', 'multiclass']
    required_quantities = ['y_true_class', 'y_pred_proba']

    def _compute(self, **kwargs) -> float:
        # could also implement manually here but that might not be equivalent in terms of clipping probabilities
        return log_loss(kwargs['y_true_class'].cpu().numpy(), kwargs['y_pred_proba'].cpu().numpy(),
                        labels=list(range(kwargs['y_pred_proba'].shape[-1])))


class TopAGOPVectorAUC(Metric):
    name = 'top_agop_vector_auc'
    display_name = 'Top AGOP Vector AUC'
    should_maximize = True
    task_types = ['binclass']
    required_quantities = ['y_true_class', 'y_pred_proba', 'agop', 'samples']

    def _compute(self, **kwargs) -> float:
        y_true_class = kwargs['y_true_class']
        y_pred_proba = kwargs['y_pred_proba']
        assert y_pred_proba.shape[1] == 2, "Top AGOP Vector AUC is only defined for binary classification"
        _, U = torch.lobpcg(kwargs['agop'], k=1)
        top_eigenvector = U[:, 0]
        projections = kwargs['samples'] @ top_eigenvector
        projections = projections.reshape(y_true_class.shape)
        # sigmoid is probably unnecessary?
        plus_auc = roc_auc_score(y_true_class.cpu().numpy(), torch.sigmoid(projections).cpu().numpy())
        minus_auc = roc_auc_score(y_true_class.cpu().numpy(), torch.sigmoid(-projections).cpu().numpy())
        return max(plus_auc, minus_auc)


class TopAGOPVectorPearsonR(Metric):
    name = 'top_agop_vector_pearson_r'
    display_name = 'Top AGOP Vector Pearson R'
    should_maximize = True
    task_types = ['binclass']
    required_quantities = ['y_true_class', 'y_pred_proba', 'agop', 'samples']

    def _compute(self, **kwargs) -> float:
        y_true_class = kwargs['y_true_class']
        y_pred_proba = kwargs['y_pred_proba']
        assert y_pred_proba.shape[1] == 2, "Top AGOP Vector Pearson R is only defined for binary classification"
        _, U = torch.lobpcg(kwargs['agop'], k=1)
        top_eigenvector = U[:, 0]
        projections = kwargs['samples'] @ top_eigenvector

        projections = projections.reshape(-1, 1)
        targets = y_true_class.float().reshape(-1, 1)  # todo??
        return torch.abs(torch.corrcoef(torch.cat((projections, targets), dim=-1).T))[0, 1].item()


class TopAGOPVectorsOLSAUC(Metric):
    name = 'top_agop_vectors_ols_auc'
    display_name = 'Top AGOP Vectors OLS AUC'
    should_maximize = True
    task_types = ['binclass', 'multiclass']
    required_quantities = ['y_true_class', 'y_pred_proba', 'agop', 'samples']

    def _compute(self, **kwargs) -> float:
        top_k = kwargs['top_k']
        y_onehot = torch.nn.functional.one_hot(kwargs['y_true_class'],
                                               num_classes=kwargs['y_pred_proba'].shape[-1]).float()
        print(f"Computing Top AGOP Vectors OLS AUC for {top_k} eigenvectors")
        start_time = time.time()
        _, U = torch.lobpcg(kwargs['agop'], k=top_k)
        end_time = time.time()
        print(f"Time taken to compute top {top_k} eigenvectors: {end_time - start_time} seconds")

        top_eigenvectors = U[:, :top_k]
        projections = kwargs['samples'] @ top_eigenvectors
        projections = projections.reshape(-1, top_k)

        start_time = time.time()
        XtX = projections.T @ projections
        Xty = projections.T @ y_onehot
        end_time = time.time()
        print(f"Time taken to compute XtX and Xty: {end_time - start_time} seconds")

        start_time = time.time()
        betas = torch.linalg.pinv(XtX) @ Xty
        end_time = time.time()
        print(f"Time taken to solve OLS: {end_time - start_time} seconds")

        start_time = time.time()
        preds = torch.sigmoid(projections @ betas).reshape(y_onehot.shape)
        end_time = time.time()
        print(f"Time taken to compute OLS predictions: {end_time - start_time} seconds")

        preds = preds.cpu().numpy()
        if preds.shape[1] == 2:
            preds = preds[:, 1]
        return roc_auc_score(kwargs['y_true_class'].cpu().numpy(), preds, multi_class='ovr')
