import os
from contextlib import nullcontext

import numpy as np

import torch
from torch import nn

from .utils import sample_trunc_gamma_int


class BaseGenerator(nn.Module):
    def __init__(self, batch_size, num_steps, data_size=1000,
                 num_datasets=8, num_trained_datasets=2,
                 eval_data_size=5000, static_eval_data=True,
                 feature_size_k=5, feature_size_mu=100,
                 split_min=0.1, split_max=0.9):
        super().__init__()
        assert batch_size % num_datasets == 0
        assert num_datasets >= num_trained_datasets
        self.sample_size = batch_size // num_datasets
        self.num_steps = num_steps
        self.data_size = data_size
        self.num_datasets = num_datasets
        self.num_trained_datasets = num_trained_datasets
        self.eval_data_size = eval_data_size
        self.static_eval_data = static_eval_data

        self.feature_size_k = feature_size_k
        self.feature_size_mu = feature_size_mu
        self.split_min = split_min
        self.split_max = split_max

        self.eval_data = None
        self.num_iters = nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False)

    def sample_feature_size(self, shift=1, high=300):
        #TODO: size
        return sample_trunc_gamma_int(self.feature_size_k, self.feature_size_mu, shift=shift, high=high)

    def sample_split(self, size=None):
        return np.random.uniform(self.split_min, self.split_max, size=size)

    def __len__(self):
        return self.num_steps

    def __iter__(self):
        self.num_iters += 1
        return iter(
            (*self.forward(), # (batch_size, data_size, feature_size), (batch_size, data_size)
            int(self.sample_split()*self.data_size), # int
            )
        for _ in range(self.num_steps))

    def set_eval_data(self, data_path, eval_data):
        self.eval_data = {}
        for d in eval_data:
            self.eval_data[d] = torch.load(os.path.join(data_path, d), map_location='cpu')

    def get_eval_data(self):
        if self.static_eval_data and self.eval_data is not None:
            return self.eval_data

        eval_data = {'_eval_data.pt': self._get_eval_data()}

        if self.static_eval_data:
            self.eval_data = eval_data
        return eval_data

    @torch.no_grad()
    def _get_eval_data(self):
        split = int(self.sample_split()*self.eval_data_size)

        sample_size, data_size, num_datasets, num_trained_datasets = (
            self.sample_size, self.data_size, self.num_datasets, self.num_trained_datasets)
        self.sample_size, self.data_size, self.num_datasets, self.num_trained_datasets = (
            1, self.eval_data_size, 1, 0)
        xs, ys = self.forward()
        xs, ys = xs.to('cpu'), ys.to('cpu')
        self.sample_size, self.data_size, self.num_datasets, self.num_trained_datasets = (
            sample_size, data_size, num_datasets, num_trained_datasets)

        x_train, x_test = xs[:, :split, :], xs[:, split:, :]
        y_train, y_test = ys[:, :split], ys[:, split:]
        return {
            "data": (x_train.squeeze(0), y_train.squeeze(0), x_test.squeeze(0), y_test.squeeze(0)),
        }
