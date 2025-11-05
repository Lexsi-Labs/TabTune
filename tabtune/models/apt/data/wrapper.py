import numpy as np

import torch

from .mlp import MultilayerPerceptron

from apt.utils import process_tensor, process_data


class DataGenerator(MultilayerPerceptron):
    def __init__(self, *args, missing_size_min=0.0, missing_size_max=0.001, **kwargs):
        super().__init__(*args, **kwargs)

        self.missing_size_min = missing_size_min
        self.missing_size_max = missing_size_max

    def sample_missing_size(self):
        return np.random.uniform(self.missing_size_min, self.missing_size_max)

    def forward(self):
        x, y = super().forward()

        # add missing
        x, mask = self.add_missing(x, return_mask=True)

        # process
        x = process_tensor(x, dim=1, mask=mask)
        if not self.classification:
            y = process_tensor(y, dim=1)

        return x, y

    def add_missing(self, x, return_mask=False):
        mask = torch.rand(x.size(), device=x.device) < self.sample_missing_size()
        x = x.masked_fill(mask, 0.0)
        if return_mask:
            return x, ~mask
        return x

    def set_eval_data(self, data_path, eval_data):
        super().set_eval_data(data_path, eval_data)

        for key, val in self.eval_data.items():
            self.eval_data[key]["data"] = process_data(val["data"], classification=self.classification)
