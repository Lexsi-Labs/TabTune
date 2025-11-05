import random

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from .base import BaseGenerator
from .utils import sample_trunc_norm_log_scaled, sample_trunc_norm_log_scaled_int, sample_trunc_gamma_int, sample_trunc_beta_min_max, sample_zero_inflated_uniform


class MultilayerPerceptron(BaseGenerator):
    def __init__(self, *args, classification=False, class_size_min=2, class_size_max=10,
                 hidden_size_min=5, hidden_size_max=130, n_hiddens_min=1, n_hiddens_max=6,
                 init_scale_min=0.01, init_scale_max=10., noise_std_min=0.0001, noise_std_max=0.3,
                 dropout_min=0.0, dropout_max=0.9, activation_choices=['leaky_relu', 'elu', 'tanh', 'identity'],
                 integer_size_p=0.65, integer_size_max=0.1, n_ints_min=2, n_ints_max=20,
                 category_size_p=0.65, category_size_max=0.2, n_cats_min=2, n_cats_max=10,
                 n_factors_min=1, n_factors_max=12, temperature=0.01, device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.classification = classification
        self.temperature = temperature

        self.class_size_min = class_size_min
        self.class_size_max = class_size_max + 1
        self.hidden_size_min = hidden_size_min
        self.hidden_size_max = hidden_size_max
        self.n_hiddens_min = n_hiddens_min
        self.n_hiddens_max = n_hiddens_max
        self.init_scale_min = init_scale_min
        self.init_scale_max = init_scale_max
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.dropout_min = dropout_min
        self.dropout_max = dropout_max
        self.activation_choices = activation_choices
        self.integer_size_p = 1 - integer_size_p
        self.integer_size_max = integer_size_max
        self.n_ints_min = n_ints_min
        self.n_ints_max = n_ints_max + 1
        self.category_size_p = 1 - category_size_p
        self.category_size_max = category_size_max
        self.n_cats_min = n_cats_min
        self.n_cats_max = n_cats_max + 1
        self.n_factors_min = n_factors_min
        self.n_factors_max = n_factors_max

        self.device = device
        self.set_models()

    def sample_class_size(self, size=None):
        return np.random.randint(self.class_size_min, self.class_size_max, size=size)

    def sample_hidden_size(self, shift=4, high=512):
        #TODO: size
        return sample_trunc_norm_log_scaled_int(self.hidden_size_min, self.hidden_size_max, shift=shift, high=high)

    def sample_n_hiddens(self, shift=2, high=32):
        #TODO: size
        return sample_trunc_norm_log_scaled_int(self.n_hiddens_min, self.n_hiddens_max, shift=shift, high=high)

    def sample_init_scale(self, min_sigma=0.01, max_sigma=1., low=0, high=1000000):
        #TODO: size
        return sample_trunc_norm_log_scaled(self.init_scale_min, self.init_scale_max,
            min_sigma=min_sigma, max_sigma=max_sigma, low=low, high=high)

    def sample_noise_std(self, min_sigma=0.01, max_sigma=1., low=0, high=1000000):
        #TODO: size
        return sample_trunc_norm_log_scaled(self.noise_std_min, self.noise_std_max,
            min_sigma=min_sigma, max_sigma=max_sigma, low=low, high=high)

    def sample_dropout(self, min_b=0.1, max_b=5.0):
        #TODO: size
        return sample_trunc_beta_min_max(self.dropout_min, self.dropout_max, min_b=min_b, max_b=max_b)

    def sample_activation(self, size=None):
        return np.random.choice(self.activation_choices, size=size)

    def sample_integer_size(self):
        #TODO: size
        return sample_zero_inflated_uniform(self.integer_size_p, self.integer_size_max)

    def sample_n_ints(self, size=None):
        return np.random.randint(self.n_ints_min, self.n_ints_max, size=size)

    def sample_category_size(self):
        #TODO: size
        return sample_zero_inflated_uniform(self.category_size_p, self.category_size_max)

    def sample_n_cats(self, size=None):
        return np.random.randint(self.n_cats_min, self.n_cats_max, size=size)

    def sample_n_factors(self, shift=1, high=64):
        #TODO: size
        return sample_trunc_norm_log_scaled_int(self.n_factors_min, self.n_factors_max, shift=shift, high=high)

    def input_sampler(self, sample_size, n_factors, eps=1e-6):
        if random.random() > 0.5:
            if random.random() > 0.5:
                mean = torch.randn(n_factors, device=self.device)
                std = torch.abs(torch.randn(n_factors, device=self.device) * mean)
                return mean + std * torch.randn(sample_size, self.data_size, n_factors, device=self.device)
            else:
                return torch.randn(sample_size, self.data_size, n_factors, device=self.device)
        else:
            if random.random() > 0.5:
                h = torch.stack(
                    [torch.multinomial(
                        torch.rand((random.randint(2, 10))),
                        sample_size*self.data_size,
                        replacement=True
                    ) for _ in range(n_factors)], dim=-1
                ).to(self.device).view(sample_size, self.data_size, n_factors).float()
                return (h - torch.mean(h, dim=(0,1))) / (torch.std(h, dim=(0,1)) + eps)
            else:
                h = torch.minimum(
                    torch.tensor(np.random.zipf(
                        2.0 + random.random() * 2,
                        size=(sample_size, self.data_size, n_factors)
                    ), device=self.device).float(),
                    torch.tensor(10.0, device=self.device)
                )
                return h - torch.mean(h, dim=(0,1))

    @torch.no_grad()
    def set_models(self):
        trained_mlps = []
        fixed_selections = []
        fixed_integers = []
        fixed_int_bounds = []
        fixed_categories = []
        fixed_cat_bounds = []
        fixed_class_bounds = []
        fixed_n_factors = []
        for _ in range(self.num_trained_datasets):
            hidden_size, n_hiddens, feature_size, n_factors = (
                self.sample_hidden_size(), self.sample_n_hiddens(), self.sample_feature_size(), self.sample_n_factors()
            )
            hidden_size = max(hidden_size, 2 * feature_size)

            mlp = MLP(n_hiddens, hidden_size, in_dim=n_factors, out_dim=1,
                init_scale=self.sample_init_scale(), noise_std=self.sample_noise_std(),
                dropout=self.sample_dropout(), activation=self.sample_activation(),
                device=self.device
            )
            trained_mlps.append(mlp)
            fixed_n_factors.append(n_factors)

            n_neurons = (n_hiddens - 1) * hidden_size
            selections = random.sample(range(n_neurons), feature_size)
            selections += range(n_neurons, n_neurons+1)
            random.shuffle(selections)
            fixed_selections.append(selections)

            integer_size = int(self.sample_integer_size()*feature_size)
            category_size = int(self.sample_category_size()*feature_size)
            discretes = random.sample(range(feature_size), integer_size + category_size)
            fixed_integers.append(discretes[:integer_size])
            fixed_int_bounds.append([np.random.normal(size=size-1) for size in self.sample_n_ints(integer_size)])
            fixed_categories.append(discretes[integer_size:])
            fixed_cat_bounds.append([np.random.normal(size=size-1) for size in self.sample_n_cats(category_size)])

            if self.classification:
                fixed_class_bounds.append(np.random.normal(size=self.sample_class_size()-1))
        self.trained_mlps = nn.ModuleList(trained_mlps)
        self.fixed_selections = fixed_selections
        self.fixed_integers = fixed_integers
        self.fixed_int_bounds = fixed_int_bounds
        self.fixed_categories = fixed_categories
        self.fixed_cat_bounds = fixed_cat_bounds
        self.fixed_class_bounds = fixed_class_bounds
        self.fixed_n_factors = fixed_n_factors

    @torch.no_grad()
    def reset_models(self):
        del self.trained_mlps

        self.set_models()

    def forward(self):
        xs, ys = [], []
        max_feature_size = 0
        for i in range(self.num_trained_datasets):
            h = self.input_sampler(self.sample_size, self.fixed_n_factors[i])
            outputs = torch.cat(self.trained_mlps[i](h), dim=-1)

            x = outputs[..., self.fixed_selections[i][:-1]]
            y = outputs[..., self.fixed_selections[i][-1]]

            x = self.discretize(x, discretes=self.fixed_integers[i], quantiles=self.fixed_int_bounds[i], perm=False)
            x = self.discretize(x, discretes=self.fixed_categories[i], quantiles=self.fixed_cat_bounds[i], perm=True)
            if self.classification:
                y = self.discretize(y.unsqueeze(-1), discretes=[0],
                    quantiles=[self.fixed_class_bounds[i]]
                ).long().squeeze(-1)

            xs.append(x)
            ys.append(y)
            max_feature_size = max(max_feature_size, x.shape[-1])
        with torch.no_grad():
            for _ in range(self.num_trained_datasets, self.num_datasets):
                hidden_size, n_hiddens, feature_size, n_factors = (
                    self.sample_hidden_size(), self.sample_n_hiddens(), self.sample_feature_size(), self.sample_n_factors()
                )
                hidden_size = max(hidden_size, 2 * feature_size)

                mlp = MLP(n_hiddens, hidden_size, in_dim=n_factors, out_dim=1,
                    init_scale=self.sample_init_scale(), noise_std=self.sample_noise_std(),
                    dropout=self.sample_dropout(), activation=self.sample_activation(),
                    device=self.device
                )

                h = self.input_sampler(self.sample_size, n_factors)
                outputs = torch.cat(mlp(h), dim=-1)

                selections = random.sample(range(outputs.shape[-1]), feature_size+1)
                x = outputs[..., selections[:-1]]
                y = outputs[..., selections[-1]]

                integer_size = int(self.sample_integer_size()*feature_size)
                category_size = int(self.sample_category_size()*feature_size)
                discretes = random.sample(range(feature_size), integer_size + category_size)
                x = self.discretize(x, discretes=discretes[:integer_size],
                    quantiles=[np.random.normal(size=size-1) for size in self.sample_n_ints(integer_size)],
                    perm=False, preserve_grad=False)
                x = self.discretize(x, discretes=discretes[integer_size:],
                    quantiles=[np.random.normal(size=size-1) for size in self.sample_n_cats(category_size)],
                    perm=True, preserve_grad=False)
                if self.classification:
                    y = self.discretize(y.unsqueeze(-1), discretes=[0],
                        quantiles=[np.random.normal(size=self.sample_class_size()-1)]
                    ).long().squeeze(-1)

                xs.append(x)
                ys.append(y)
                max_feature_size = max(max_feature_size, x.shape[-1])
        xs = torch.cat([F.pad(x, (0, max_feature_size-x.shape[-1]), "constant", 0) for x in xs], dim=0)
        ys = torch.cat(ys, dim=0)

        return xs, ys

    def discretize(self, x, discretes, quantiles, perm=True, preserve_grad=True, eps=1e-6):
        for col, z in zip(discretes, quantiles):
            d = x.detach()[..., col]
            z = torch.as_tensor(z, dtype=d.dtype, device=d.device).sort()[0]
            class_boundaries = d.mean(-1, keepdim=True) + z * d.std(-1, keepdim=True)

            idx = (d.unsqueeze(-1) > class_boundaries.unsqueeze(1)).sum(-1)
            class_boundaries = torch.cat(
                (torch.min(d, dim=-1)[0].unsqueeze(-1),
                 class_boundaries,
                 torch.max(d, dim=-1)[0].unsqueeze(-1)
                ), dim=1
            )

            class_floors = class_boundaries.gather(dim=1, index=idx)
            class_ceils = class_boundaries.gather(dim=1, index=idx+1)

            if perm:
                rand_perm = torch.randperm(len(z)+1, device=idx.device)
                idx = rand_perm[idx]

            if preserve_grad:
                x[:, :, col] = idx + self.temperature * (x[:, :, col] - class_floors)/(class_ceils - class_floors + eps)
            else:
                x[:, :, col] = idx
        return x


class MLP(torch.nn.Module):
    def __init__(self, n_hid, dim, in_dim=None, out_dim=None,
                 init_scale=1., noise_std=0.01, dropout=0.1, activation='relu',
                 device=None):
        super(MLP, self).__init__()
        assert n_hid > 0

        self.init_scale = init_scale
        self.noise_std = noise_std
        self.dropout = dropout

        in_dim = dim if in_dim is None else in_dim
        out_dim = dim if out_dim is None else out_dim

        in_dims = [in_dim] + n_hid*[dim]
        out_dims = n_hid*[dim] + [out_dim]
        linear = [nn.Linear(i, o, device=device) for i, o in zip(in_dims, out_dims)]
        self.linears = nn.ModuleList(linear)

        if activation == "relu":
            act = nn.ReLU
        elif activation == "leaky_relu":
            act = nn.LeakyReLU
        elif activation == "elu":
            act = nn.ELU
        elif activation == "silu":
            act = nn.SiLU
        elif activation == "gelu":
            act = nn.GELU
        elif activation == "tanh":
            act = nn.Tanh
        elif activation == "sigmoid":
            act = nn.Sigmoid
        elif activation == "identity":
            act = nn.Identity
        else:
            act = activation
        self.act = act()

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        if self.dropout > 0.0:
            for linear in self.linears[1:-1]:
                #nn.init.normal_(linear.weight, std=self.init_scale / (1. - self.dropout)**0.5)
                linear.weight *= self.init_scale / (1. - self.dropout)**0.5
                linear.weight *= torch.bernoulli(torch.full_like(linear.weight, 1. - self.dropout))

    def forward(self, x):
        x = self.linears[0](x)
        x = x + torch.randn_like(x) * self.noise_std
        outs = []
        for linear in self.linears[1:]:
            x = linear(self.act(x))
            x = x + torch.randn_like(x) * self.noise_std
            outs.append(x)
        return outs
