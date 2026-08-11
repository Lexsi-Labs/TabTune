import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import forward_linear_layer, TorchPCA, SimpleBatchNorm1dFixed

logger = logging.getLogger(__name__)

class iLTM(nn.Module):
    def __init__(self,
                 n_dims: int = 512,
                 hn_n_layers: int = 4,
                 hn_hidden_size: int = 1024,
                 clip_data_value: float = 27.6041,
                 rf_size: int = 2**15,
                 main_n_layers: int = 3,
                 n_classes_limit: int = 100,
                 bottleneck_size: int = 0,
                 dim_exp_type: str = 'rf',
                 pca_fit: str = 'lowrank',
                 pca_svd_driver: str | None = None,
                 hyper_dropout: float = 0.0,
                 pca_sampling: str = 'repeat'):
        super().__init__()
        self.n_dims = n_dims
        self.hn_n_layers = hn_n_layers
        self.hn_hidden_size = hn_hidden_size
        self.clip_data_value = clip_data_value
        self.rf_size = rf_size
        self.main_n_layers = main_n_layers
        self.n_classes_limit = n_classes_limit
        self.bottleneck_size = bottleneck_size
        self.dim_exp_type = dim_exp_type
        self.pca_fit = pca_fit
        self.pca_svd_driver = pca_svd_driver
        self.hyper_dropout = hyper_dropout
        self.pca_sampling = pca_sampling

        logger.debug("Initializing iLTM model.")

        # Initial Transformation Block
        self.initial_transformation_block = InitialTransformationBlock(
            rf_size=self.rf_size,
            n_dims=self.n_dims,
            clip_data_value=self.clip_data_value,
            dim_exp_type=self.dim_exp_type,
            pca_fit=self.pca_fit,
            pca_svd_driver=self.pca_svd_driver,
            pca_sampling=self.pca_sampling
        )

        # Hypernetwork Block
        self.hypernetwork_block = HypernetworkBlock(
            n_classes_limit=self.n_classes_limit,
            hn_n_layers=self.hn_n_layers,
            hn_hidden_size=self.hn_hidden_size,
            main_n_layers=self.main_n_layers,
            n_dims=self.n_dims,
            bottleneck_size=self.bottleneck_size,
            hyper_dropout=self.hyper_dropout
        )

        logger.debug("iLTM model initialized.")

    def forward(self, X, y, n_classes, training=False):
        
        if X.dtype != torch.float32:
            X = X.to(torch.float32)
        
        X_transformed = self.initial_transformation_block(X)

        generated_network = self.hypernetwork_block(X_transformed, y, n_classes, training=training)
        rf, pca, norm = self.initial_transformation_block.get_submodules()
        return rf, pca, generated_network, norm


class InitialTransformationBlock(nn.Module):
    def __init__(self, rf_size, n_dims, clip_data_value, dim_exp_type='rf', dim_red_type='pca',
                 pca_fit='lowrank', pca_svd_driver=None, pca_sampling='repeat'):
        super().__init__()
        self.rf_size = rf_size
        self.n_dims = n_dims
        self.clip_data_value = clip_data_value
        self.dim_exp_type = dim_exp_type
        self.dim_red_type = dim_red_type
        self.pca_fit = pca_fit
        self.pca_svd_driver = pca_svd_driver
        self.pca_sampling = pca_sampling
        self.norm = None

    def forward(self, X):
        """Process input data X through the transformation block."""
        
        X = X.flatten(start_dim=1)
        X = self._apply_dimensionality_expansion(X)
        X = self._apply_dimensionality_reduction(X)
        X = self._apply_normalization(X)
        return X

    def _apply_dimensionality_expansion(self, X):
        if self.dim_exp_type == 'rf':
            X = self._apply_random_features(X)
        elif self.dim_exp_type == 'none':
            # No expansion
            pass
        else:
            # Other method
            raise ValueError(f"Dimensionality expansion method {self.dim_exp_type} not recognized.")
        return X

    def _apply_random_features(self, X):
        rf_size = self.rf_size
        rf_linear = nn.Linear(X.shape[1], rf_size, bias=False)
        nn.init.kaiming_normal_(rf_linear.weight, mode="fan_out", nonlinearity="relu")
        rf_linear.weight.requires_grad = False
        self.rf = nn.Sequential(rf_linear, nn.ReLU()).to(X.device)
        with torch.no_grad():
            X = self.rf(X)
        return X

    def _apply_dimensionality_reduction(self, X):
        if self.dim_red_type == 'pca':
            X = self._apply_pca(X)
        else:
            # Other method
            raise ValueError(f"Dimensionality reduction method {self.dim_red_type} not recognized.")
        return X

    def _apply_pca(self, X):
        # PCA
        self.pca = TorchPCA(n_components=self.n_dims, fit=self.pca_fit, svd_driver=self.pca_svd_driver)
        with torch.autocast(device_type=X.device.type, enabled=False):
            X = X.to(torch.float32)
            X = self.pca.fit_transform(X)
            if self.pca_sampling == 'zeropad' and X.shape[1] < self.n_dims:
                # Zero-pad the data to the fixed number of dimensions
                X = F.pad(X, (0, self.n_dims - X.shape[1]), value=0)
            X = torch.clamp(X, -self.clip_data_value, self.clip_data_value)
        return X

    def _apply_normalization(self, X):
        self.norm = SimpleBatchNorm1dFixed(X.shape[1]).to(X.device)
        self.norm.fit(X)
        X = self.norm(X)
        return X

    def get_submodules(self):
        assert hasattr(self, 'pca'), "PCA submodule not found."
        if self.dim_exp_type == 'rf':
            expansion = self.rf
        elif self.dim_exp_type == 'none':
            expansion = nn.Identity()
        reduction = self.pca
        assert self.norm is not None
        normalization = self.norm

        # Delete the submodules to avoid saving them in the model checkpoint
        if hasattr(self, 'rf'):
            del self.rf
        del self.pca

        return expansion, reduction, normalization


class HypernetworkBlock(nn.Module):
    def __init__(self, n_classes_limit, hn_n_layers, hn_hidden_size,
                 main_n_layers, n_dims, bottleneck_size, hyper_dropout=0.0):
        super().__init__()
        self.n_classes_limit = n_classes_limit
        self.hn_n_layers = hn_n_layers
        self.hn_hidden_size = hn_hidden_size
        self.main_n_layers = main_n_layers
        self.n_dims = n_dims
        self.num_input_features_hn = n_dims + n_classes_limit
        self.hypernetworks = nn.ModuleList()
        self.hn_emb_to_weights = nn.ModuleList()
        self.bottleneck_size = bottleneck_size
        self.hyper_dropout = hyper_dropout

        # Initialize hypernetwork components
        self._init_hypernetworks()

        logger.debug("HypernetworkBlock initialized.")

    def _init_hypernetworks(self):
        """Initialize the layers for the hypernetworks."""
        middle_layers = self._create_middle_layers()
        for n in range(self.main_n_layers - 1):
            if n > 0:
                self.num_input_features_hn = self.n_dims * 2 + self.n_classes_limit
            num_input_features_hn = self.num_input_features_hn + self.n_dims * 2

            hn_layers = [nn.Linear(num_input_features_hn, self.hn_hidden_size), nn.ReLU()] + middle_layers

            self.hypernetworks.append(nn.Sequential(*hn_layers))

            output_size_hn = (self.n_dims + 1) * self.n_dims
            if self.bottleneck_size > 0 and self.bottleneck_size < self.hn_hidden_size:
                bottleneck_layers = self._build_bottleneck_layers(self.hn_hidden_size, output_size_hn)
                self.hn_emb_to_weights.append(nn.Sequential(*bottleneck_layers))

            else:
                self.hn_emb_to_weights.append(nn.Linear(self.hn_hidden_size, output_size_hn))

        # Setup the last hypernetwork layer
        self._setup_last_hypernetwork_layer(middle_layers)

    def _build_bottleneck_layers(self, input_size, output_size):

        return nn.Sequential(
            nn.Linear(input_size, self.bottleneck_size),
            nn.ReLU(),
            nn.Linear(self.bottleneck_size, output_size)
        )

    def _create_middle_layers(self):
        """Create middle layers used in multiple places in the hypernetwork."""
        middle_layers = []
        for _ in range(self.hn_n_layers - 2):
            middle_layers += [nn.Linear(self.hn_hidden_size, self.hn_hidden_size), nn.ReLU()]
        return middle_layers

    def _setup_last_hypernetwork_layer(self, middle_layers):
        """Setup the last layer of the hypernetworks, handling the special case of the last layer."""
        last_hn_output_size = self.n_dims + 1
        self.num_input_features_hn += self.n_dims * 2
        hn_layers = [nn.Linear(self.num_input_features_hn, self.hn_hidden_size), nn.ReLU()] + middle_layers
        hn_layers.append(nn.Linear(self.hn_hidden_size, last_hn_output_size))
        self.hypernetworks.append(nn.Sequential(*hn_layers))

    def _calculate_class_means(self, X, y, n_classes):
        """
        Calculate per-class means if in classification mode (n_classes > 1).
        For regression (n_classes == 1), return the global mean.

        Args:
            X (Tensor): Feature matrix.
            y (Tensor): Targets.
            n_classes (int): Number of classes (set to 1 for regression).

        Returns:
            Tensor: If classification, concatenated means per class (shape: [n_classes, X.shape[1]]);
                    if regression, global mean (shape: [1, X.shape[1]]).
        """
        if n_classes == 1:
            # Regression: return global mean.
            return torch.mean(X, dim=0, keepdim=True)
        
        perclass_mean = []
        for lab in range(n_classes):
            if torch.sum((y == lab)) > 0:
                class_mean = torch.mean(X[y == lab], dim=0, keepdim=True)
            else:
                class_mean = torch.mean(X, dim=0, keepdim=True)
            perclass_mean.append(class_mean)
        return torch.cat(perclass_mean, dim=0)


    def _concatenate_input_features(self, X, X_global_mean, X_perclass_mean, y):
        """Concatenate global and per-class means with the transformed input data."""
        B = X.shape[0]
        global_rep = X_global_mean.expand(B, -1)
        if X_perclass_mean.size(0) == 1:
            # If there's only one per-class mean (or regression for now), repeat it for all samples
            perclass = X_perclass_mean.expand(B, -1)
        else:
            idx = y.clamp_max(X_perclass_mean.size(0) - 1)  # safe guard
            perclass = X_perclass_mean[idx]
        return torch.cat((X, global_rep, perclass), dim=1)


    def _forward_hypernetwork_modules(self, out, dataset_representation, y, n_classes, training=False):
        """Process layers of the hypernetworks."""
        main_network = []
        
        # All hypernetwork modules except the last one
        for n, layer in enumerate(self.hypernetworks[:-1]):
            
            if n == 0:
                data = dataset_representation
            else:
                data = torch.cat((out, dataset_representation), dim=1)

            if n % 2 == 0:
                residual_connection = out
            weights = self.get_main_weights(data, layer, self.hn_emb_to_weights[n])
            out, main_linear_layer = forward_linear_layer(out, weights, self.n_dims)

            if n % 2 == 1:
                out = out + residual_connection
            out = F.relu(out)
            out = F.dropout(out, p=self.hyper_dropout, training=training)

            main_network.append(main_linear_layer)

        # Last hypernetwork module
        last_layer = self._forward_last_hypernetwork_module(out, dataset_representation, y, n_classes)
        main_network.append(last_layer)
        return main_network

    def _forward_last_hypernetwork_module(self, out, dataset_representation, y, n_classes):
        """Forward pass for the last hypernetwork module that generates the final layer weights.

        This method handles both regression (n_classes=1) and classification (n_classes>1) cases.
        For regression, it generates a single set of weights based on the mean of all samples.
        For classification, it generates separate weights for each class based on class-specific means.

        Args:
            out (torch.Tensor): Output from previous layer, shape [batch_size, n_dims]
            dataset_representation (torch.Tensor): Concatenated features representing the dataset
            y (torch.Tensor): Target labels, shape [batch_size]
            n_classes (int): Number of classes (1 for regression)

        Returns:
            last_layer (nn.Linear): Generated final linear layer with appropriate weights
        """
        data = torch.cat((out, dataset_representation), dim=1)
        weights_per_sample = self.get_main_weights(data, self.hypernetworks[-1])

        if n_classes == 1:  # Regression case
            # For regression, take mean of all weights and inputs
            weights = torch.mean(weights_per_sample, dim=0, keepdim=True)
            last_input_mean = torch.mean(out, dim=0, keepdim=True)
            
            # Adjust weights based on mean input
            weights[:, :-1] = weights[:, :-1] + last_input_mean
            weights = weights.transpose(0, 1)

        else:  # Classification case
            # Compute the mean weights and output for each class
            weights = []
            last_input_mean = []
            for lab in range(n_classes):
                if torch.sum((y == lab)) > 0:
                    class_weights = torch.mean(weights_per_sample[y == lab], dim=0, keepdim=True)
                    class_input_mean = torch.mean(out[y == lab], dim=0, keepdim=True)
                else:
                    class_weights = torch.mean(weights_per_sample, dim=0, keepdim=True)
                    class_input_mean = torch.mean(out, dim=0, keepdim=True)
                weights.append(class_weights)
                last_input_mean.append(class_input_mean)

            # Concatenate all class weights and means
            weights = torch.cat(weights)
            last_input_mean = torch.cat(last_input_mean)
            
            # Adjust weights based on the class mean inputs
            weights[:, :-1] = weights[:, :-1] + last_input_mean
            weights = weights.transpose(0, 1)

        # Apply the final transformed weights to the output
        out, last_layer = forward_linear_layer(out, weights, n_classes)

        return last_layer
    
    

    def get_main_weights(self, x, hn, weight_gen=None):
        emb = hn(x)
        if weight_gen is not None:
            global_emb = torch.mean(emb, dim=0)
            w = weight_gen(global_emb)
        else:
            w = emb
        return w

    def forward(self, X_transformed, y, n_classes, training=False):
        """Forward pass for the HypernetworkBlock."""
        X_transformed_mean = torch.mean(X_transformed, axis=0)
        X_transformed_perclass_mean = self._calculate_class_means(X_transformed, y, n_classes)
        X_concat = self._concatenate_input_features(X_transformed, X_transformed_mean, X_transformed_perclass_mean, y)

        ## onehot only if n_classes > 1, else is regression
        if n_classes > 1:
            y_onehot = F.one_hot(y, self.n_classes_limit)
        else:
            # For regression, pad with zeros to match classification shape
            y_onehot = torch.zeros((y.shape[0], self.n_classes_limit), device=y.device)
            y_onehot[:, 0] = y  # Put regression target in first column

        dataset_representation = torch.cat((X_concat, y_onehot), dim=1)
        X_input_to_generated_linears = X_transformed
        main_network = self._forward_hypernetwork_modules(X_input_to_generated_linears, dataset_representation, y, n_classes, training=training)
        return main_network
