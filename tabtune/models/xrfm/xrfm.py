import sys
import time
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .rfm_src import RFM, matrix_power
from .rfm_src.gpu_utils import (
    memory_scaling_factor,
)
from tqdm import tqdm
import copy

from .rfm_src.class_conversion import ClassificationConverter
from .rfm_src.metrics import Metric
from .rfm_src.utils import get_top_eigenvector
from .tree_utils import get_param_tree

DEFAULT_TEMP_TUNING_SPACE = [0.0] + list(np.logspace(np.log10(0.025), np.log10(4.5), num=20))

class xRFM:
    """
    Tree-based Recursive Feature Machine (RFM).
    
    This model recursively splits the training data using random projections 
    and fits the base RFM model on each subset once the subset size is small enough.
    
    Parameters
    ----------
    rfm_params : dict, default=None
        Parameters to pass to the RFM model at each leaf node.
        If None, default parameters are used.
    
    max_leaf_size : int, default=60000
        The maximum size of a leaf (i.e. the minimum size of a subset to further split).
        If a subset has fewer samples than this, a base RFM model is fit on it directly.
        The deprecated alias ``min_subset_size`` is still accepted in the constructor.
    
    number_of_splits : int, optional
        Minimum number of splits to perform while building each tree. If None,
        splitting is only constrained by the subset size.
    
    device : str, default=None
        Device to use for computation. If None, uses cuda if available, otherwise cpu.
        
    n_trees : int, default=1
        Number of trees to build. Predictions will be averaged across all trees.
        
    n_tree_iters : int, default=0
        Number of iterations to build each tree. Later iterations use the average
        of model.M from all leaf nodes to generate better projection directions.
        If n_tree_iters=0, the original random projection method is used.

    split_method : str, default='top_vector_agop_on_subset'
        Method to use for splitting the data.
        'top_vector_agop_on_subset' : use the top eigenvector of the AGOP on the subset
        'random_agop_on_subset' : use a random eigenvector of the AGOP on the subset
        'top_pc_agop_on_subset' : use the top principal component of data transformed with the AGOP
        'random_pca' : use a random principal component of the data
        'linear' : use linear regression coefficients as projection direction
        'fixed_vector' : use a fixed vector for projection (requires fixed_vector parameter)

    tuning_metric : str, default=None
        Metric to use for tuning the model (defaults to 'mse' for regression and 'brier' for classification).
        'mse' : mean squared error
        'rmse' : root mean squared error
        'mae' : mean absolute error
        'accuracy' : accuracy
        'brier' : Brier loss
        'logloss' : Log loss
        'f1' : F1 score
        'auc' : area under the ROC curve
        
    categorical_info : dict, default=None
        Information about the categorical features.
        If None, it is assumed that there are no categorical features.
        If not None, it should be a dictionary with the following keys:
        'categorical_indices' : list of indices of the categorical features
        'categorical_vectors' : list of vectors of the categorical features
    
    default_rfm_params : dict, default=None
        Default parameters for the RFM model used for generating split directions
        when using AGOP-based split methods. If None, uses built-in default parameters
        with kernel='l2', exponent=1.0, bandwidth=10.0, etc.
    
    fixed_vector : torch.Tensor, default=None
        Fixed projection vector to use when split_method='fixed_vector'.
        Must be provided if using 'fixed_vector' split method.
    
    callback : function, default=None
        Callback function to call after each iteration of each Leaf RFM.
        The function must accept an 'iteration' argument.

    classification_mode : str, default='zero_one'
        How to convert classification problems to regression problems.
        'zero_one': Binary problems are converted to {0, 1}, multiclass to one-hot labels.
        'prevalence': Problems with $k$ classes are encoded to a k-1 dimensional simplex,
        such that zero corresponds to the empirical probability distribution of train labels.
        This way, the predictions will converge to this empirical distribution far away from the training data.
        This mode will also be slightly faster than 'zero_one' for multiclass problems
        since only k-1 instead of k linear systems need to be solved for each leaf RFM.

    time_limit_s : float, optional
        Time limit in seconds.

    n_threads : int, optional
        Number of CPU threads to use.

    split_temperature : float, optional
        Global temperature constant controlling soft routing and ensembling during prediction.
        Each split node j uses an adaptive temperature equal to ``split_temperature * IQR_j``,
        where ``IQR_j`` is the inter-quartile range of the projection values observed at that node.
        If None, predictions use the original hard routing that follows a single leaf.
        Smaller positive values sharpen the routing distribution, approaching hard decisions.
    
    overlap_fraction : float
        Fraction of the dataset (per side) to include around the split point in both child leaves.
        Each leaf receives an additional overlap of size 2 * overlap_fraction of the original data.

    keep_weight_frac_in_predict : float
        Fraction of cumulative leaf weight mass to retain per sample during soft prediction.
        The top-weighted leaves covering this fraction are evaluated and their weights
        are renormalized before aggregation.

    max_leaf_count_in_ensemble : int
        Maximum number of leaves evaluated per sample during soft prediction.
        Acts as a hard cap after enforcing keep_weight_frac_in_predict.
    
    Notes
    -----
    The model follows sklearn's estimator interface with fit, predict, predict_proba, and score methods,
    but does not comply with all requirements.
    """

    def __init__(self, rfm_params=None, max_leaf_size=60_000,
                 number_of_splits=None, device=None, n_trees=1, n_tree_iters=0,
                 split_method='top_vector_agop_on_subset', tuning_metric=None,
                 categorical_info=None, default_rfm_params=None,
                 fixed_vector=None, callback=None, classification_mode='zero_one',
                 time_limit_s=None, n_threads=None, refill_size=1500, random_state=None,
                 split_temperature=None, overlap_fraction=0.0, use_temperature_tuning=True,
                 keep_weight_frac_in_predict=0.99, max_leaf_count_in_ensemble=12,
                 temp_tuning_space: Optional[List[float]]= None,
                 verbose: bool = True,
                 **kwargs):
        deprecated_min_subset_size = kwargs.pop('min_subset_size', None)
        if deprecated_min_subset_size is not None:
            if max_leaf_size != 60_000:
                raise ValueError("Cannot specify both max_leaf_size and min_subset_size.")
            max_leaf_size = deprecated_min_subset_size

        self._base_max_leaf_size = int(max_leaf_size)
        self.rfm_params = rfm_params
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trees = None
        self.projections = None
        self.models = None
        self.n_trees = n_trees
        self.n_tree_iters = n_tree_iters
        self.tuning_metric = tuning_metric
        self.split_method = split_method
        self.number_of_splits = number_of_splits
        self.maximizing_metric = False if tuning_metric is None else Metric.from_name(tuning_metric).should_maximize
        self.categorical_info = categorical_info
        self.fixed_vector = fixed_vector
        self.callback = callback
        self.classification_mode = classification_mode
        self.time_limit_s = time_limit_s
        self.n_threads = n_threads
        self.extra_rfm_params_ = {}
        if not (0.0 <= overlap_fraction <= 0.5):
            raise ValueError("overlap_fraction must be in [0.0, 0.5].")
        self.overlap_fraction = overlap_fraction
        self.use_temperature_tuning = use_temperature_tuning
        if not (0.0 <= keep_weight_frac_in_predict <= 1.0):
            raise ValueError("keep_weight_frac_in_predict must lie in [0.0, 1.0].")
        self.keep_weight_frac_in_predict = keep_weight_frac_in_predict
        if max_leaf_count_in_ensemble < 1:
            raise ValueError("max_leaf_count_in_ensemble must be at least 1.")
        self.max_leaf_count_in_ensemble = int(max_leaf_count_in_ensemble)

        # scale the maximum leaf size relative to a 40GB GPU; assume quadratic memory growth
        subset_scale = memory_scaling_factor(self.device, quadratic=True)
        self.max_leaf_size = max(int(self._base_max_leaf_size * subset_scale), 1)
        # Backwards compatibility for downstream users referencing min_subset_size directly.
        self.min_subset_size = self.max_leaf_size

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)

        if split_temperature is not None and split_temperature < 0:
            raise ValueError("split_temperature must be positive when specified.")
        self.split_temperature = split_temperature

        if temp_tuning_space is None:
            temp_tuning_space = DEFAULT_TEMP_TUNING_SPACE
        self.temp_tuning_space = temp_tuning_space

        self.verbose = verbose

        # parameters for refilling the validation set at leaves
        self.min_val_size = refill_size
        self.val_size_frac = 0.2

        # default parameters for the split direction model
        if default_rfm_params is None:
            self.default_rfm_params = {
                'model': {
                    "kernel": 'l2_high_dim',
                    "exponent": 1.0,
                    "bandwidth": 10.0,
                    "diag": False,
                    "bandwidth_mode": "constant"
                },
                'fit': {
                    "get_agop_best_model": True,
                    "return_best_params": False,
                    "reg": 1e-3,
                    "iters": 0,
                    "early_stop_rfm": False,
                }
            }
        else:
            self.default_rfm_params = default_rfm_params

        if self.rfm_params is None:
            self.rfm_params = self.default_rfm_params
            self.rfm_params['fit']['return_best_params'] = True
            self.rfm_params['fit']['iters'] = 5

    def to(self, device):
        """Move all internal tensors and leaf RFM models to *device*.

        Parameters
        ----------
        device : str or torch.device
            Target device ('cpu', 'cuda', etc.)

        Returns
        -------
        self
        """
        device = torch.device(device) if isinstance(device, str) else device
        self.device = device

        if self.trees is not None:
            for tree in self.trees:
                # Drop the routing cache before moving: it stores tensor
                # references captured in _build_tree_cache, which a node-dict
                # reassignment in _move_tree_to_device cannot reach. Leaving it
                # in place would keep source-device tensors live and break
                # pickling onto another device.
                tree.pop('_cache', None)
                self._move_tree_to_device(tree, device)

        # Move categorical_info tensors
        cat_info = getattr(self, 'categorical_info', None)
        if isinstance(cat_info, dict):
            for key in list(cat_info.keys()):
                val = cat_info[key]
                if torch.is_tensor(val):
                    cat_info[key] = val.to(device)
                elif isinstance(val, list):
                    cat_info[key] = [
                        v.to(device) if torch.is_tensor(v) else v
                        for v in val
                    ]

        return self

    def _move_tree_to_device(self, node, device):
        """Recursively move all tensors in a tree node to *device*."""
        if node is None:
            return

        # Move every tensor value in this node dict
        for key in list(node.keys()):
            val = node[key]
            if torch.is_tensor(val):
                node[key] = val.to(device)

        # Recurse into the leaf model and child nodes
        if node.get('type') == 'leaf':
            model = node.get('model')
            if model is not None and hasattr(model, 'to'):
                model.to(device)
        else:
            self._move_tree_to_device(node.get('left'), device)
            self._move_tree_to_device(node.get('right'), device)

    def tree_copy(self, tree):
        """
        Deep copy a tree structure.
        
        Parameters
        ----------
        tree : dict
            Tree to copy
            
        Returns
        ------- 
        dict
            Copied tree structure
        """
        return copy.deepcopy(tree)

    def _generate_random_projection(self, dim):
        """
        Generate a random unit vector for data projection.
        
        Parameters
        ----------
        dim : int
            Dimension of the projection vector
            
        Returns
        -------
        torch.Tensor
            Random unit vector of shape (dim,)
        """
        projection = torch.randn(dim, device=self.device)
        return projection / torch.norm(projection)

    def _generate_projection_from_M(self, dim, M):
        """
        Generate a projection vector using the covariance matrix M.
        
        This method samples from a multivariate normal distribution with 
        covariance M (typically the AGOP matrix).
        
        Parameters
        ----------
        dim : int
            Dimension of the projection vector
        M : torch.Tensor
            Covariance matrix, either diagonal (1D) or full matrix (2D)
            
        Returns
        -------
        torch.Tensor
            Projection vector of shape (dim,), normalized to unit length
        """
        if M.dim() == 1:  # If M is diagonal
            std_devs = torch.sqrt(M)
            projection = torch.normal(0, std_devs).to(self.device)
        else:  # If M is a full matrix
            # Generate random vector from standard normal distribution
            z = torch.randn(dim, device=self.device)

            try:
                sqrtM = matrix_power(M, 0.5)

                # Transform z to get vector with covariance M
                projection = sqrtM @ z
            except:
                if self.verbose:
                    print(f"Matrix power failed, defaulting to random projection")

                # Fallback to random projection if matrix power fails
                projection = torch.randn(dim, device=self.device)

        # Normalize to unit vector
        return projection / torch.norm(projection)

    def _collect_leaf_nodes(self, node):
        """
        Recursively collect all leaf nodes in a tree.
        
        Parameters
        ----------
        node : dict
            Current tree node
            
        Returns
        -------
        list
            List of all xRFM models at leaf nodes
        """
        if node['type'] == 'leaf':
            return [node]

        left_nodes = self._collect_leaf_nodes(node['left'])
        right_nodes = self._collect_leaf_nodes(node['right'])

        return left_nodes + right_nodes

    def _collect_attr(self, attr_name):
        """
        Collect a specific attribute from all leaf nodes in all trees.
        
        Parameters
        ----------
        attr_name : str
            Name of the attribute to collect from each leaf model
            
        Returns
        -------
        list
            List of attribute values from all leaf models across all trees
        """
        best_agops = []
        for t in self.trees:
            leaf_nodes = self._collect_leaf_nodes(t)
            best_agops += [getattr(node['model'], attr_name) for node in leaf_nodes]
        return best_agops

    def _build_tree_cache(self, tree):
        """
        Construct lookup tables for a tree to support soft routing.

        Parameters
        ----------
        tree : dict
            Root node of the tree to index.

        Returns
        -------
        dict
            Dictionary containing hash tables for leaf models, split metadata,
            and traversal paths per leaf.
        """
        leaf_models = {}
        leaf_paths = {}
        leaf_order = []
        split_directions = {}
        split_thresholds = {}
        split_temp_scalings = {}

        stack = [(tree, [])]
        next_node_id = 0
        next_leaf_id = 0

        while stack:
            node, path = stack.pop()

            if node['type'] == 'leaf':
                leaf_id = next_leaf_id
                next_leaf_id += 1
                leaf_models[leaf_id] = node['model']
                leaf_paths[leaf_id] = tuple(path)
                leaf_order.append(leaf_id)
            else:
                node_id = next_node_id
                next_node_id += 1
                split_directions[node_id] = node['split_direction']
                split_thresholds[node_id] = node['split_point']
                split_temp_scalings[node_id] = node.get('adaptive_temp_scaling', 1.0)

                stack.append((node['right'], path + [(node_id, False)]))
                stack.append((node['left'], path + [(node_id, True)]))

        cache = {
            'leaf_models': leaf_models,
            'leaf_paths': leaf_paths,
            'leaf_order': leaf_order,
            'split_directions': split_directions,
            'split_thresholds': split_thresholds,
            'split_temp_scalings': split_temp_scalings,
        }
        tree['_cache'] = cache
        return cache

    def _ensure_tree_cache(self, tree):
        """
        Ensure lookup tables are available for the provided tree.

        Parameters
        ----------
        tree : dict
            Root node of the tree.

        Returns
        -------
        dict
            Lookup cache for the tree.
        """
        cache = tree.get('_cache')
        if cache is None:
            cache = self._build_tree_cache(tree)
        return cache

    def collect_best_agops(self):
        """
        Collect the best AGOP matrices from all leaf nodes across all trees.
        
        Returns
        -------
        list
            List of AGOP matrices from all leaf models
        """
        return self._collect_attr('agop_best_model')
        # best_agops = []
        # for t in self.trees:
        #     leaf_nodes = self._collect_leaf_nodes(t)
        #     best_agops += [node['model'].agop_best_model for node in leaf_nodes]
        # return best_agops

    def collect_Ms(self):
        """
        Collect the Mahalanobis matrices (M) from all leaf nodes across all trees.
        
        Returns
        -------
        list
            List of M matrices from all leaf models
        """
        return self._collect_attr('M')

    def _average_M_across_leaves(self, tree):
        """
        Average the M parameter across all leaf nodes in a tree.
        
        This method collects the Mahalanobis matrices from all leaf nodes
        and computes their average. This averaged matrix is used to generate
        better projection directions in subsequent iterations.
        
        Parameters
        ----------
        tree : dict
            Tree to analyze
            
        Returns
        -------
        torch.Tensor
            Averaged M parameter, either diagonal (1D) or full matrix (2D)
        """
        leaf_nodes = self._collect_leaf_nodes(tree)
        leaf_models = [node['model'] for node in leaf_nodes]

        # Collect M matrices from all leaf models
        M_matrices = []
        for model in leaf_models:
            if hasattr(model, 'M') and model.M is not None:
                M_matrices.append(model.M)
            else:
                identity = torch.ones(self.data_dim, device=self.device) if model.diag else torch.eye(self.data_dim,
                                                                                                      device=self.device)
                M_matrices.append(identity)

        if M_matrices[0].dim() == 1:  # If M is diagonal
            avg_M = torch.stack(M_matrices).mean(dim=0)
        else:  # If M is a full matrix
            avg_M = torch.stack(M_matrices).mean(dim=0)

        return avg_M

    def _get_balanced_split(self, projections, train_median):
        """
        Construct balanced boolean masks with an optional central overlap.

        The samples are ordered by their projection values. The lowest portion
        goes uniquely to the left leaf, the highest portion uniquely to the right
        leaf, and the middle ``2 * overlap_fraction`` share of samples are added
        to both leaves.
        
        Parameters
        ----------
        projections : torch.Tensor
            Projected values for all samples
        train_median : float
            Median value to split on
            
        Returns
        -------
        tuple
            (left_mask, right_mask) balanced masks
        """
        _ = train_median  # kept for interface compatibility

        n_samples = projections.numel()
        if n_samples == 0:
            raise ValueError("Cannot split an empty projection tensor.")

        _, sorted_indices = torch.sort(projections)

        overlap_count = int(round(2 * self.overlap_fraction * n_samples))
        overlap_count = max(0, min(overlap_count, n_samples))

        remaining = n_samples - overlap_count
        left_unique_count = (remaining + 1) // 2  # ceil division
        right_unique_count = remaining // 2

        overlap_start = left_unique_count
        overlap_end = overlap_start + overlap_count

        left_unique_indices = sorted_indices[:left_unique_count]
        overlap_indices = sorted_indices[overlap_start:overlap_end]
        right_unique_indices = sorted_indices[overlap_end:]

        assert right_unique_indices.numel() == right_unique_count, "Right split size mismatch"

        left_mask = torch.zeros(n_samples, dtype=torch.bool, device=projections.device)
        right_mask = torch.zeros_like(left_mask)

        if left_unique_indices.numel() > 0:
            left_mask[left_unique_indices] = True
        if right_unique_indices.numel() > 0:
            right_mask[right_unique_indices] = True
        if overlap_indices.numel() > 0:
            left_mask[overlap_indices] = True
            right_mask[overlap_indices] = True

        assert left_mask.any() and right_mask.any(), "Each split must contain at least one element"
        assert left_mask.sum() - right_mask.sum() <= 1, "Left and right masks should have the same number of elements"
        return left_mask, right_mask

    def _build_tree(self, X, y, X_val, y_val, train_indices=None, avg_M=None, is_root=False,
                    time_limit_s=None, split_tracker=None, **kwargs):
        """
        Recursively build the tree by splitting data based on random projections.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features
        y : torch.Tensor
            Target values
        X_val : torch.Tensor
            Validation features
        y_val : torch.Tensor
            Validation target values
        avg_M : torch.Tensor, optional
            Averaged M matrix to use for generating projections
        is_final_iter : bool, default=False
            Whether this is the final iteration of tree building
        time_limit_s : float, optional
            Time limit in seconds.
        split_tracker : dict, optional
            Mutable counter tracking the number of splits performed for the current tree.
            
        Returns
        -------
        dict
            A tree node (either a leaf with a model or an internal node with split information)
        """
        start_time = time.time()
        n_samples = X.shape[0]
        if train_indices is None:
            train_indices = torch.arange(n_samples, device=self.device)
        if split_tracker is None:
            split_tracker = {'count': 0}

        # Check terminal conditions
        should_create_leaf = False
        if n_samples <= self.max_leaf_size:
            if self.number_of_splits is None or split_tracker['count'] >= self.number_of_splits:
                should_create_leaf = True

        if should_create_leaf:
            if not is_root:  # refill the validation set if you've split the data before
                if self.verbose:
                    print("Refilling validation set, because at least one split has been made.")
                X, y, X_val, y_val, train_indices = self._refill_val_set(X, y, X_val, y_val, train_indices)

            # Create and fit a xRFM model on this subset
            model = RFM(**self.rfm_params['model'], tuning_metric=self.tuning_metric,
                        categorical_info=self.categorical_info, device=self.device, time_limit_s=time_limit_s,
                        verbose=self.verbose, **self.extra_rfm_params_)

            model.fit((X, y), (X_val, y_val), **self.rfm_params['fit'], callback=self.callback, **kwargs)
            return {'type': 'leaf', 'model': model, 'train_indices': train_indices, 'is_root': is_root}

        split_tracker['count'] += 1

        # Generate projection vector
        if avg_M is not None and self.split_method == 'random_global_agop':
            projection = self._generate_projection_from_M(X.shape[1], avg_M)
        elif self.split_method == 'pca':
            Xb = X - X.mean(dim=0, keepdim=True)
            _, _, Vt = torch.linalg.svd(Xb.T @ Xb,
                                        full_matrices=False)  # do computation on Xb.T @ Xb assuming n >> d
            projection = Vt[0]
        elif self.split_method == 'rf_criterion':
            def _sum_squared_error(values: torch.Tensor) -> torch.Tensor:
                centered = values - values.mean(dim=0, keepdim=True)
                return torch.sum(centered**2).item()

            best_dim = 0
            best_score = float('inf')
            for dim in range(X.shape[1]):
                projections_dim = X[:, dim]
                dim_median = torch.median(projections_dim)
                left_mask_dim, right_mask_dim = self._get_balanced_split(projections_dim, dim_median)
                y_left_dim = y[left_mask_dim]
                y_right_dim = y[right_mask_dim]

                score = _sum_squared_error(y_left_dim) + _sum_squared_error(y_right_dim)
                if score < best_score:
                    best_score = score
                    best_dim = dim

            projection = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
            projection[best_dim] = 1.0
        elif self.split_method == 'random_pca':
            Xb = X - X.mean(dim=0, keepdim=True)
            Xcov = Xb.T @ Xb
            projection = self._generate_projection_from_M(X.shape[1], Xcov)
        elif self.split_method == 'linear':
            XtX = X.T @ X
            beta = torch.linalg.solve(XtX + 1e-6 * torch.eye(X.shape[1], device=self.device), X.T @ y)
            beta = beta.mean(dim=1)  # probably not the best way to do this
            projection = beta / torch.norm(beta)
        elif 'agop_on_subset' in self.split_method:
            if self.verbose:
                print(f"Using {self.split_method} split method")
            sub_time_limit_s = None
            if time_limit_s is not None:
                # spend ~half of the time for fitting agop_on_subset and the other half for fitting the leaves
                n_leaves = 2 ** np.ceil(np.log2(n_samples / self.max_leaf_size))
                sub_time_limit_s = 0.5 * time_limit_s / (n_leaves - 1)
            M = self._get_agop_on_subset(X, y, time_limit_s=sub_time_limit_s)
            if self.split_method == 'top_vector_agop_on_subset':
                projection = get_top_eigenvector(M)
            elif self.split_method == 'random_agop_on_subset':
                projection = self._generate_projection_from_M(X.shape[1], M)
            elif self.split_method == 'top_pc_agop_on_subset':
                sqrtM = matrix_power(M, 0.5)
                XM = X @ sqrtM
                Xb = XM - XM.mean(dim=0, keepdim=True)
                projection = get_top_eigenvector(Xb.T @ Xb)
        elif self.split_method == 'fixed_vector':
            projection = self.fixed_vector
        else:
            projection = self._generate_random_projection(X.shape[1])

        # Project data onto the random direction
        projections = X @ projection

        # Find median as split point
        train_median = torch.median(projections)

        # Compute inter-quartile range to scale the gating temperature adaptively
        q1 = torch.quantile(projections, 0.25)
        q3 = torch.quantile(projections, 0.75)
        iqr = (q3 - q1).clamp_min(1e-6)
        adaptive_temp_scaling = float(iqr.item())

        # Get balanced split for training set to avoid infinite recursion with repeated data
        left_mask, right_mask = self._get_balanced_split(projections, train_median)

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        # Possibly imbalanced split for validation set
        projections_val = X_val @ projection
        left_mask_val = projections_val <= train_median
        right_mask_val = ~left_mask_val

        X_val_left, y_val_left = X_val[left_mask_val], y_val[left_mask_val]
        X_val_right, y_val_right = X_val[right_mask_val], y_val[right_mask_val]

        # Build subtrees
        left_tree = self._build_tree(X_left, y_left, X_val_left, y_val_left,
                                     train_indices=train_indices[left_mask],
                                     avg_M=avg_M,
                                     is_root=False,
                                     split_tracker=split_tracker,
                                     time_limit_s=None if time_limit_s is None
                                     else 0.5 * (time_limit_s - (time.time() - start_time)),
                                     **kwargs)
        right_tree = self._build_tree(X_right, y_right, X_val_right, y_val_right,
                                      train_indices=train_indices[right_mask],
                                      avg_M=avg_M,
                                      is_root=False,
                                      split_tracker=split_tracker,
                                      time_limit_s=None if time_limit_s is None
                                      else time_limit_s - (time.time() - start_time),
                                      **kwargs
                                      )

        return {
            'type': 'split',
            'split_direction': projection,
            'split_point': train_median,
            'left': left_tree,
            'right': right_tree,
            'is_root': is_root,
            'adaptive_temp_scaling': adaptive_temp_scaling,
        }

    def _refill_val_set(self, X, y, X_val, y_val, train_indices):
        """
        Refill the validation set with samples from the training set.
        
        This method ensures that each leaf node has a sufficient validation set
        for proper model tuning. When the validation set becomes too small after
        tree splitting, it transfers samples from the training set to the 
        validation set.
        
        Parameters
        ----------
        X : torch.Tensor
            Training features
        y : torch.Tensor
            Training targets
        X_val : torch.Tensor
            Validation features
        y_val : torch.Tensor
            Validation targets
        train_indices : torch.Tensor
            Indices of training samples in the original dataset
            
        Returns
        -------
        tuple
            Updated (X, y, X_val, y_val, train_indices) with refilled validation set
        """

        if len(X_val) <= self.min_val_size:
            n_orig_val = len(X_val)
            n_orig_train = len(X)

            num_val_to_add = self.min_val_size - len(X_val)
            num_val_to_add = min(num_val_to_add, int(len(X) * self.val_size_frac))
            shuffled_indices = torch.randperm(len(X))
            val_indices = shuffled_indices[:num_val_to_add]
            local_train_indices_to_keep = shuffled_indices[num_val_to_add:]

            X_val = torch.cat([X_val, X[val_indices]])
            y_val = torch.cat([y_val, y[val_indices]])
            X = X[local_train_indices_to_keep]
            y = y[local_train_indices_to_keep]

            train_indices = train_indices[local_train_indices_to_keep]

            assert n_orig_val + num_val_to_add == len(X_val) == len(y_val)
            assert n_orig_train - num_val_to_add == len(X) == len(y)

        return X, y, X_val, y_val, train_indices

    def _build_tree_with_iterations(self, X, y, X_val, y_val, time_limit_s=None, **kwargs):
        """
        Build a tree using multiple iterations, where each iteration uses
        information from the previous iteration's leaf models.
        
        Parameters
        ----------
        X : torch.Tensor
            Input features
        y : torch.Tensor
            Target values
        X_val : torch.Tensor
            Validation features
        y_val : torch.Tensor
            Validation target values
        time_limit_s : float, optional
            Time limit in seconds.
            
        Returns
        -------
        dict
            Final tree structure with the best validation performance
        """
        avg_M = None
        start_time = time.time()

        # First iteration: use random projections
        tree = self._build_tree(X, y, X_val, y_val, avg_M=None, is_root=True,
                                time_limit_s=None if time_limit_s is None else time_limit_s / (1 + self.n_tree_iters),
                                split_tracker={'count': 0})

        # Evaluate the first tree on validation data
        best_val_score = self.score_tree(X_val, y_val, tree)
        best_tree = self.tree_copy(tree)

        val_scores = [best_val_score + 0]

        for iter in tqdm(range(self.n_tree_iters), desc="Iterating tree", disable=not self.verbose):
            if time_limit_s is not None and (iter + 2) / (iter + 1) * (time.time() - start_time) > time_limit_s:
                break  # stop early because we expect to exceed the time limit

            # Later iterations: use averaged M from previous iterations
            avg_M = self._average_M_across_leaves(tree)

            del tree

            # Build new tree with improved projections
            tree = self._build_tree(X, y, X_val, y_val, avg_M=avg_M, is_root=False,
                                    time_limit_s=None if time_limit_s is None
                                    else (time_limit_s - (time.time() - start_time)) / (self.n_tree_iters - iter),
                                    split_tracker={'count': 0},
                                    **kwargs)

            # Evaluate this iteration's tree on validation data
            val_score = self.score_tree(X_val, y_val, tree)
            val_scores.append(val_score)

            if self.maximizing_metric and val_score > best_val_score:
                best_val_score = val_score
                best_tree = self.tree_copy(tree)
            elif not self.maximizing_metric and val_score < best_val_score:
                best_val_score = val_score
                best_tree = self.tree_copy(tree)

        if self.verbose:
            print("==========================Tree iteration results==========================")
            print("Validation scores over tree iterations:", val_scores)
            print("Best validation score:", best_val_score)
            print("==========================================================================")
        return best_tree

    def fit(self, X, y, X_val, y_val, **kwargs):
        """
        Fit the xRFM model to the training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values
        X_val : array-like of shape (n_samples, n_features)
            Validation feature matrix
        y_val : array-like of shape (n_samples,) or (n_samples, n_targets)
            Validation target values
            
        Returns
        -------
        self : object
            Returns self.
        """
        if self.verbose:
            print(f"Fitting xRFM with {self.n_trees} trees and {self.n_tree_iters} iterations per tree")

        if self.n_threads is not None:
            old_n_threads = torch.get_num_threads()
            torch.set_num_threads(self.n_threads)

        # Convert to torch tensors if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if not isinstance(X_val, torch.Tensor):
            X_val = torch.tensor(X_val, dtype=torch.float32, device=self.device)

        X = X.to(self.device)
        X_val = X_val.to(self.device)

        y = torch.as_tensor(y).to(self.device)
        y_val = torch.as_tensor(y_val).to(self.device)
        y_train_and_val = torch.cat([y, y_val], dim=0)

        # automatically determine whether it's classification or regression
        if self.tuning_metric is not None:
            metric = Metric.from_name(self.tuning_metric)
            is_class = not ('reg' in metric.task_types)
            if is_class and y.is_floating_point():
                print(f'Warning: Using floating point y with a classification metric. '
                      f'Assuming that y is already binarized / one-hot encoded.', file=sys.stderr, flush=True)
        else:
            is_class = not y.is_floating_point()
            self.tuning_metric = 'brier' if is_class else 'mse'

        # determine n_classes and convert automatically
        if is_class:
            if y.is_floating_point():
                if len(y.shape) == 1:
                    y = y[:, None]
                assert len(y.shape) == 2

                self.n_classes_ = max(2, y.shape[1])
                self.class_converter_ = ClassificationConverter(mode=self.classification_mode,
                                                                n_classes=self.n_classes_)
            else:
                self.n_classes_ = max(2, y_train_and_val.max().item() + 1)

                self.class_converter_ = ClassificationConverter(mode=self.classification_mode, labels=y,
                                                                n_classes=self.n_classes_)

                y = self.class_converter_.labels_to_numerical(y)
                y_val = self.class_converter_.labels_to_numerical(y_val)

            self.extra_rfm_params_ = dict(class_converter=self.class_converter_)
        else:
            self.n_classes_ = 0
            y = y.float()
            y_val = y_val.float()

            # Ensure y has the right shape
            if len(y.shape) == 1:
                y = y.unsqueeze(-1)
            if len(y_val.shape) == 1:
                y_val = y_val.unsqueeze(-1)
            assert len(y.shape) == 2
            self.extra_rfm_params_ = dict()

        self.data_dim = X.shape[1]

        # Build multiple trees
        self.trees = []
        start_time = time.time()
        has_split = False
        for iter in tqdm(range(self.n_trees), desc="Building trees", disable=not self.verbose):
            if iter > 0 and self.time_limit_s is not None and (iter + 1) / iter * (
                    time.time() - start_time) > self.time_limit_s:
                break
            time_limit_s = None if self.time_limit_s is None else (self.time_limit_s - (time.time() - start_time)) / (
                    self.n_trees - iter)
            if self.n_tree_iters > 0:
                tree = self._build_tree_with_iterations(X, y, X_val, y_val,
                                                        time_limit_s=time_limit_s, **kwargs)
            else:
                tree = self._build_tree(X, y, X_val, y_val, is_root=True, time_limit_s=time_limit_s,
                                        split_tracker={'count': 0}, **kwargs)
            self.trees.append(tree)
            self._ensure_tree_cache(tree)

            if tree['type'] == 'leaf':
                if self.verbose:
                    print("Tree has no split, stopping training")
                break
            has_split = True

        if self.n_threads is not None:
            torch.set_num_threads(old_n_threads)
        
        if has_split and self.use_temperature_tuning:
            self.fit_temperature(X_val, y_val, self.temp_tuning_space)

        return self


    def fit_temperature(self, X_val, y_val, temp_tuning_space):
        """
        Tune split_temperature on the validation set using self.tuning_metric.

        Parameters
        ----------
        X_val : torch.Tensor
            Validation features.
        y_val : torch.Tensor
            Validation targets (potentially converted for classification).
        temp_tuning_space : sequence of floats, optional
            Candidate temperatures to evaluate. A value of 0 corresponds to
            hard routing (split_temperature=None).

        Returns
        -------
        float or None
            Selected split temperature. None denotes hard routing.
        """
        if self.trees is None or len(self.trees) == 0:
            return self.split_temperature

        metric = Metric.from_name(self.tuning_metric)
        if 'agop' in metric.required_quantities or 'topk' in metric.required_quantities:
            raise NotImplementedError(
                f"Temperature tuning does not support metric '{self.tuning_metric}' "
                "because it requires AGOP statistics."
            )

        maximizing = metric.should_maximize
        best_score = float("-inf") if maximizing else float("inf")
        best_temp_attr = self.split_temperature if self.split_temperature is not None else None
        best_temp_value = 0.0 if best_temp_attr is None else float(best_temp_attr)

        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        metric_inputs = {}
        if 'y_true_reg' in metric.required_quantities:
            metric_inputs['y_true_reg'] = y_val
        if 'y_true_class' in metric.required_quantities:
            if not hasattr(self, 'class_converter_'):
                raise ValueError("Classification converter is required for classification metrics.")
            metric_inputs['y_true_class'] = self.class_converter_.numerical_to_labels(y_val)
        if 'samples' in metric.required_quantities:
            metric_inputs['samples'] = X_val

        tuning_results = []

        def _aggregate_predictions(use_soft, proba):
            preds = []
            for tree in self.trees:
                if use_soft:
                    preds.append(self._predict_tree_soft(X_val, tree, proba=proba))
                else:
                    preds.append(self._predict_tree_hard(X_val, tree, proba=proba))
            if len(preds) == 1:
                return preds[0]
            return torch.mean(torch.stack(preds, dim=0), dim=0)

        for temp_candidate in tqdm(temp_tuning_space, desc="Tuning split temperature", disable=not self.verbose):
            temp_candidate = float(temp_candidate)
            if temp_candidate <= 0.0:
                self.split_temperature = None
                use_soft = False
            else:
                self.split_temperature = temp_candidate
                use_soft = True

            if 'y_pred' in metric.required_quantities:
                metric_inputs['y_pred'] = _aggregate_predictions(use_soft=use_soft, proba=False)
            if 'y_pred_proba' in metric.required_quantities:
                metric_inputs['y_pred_proba'] = _aggregate_predictions(use_soft=use_soft, proba=True)

            score = metric.compute(**metric_inputs)
            tuning_results.append((temp_candidate, score))

            is_better = score > best_score if maximizing else score < best_score
            if is_better or (temp_candidate == best_temp_value and score == best_score):
                best_score = score
                best_temp_attr = None if temp_candidate <= 0.0 else temp_candidate

        self.split_temperature = best_temp_attr
        self.best_split_temperature_score_ = best_score
        self.temperature_tuning_results_ = tuning_results

        if self.verbose:
            print(
                f"Selected split_temperature={self.split_temperature if self.split_temperature is not None else 0.0} "
                f"based on validation {self.tuning_metric}={best_score:.6f}"
            )

        return self.split_temperature


    def score(self, samples, targets):
        """
        Return the score of the model on the given samples and targets on self.tuning_metric.

        Parameters
        ----------
        samples : array-like of shape (n_samples, n_features)
            Test samples
        targets : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for samples

        Returns
        -------
        float
            Score of the model on the given samples and targets
        """

        metric = Metric.from_name(self.tuning_metric)
        assert len(targets.shape) == 2 and targets.shape[1] >= 2
        kwargs = dict(y_true_reg=targets)
        if 'y_pred' in metric.required_quantities:
            kwargs['y_pred'] = self.predict(samples.to(self.device)).to(targets.device)
        if 'y_pred_proba' in metric.required_quantities:
            kwargs['y_pred_proba'] = self.predict_proba(samples.to(self.device)).to(targets.device)
        if 'y_true_class' in metric.required_quantities:
            kwargs['y_true_class'] = self.class_converter_.numerical_to_labels(targets)

        return metric.compute(**kwargs)


    def score_tree(self, samples, targets, tree):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        samples : array-like of shape (n_samples, n_features)
            Test samples
        targets : array-like of shape (n_samples,) or (n_samples, n_targets)
            True values for samples
        tree : dict
            Tree to use for prediction

        Returns
        -------
        float
            Metric value for self.tuning_metric
        """

        metric = Metric.from_name(self.tuning_metric)
        assert len(targets.shape) == 2 and targets.shape[1] >= 2
        kwargs = dict(y_true_reg=targets)
        if 'y_pred' in metric.required_quantities:
            kwargs['y_pred'] = self._predict_tree(samples.to(self.device), tree).to(targets.device)
        if 'y_pred_proba' in metric.required_quantities:
            kwargs['y_pred_proba'] = self._predict_tree(samples.to(self.device), tree, proba=True).to(targets.device)
        if 'y_true_class' in metric.required_quantities:
            kwargs['y_true_class'] = self.class_converter_.numerical_to_labels(targets)

        return metric.compute(**kwargs)


    def predict(self, X):
        """
        Predict using the xRFM model by averaging predictions across all trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        array-like
            Returns predicted values
        """
        if self.trees is None:
            raise ValueError("Model has not been fitted yet.")

        if self.n_threads is not None:
            old_n_threads = torch.get_num_threads()
            torch.set_num_threads(self.n_threads)

        # Convert to torch tensor if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        X = X.to(self.device)

        all_predictions = []

        # Get predictions from each tree
        for tree in self.trees:
            tree_predictions = self._predict_tree(X, tree)
            all_predictions.append(tree_predictions)

        # Average predictions across trees
        pred = torch.mean(torch.stack(all_predictions), dim=0)

        if self.n_threads is not None:
            torch.set_num_threads(old_n_threads)

        if self.n_classes_ > 0:
            return self.class_converter_.numerical_to_labels(pred).cpu().numpy()
        else:
            return pred.cpu().numpy()


    def predict_proba(self, X):
        """
        Predict class probabilities by averaging across all trees.
        Only usable if the underlying xRFM models were fitted for classification.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict

        Returns
        -------
        array-like
            Returns predicted probabilities
        """
        if self.trees is None:
            raise ValueError("Model has not been fitted yet.")

        if self.n_threads is not None:
            old_n_threads = torch.get_num_threads()
            torch.set_num_threads(self.n_threads)

        # Convert to torch tensor if needed
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        all_probas = []
        for tree in self.trees:
            tree_probas = self._predict_tree(X, tree, proba=True)
            all_probas.append(tree_probas)

        result = torch.mean(torch.stack(all_probas), dim=0)

        if self.n_threads is not None:
            torch.set_num_threads(old_n_threads)

        return result.cpu().numpy()


    def _predict_tree(self, X, tree, proba=False):
        """
        Dispatch tree prediction to hard or soft routing depending on configuration.
        """
        if not self.split_temperature:
            # print("Using hard routing for tree prediction")
            return self._predict_tree_hard(X, tree, proba=proba)
        # print("Using soft routing for tree prediction")
        return self._predict_tree_soft(X, tree, proba=proba)

    def _predict_tree_hard(self, X, tree, proba=False):
        """
        Make predictions for all samples using a single tree.

        Parameters
        ----------
        X : torch.Tensor
            Input features
        tree : dict
            Tree to use for prediction

        Returns
        -------
        torch.Tensor
            Predictions for all samples
        """

        X_leaf_groups, X_leaf_group_indices, leaf_nodes = self._get_leaf_groups_and_models_on_samples(X, tree)
        predictions = []
        for X_leaf, leaf_node in zip(X_leaf_groups, leaf_nodes):
            if proba:
                preds = leaf_node['model'].predict_proba(X_leaf)
            else:
                preds = leaf_node['model'].predict(X_leaf)
            predictions.append(preds)

        def reorder_tensor(original_tensor, order_tensor):
            """
            Args:
                original_tensor: The tensor to be reordered
                order_tensor: A tensor containing the new positions for each element

            Returns:
                The reordered tensor
            """
            # Sort the indices based on the order tensor
            # This gives us the inverse permutation needed
            _, sorted_indices = torch.sort(order_tensor)

            # Use the sorted indices to reorder the original tensor
            return original_tensor[sorted_indices]

        order = torch.cat(X_leaf_group_indices, dim=0)
        return reorder_tensor(torch.cat(predictions, dim=0), order)

    def _predict_tree_soft(self, X, tree, proba=False):
        """
        Perform soft routing over all leaves and aggregate predictions.

        Parameters
        ----------
        X : torch.Tensor
            Input features
        tree : dict
            Tree to use for prediction

        Returns
        -------
        torch.Tensor
            Aggregated predictions for all samples
        """
        cache = self._ensure_tree_cache(tree)
        
        leaf_models = cache['leaf_models']
        leaf_paths = cache['leaf_paths']
        leaf_order = cache['leaf_order']
        split_directions = cache['split_directions']
        split_thresholds = cache['split_thresholds']

        if not leaf_order:
            # Tree reduced to a single leaf
            sole_leaf = next(iter(leaf_models.values()))
            return sole_leaf.predict_proba(X) if proba else sole_leaf.predict(X)

        temperature_constant = self.split_temperature
        if temperature_constant <= 0:
            raise ValueError("split_temperature must be positive.")

        # Compute logits for each split node once for all samples
        node_logits = {}
        temp_scalings = cache.get('split_temp_scalings', {})
        for node_id, direction in split_directions.items():
            split_point = split_thresholds[node_id]
            logits = (X @ direction) - split_point
            node_scale = temp_scalings.get(node_id, 1.0)
            node_temperature = temperature_constant * node_scale
            node_logits[node_id] = logits / node_temperature
        
        # Aggregate log probabilities for each leaf path
        log_leaf_probs = []
        for leaf_id in leaf_order:
            path = leaf_paths[leaf_id]
            log_prob = torch.zeros(X.shape[0], device=X.device)
            for node_id, took_left in path:
                logits = node_logits[node_id]
                if took_left:
                    log_prob = log_prob + F.logsigmoid(-logits)
                else:
                    log_prob = log_prob + F.logsigmoid(logits)
            log_leaf_probs.append(log_prob)

        leaf_log_prob_tensor = torch.clamp(torch.stack(log_leaf_probs, dim=1), min=-50.0)  # (n_samples, n_leaves)
        max_log_prob = torch.max(leaf_log_prob_tensor, dim=1, keepdim=True).values
        stable_log_probs = leaf_log_prob_tensor - max_log_prob
        leaf_probs = torch.exp(stable_log_probs)
        
        normalizer = torch.clamp(
            leaf_probs.sum(dim=1, keepdim=True),
            min=torch.finfo(leaf_probs.dtype).tiny
        )
        weights = leaf_probs / normalizer

        sorted_weights, sorted_indices = torch.sort(weights, dim=1, descending=True)
        n_leaves = weights.shape[1]

        cumulative = torch.cumsum(sorted_weights, dim=1)
        keep_counts = torch.sum(cumulative < self.keep_weight_frac_in_predict, dim=1)

        max_allowed = min(self.max_leaf_count_in_ensemble, n_leaves) - 1
        max_allowed = max(max_allowed, 0)
        keep_counts = torch.clamp(keep_counts, max=max_allowed)
        position_range = torch.arange(n_leaves, device=weights.device).view(1, -1).expand_as(weights)
        keep_mask_sorted = position_range <= keep_counts.unsqueeze(1)
        active_mask = torch.zeros_like(weights, dtype=torch.bool)
        active_mask.scatter_(1, sorted_indices, keep_mask_sorted)

        weights = torch.where(active_mask, weights, torch.zeros_like(weights))
        renorm = torch.clamp(weights.sum(dim=1, keepdim=True), min=torch.finfo(weights.dtype).tiny)
        weights = weights / renorm

        aggregated = None
        expected_dim = None
        n_samples = X.shape[0]

        for leaf_idx, leaf_id in enumerate(leaf_order):
            sample_indices = torch.nonzero(active_mask[:, leaf_idx], as_tuple=False).squeeze(1)
            if sample_indices.numel() == 0:
                continue

            model = leaf_models[leaf_id]
            X_subset = X[sample_indices]
            preds = model.predict_proba(X_subset) if proba else model.predict(X_subset)
            preds = torch.as_tensor(preds, device=weights.device)
            if preds.dim() == 1:
                preds = preds.unsqueeze(-1)
            preds = preds.to(dtype=weights.dtype)

            if aggregated is None:
                expected_dim = preds.shape[1]
                aggregated = torch.zeros((n_samples, expected_dim), device=weights.device, dtype=preds.dtype)
            elif preds.shape[1] != expected_dim:
                raise ValueError("Leaf predictions have inconsistent output dimensions.")

            leaf_weights = weights[sample_indices, leaf_idx].unsqueeze(-1)
            aggregated[sample_indices] += leaf_weights * preds

        return aggregated



    def load_state_dict(self, state_dict, X_train):
        """
        Load model state from a state dictionary.

        This method reconstructs the model from saved parameters, including
        the tree structure and leaf model parameters. The training data is
        needed to set the centers for each leaf model.

        Parameters
        ----------
        state_dict : dict
            Dictionary containing model parameters from get_state_dict()
        X_train : torch.Tensor
            Training data used to set leaf model centers
        """
        self.rfm_params = state_dict['rfm_params']
        self.categorical_info = state_dict['categorical_info']
        self.n_classes_ = state_dict['n_classes']
        self.extra_rfm_params_ = state_dict['extra_rfm_params_']
        self.solver = state_dict.get('solver', None)

        if self.n_classes_ > 0:
            self.classification_mode = state_dict['classification_mode']
            self.class_converter_ = ClassificationConverter(mode=self.classification_mode,
                                                            n_classes=self.n_classes_,
                                                            init_from_params=True)
            self.class_converter_._prior = state_dict['class_converter']['_prior']
            self.class_converter_._C = state_dict['class_converter']['_C']
            self.class_converter_._invA = state_dict['class_converter']['_invA']
            self.class_converter_._numerical_type = state_dict['class_converter']['_numerical_type']
            self.extra_rfm_params_['class_converter'] = self.class_converter_

        self._build_leaf_models_from_param_trees(state_dict['param_trees'])

        # set centers for leaf models
        for tree in self.trees:
            assert tree['is_root']
            leaf_nodes = self._collect_leaf_nodes(tree)
            for leaf_node in leaf_nodes:
                leaf_model = leaf_node['model']
                leaf_center_indices = leaf_node['train_indices']
                leaf_model.centers = X_train[leaf_center_indices]
        return


    def _build_leaf_models_from_param_trees(self, param_trees):
        """
        Build leaf models from parameter trees during model loading.

        This method reconstructs the tree structure and instantiates RFM models
        at each leaf node using the saved parameters. It traverses the tree
        structure and sets the model attributes at leaf nodes.

        Parameters
        ----------
        param_trees : list
            List of parameter trees from the state dictionary
        """
        self.trees = []
        def set_leaf_model_single_tree(tree):
            if tree['type'] == 'leaf':
                leaf_model = RFM(**self.rfm_params['model'],
                                 categorical_info=self.categorical_info,
                                 device=self.device, verbose=self.verbose, **self.extra_rfm_params_)
                leaf_model.kernel_obj.bandwidth = tree['bandwidth']
                leaf_model.weights = tree['weights']
                leaf_model.M = tree['M']
                leaf_model.sqrtM = tree['sqrtM']
                tree['model'] = leaf_model
                return tree
            else:
                tree['left'] = set_leaf_model_single_tree(tree['left'])
                tree['right'] = set_leaf_model_single_tree(tree['right'])
                tree.setdefault('adaptive_temp_scaling', 1.0)
                return tree

        for param_tree in param_trees:
            tree = set_leaf_model_single_tree(param_tree)
            self.trees.append(tree)
            self._ensure_tree_cache(tree)

        return


    def get_state_dict(self):
        """
        Get the state dictionary containing all model parameters for serialization.

        The state dictionary contains the tree structure and all parameters needed
        to reconstruct the model, including individual weights, M/sqrtM matrices,
        and bandwidths for each leaf model. This enables model saving and loading.

        Returns
        -------
        dict
            State dictionary with keys:
            - 'rfm_params': RFM model parameters
            - 'categorical_info': Categorical feature information
            - 'param_trees': List of parameter trees containing leaf model parameters
        """
        param_trees = []
        for tree in self.trees:
            param_trees.append(get_param_tree(tree, is_root=True))
        state_dict = {
            'rfm_params': self.rfm_params,
            'categorical_info': self.categorical_info,
            'param_trees': param_trees,
            'n_classes': self.n_classes_,
        }

        if 'solver' in self.rfm_params['fit']:
            state_dict['solver'] = self.rfm_params['fit']['solver']
        if 'solver' in self.rfm_params['model']:
            state_dict['solver'] = self.rfm_params['model']['solver']
        
        clean_extra_rfm_params = self.extra_rfm_params_.copy()
        if self.n_classes_ > 0:
            state_dict['classification_mode'] = self.classification_mode
            state_dict['class_converter'] = {
                '_prior': self.class_converter_._prior,
                '_C': self.class_converter_._C,
                '_invA': self.class_converter_._invA,
                '_numerical_type': self.class_converter_._numerical_type
            }
            clean_extra_rfm_params.pop('class_converter')
        state_dict['extra_rfm_params_'] = clean_extra_rfm_params
        return state_dict


    def _get_agop_on_subset(self, X, y, subset_size=50_000, time_limit_s=None, max_subset_size_for_split_rfm=60_000):
        """

        This method fits a base RFM model on a subset of the data to compute the AGOP matrix,
        whose eigenvectors are used to generate projection direction for data splitting.

        Parameters
        ----------
        X : torch.Tensor
            Input features of shape (n_samples, n_features)
        y : torch.Tensor
            Target values of shape (n_samples, n_targets)
        subset_size : int, default=50000
            Maximum size of the subset to use for AGOP computation
        max_subset_size_for_split_rfm : int, default=60000
            Maximum size of the subset to use for AGOP computation

        Returns
        -------
        torch.Tensor
            AGOP matrix of shape (n_features, n_features)
        """
        model = RFM(**self.default_rfm_params['model'], device=self.device, time_limit_s=time_limit_s,
                    verbose=self.verbose, **self.extra_rfm_params_)

        base_subset_size = int(subset_size)
        scaled_subset_size = max(int(base_subset_size * memory_scaling_factor(self.device, quadratic=True)), 1)
        subset_size = min(scaled_subset_size, len(X), max_subset_size_for_split_rfm)
        subset_train_size = max(int(subset_size * 0.95), 1)  # 95/5 split, probably won't need the val data.

        subset_indices = torch.randperm(len(X))
        subset_train_indices = subset_indices[:subset_train_size]
        subset_val_indices = subset_indices[subset_train_size:subset_size]

        X_train = X[subset_train_indices]
        y_train = y[subset_train_indices]
        X_val = X[subset_val_indices]
        y_val = y[subset_val_indices]

        if self.verbose:
            print("Getting AGOP on subset")
            print("X_train", X_train.shape, "y_train", y_train.shape, "X_val", X_val.shape, "y_val", y_val.shape)

        model.fit((X_train, y_train), (X_val, y_val), **self.default_rfm_params['fit'])
        agop = model.agop_best_model
        return agop


    def _get_leaf_groups_and_models_on_samples(self, X, tree):
        """
        Assign samples to leaf nodes and return grouped data with corresponding models.

        This method traverses the tree to determine which leaf node each sample
        belongs to, then groups the samples by leaf and returns the corresponding
        models for making predictions.

        Parameters
        ----------
        X : torch.Tensor
            Input data matrix of shape (n_samples, n_features)
        tree : dict
            Tree structure with split directions and leaf models

        Returns
        -------
        X_leaf_groups : list of torch.Tensor
            List of data tensors, one for each leaf node containing the samples
            that belong to that leaf
        X_leaf_group_indices : list of torch.Tensor
            List of tensors containing the original indices of samples in each
            leaf group, used for reordering predictions
        leaf_nodes : list of dict
            List of leaf node dictionaries containing the trained models
        """
        # Initialize results lists
        X_leaf_groups = []
        X_leaf_group_indices = []
        leaf_nodes = []

        # Initialize stack with the root node and all sample indices
        sample_indices = torch.arange(X.shape[0], device=self.device)
        stack = [(X, sample_indices, tree)]

        # Iterative traversal of the tree
        while stack:
            current_X, current_indices, current_node = stack.pop()

            # If we've reached a leaf node, store the results
            if current_node['type'] == 'leaf':
                X_leaf_groups.append(current_X)
                X_leaf_group_indices.append(current_indices)
                leaf_nodes.append(current_node)
                continue

            # Compute projections for all samples in current_X
            projections = current_X @ current_node['split_direction']

            # Split samples based on projection values
            left_mask = projections <= current_node['split_point']
            right_mask = ~left_mask

            # Add right child to stack (will be processed first since we're using pop())
            if right_mask.sum() > 0:
                stack.append((
                    current_X[right_mask],
                    current_indices[right_mask],
                    current_node['right']
                ))

            # Add left child to stack
            if left_mask.sum() > 0:
                stack.append((
                    current_X[left_mask],
                    current_indices[left_mask],
                    current_node['left']
                ))

        return X_leaf_groups, X_leaf_group_indices, leaf_nodes


    def _get_tree_grads_hard(self, X, tree):
        """
        Compute gradients for a single tree under hard routing.
        """
        X_leaf_groups, X_leaf_group_indices, leaf_nodes = self._get_leaf_groups_and_models_on_samples(X, tree)
        grads = []
        for X_leaf, leaf_node in zip(X_leaf_groups, leaf_nodes):
            leaf_grads = leaf_node['model'].get_grads(X_leaf.contiguous())
            grads.append(leaf_grads.to(X.device))

        order = torch.cat(X_leaf_group_indices, dim=0)
        stacked = torch.cat(grads, dim=0)
        _, sorted_indices = torch.sort(order)
        return stacked[sorted_indices]

    def get_grads(self, X):
        """
        Compute input gradients of the ensemble prediction using hard routing.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples where gradients should be evaluated.

        Returns
        -------
        torch.Tensor
            Gradient tensor with shape (n_samples, n_outputs, n_features).
        """
        if self.trees is None:
            raise ValueError("Model has not been fitted yet.")
        if self.split_temperature:
            raise NotImplementedError("Gradient computation for soft routing is not supported.")

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device)

        per_tree_grads = [self._get_tree_grads_hard(X, tree) for tree in self.trees]
        if len(per_tree_grads) == 1:
            return per_tree_grads[0]
        return torch.mean(torch.stack(per_tree_grads, dim=0), dim=0)
