import time
from typing import Union

import torch
import numpy as np
from tqdm.contrib import tenumerate

from .class_conversion import ClassificationConverter
from .eigenpro import KernelModel
from .utils import device_from_str

from .kernels import (
    Kernel,
    LaplaceKernel,
    ProductLaplaceKernel,
    SumPowerLaplaceKernel,
    LightLaplaceKernel,
    LpqLaplaceKernel,
    KermacProductLaplaceKernel,
    KermacLpqLaplaceKernel,
    KermacSumPowerLaplaceKernel,
    kermac,
)
from .metrics import Metrics, Metric
from .utils import matrix_power
from .gpu_utils import with_env_var, get_gpu_memory_bytes, memory_scaling_factor


class RFM(torch.nn.Module):
    """
    Recursive Feature Machine (RFM) - A kernel-based learning algorithm with iterative feature transformation.

    Parameters
    ----------
    kernel : Union[Kernel, str]
        Kernel function specification. Can be a Kernel object or string identifier.
    iters : int, default=5
        Number of RFM iterations for feature learning
    bandwidth : float, default=10.0
        Kernel bandwidth parameter (used with string kernel specification)
    exponent : float, default=1.0
        Kernel exponent parameter (used with string kernel specification)
    bandwidth_mode : str, default='constant'
        Bandwidth adaptation: 'constant' or 'adaptive'
    agop_power : float, default=0.5
        Power for matrix square root in AGOP computation
    device : str or torch.device, optional
        Computation device. Auto-selects if None.
    diag : bool, default=False
        Whether to use diagonal Mahalanobis matrix
    verbose : bool, default=True
        Whether to print progress information
    mem_gb : float, optional
        Memory limit in GB for computation
    tuning_metric : str, default='mse'
        Metric for model selection and early stopping
    categorical_info : dict, optional
        Configuration for categorical features
    fast_categorical : bool, default=True
        Whether to use optimized categorical handling
    
    Attributes
    ----------
    kernel_obj : Kernel
        The kernel object used for computations
    M : torch.Tensor
        Learned Mahalanobis matrix for feature transformation
    sqrtM : torch.Tensor
        Square root of Mahalanobis matrix (if used)
    centers : torch.Tensor
        Training centers (support vectors)
    weights : torch.Tensor
        Kernel regression coefficients
    best_iter : int
        Iteration achieving best validation performance
    
    Examples
    --------
    >>> from xrfm.rfm_src import RFM
    >>> import torch
    >>> 
    >>> # Basic usage
    >>> model = RFM(kernel='laplace', bandwidth=1.0, iters=3)
    >>> X_train, y_train = torch.randn(100, 10), torch.randn(100, 1)
    >>> X_val, y_val = torch.randn(20, 10), torch.randn(20, 1)
    >>> model.fit((X_train, y_train), (X_val, y_val))
    >>> predictions = model.predict(X_val)
    >>> 
    >>> # Classification example
    >>> model = RFM(kernel='laplace', tuning_metric='accuracy')
    >>> y_class = torch.randint(0, 2, (100, 1)).float()
    >>> model.fit((X_train, y_class), (X_val, y_val))
    >>> probabilities = model.predict_proba(X_val)
    """

    def __init__(
        self,
        kernel: Union[Kernel, str],
        iters=5,
        bandwidth=10.0,
        exponent=1.0,
        norm_p=None,
        bandwidth_mode="constant",
        agop_power=0.5,
        device=None,
        diag=False,
        verbose=True,
        mem_gb=None,
        tuning_metric="mse",
        categorical_info=None,
        fast_categorical=False,
        class_converter=None,
        time_limit_s=None,
        solver="solve",
        # Kernel-specific kwargs (used by SumPowerLaplace kernels; accepted here for config compatibility)
        const_mix: float = 0.0,
        power: int = 2,
        eps: float | None = None,
    ):
        """
        Parameters
        ----------
        kernel : Union[Kernel, str]
            Kernel function to use. Can be either:
            - A Kernel object (LaplaceKernel, ProductLaplaceKernel, etc.)
            - A string: 'laplace'/'l2', 'l2_high_dim'/'l2_light', 'product_laplace'/'l1', 'sum_power_laplace'/'l1_power'
            
        iters : int, default=5
            Number of iterations for the RFM algorithm. Each iteration refines the
            Mahalanobis matrix M through AGOP computation.
            
        bandwidth : float, default=10.0
            Kernel bandwidth parameter. Used when kernel is specified as string.
            Controls the width of the kernel function.
            
        exponent : float, default=1.0
            Kernel exponent parameter. Used when kernel is specified as string.
            Controls the shape of the kernel function.
            
        bandwidth_mode : str, default='constant'
            Bandwidth adaptation mode. Options:
            - 'constant': Fixed bandwidth throughout training
            - 'adaptive': Bandwidth adapts during training
            
        agop_power : float, default=0.5
            Power for the matrix square root in AGOP computation.
            Controls the strength of the feature transformation.
            
        device : str or torch.device, optional
            Device for computation ('cpu', 'cuda', or torch.device object).
            If None, automatically selects GPU if available, otherwise CPU.
            
        diag : bool, default=False
            Whether to use diagonal Mahalanobis matrix M. If True, only diagonal
            elements are learned, reducing computational complexity.
            
        verbose : bool, default=True
            Whether to print training progress and diagnostic information.
            
        mem_gb : float, optional
            Memory limit in GB for AGOP/EigenPro computation. If None, automatically
            determined based on available GPU memory (with 1GB safety margin).
            
        tuning_metric : str, default='mse'
            Metric for model selection and early stopping. Options:
            - 'mse': Mean squared error (for regression)
            - 'rmse': Root mean squared error (for regression)
            - 'mae': Mean absolute error (for regression)
            - 'accuracy': Classification accuracy
            - 'auc': Area under ROC curve
            - 'f1': F1 score
            - 'top_agop_vector_auc': AUC using top AGOP eigenvector
            - 'top_agop_vector_pearson_r': Pearson correlation with top AGOP eigenvector
            - 'top_agop_vectors_ols_auc': AUC using OLS on top AGOP eigenvectors
            
        categorical_info : dict, optional
            Information for handling categorical features. Should contain:
            - 'numerical_indices': Indices of numerical features
            - 'categorical_indices': List of indices for each categorical feature
            - 'categorical_vectors': Encoding vectors for categorical features
            
        fast_categorical : bool, default=True
            Whether to use optimized categorical feature handling.
            Only applies to ProductLaplaceKernel.

        class_converter : ClassificationConverter, optional, default=None
            A classification converter for converting between numerical representations predicted by the model
            and classification probabilities or thresholded predictions used by the metrics.
            Only needed for classification.

        time_limit_s : float, optional
            If specified as a float, imposes a time limit (in seconds) on the fitting process.
            RFM will try to fit fewer iterations if that is estimated to be needed to stay below the time limit,
            however, it will always fit at least one iteration.
        
        Attributes
        ----------
        kernel_obj : Kernel
            The kernel object used for computations
        M : torch.Tensor
            Mahalanobis matrix for feature space transformation
        sqrtM : torch.Tensor
            Square root of Mahalanobis matrix (if used by kernel)
        centers : torch.Tensor
            Training centers (support vectors) for kernel regression
        weights : torch.Tensor
            Alpha coefficients for kernel regression
        best_iter : int
            Iteration number that achieved the best validation performance
        
        Examples
        --------
        >>> from xrfm.rfm_src import RFM
        >>> from xrfm.rfm_src.kernels import LaplaceKernel
        >>> 
        >>> # Using string kernel specification
        >>> model = RFM(kernel='laplace', bandwidth=1.0, iters=3)
        >>> 
        >>> # Using explicit kernel object
        >>> kernel = LaplaceKernel(bandwidth=1.0, exponent=1.2)
        >>> model = RFM(kernel=kernel, device='cuda', tuning_metric='accuracy')
        >>> 
        >>> # For categorical data
        >>> categorical_info = {
        ...     'numerical_indices': torch.tensor([0, 1, 2]),
        ...     'categorical_indices': [torch.tensor([3, 4])],
        ...     'categorical_vectors': [torch.eye(2)]
        ... }
        >>> model = RFM(kernel='product_laplace', categorical_info=categorical_info)
        """
        super().__init__()
        if isinstance(kernel, str):
            kernel = self.kernel_from_str(
                kernel,
                bandwidth=bandwidth,
                exponent=exponent,
                norm_p=norm_p,
                device=device,
                const_mix=const_mix,
                power=power,
                eps=eps,
            )
        self.kernel_obj = kernel
        self.agop_power = agop_power
        self.M = None
        self.sqrtM = None
        self.iters = iters
        self.diag = diag # if True, Mahalanobis matrix M will be diagonal
        self.device = device_from_str(device)
        self.agop_power = 0.5 # power for root of agop
        self.max_lstsq_size = 70_000 # max number of points to use for direct solve
        self.bandwidth_mode = bandwidth_mode
        self.proba_beta = 500
        self.verbose = verbose
        self.tuning_metric = tuning_metric
        self.use_sqrtM = self.kernel_obj.use_sqrtM
        self.class_converter = class_converter
        self.time_limit_s = time_limit_s
        self.solver = solver

        if categorical_info is not None and fast_categorical:
            self.set_categorical_indices(**categorical_info)

        if mem_gb is not None:
            self.mem_gb = mem_gb
        elif self.device.type == "cuda":
            # find GPU memory in GB, keeping aside 1GB for safety
            self.mem_gb = torch.cuda.get_device_properties(self.device).total_memory//1024**3 - 1 
        else:
            self.mem_gb = 8
        
    def to(self, device, **kwargs):
        """Move all internal tensors and state to *device*.

        Parameters
        ----------
        device : str or torch.device
            Target device ('cpu', 'cuda', etc.)

        Returns
        -------
        self
        """
        device = device_from_str(device)
        self.device = device

        # Core model tensors
        for attr in ("centers", "weights", "M", "sqrtM"):
            val = getattr(self, attr, None)
            if val is not None and torch.is_tensor(val):
                setattr(self, attr, val.to(device))

        # Kernel object categorical tensors
        kernel = getattr(self, "kernel_obj", None)
        if kernel is not None:
            for attr in ("numerical_indices",):
                val = getattr(kernel, attr, None)
                if val is not None and torch.is_tensor(val):
                    setattr(kernel, attr, val.to(device))
            for list_attr in ("categorical_indices", "categorical_vectors"):
                lst = getattr(kernel, list_attr, None)
                if lst is not None:
                    for i, val in enumerate(lst):
                        if torch.is_tensor(val):
                            lst[i] = val.to(device)

        # Let nn.Module.to() handle any registered parameters/buffers
        return super().to(device, **kwargs)

    def kernel(self, x, z):
        """
        Compute kernel matrix between two sets of points.

        This method delegates to the kernel object's get_kernel_matrix method,
        applying the learned Mahalanobis matrix transformation.

        Parameters
        ----------
        x : torch.Tensor
            First set of points of shape (n_x, d)
        z : torch.Tensor
            Second set of points of shape (n_z, d)

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape (n_x, n_z) where entry (i,j) is k(x[i], z[j])
        """
        return self.kernel_obj.get_kernel_matrix(x, z, self.sqrtM if self.use_sqrtM else self.M)

    def kernel_from_str(
        self,
        kernel_str,
        bandwidth,
        exponent,
        norm_p=2.0,
        device=None,
        const_mix: float = 0.0,
        power: int = 2,
        eps: float | None = None,
    ):
        """
        Create kernel object from string specification.
        
        Parameters
        ----------
        kernel_str : str
            Kernel type specification. Supported options:
            - 'laplace', 'l2': Standard Laplace kernel
            - 'l2_high_dim', 'l2_light': Lightweight Laplace kernel for high dimensions
            - 'product_laplace', 'l1': Product Laplace kernel for categorical features
            - 'lpq': Mixed L^p/L^q Laplace kernel
            - 'sum_power_laplace', 'l1_power': Sum of power Laplace kernel
            
        bandwidth : float
            Kernel bandwidth parameter
            
        exponent : float
            Kernel exponent parameter
            
        Returns
        -------
        Kernel
            Instantiated kernel object
            
        Raises
        ------
        ValueError
            If kernel_str is not recognized
        """
        def _should_use_kermac(dev):
            if kermac is None:
                return False
            if not torch.cuda.is_available():
                return False
            if dev is None:
                return True
            try:
                return torch.device(dev).type == 'cuda'
            except (TypeError, RuntimeError, ValueError):
                return False

        use_kermac = _should_use_kermac(device)

        if kernel_str in ['laplace', 'l2']:
            return LaplaceKernel(bandwidth=bandwidth, exponent=exponent)
        elif kernel_str in ['l2_high_dim', 'l2_light']:
            return LightLaplaceKernel(bandwidth=bandwidth, exponent=exponent)
        elif kernel_str in ["sum_power_laplace", "kermac_sum_power_laplace"]:
            eps_val = 1e-10 if eps is None else eps
            if use_kermac:
                return KermacSumPowerLaplaceKernel(
                    bandwidth=bandwidth,
                    exponent=exponent,
                    eps=eps_val,
                    const_mix=const_mix,
                    power=power,
                )
            return SumPowerLaplaceKernel(
                bandwidth=bandwidth,
                exponent=exponent,
                eps=eps_val,
                const_mix=const_mix,
                power=power,
            )
        elif kernel_str == 'l1_legacy':
            return ProductLaplaceKernel(bandwidth=bandwidth, exponent=exponent)
        elif kernel_str in ['product_laplace', 'l1', 'kermac_product_laplace', 'l1_kermac']:
            if use_kermac:
                return KermacProductLaplaceKernel(bandwidth=bandwidth, exponent=exponent)
            return ProductLaplaceKernel(bandwidth=bandwidth, exponent=exponent)
        elif kernel_str == 'lpq_legacy':
            return LpqLaplaceKernel(bandwidth=bandwidth, p=norm_p, q=exponent)
        elif kernel_str in ['lpq', 'kermac_lpq_laplace', 'lpq_kermac']:
            if use_kermac:
                return KermacLpqLaplaceKernel(bandwidth=bandwidth, q=exponent, p=norm_p)
            return LpqLaplaceKernel(bandwidth=bandwidth, p=norm_p, q=exponent)
        else:
            raise ValueError(f"Invalid kernel: {kernel_str}")
        
    def update_M(self, samples):
        """
        Update the Mahalanobis matrix M using AGOP on a batch of samples.
        
        Parameters
        ----------
        samples : torch.Tensor
            Input samples of shape (n_samples, n_features)
            
        Returns
        -------
        torch.Tensor
            AGOP matrix of shape (n_features, n_features) or (n_features,) if diagonal
        """
        samples = samples.to(self.device)
        self.centers = self.centers.to(self.device)
        
        if self.M is None:
            if self.diag:
                self.M = torch.ones(samples.shape[-1], device=samples.device, dtype=samples.dtype)
            else:
                self.M = torch.eye(samples.shape[-1], device=samples.device, dtype=samples.dtype)

        if self.use_sqrtM and self.sqrtM is None:
            if self.diag:
                self.sqrtM = torch.ones(samples.shape[-1], device=samples.device, dtype=samples.dtype)
            else:
                self.sqrtM = torch.eye(samples.shape[-1], device=samples.device, dtype=samples.dtype)

        agop_func = self.kernel_obj.get_agop_diag if self.diag else self.kernel_obj.get_agop
        agop = agop_func(x=self.centers, z=samples, coefs=self.weights.t(), mat=self.sqrtM if self.use_sqrtM else self.M, center_grads=self.center_grads)
        return agop
    
    def reset_adaptive_bandwidth(self):
        """
        Reset the adaptive bandwidth mechanism in the kernel.
        
        This method is called when using adaptive bandwidth mode to reset
        the bandwidth adaptation state at the beginning of each training round.
        """
        self.kernel_obj._reset_adaptive_bandwidth()
        return 

    def tensor_copy(self, tensor):
        """
        Create a CPU copy of a tensor.
        
        Parameters
        ----------
        tensor : torch.Tensor or None
            Tensor to copy. If None, returns None.
            
        Returns
        -------
        torch.Tensor or None
            Copied tensor, potentially moved to CPU
        """
        if tensor is None:
            return None
        elif self.keep_device or tensor.device.type == 'cpu':
            return tensor.clone()
        else:
            return tensor.cpu()
        
    def set_categorical_indices(self, numerical_indices, categorical_indices, categorical_vectors, device=None):
        """
        Configure categorical feature handling for the kernel.
        
        This method sets up the kernel to handle categorical features by
        specifying which features are numerical vs categorical and providing
        encoding vectors for categorical features.
        
        Parameters
        ----------
        numerical_indices : torch.Tensor
            Indices of numerical features in the input data
            
        categorical_indices : list of torch.Tensor
            List where each element contains indices for one categorical feature
            
        categorical_vectors : list of torch.Tensor
            List where each element is an encoding matrix for one categorical feature.
            Each row represents the encoding for that categorical value.
            
        device : str or torch.device, optional
            Device to store the categorical information. If None, uses self.device.
            
        Notes
        -----
        - Only applies to ProductLaplaceKernel
        """
        if numerical_indices is None and categorical_indices is None and categorical_vectors is None:
            if self.verbose:
                print("No categorical indices provided, ignoring")
            return
        assert numerical_indices is not None, "Numerical indices must be provided if one of categorical indices/vectors are provided"
        assert categorical_vectors is not None, "Categorical vectors must be provided if categorical indices are provided"
        assert len(categorical_indices) == len(categorical_vectors), "Number of categorical index and vector groups must match"
        assert len(numerical_indices) > 0 or len(categorical_indices) > 0, "No numerical or categorical features"
        self.kernel_obj.set_categorical_indices(numerical_indices, categorical_indices, categorical_vectors, device=self.device if device is None else device)
        return

    def update_best_params(self, best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth, current_metric, current_iter):
        """
        Update best parameters if current model performance is better.
        
        Parameters
        ----------
        best_metric : float
            Best validation metric seen so far
        best_alphas : torch.Tensor
            Best model weights (alpha coefficients)
        best_M : torch.Tensor
            Best Mahalanobis matrix
        best_sqrtM : torch.Tensor
            Best square root Mahalanobis matrix
        best_iter : int
            Iteration number that achieved best performance
        best_bandwidth : float
            Best kernel bandwidth
        current_metric : float
            Current validation metric
        current_iter : int
            Current iteration number
            
        Returns
        -------
        tuple
            Updated (best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth)
            
        Notes
        -----
        - For maximization metrics (accuracy, AUC, F1), improvement means higher values
        - For minimization metrics (MSE), improvement means lower values
        - Parameters are copied to avoid reference issues
        """
        # if classification and accuracy higher, or if regression and mse lower
        # maximize_metric = self.tuning_metric in ['accuracy', 'auc', 'f1', 'top_agop_vector_auc', 'top_agop_vector_pearson_r', 'top_agop_vectors_ols_auc']
        maximize_metric = Metric.from_name(self.tuning_metric).should_maximize
        if maximize_metric and current_metric > best_metric:
            best_metric = current_metric
            best_alphas = self.tensor_copy(self.weights)
            best_iter = current_iter
            best_bandwidth = self.kernel_obj.bandwidth+0
            best_M = self.tensor_copy(self.M)
            best_sqrtM = self.tensor_copy(self.sqrtM)

        elif not maximize_metric and current_metric < best_metric:
            best_metric = current_metric
            best_alphas = self.tensor_copy(self.weights)
            best_iter = current_iter
            best_bandwidth = self.kernel_obj.bandwidth+0
            best_M = self.tensor_copy(self.M)
            best_sqrtM = self.tensor_copy(self.sqrtM)

        return best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth
        
    def fit_predictor(self, centers, targets, bs=None, lr_scale=1, **kwargs):
        """
        Fit the kernel regression predictor using either least squares or EigenPro.
        
        Parameters
        ----------
        centers : torch.Tensor
            Training centers (support vectors) of shape (n_centers, n_features)
        targets : torch.Tensor
            Target values of shape (n_centers, n_outputs)
        bs : int, optional
            Batch size for EigenPro optimization. If None, uses default.
        lr_scale : float, default=1
            Learning rate scale factor for EigenPro
        **kwargs : dict
            Additional arguments passed to the predictor fitting methods
            
        Notes
        -----
        - Method selection depends on self.fit_using_eigenpro (set during initialization)
        - For EigenPro, can optionally prefit with a subset for initialization
        - Adaptive bandwidth is reset if bandwidth_mode is 'adaptive'
        - Results are stored in self.weights
        """
        
        if self.bandwidth_mode == 'adaptive':
            if isinstance(self.kernel_obj, SumPowerLaplaceKernel):
                raise ValueError("Adaptive bandwidth is not yet supported for SumPowerLaplaceKernel.")

            # adaptive bandwidth will be reset on next kernel computation
            if self.verbose:
                print("Resetting adaptive bandwidth")
            self.reset_adaptive_bandwidth()

        self.centers = centers

        # Route logistic solver to a dedicated IRLS method with validation early stopping
        if self.solver == 'log_reg':
            self.weights = self.fit_predictor_logistic(centers, targets, **kwargs)
            return

        if self.fit_using_eigenpro:
            assert not self.label_centering, "EigenPro does not yet support label centering"
            if self.prefit_eigenpro:
                random_indices = torch.randperm(centers.shape[0])[:self.max_lstsq_size]
                if self.verbose:
                    print(f"Prefitting Eigenpro with {len(random_indices)} points")
                sub_weights = self.fit_predictor_lstsq(centers[random_indices], targets[random_indices])
                initial_weights = torch.zeros_like(targets)
                initial_weights[random_indices] = sub_weights.to(targets.device, dtype=targets.dtype)
            else:
                initial_weights = None

            self.weights = self.fit_predictor_eigenpro(centers, targets, bs=bs, lr_scale=lr_scale, 
                                                       initial_weights=initial_weights, **kwargs)
        else:
            self.weights = self.fit_predictor_lstsq(centers, targets)

    def fit_predictor_logistic(self, centers, targets, X_val, y_val, **kwargs):
        """
        Fit kernel logistic regression using IRLS with validation early stopping.

        Parameters
        ----------
        centers : torch.Tensor
            Training centers of shape (n, d).
        targets : torch.Tensor
            Binary labels of shape (n, 1) or (n,) with values in {0,1} or {-1,1}.
        X_val : torch.Tensor
            Validation features.
        y_val : torch.Tensor
            Validation labels aligned with X_val.

        Keyword Args
        ------------
        log_max_iters : int, default=6
            Maximum number of IRLS steps (outer control). Each step calls a single-iteration Newton update.
        lr : float, default=1.0
            Learning rate for Newton updates.
        tol : float, default=1e-6
            Tolerance passed to the inner solver (not used when doing single-step updates).

        Returns
        -------
        torch.Tensor
            Learned dual coefficients alpha of shape (n, 1).
        """

        assert X_val is not None and y_val is not None, "X_val and y_val must be provided for logistic solver"

        # Ensure tensors on device
        if centers.device != self.device:
            centers = centers.to(self.device)
            targets = targets.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        # Build Gram matrix once
        K = self.kernel(centers, centers)

        # IRLS control fully inside kernel_log_solve via callback
        max_steps = kwargs.get('log_max_iters', 7)
        lr = kwargs.get('lr', 1.0)
        tol = kwargs.get('tol', 1e-6)

        best_metric = float('inf') if self.should_minimize else float('-inf')
        best_alphas = None
        best_iter = -1

        def _early_stop_cb(iteration, alpha, f):
            # Temporarily attach weights for scoring
            self.weights = alpha
            val_metrics = self._compute_validation_metrics(centers, targets, X_val, y_val, iteration_num=iteration, **kwargs)
            current = val_metrics[self.tuning_metric]

            # Check early stopping w.r.t. best so far
            nonlocal best_metric, best_alphas, best_iter
            if iteration > 0 and self._should_early_stop(current, best_metric, es_multiplier=0.9995):
                if self.verbose:
                    print(f"Logistic early stopping at IRLS step {iteration}")
                should_stop = True
            else:
                should_stop = False
            
            if (not self.should_minimize and current > best_metric) or (self.should_minimize and current < best_metric):
                best_metric = current
                best_alphas = self.tensor_copy(self.weights)
                best_iter = iteration
            
            return should_stop


        from .kernel_log_reg import kernel_log_solve
        if self.verbose:
            print('-'*100)
            print("Starting kernel log solve")
        alpha = kernel_log_solve(
            K, targets,
            reg=float(self.reg) if self.reg is not None else 0.0,
            max_iters=max_steps,
            tol=tol,
            lr=lr,
            initial_alpha=None,
            callback=_early_stop_cb,
        )
        if self.verbose:
            print('-'*100)

        # Choose best seen alpha if recorded; otherwise use final alpha
        self.weights = (best_alphas.to(self.device) if best_alphas is not None else alpha)
        self.best_iter = best_iter
        return self.weights
    
    def fit_predictor_lstsq(self, centers, targets):
        """
        Fit kernel regression using direct least squares solution.
        
        Parameters
        ----------
        centers : torch.Tensor
            Training centers (support vectors) of shape (n_centers, n_features)
        targets : torch.Tensor
            Target values of shape (n_centers, n_outputs)
        solver : str, default='solve'
            Matrix factorization method to use:
            - 'solve': LU decomposition with partial pivoting
            - 'cholesky': Cholesky decomposition (assumes positive definite)
            - 'lu': Explicit LU decomposition
            
        Returns
        -------
        torch.Tensor
            Alpha coefficients of shape (n_centers, n_outputs)
        """
        assert(len(centers)==len(targets))

        if centers.device != self.device:
            centers = centers.to(self.device)
            targets = targets.to(self.device)

        kernel_matrix = self.kernel(centers, centers)  

        # Apply direct K diagonal regularization only for closed-form LSTSQ solvers.
        # For logistic IRLS, regularization is handled inside the Newton step as (WK + reg I).
        if self.reg > 0:
            kernel_matrix.diagonal().add_(self.reg)

        try:
            if self.solver == 'solve':
                out = torch.linalg.solve(kernel_matrix, targets)
            elif self.solver == 'cholesky':
                L = torch.linalg.cholesky(kernel_matrix, out=kernel_matrix)
                out = torch.cholesky_solve(targets, L)
            elif self.solver == 'lu':
                P, L, U = torch.linalg.lu(kernel_matrix)
                out = torch.linalg.lu_solve(P, L, U, targets)
        except Exception as e:
            if self.verbose:
                print(f"Error in previous solver: {e}, re-trying with large regularization")
            
            # Gershgorin circle theorem to upper bound maximum eigenvalue
            row_sums = kernel_matrix.abs().sum(dim=1)
            max_row_sum = row_sums.max()
            kernel_matrix.diagonal().add_(max_row_sum*1e-2) # 1% of max row eigenvalue bound 

            out = torch.linalg.solve(kernel_matrix, targets)
        
        return out

    def fit_predictor_eigenpro(self, centers, targets, bs, lr_scale, initial_weights=None, **kwargs):
        """
        Fit kernel regression using EigenPro iterative optimization.
        
        Parameters
        ----------
        centers : torch.Tensor
            Training centers (support vectors) of shape (n_centers, n_features)
        targets : torch.Tensor
            Target values of shape (n_centers, n_outputs)
        bs : int
            Batch size for EigenPro iterations
        lr_scale : float
            Learning rate scale factor
        initial_weights : torch.Tensor, optional
            Initial weights for warm start. If None, uses zero initialization.
        **kwargs : dict
            Additional arguments passed to EigenPro fit method
            
        Returns
        -------
        torch.Tensor
            Alpha coefficients of shape (n_centers, n_outputs)
            
        """
        n_classes = 1 if targets.dim()==1 else targets.shape[-1]
        ep_model = KernelModel(self.kernel, centers, n_classes, device=self.device)
        if initial_weights is not None:
            ep_model.weight = initial_weights.to(ep_model.weight.device, dtype=ep_model.weight.dtype)
        _ = ep_model.fit(centers, targets, verbose=self.verbose, mem_gb=self.mem_gb, bs=bs, 
                         lr_scale=lr_scale, classification=self.classification, **kwargs)
        return ep_model.weight.clone()

    @with_env_var("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    def predict(self, samples, max_batch_size=50_000):
        """
        Parameters
        ----------
        samples : torch.Tensor or numpy.ndarray
            Input samples of shape (n_samples, n_features)
        max_batch_size : int, default=50_000
            Maximum batch size for prediction to control memory usage
            
        Returns
        -------
        torch.Tensor or numpy.ndarray
            Predictions of shape (n_samples, n_outputs). Return type matches input type.
            
        """
        samples, original_format = self.validate_samples(samples)
        out = []
        for i in range(0, samples.shape[0], max_batch_size):
            out_batch = self.kernel(samples[i:i+max_batch_size].to(self.device), self.centers.to(self.device)) @ self.weights.to(self.device)
            out.append(out_batch)
        out = torch.cat(out, dim=0)
        return self.convert_to_format(out, original_format)

    def validate_samples(self, samples):
        """
        Validate and normalize input samples format.
        
        This method ensures samples are in the correct format (torch.Tensor)
        and device, while keeping track of the original format for later conversion.
        
        Parameters
        ----------
        samples : torch.Tensor or numpy.ndarray
            Input samples to validate
            
        Returns
        -------
        tuple
            (normalized_samples, original_format) where:
            - normalized_samples: torch.Tensor on self.device
            - original_format: dict with 'type' and 'device' keys
            
        Raises
        ------
        ValueError
            If samples are not torch.Tensor or numpy.ndarray
        """
        original_format = {}
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples)
            original_format['type'] = 'numpy'
            original_format['device'] = 'cpu'
        elif isinstance(samples, torch.Tensor):
            original_format['type'] = 'torch'
            original_format['device'] = samples.device
        else:
            raise ValueError(f"Invalid sample type: {type(samples)}")
        return samples.to(self.device), original_format
    
    def convert_to_format(self, tensor, original_format):
        """
        Convert tensor back to original input format.
        
        This method converts the processed tensor back to the format
        of the original input (NumPy array or PyTorch tensor on original device).
        
        Parameters
        ----------
        tensor : torch.Tensor
            Processed tensor to convert
        original_format : dict
            Original format information from validate_samples
            
        Returns
        -------
        torch.Tensor or numpy.ndarray
            Tensor converted to original format
        """
        if original_format['type'] == 'numpy':
            return tensor.cpu().numpy()
        elif original_format['type'] == 'torch':
            return tensor.to(original_format['device'])

    def get_grads(self, samples):
        """
        Compute gradients of the model predictions with respect to the inputs.

        Parameters
        ----------
        samples : torch.Tensor or numpy.ndarray
            Points at which to evaluate gradients, shape (n_samples, n_features).

        Returns
        -------
        torch.Tensor
            Gradients with shape (n_samples, n_outputs, n_features) on the
            model device.
        """
        if self.centers is None or self.weights is None:
            raise ValueError("Model must be fitted before calling get_grads.")

        samples, _ = self.validate_samples(samples)
        transform = self.sqrtM if self.use_sqrtM else self.M

        grads = self.kernel_obj.get_function_grads(
            self.centers.to(self.device),
            samples.to(self.device),
            self.weights.t().to(self.device),
            mat=None if transform is None else transform.to(self.device),
        )
        if grads.dim() == 4 and grads.shape[0] == 1:
            grads = grads.squeeze(0)
        if grads.dim() != 3:
            raise ValueError(f"Unexpected gradient tensor shape {grads.shape}")
        return grads.permute(1, 0, 2)

    def validate_data(self, train_data, val_data):
        """
        Validate and preprocess training and validation data.
        
        This method ensures both training and validation data are in the correct
        format and shape, converting them to PyTorch tensors and ensuring targets
        have the appropriate dimensionality.
        
        Parameters
        ----------
        train_data : tuple
            Training data as (X_train, y_train) where:
            - X_train: features of shape (n_train, n_features)
            - y_train: targets of shape (n_train,) or (n_train, n_outputs)
        val_data : tuple
            Validation data as (X_val, y_val) where:
            - X_val: features of shape (n_val, n_features)
            - y_val: targets of shape (n_val,) or (n_val, n_outputs)
            
        Returns
        -------
        tuple
            (X_train, y_train, X_val, y_val) all as PyTorch tensors with:
            - X tensors of shape (n_samples, n_features)
            - y tensors of shape (n_samples, n_outputs) with n_outputs >= 1
        """
        assert train_data is not None, "Train data must be provided"
        assert val_data is not None, "Validation data must be provided"

        X_train, y_train = train_data
        X_val, y_val = val_data

        X_train, _ = self.validate_samples(X_train)
        X_val, _ = self.validate_samples(X_val)
        y_train, _ = self.validate_samples(y_train)
        y_val, _ = self.validate_samples(y_val)

        if len(y_val.shape) == 1:
            y_val = y_val.unsqueeze(-1)
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(-1)

        return X_train, y_train, X_val, y_val
    
    def adapt_params_to_data(self, n, d):
        """
        Adapt RFM parameters based on dataset characteristics.
        
        This method automatically adjusts key parameters (iterations, sample size,
        epochs) based on the dataset size and dimensionality to optimize performance
        and computational efficiency.
        
        Parameters
        ----------
        n : int
            Number of training samples
        d : int
            Number of features (dimensionality)
            
        Notes
        -----
        - Adjusts early stopping multiplier for accuracy metric
        - Sets different parameters for ProductLaplaceKernel vs other kernels
        - Reduces iterations and sample sizes for large datasets
        - Considers both sample size and dimensionality in parameter selection
        - Updates self.iters, self.total_points_to_sample, self.ep_epochs
        - Sets self.keep_device based on dimensionality vs sample size ratio
        """

        # if self.tuning_metric == 'accuracy' and self.early_stop_rfm:
        #     if n <= 30_000:
        #         self.early_stop_multiplier = min(self.early_stop_multiplier, 1.01)
        #     else:
        #         self.early_stop_multiplier = min(self.early_stop_multiplier, 1.02)
        #     print(f"More aggressive early stop multiplier for accuracy: {self.early_stop_multiplier}")

        self.keep_device = d > n # keep previous Ms on GPU if more features than samples
        ep_epochs = 8
        total_points_to_sample = 20_000
        iters_to_use = 4
        if isinstance(self.kernel_obj, ProductLaplaceKernel):
            ep_epochs = 2
            if n > 1000: # only handle cateogricals specially for high-dimensional data
                if n <= 10_000:
                    # For smallest datasets: use default values
                    pass
                elif 10_000 < n <= 20_000 and d <= 2000:
                    # Medium-small datasets with moderate dimensionality
                    total_points_to_sample = min(total_points_to_sample, 10_000)
                    iters_to_use = min(iters_to_use, 4)
                elif 20_000 < n <= 50_000 and d <= 2000:
                    # Medium-sized datasets with moderate dimensionality
                    total_points_to_sample = min(total_points_to_sample, 2500)
                    iters_to_use = min(iters_to_use, 2)
                elif 10_000 < n <= 20_000 and d <= 3000:
                    # Medium-small datasets with higher dimensionality
                    total_points_to_sample = 2500
                    iters_to_use = min(iters_to_use, 2)
                elif d < 1000:
                    # Largest datasets or highest dimensionality
                    total_points_to_sample = 2000
                    iters_to_use = min(iters_to_use, 1)
                elif d < 4000:
                    # Largest datasets or highest dimensionality
                    total_points_to_sample = 1000
                    iters_to_use = min(iters_to_use, 1)
                else:
                    # For highest dimensionality
                    total_points_to_sample = 250
                    iters_to_use = min(iters_to_use, 1)
        if n >= 70_000:
            # for large datasets, use fewer iterations for all kernel types
            iters_to_use = min(iters_to_use, 2)

        ep_epochs = ep_epochs if self.ep_epochs is None else self.ep_epochs
        total_points_to_sample = total_points_to_sample if self.total_points_to_sample is None else self.total_points_to_sample
        iters_to_use = iters_to_use if self.iters is None else self.iters

        self.iters = iters_to_use
        self.total_points_to_sample = total_points_to_sample
        self.ep_epochs = ep_epochs
        return
    
    def _initialize_fit_parameters(self, iters, method, reg, verbose, M_batch_size, total_points_to_sample, 
                                   ep_epochs, tuning_metric, early_stop_rfm, early_stop_multiplier, 
                                   center_grads, prefit_eigenpro, solver, **kwargs):
        """Initialize parameters for the fit method."""
        self.verbose = verbose if verbose is not None else self.verbose
        self.fit_using_eigenpro = (method.lower()=='eigenpro')
        self.prefit_eigenpro = prefit_eigenpro
        self.reg = reg if reg is not None else self.reg
        self.M_batch_size = M_batch_size
        self.total_points_to_sample = total_points_to_sample
        self.iters = iters if iters is not None else self.iters
        self.ep_epochs = ep_epochs
        self.tuning_metric = tuning_metric if tuning_metric is not None else self.tuning_metric
        self.should_minimize = not Metric.from_name(self.tuning_metric).should_maximize
        self.early_stop_rfm = early_stop_rfm
        self.early_stop_multiplier = early_stop_multiplier
        self.center_grads = center_grads
        self.solver = solver if solver is not None else self.solver
        self.top_k = kwargs.get('top_k', None)

        if self.solver == 'log_reg':
            self.class_converter._numerical_type = 'logit_diff'
            
        assert 'diag' not in kwargs, "diag should be set in the constructor"

    def _compute_validation_metrics(self, X_train, y_train, X_val, y_val, iteration_num=None, is_final=False, M_batch_size=None, **kwargs):
        """Compute validation metrics based on tuning_metric."""

        metric = Metric.from_name(self.tuning_metric)
        if 'agop' in metric.required_quantities:
            self.agop = self.fit_M(X_train, self.n_classes, M_batch_size=M_batch_size, inplace=False, **kwargs)
        val_metrics = self.score(X_val, y_val, metrics=[self.tuning_metric])
        if self.verbose:
            prefix = "Final" if is_final else f"Round {iteration_num}"
            print(f"{prefix} Val {metric.display_name}: {val_metrics[self.tuning_metric]:.4f}")
        return val_metrics

    def _should_early_stop(self, current_metric, best_metric, es_multiplier=None):
        if es_multiplier is None:
            es_multiplier = self.early_stop_multiplier
        """Check if early stopping criteria is met."""
        if self.should_minimize:
            return current_metric > best_metric * es_multiplier
        else:
            return current_metric < best_metric / es_multiplier

    # TODO: to prevent OOM issue due to memory fragmentation but doesn't actually set the environment variable!
    @with_env_var("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True") 
    def fit(self, train_data, val_data=None, iters=None, method='lstsq', reg=None, center_grads=False,
            verbose=False, M_batch_size=None, ep_epochs=None, return_best_params=True, bs=None, 
            return_Ms=False, lr_scale=1, total_points_to_sample=None, solver=None, 
            tuning_metric=None, prefit_eigenpro=True, early_stop_rfm=True, early_stop_multiplier=1.1, 
            callback=None, **kwargs):
        """
        :param train_data: tuple of (X, y)
        :param val_data: tuple of (X, y)
        :param iters: number of iterations to run
        :param method: 'lstsq' or 'eigenpro'
        :param reg: Regularization coefficient (higher is more regularization).
        :param classification: if True, the model will tune for (and report) accuracy, else just MSE loss
        :param verbose: if True, print progress
        :param M_batch_size: batch size over samples for AGOP computation
        :param return_best_params: if True, return the best parameters
        :param bs: batch size for eigenpro
        :param return_Ms: if True, return the Mahalanobis matrix at each iteration
        :param lr_scale: learning rate scale for EigenPro
        :param total_points_to_sample: number of points to sample for AGOP computation
        :param solver: 'solve' or 'cholesky' or 'lu', used in LSTSQ computation
        :param prefit_eigenpro: if True, prefit EigenPro with a subset of <= max_lstsq_size samples
        """

        # Initialize parameters
        self._initialize_fit_parameters(iters, method, reg, verbose, M_batch_size, total_points_to_sample,
                                       ep_epochs, tuning_metric, early_stop_rfm, early_stop_multiplier,
                                       center_grads, prefit_eigenpro, solver, **kwargs)
        
        

        # Validate and prepare data
        X_train, y_train, X_val, y_val = self.validate_data(train_data, val_data)
        n, d = X_train.shape
        assert len(y_train.shape) == 2, "y_train must be a 2D tensor"

        if self.verbose:
            print("="*70)
            print(f"Fitting RFM with ntrain: {n}, d: {d}, and nval: {X_val.shape[0]}")
            print("="*70)

        self.n_classes = y_train.shape[1]
        if self.class_converter is None and self.n_classes > 1:
            # Classification converter is needed for classification tasks
            self.class_converter = ClassificationConverter(mode='zero_one', n_classes=self.n_classes)

        self.adapt_params_to_data(n, d)
        
        # Initialize tracking variables
        metrics, Ms = [], []
        best_alphas, best_M, best_sqrtM = None, None, None
        best_metric = float('inf') if self.should_minimize else float('-inf')
        best_iter = None
        early_stopped = False
        best_bandwidth = self.kernel_obj.bandwidth+0

        start_time = time.time()

        # Main training loop
        for i in range(self.iters):
            # check time limit
            if i > 0 and self.time_limit_s is not None and (i+1)/i*(time.time()-start_time) > self.time_limit_s:
                break  # would expect to exceed the time limit, so stop


            if callback is not None:
                callback(iteration=i)

            start = time.time()
            self.fit_predictor(X_train, y_train, X_val=X_val, y_val=y_val, 
                               bs=bs, lr_scale=lr_scale, **kwargs)
                        
            # Compute validation metrics
            val_metrics = self._compute_validation_metrics(X_train, y_train, X_val, y_val, iteration_num=i, M_batch_size=M_batch_size, **kwargs)

            # Update best parameters if needed
            if return_best_params:
                best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth = self.update_best_params(
                    best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth, 
                    val_metrics[self.tuning_metric], i)
             
            # Check for early stopping
            if self.early_stop_rfm:
                val_metric = val_metrics[self.tuning_metric]
                if self._should_early_stop(val_metric, best_metric):
                    if self.verbose:
                        print(f"Early stopping at iteration {i}")
                    if not return_best_params:
                        self.fit_M(X_train, self.n_classes, M_batch_size=M_batch_size, **kwargs)
                    early_stopped = True
                    break

            # Fit M matrix and cleanup
            self.fit_M(X_train, self.n_classes, M_batch_size=M_batch_size, **kwargs)
            del self.weights
            
            if return_Ms:
                Ms.append(self.tensor_copy(self.M))
                metrics.append(val_metrics[self.tuning_metric])

            if self.verbose:
                print(f"Time taken for round {i}: {time.time() - start} seconds")

        if callback is not None:
            callback(iteration=self.iters)

        # Handle final iteration if no early stopping occurred
        if not early_stopped:
            self.fit_predictor(X_train, y_train, X_val=X_val, y_val=y_val, bs=bs, **kwargs)        
            final_val_metrics = self._compute_validation_metrics(X_train, y_train, X_val, y_val, is_final=True, **kwargs)

            if return_best_params:
                best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth = self.update_best_params(
                    best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth, 
                    final_val_metrics[self.tuning_metric], iters)
                
        # Restore best parameters
        if return_best_params:
            self.M = None if best_M is None else best_M.to(self.device)
            self.sqrtM = None if best_sqrtM is None else best_sqrtM.to(self.device)
            self.weights = best_alphas.to(self.device)
            self.kernel_obj.bandwidth = best_bandwidth

        self.best_iter = best_iter

        if self.verbose:
            print(f"{self.best_iter=}")

        if kwargs.get('get_agop_best_model', False):
            # fit AGOP of best model
            self.agop_best_model = self.fit_M(X_train, self.n_classes, M_batch_size=M_batch_size, inplace=False, **kwargs)

        return Ms if return_Ms else None
    
    def _compute_optimal_M_batch(
        self,
        n,
        c,
        d,
        scalar_size=4,
        mem_constant=2.0,
        max_batch_size=10_000,
        max_cheap_batch_size=10_000,
        light_kernels=(LaplaceKernel, LightLaplaceKernel, 
        KermacLpqLaplaceKernel, KermacProductLaplaceKernel, KermacSumPowerLaplaceKernel),
    ):
        """Computes the optimal batch size for AGOP."""
        # # SumPowerLaplace gradients can be extremely memory-hungry (large K×M intermediates),
        # # so be conservative when choosing batch sizes to avoid CUDA OOM.
        # if isinstance(self.kernel_obj, (SumPowerLaplaceKernel, KermacSumPowerLaplaceKernel)):
        #     mem_constant = max(float(mem_constant), 10.0)
        #     max_batch_size = min(int(max_batch_size), 8096)

        if self.device in ['cpu', torch.device('cpu')] or isinstance(self.kernel_obj, light_kernels):
            if self.verbose:
                print("Using cheap batch size")
            # cpu and light kernels are less memory intensive, use fewer but larger batches scaled by free GPU memory
            cheap_batch_cap = int(max_cheap_batch_size * memory_scaling_factor(self.device))
            cheap_batch_cap = max(cheap_batch_cap, 1)
            M_batch_size = max(min(n, cheap_batch_cap), 1)
        else:
            if self.verbose:
                print("Using expensive batch size")
            available_memory, _ = get_gpu_memory_bytes(self.device)
            if not available_memory:
                M_batch_size = max(min(n, max_batch_size), 1)
            else:
                denom = mem_constant * n * c * d * scalar_size
                M_batch_size = int(available_memory / denom) if denom else max_batch_size
                M_batch_size = max(M_batch_size, 1)
                M_batch_size = min(M_batch_size, max_batch_size, n)
        if self.verbose:
            print(f"Optimal M batch size: {M_batch_size}")
        return M_batch_size
    
    def fit_M(self, samples, num_classes, M_batch_size=None, inplace=True, **kwargs):
        """
        Fit the Mahalanobis matrix M using AGOP.
        
        Parameters
        ----------
        samples : torch.Tensor
            Input samples of shape (n_samples, n_features)
        num_classes : int
            Number of output classes/dimensions
        M_batch_size : int, optional
            Batch size for AGOP computation. If None, computed automatically
            based on available memory.
        inplace : bool, default=True
            Whether to update self.M and self.sqrtM in place. If False, returns
            the computed M matrix without modifying the object.
        **kwargs : dict
            Additional arguments (unused, for compatibility)
            
        Returns
        -------
        torch.Tensor or None
            If inplace=False, returns the computed M matrix. Otherwise returns None.
            
        Notes
        -----
        - AGOP matrix is computed by averaging gradients across batches
        - Total sample size is limited by self.total_points_to_sample
        - Matrix is normalized by dividing by its maximum value
        - For kernels using sqrtM, computes matrix power using self.agop_power
        - Batch size is automatically optimized based on available GPU memory
        """
        
        n, d = samples.shape
        M = torch.zeros_like(self.M) if self.M is not None else (
            torch.zeros(d, dtype=samples.dtype, device=self.device) 
            if self.diag else torch.zeros(d, d, dtype=samples.dtype, device=self.device))
        

        if M_batch_size is None: 
            BYTES_PER_SCALAR = samples.element_size()
            M_batch_size = self._compute_optimal_M_batch(n, num_classes, d, scalar_size=BYTES_PER_SCALAR)
        
        batches = torch.arange(n).split(M_batch_size)

        num_batches = 1 + self.total_points_to_sample//M_batch_size
        batches = batches[:num_batches]
        if self.verbose:
            print(f'Sampling AGOP on maximum of {num_batches*M_batch_size} total points')

        if self.verbose:
            for i, bids in tenumerate(batches):
                M.add_(self.update_M(samples[bids]))
        else:
            for bids in batches:
                M.add_(self.update_M(samples[bids]))
        
        scaled_M = M / (M.max() + 1e-30)
        if self.use_sqrtM:
            sqrtM = matrix_power(scaled_M, self.agop_power, verbose=self.verbose)
        else:
            sqrtM = None
        
        if inplace:
            self.M = scaled_M
            self.sqrtM = sqrtM
        else:
            return scaled_M
        
    def score(self, samples, targets, metrics):
        """
        Evaluate model performance using specified metrics.
        
        Parameters
        ----------
        samples : torch.Tensor
            Input samples of shape (n_samples, n_features)
        targets : torch.Tensor
            Target values of shape (n_samples, n_outputs)
        metrics : list of str
            List of metrics to compute. Supported metrics:
            - 'accuracy': Classification accuracy
            - 'mse': Mean squared error
            - 'rmse': Root mean squared error
            - 'mae': Mean absolute error
            - 'brier': Brier loss (= MSE to one-hot encoded labels for classification)
            - 'logloss': Log loss
            - 'f1': F1 score
            - 'auc': Area under ROC curve (one-vs-rest for multiclass)
            - 'top_agop_vector_auc': AUC using top AGOP eigenvector projection
            - 'top_agop_vector_pearson_r': Pearson correlation of targets with top AGOP eigenvector projection
            - 'top_agop_vectors_ols_auc': AUC using OLS regression on top AGOP eigenvectors
            
        Returns
        -------
        dict
            Dictionary mapping metric names to their computed values
        """

        metrics = Metrics(metrics)
        assert len(targets.shape) == 2 and targets.shape[1] >= 1
        kwargs = dict(top_k=self.top_k, y_true_reg=targets)
        if 'y_pred' in metrics.required_quantities:
            kwargs['y_pred'] = self.predict(samples.to(self.device))
        if 'y_pred_proba' in metrics.required_quantities:
            kwargs['y_pred_proba'] = self.predict_proba(samples.to(self.device))
        if 'agop' in metrics.required_quantities:
            kwargs['agop'] = self.agop
        if 'y_true_class' in metrics.required_quantities:
            kwargs['y_true_class'] = self.class_converter.numerical_to_labels(targets)

        return metrics.compute(**kwargs)
    
    @with_env_var("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    def predict_proba(self, samples, eps=1e-3):
        """
        Predict class probabilities for new samples.
        
        This method computes the probability of each class for the given samples
        by first getting raw predictions from the kernel regression model, then
        transforming them into proper probabilities through normalization.
        
        Parameters
        ----------
        samples : torch.Tensor or numpy.ndarray
            Input samples of shape (n_samples, n_features)
        eps : float, default=1e-3
            Clamping value for probabilities to avoid numerical issues.
            Probabilities are clamped to [eps, 1-eps] range.
            
        Returns
        -------
        torch.Tensor
            Probability matrix of shape (n_samples, n_classes) where each row
            sums to 1 and represents the probability distribution over classes.
            
        Notes
        -----
        - For binary classification, converts single-column predictions to two-column format
        - Applies clamping to avoid log(0) or division by zero in downstream computations
        - Normalizes probabilities to ensure they sum to 1
        - Inherits batch processing and device management from predict() method
        """
        predictions = self.predict(samples) 
        if self.solver == 'log_reg':
            assert self.class_converter.mode == 'zero_one'
            predictions = torch.sigmoid(predictions)
            eps = 1e-10
        return self.class_converter.numerical_to_probas(predictions, eps=eps)
