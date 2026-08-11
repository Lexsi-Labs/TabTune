'''Helper functions.'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Threshold for using iterative vs full SVD for top eigenvector extraction
_TOP_EIGENVECTOR_ITERATIVE_THRESHOLD = 256


def device_from_str(device):
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def get_top_eigenvector(M: torch.Tensor, method: str = 'auto', eps: float = 1e-6) -> torch.Tensor:
    """
    Extract the top eigenvector from a symmetric positive semi-definite matrix.

    Uses iterative methods (lobpcg) for large matrices to avoid O(d³) full SVD,
    falling back to full SVD for small matrices or when iterative methods fail.

    Parameters
    ----------
    M : torch.Tensor
        Symmetric positive semi-definite matrix of shape (d, d)
    method : str
        'auto' (default): Use lobpcg for d > threshold, else full SVD
        'lobpcg': Force iterative method
        'svd': Force full SVD

    Returns
    -------
    torch.Tensor
        Top eigenvector of shape (d,)
    """
    d = M.shape[0]

    if method == 'auto':
        use_iterative = d > _TOP_EIGENVECTOR_ITERATIVE_THRESHOLD
    elif method == 'lobpcg':
        use_iterative = True
    else:
        use_iterative = False

    if use_iterative:
        try:
            # lobpcg is much faster for large matrices when only top-k needed
            # Add small regularization for numerical stability
            M_reg = M + eps * torch.eye(d, device=M.device, dtype=M.dtype)
            eigenvalues, eigenvectors = torch.lobpcg(M_reg, k=1, largest=True)
            return eigenvectors[:, 0]
        except Exception:
            # Fall back to SVD if lobpcg fails (can happen with ill-conditioned matrices)
            pass

    # Full SVD fallback - more stable but O(d³)
    _, _, Vt = torch.linalg.svd(M, full_matrices=False)
    return Vt[0]

def matrix_power(M, power, verbose=True):
    """
    Compute the power of a matrix.
    :param M: Matrix to power.
    :param power: Power to raise the matrix to.
    :param verbose: Whether to print progress information.
    :return: Matrix raised to the power - M^{power}.
    """
    return stable_matrix_power(M, power, verbose=verbose)
    # if len(M.shape) == 2:
    #     assert M.shape[0] == M.shape[1], "Matrix must be square"

    #     # gpu square root
    #     S, U = torch.linalg.eigh(M)
    #     S[S<0] = 0.
    #     return U @ torch.diag(S**power) @ U.T
    # elif len(M.shape) == 1:
    #     assert M.shape[0] > 0, "Vector must be non-empty"
    #     M[M<0] = 0.
    #     return M**power
    # else:
    #     raise ValueError(f"Invalid matrix shape for square root: {M.shape}")
    
def stable_matrix_power(M, power, MAX_DIMENSIONS_FOR_SVD=5000, verbose=True):
    """
    Compute the power of a matrix.
    :param M: Matrix to power.
    :param power: Power to raise the matrix to.
    :param MAX_DIMENSIONS_FOR_SVD: Threshold for using full vs lowrank SVD.
    :param verbose: Whether to print progress information.
    :return: Matrix raised to the power - M^{power}.
    """
    if len(M.shape) == 2:
        assert M.shape[0] == M.shape[1], "Matrix must be square"

        # Handle NaNs
        if torch.isnan(M).all():
            if verbose:
                print("All NaNs in matrix, returning identity")
            return torch.eye(M.shape[0], device=M.device, dtype=M.dtype)

        if torch.isnan(M).any():
            if verbose:
                print("Some NaNs in matrix, replacing with 0")
            M = torch.nan_to_num(M, nan=0.0, posinf=1e12, neginf=-1e12)
            # Optional: scale to a reasonable magnitude
            scale = M.abs().max()
            if scale > 0:
                M = M / scale

        M.diagonal().add_(1e-8)
        if M.shape[0] < MAX_DIMENSIONS_FOR_SVD:
            if verbose:
                print("Using SVD")
            U, S, _ = torch.linalg.svd(M)
        else:
            if verbose:
                print("Using SVD lowrank with q=", MAX_DIMENSIONS_FOR_SVD)
                print("M.shape", M.shape)
            start_time = time.time()
            U, S, _ = torch.svd_lowrank(M, q=MAX_DIMENSIONS_FOR_SVD)
            end_time = time.time()
            if verbose:
                print(f"Time taken for SVD lowrank: {end_time - start_time} seconds")

        S[S<0] = 0.
        return (U @ torch.diag(S**power) @ U.T).to(device=M.device, dtype=M.dtype)

    elif len(M.shape) == 1:
        # Handle NaNs
        if torch.isnan(M).all():
            if verbose:
                print("All NaNs in vector, returning all ones")
            return torch.ones(M.shape[0], device=M.device, dtype=M.dtype)

        if torch.isnan(M).any():
            if verbose:
                print("Some NaNs in vector, replacing with 0")
            M = torch.nan_to_num(M, nan=0.0, posinf=1e12, neginf=-1e12)
            # Optional: scale to a reasonable magnitude
            scale = M.abs().max()
            if scale > 0:
                M = M / scale

        assert M.shape[0] > 0, "Vector must be non-empty"
        M[M<0] = 0.
        return M**power
    else:
        raise ValueError(f"Invalid matrix shape for square root: {M.shape}")
