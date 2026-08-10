'''Utility functions for performing fast SVD.'''
import torch
import torch.linalg as linalg
import time
from math import sqrt
import scipy.sparse.linalg

def nystrom_kernel_svd(samples, kernel_fn, top_q, method='eigh'):
    """Compute top eigensystem of kernel matrix using Nystrom method.

    Arguments:
        samples: data matrix of shape (n_sample, n_feature).
        kernel_fn: tensor function k(X, Y) that returns kernel matrix.
        top_q: top-q eigensystem.

    Returns:
        eigvals: top eigenvalues of shape (top_q).
        eigvecs: (rescaled) top eigenvectors of shape (n_sample, top_q).
    """

    n_samples, _ = samples.shape
    kmat = kernel_fn(samples, samples)
    scaled_kmat = kmat / n_samples

    if method == 'lobpcg':
        # this seems unstable
        start_time = time.time()
        converged_count = 0  # Add this to store the count
        
        def custom_tracker(lobpcg_instance):
            nonlocal converged_count  # Allow access to outer variable
            current_step = lobpcg_instance.ivars["istep"]
            converged_count = lobpcg_instance.ivars["converged_count"]  # Store the count
            current_residual = lobpcg_instance.tvars["rerr"]
            elapsed_time = time.time() - start_time
            print(f"LOBPCG Step {current_step}: Converged {converged_count} eigenpairs, Time: {elapsed_time:.2f}s")

        print(f"Using LOBPCG with n_samples={n_samples}, top_q={top_q}")
        eigenvalues, eigenvectors = torch.lobpcg(
            scaled_kmat.cuda().double(), 
            k=top_q, 
            niter=30,
            tol=None,
            X=None,  # Pass the initial guess
            tracker=custom_tracker
        )

        vals = torch.flip(eigenvalues, dims=(0,)).float()
        vecs = torch.flip(eigenvectors, dims=(1,)).float()

    elif method == 'eigh':
        # this is stable, but computing the full eigendecomposition is unnecessary
        vals, vecs = linalg.eigh(scaled_kmat.cuda())

    elif method == 'eigsh':
        vals, vecs = scipy.sparse.linalg.eigsh(scaled_kmat.cpu().numpy(), k=top_q, which='LM')
        vals = torch.from_numpy(vals).float()
        vecs = torch.from_numpy(vecs).float()

    vals = vals.float()
    vecs = vecs.float()
    eigvals = torch.flip(vals, dims=(0,))[:top_q]
    eigvecs = torch.flip(vecs, dims=(1,))[:, :top_q] / sqrt(n_samples)
    return eigvals.float(), eigvecs.float()
