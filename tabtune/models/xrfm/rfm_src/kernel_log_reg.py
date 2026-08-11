import torch
import time

def _ensure_2d_column(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        return t.unsqueeze(-1)
    return t

def kernel_log_solve(kernel_matrix: torch.Tensor,
                     targets: torch.Tensor,
                     reg: float = 0.0,
                     max_iters: int = 7,
                     tol: float = 5e-3,
                     lr: float = 1.0,
                     initial_alpha: torch.Tensor = None,
                     callback=None) -> torch.Tensor:
    """
    Solve kernel logistic regression via IRLS/Newton on the dual coefficients.

    This minimizes: sum_i logloss(y_i, f_i) + (reg/2) * alpha^T K alpha, with f = K alpha.

    Parameters
    ----------
    kernel_matrix : torch.Tensor
        Gram matrix K of shape (n, n). Should be symmetric positive semi-definite.
    targets : torch.Tensor
        Labels of shape (n,) or (n, 1). Values in {0,1} expected.
    reg : float, default=0.0
        Ridge regularization strength applied to the total function.
    max_iters : int, default=6
        Maximum IRLS iterations.
    tol : float, default=1e-6
        Convergence tolerance on relative change of f = K alpha.
    lr : float, default=1.0
        Deprecated alias for damping; retained for backward compatibility.
    damping : float, optional
        Step size (0,1] applied to the Newton/boosting update α ← α + damping · Δα.
    initial_alpha : torch.Tensor, optional
        Optional warm-start alpha of shape (n, 1).

    callback : callable, optional
        If provided, called after each Newton step with signature
        callback(iteration=i, alpha=alpha, f=f). If it returns True, the
        iterations stop early.

    Returns
    -------
    torch.Tensor
        Dual coefficients alpha of shape (n, 1).
    """

    device = kernel_matrix.device
    dtype = kernel_matrix.dtype

    K = kernel_matrix
    y = _ensure_2d_column(targets).to(device=device, dtype=dtype)

    if y.shape[1] != 1:
        raise NotImplementedError("kernel_log_solve currently supports only binary labels with a single column.")

    n = y.shape[0]

    # Initialize alpha and f = K alpha
    if initial_alpha is not None:
        alpha = _ensure_2d_column(initial_alpha).to(device=device, dtype=dtype)
        f = K @ alpha
    else:
        alpha = torch.zeros((n, 1), device=device, dtype=dtype)
        f = torch.zeros((n, 1), device=device, dtype=dtype)

    prev_f = f

    for i in range(max_iters):
        p = torch.sigmoid(f)
        # Per the derivation: W = diag(0.5 * ∂^2ℓ/∂F^2) = 0.5 * p(1-p)
        w = (0.5 * p * (1.0 - p)).clamp_min(1e-12)

        # b = ∂ℓ/∂F = p - y
        b = p - y
        # b2 = b + 2λ α_F, where α_F is the current coefficient vector
        b2 = b + (2.0 * reg) * alpha

        # Solve for the Newton/boosting increment: Δα = -0.5 * (W K + λ I)^{-1} b2
        A = w * K 
        A.diagonal().add_(reg)

        delta_alpha = -0.5 * torch.linalg.solve(A, b2)

        # Damped update and refresh f = K α
        alpha = alpha + lr * delta_alpha
        f = K @ alpha

        # External early-stopping callback hook
        if callback is not None:
            should_stop = bool(callback(iteration=i, alpha=alpha, f=f))
            if should_stop:
                break

        # Convergence check on relative change of f
        delta = torch.linalg.norm(f - prev_f)
        scale = torch.linalg.norm(prev_f).clamp_min(1.0)
        if (delta / scale) < tol:
            break
        prev_f = f

    return alpha