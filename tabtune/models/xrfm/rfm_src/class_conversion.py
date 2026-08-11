from typing import Literal, Optional
import torch
import torch.nn.functional as F

class ClassificationConverter:
    def __init__(self, mode: Literal['zero_one', 'prevalence'], n_classes: int, 
                 labels: Optional[torch.Tensor] = None, init_from_params: bool = False):
        """
        Args:
            mode:
              - 'zero_one': binary -> {0,1} (shape (N,1)); multiclass -> one-hot (shape (N,K)).
              - 'prevalence': encode to a regular simplex in R^{K-1}, shifted so the empirical
                class prior maps to the origin (so a prediction of 0 decodes to the prior).
            n_classes: K >= 2
            labels: (N,), torch.long with class ids; REQUIRED for 'prevalence'. Ignored for 'zero_one'.

        Notes:
            - No device/dtype assumptions are made here; tensors are moved to the caller's device on use.
            - Prior is computed exactly as counts / total (no clamp_min).
        """
        assert mode in ['zero_one', 'prevalence']
        assert n_classes >= 2
        self.mode = mode
        self.n_classes = n_classes
        self._numerical_type = None

        if init_from_params:
            return

        if self.mode == 'prevalence':
            if labels is None:
                raise ValueError("labels must be provided for mode='prevalence'.")
            counts = torch.bincount(labels.cpu().long().squeeze(-1), minlength=n_classes).float()
            total = counts.sum()
            if total.item() == 0:
                raise ValueError("labels must contain at least one element for mode='prevalence'.")
            prior = counts / total  # (K,)
            K = n_classes
            # Build regular simplex (rows equidistant) via QR on [e_i - e_K]
            I = torch.eye(K, dtype=torch.float32)          # CPU build
            M = I[:, :-1] - I[:, [-1]]                     # (K, K-1)
            Q, _ = torch.linalg.qr(M, mode='reduced')      # (K, K-1), columns orthonormal
            # Shift so empirical prior maps to origin
            mu = prior @ Q                                  # (K-1,)
            C = Q - mu                                      # (K, K-1)
            # Precompute inverse for decoding: A = [C^T; 1^T] \in R^{K x K}
            A = torch.cat([C.T, torch.ones(1, K, dtype=torch.float32)], dim=0)  # (K, K)
            invA = torch.linalg.inv(A)

            self._prior = prior          # (K,) CPU float32
            self._C = C                  # (K, K-1) CPU float32
            self._invA = invA            # (K, K) CPU float32
        else:
            self._prior = None
            self._C = None
            self._invA = None

    # ---------- public API ----------
    def labels_to_numerical(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Encode integer class labels to regression targets.

        Args:
            labels: (N,), dtype=torch.long

        Returns:
            - mode='zero_one' & K=2  -> (N, 1), float32 in {0.0, 1.0}
            - mode='zero_one' & K>2  -> (N, K), float32 one-hot
            - mode='prevalence'      -> (N, K-1), float32 (regular simplex codes, zero ↔ empirical prior)
        """
        if self.mode == 'zero_one':
            if self.n_classes == 2:
                return labels.float().reshape(-1, 1)
            return F.one_hot(labels.long().squeeze(-1), num_classes=self.n_classes).float()

        # prevalence
        C = self._C.to(labels.device)
        return C[labels.long().squeeze(-1)]  # (N, K-1)

    def numerical_to_probas(self, num: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """
        Convert regression outputs back to class probabilities.

        Args:
            num:
              - mode='zero_one':
                  * K=2: (N,1) floats → normalized to (N,2)
                  * K>2: (N,K) floats → normalized to (N,K)
              - mode='prevalence':
                  * (N, K-1) floats (or (K-1,) which is treated as (1, K-1))

            eps: clamp to [eps, 1-eps] before normalization.

        Returns:
            probs: (N, K), float32, rows sum to 1.
        """
        if self.mode == 'zero_one':
            if num.ndim == 1:
                num = num.unsqueeze(-1)
            if num.shape[1] == 1:
                num = torch.cat([1 - num, num], dim=1)  # (N,2)
            num = torch.clamp(num, eps, 1 - eps)
            return num / num.sum(dim=1, keepdim=True)

        # prevalence
        if num.ndim == 1:
            num = num.unsqueeze(0)  # (1, K-1)
        invA = self._invA.to(num.device)
        N = num.shape[0]
        ones = torch.ones((N, 1), dtype=invA.dtype, device=num.device)
        B = torch.cat([num.to(dtype=invA.dtype), ones], dim=1)  # (N, K)
        pi = B @ invA.T                                         # (N, K)
        pi = torch.clamp(pi, eps, 1 - eps)
        return pi / pi.sum(dim=1, keepdim=True)

    def numerical_to_labels(self, num: torch.Tensor) -> torch.Tensor:
        """
        Convert regression outputs to hard labels via argmax over probabilities.

        Args:
            num:
              - mode='zero_one': (N,1) if K=2, else (N,K)
              - mode='prevalence': (N, K-1) or (K-1,)

        Returns:
            labels: (N,), dtype=torch.long
        """
        if self._numerical_type == 'logit_diff' and self.n_classes==2:
            assert self.mode == 'zero_one'
            num = torch.sigmoid(num)
        probs = self.numerical_to_probas(num)
        return probs.argmax(dim=-1)
