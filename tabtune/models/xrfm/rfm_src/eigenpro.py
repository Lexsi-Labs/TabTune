'''Construct kernel model with EigenPro optimizer.'''
import collections
import time
import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from .svd import nystrom_kernel_svd

def asm_eigenpro_fn(samples, map_fn, top_q, bs_gpu, alpha, min_q=5, seed=1, verbose=True):
    """Prepare gradient map for EigenPro and calculate
    scale factor for learning ratesuch that the update rule,
        p <- p - eta * g
    becomes,
        p <- p - scale * eta * (g - eigenpro_fn(g))

    Arguments:
        samples:	matrix of shape (n_sample, n_feature).
        map_fn:    	kernel k(samples, centers) where centers are specified.
        top_q:  	top-q eigensystem for constructing eigenpro iteration/kernel.
        bs_gpu:     maxinum batch size corresponding to GPU memory.
        alpha:  	exponential factor (<= 1) for eigenvalue rescaling due to approximation.
        min_q:  	minimum value of q when q (if None) is calculated automatically.
        seed:   	seed for random number generation.
        verbose:    whether to print outputs.

    Returns:
        eigenpro_fn:	tensor function.
        scale:  		factor that rescales learning rate.
        top_eigval:  	largest eigenvalue.
        beta:   		largest k(x, x) for the EigenPro kernel.
    """

    np.random.seed(seed)  # set random seed for subsamples
    start = time.time()
    n_sample, _ = samples.shape

    if top_q is None:
        svd_q = min(n_sample - 1, 400)
    else:
        svd_q = top_q

    
    eigvals, eigvecs = nystrom_kernel_svd(samples, map_fn, svd_q)

    # Choose q such that the batch size is bounded by
    #   the subsample size and the memory size.
    #   Keep the original q if it is pre-specified.
    if top_q is None:
        print("Computing top_q")
        max_bs = min(max(n_sample / 5, bs_gpu), n_sample)
        top_q = torch.sum(torch.pow(1 / eigvals, alpha) < max_bs) - 1
        top_q = max(top_q, min_q)

    if verbose:
        print("top_q", top_q, "svd_q", svd_q)
    eigvals, tail_eigval = eigvals[:top_q - 1], eigvals[top_q - 1]
    eigvecs = eigvecs[:, :top_q - 1]

    device = samples.device
    eigvals_t = eigvals.to(device)
    eigvecs_t = eigvecs.to(device)
    tail_eigval_t = tail_eigval.to(dtype=samples.dtype, device=device)

    scale = torch.pow(eigvals[0] / tail_eigval, alpha).to(samples.dtype)
    diag_t = (1 - torch.pow(tail_eigval_t / eigvals_t, alpha)) / eigvals_t

    def eigenpro_fn(grad, kmat):
        '''Function to apply EigenPro preconditioner.'''
        return torch.mm(eigvecs_t * diag_t,
                        torch.t(torch.mm(torch.mm(torch.t(grad),
                                                  kmat),
                                         eigvecs_t)))

    if verbose:
        print("SVD time: %.2f, top_q: %d, top_eigval: %.2f, new top_eigval: %.2e" %
              (time.time() - start, top_q, eigvals[0], eigvals[0] / scale))

    #beta = kmat.diag().max()
    knorms = 1 - torch.sum(eigvecs ** 2, dim=1) * n_sample
    beta = torch.max(knorms)

    return eigenpro_fn, scale.item(), eigvals[0].item(), beta.to(samples.dtype).item()


class KernelModel(nn.Module):
    '''Fast Kernel Regression using EigenPro iteration.'''
    def __init__(self, kernel_fn, centers, y_dim, device="cuda"):
        super(KernelModel, self).__init__()
        self.kernel_fn = kernel_fn
        self.n_centers, self.x_dim = centers.shape
        self.device = device
        self.pinned_list = []

        self.centers = self.tensor(centers, release=True, dtype=centers.dtype)
        self.weight = self.tensor(torch.zeros(
            self.n_centers, y_dim), release=True, dtype=centers.dtype)
        
        self.save_kernel_matrix = False
        self.kernel_matrix = [] if self.save_kernel_matrix else None

    def __del__(self):
        for pinned in self.pinned_list:
            _ = pinned.to("cpu")

    def tensor(self, data, dtype=None, release=False):
        if torch.is_tensor(data) and data.device == self.device:
            tensor = data.detach()
        elif torch.is_tensor(data):
            tensor = data.detach().to(self.device)
        else:
            tensor = torch.tensor(data, requires_grad=False, device=self.device)

        if release:
            self.pinned_list.append(tensor)
        return tensor

    def get_kernel_matrix(self, batch, batch_ids, samples=None, sample_ids=None):
        if batch_ids is not None and self.save_kernel_matrix and isinstance(self.kernel_matrix, torch.Tensor):
            if samples is None or sample_ids is None:
                kmat = self.kernel_matrix[batch_ids]
            else:
                kmat = self.kernel_matrix[batch_ids][:, sample_ids]
        else:
            if samples is None or sample_ids is None:
                kmat = self.kernel_fn(batch, self.centers)
            else:
                kmat = self.kernel_fn(batch, samples)
        return kmat

    def forward(self, batch, batch_ids=None, weight=None, save_kernel_matrix=False):
        if weight is None:
            weight = self.weight
        kmat = self.get_kernel_matrix(batch, batch_ids)
        if save_kernel_matrix: # only call if self.kernel_matrix is a list
            self.kernel_matrix.append((batch_ids.cpu(), kmat.cpu()))
        pred = kmat.mm(weight)
        return pred

    def primal_gradient(self, batch, labels, weight, batch_ids, save_kernel_matrix=False):
        pred = self.forward(batch, batch_ids, weight, save_kernel_matrix)
        grad = pred - labels
        return grad

    @staticmethod
    def _compute_opt_params(bs, bs_gpu, beta, top_eigval):
        if bs is None:
            bs = min(np.int32(beta / top_eigval + 1), bs_gpu)

        if bs < beta / top_eigval + 1:
            eta = bs / beta
        else:
            eta = 0.99 * 2 * bs / (beta + (bs - 1) * top_eigval)
        return bs, float(eta)

    def eigenpro_iterate(self, samples, x_batch, y_batch, eigenpro_fn,
                         eta, sample_ids, batch_ids, save_kernel_matrix=False):
        # update random coordiate block (for mini-batch)
        grad = self.primal_gradient(x_batch, y_batch, self.weight, batch_ids, save_kernel_matrix)
        self.weight.index_add_(0, batch_ids, -eta * grad)

        # update fixed coordinate block (for EigenPro)
        kmat = self.get_kernel_matrix(x_batch, batch_ids, samples, sample_ids)
        correction = eigenpro_fn(grad, kmat)
        self.weight.index_add_(0, sample_ids, eta * correction)
        return

    def evaluate(self, X_eval, y_eval, bs=None, metrics=('mse')):
        p_list = []
        n_eval, _ = X_eval.shape

        if bs is None:
            n_batch = 1
        else:
            n_batch = n_eval // min(n_eval, bs)

        for batch_ids in np.array_split(np.arange(n_eval), n_batch):
            x_batch = self.tensor(X_eval[batch_ids], dtype=X_eval.dtype)
            p_batch = self.forward(x_batch)
            p_list.append(p_batch)
        p_eval = torch.concat(p_list, dim=0).to(self.device)

        eval_metrics = collections.OrderedDict()
        if 'mse' in metrics:
            eval_metrics['mse'] = torch.mean(torch.square(p_eval - y_eval)).item()
        if 'multiclass-acc' in metrics:
            y_class = torch.argmax(y_eval, dim=-1)
            p_class = torch.argmax(p_eval, dim=-1)
            eval_metrics['multiclass-acc'] = torch.sum(y_class == p_class).item() / len(y_eval)
        if 'binary-acc' in metrics:
            y_class = torch.where(y_eval > 0.5, 1, 0).reshape(-1)
            p_class = torch.where(p_eval > 0.5, 1, 0).reshape(-1)
            eval_metrics['binary-acc'] = torch.sum(y_class == p_class).item() / len(y_eval)
        if 'f1' in metrics:
            y_class = torch.where(y_eval > 0.5, 1, 0).reshape(-1)
            p_class = torch.where(p_eval > 0.5, 1, 0).reshape(-1)
            eval_metrics['f1'] = torch.mean(2 * (y_class * p_class) / (y_class + p_class + 1e-8)).item()
        if 'auc' in metrics:
            eval_metrics['auc'] = roc_auc_score(y_eval.cpu().flatten(), p_eval.cpu().flatten())

        return eval_metrics

    def fit(self, X_train, y_train, X_val, y_val, epochs, mem_gb,
            n_subsamples=None, top_q=None, bs=None, eta=None,
            n_eval=1000, run_epoch_eval=True, lr_scale=1, 
            verbose=True, seed=1, classification=False, threshold=1e-5,
            early_stopping_window_size=7, eval_interval=1):
        
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)

        n_eval = min(n_eval, len(X_train), len(X_val))
        train_eval_ids = np.random.choice(len(X_train), n_eval, replace=False)

        X_train_eval = X_train[train_eval_ids].clone()
        y_train_eval = y_train[train_eval_ids].clone()

        assert(len(X_train)==len(y_train))
        assert(len(X_val)==len(y_val))

        metrics = ('mse',)
        if classification:
            if y_train.shape[-1] == 1:
                metrics += ('binary-acc', 'f1', 'auc')
            else:
                metrics += ('multiclass-acc',)

        n_samples, n_labels = y_train.shape
        if n_subsamples is None:
            n_subsamples = min(n_samples, 12000)
        n_subsamples = min(n_subsamples, n_samples)

        mem_bytes = (mem_gb - 1) * 1024**3  # preserve 1GB
        bsizes = np.arange(n_subsamples)
        mem_usages = ((self.x_dim + 3 * n_labels + bsizes + 1)
                      * self.n_centers + n_subsamples * 1000) * 4
        bs_gpu = np.sum(mem_usages < mem_bytes)  # device-dependent batch size

        # Calculate batch size / learning rate for improved EigenPro iteration.
        np.random.seed(seed)
        sample_ids = np.random.choice(n_samples, n_subsamples, replace=False)
        sample_ids = self.tensor(sample_ids)
        samples = self.centers[sample_ids]
        eigenpro_f, gap, top_eigval, beta = asm_eigenpro_fn(
            samples, self.kernel_fn, top_q, bs_gpu, alpha=.95, seed=seed, verbose=verbose)
        new_top_eigval = top_eigval / gap

        if eta is None:
            bs, eta = self._compute_opt_params(
                bs, bs_gpu, beta, new_top_eigval)
        else:
            bs, _ = self._compute_opt_params(bs, bs_gpu, beta, new_top_eigval)

        if verbose:
            print("n_subsamples=%d, bs_gpu=%d, eta=%.2f, bs=%d, top_eigval=%.2e, beta=%.2f" %
                  (n_subsamples, bs_gpu, eta, bs, top_eigval, beta))
        eta = self.tensor(lr_scale * eta / bs, dtype=torch.float)


        res = dict()
        initial_epoch = 0
        train_sec = 0  # training time in seconds
        best_weights = None
        if classification:
            best_metric = 0
        else:
            best_metric = float('inf')
        
        # Add early stopping variables
        val_loss_history = []

        for epoch in range(epochs):
            start = time.time()
            for _ in range(epoch - initial_epoch):
                # Create a permutation of all indices
                epoch_ids = np.random.permutation(n_samples)

                save_kernel_matrix = epoch==1 and self.save_kernel_matrix

                for batch_ids in tqdm(np.array_split(epoch_ids, n_samples // bs)):
                    batch_ids = self.tensor(batch_ids)
                    x_batch = self.tensor(X_train[batch_ids], dtype=X_train.dtype)
                    y_batch = self.tensor(y_train[batch_ids], dtype=y_train.dtype)
                    self.eigenpro_iterate(samples, x_batch, y_batch, eigenpro_f,
                                          eta, sample_ids, batch_ids, save_kernel_matrix)
                    del x_batch, y_batch, batch_ids

                if save_kernel_matrix:
                    print(f"Storing kernel matrix")
                    # First concatenate all rows
                    concat_matrix = torch.cat([pair[1] for pair in self.kernel_matrix], dim=0)
                    # Get all batch indices and their positions
                    all_batch_ids = torch.cat([pair[0] for pair in self.kernel_matrix])
                    # Get sorting indices and reorder the matrix
                    _, sort_indices = torch.sort(all_batch_ids)
                    self.kernel_matrix = concat_matrix[sort_indices]
                    self.kernel_matrix = self.kernel_matrix.to(self.device)

            if run_epoch_eval and epoch%eval_interval==0:
                train_sec += time.time() - start
                eval_start = time.time()
                tr_score = self.evaluate(X_train_eval, y_train_eval, bs=bs, metrics=metrics)
                eval_time = time.time() - eval_start
                print(f"Train Eval time: {eval_time} seconds")

                eval_start = time.time()
                tv_score = self.evaluate(X_val, y_val, bs=bs, metrics=metrics)
                eval_time = time.time() - eval_start
                print(f"Val Eval time: {eval_time} seconds")
                
                if verbose:
                    out_str = f"({epoch} epochs, {train_sec} seconds)\t train l2: {tr_score['mse']} \tval l2: {tv_score['mse']}"
                    if classification:
                        if 'binary-acc' in tr_score:
                            out_str += f"\ttrain binary acc: {tr_score['binary-acc']} \tval binary acc: {tv_score['binary-acc']}"
                        else:
                            out_str += f"\ttrain multiclass acc: {tr_score['multiclass-acc']} \tval multiclass acc: {tv_score['multiclass-acc']}"
                        if 'f1' in tr_score:
                            out_str += f"\ttrain f1: {tr_score['f1']} \tval f1: {tv_score['f1']}"
                        if 'auc' in tr_score:
                            out_str += f"\ttrain auc: {tr_score['auc']} \tval auc: {tv_score['auc']}"
                    print(out_str)

                res[epoch] = (tr_score, tv_score, train_sec)

                # Track validation loss changes
                if 'binary-acc' in tv_score:
                    val_loss_history.append(tv_score['binary-acc'] <= best_metric)
                elif 'multiclass-acc' in tv_score:
                    val_loss_history.append(tv_score['multiclass-acc'] <= best_metric)
                else:
                    val_loss_history.append(tv_score['mse'] >= best_metric)
                if len(val_loss_history) > early_stopping_window_size:
                    val_loss_history.pop(0)
                    # Check if validation loss increased in majority of recent iterations
                    if sum(val_loss_history) / len(val_loss_history) >= 0.8:
                        if verbose:
                            print(f"Early stopping triggered: validation loss increased in majority of last {early_stopping_window_size} epochs")
                        break

                if classification:
                    if 'auc' in tv_score:
                        if tv_score['auc'] > best_metric:
                            best_metric = tv_score['auc']
                            best_weights = self.weight.cpu().clone()
                            val_loss_history = []
                            print(f"New best auc: {best_metric}")
                    elif 'binary-acc' in tv_score:
                        if tv_score['binary-acc'] > best_metric:
                            best_metric = tv_score['binary-acc']
                            best_weights = self.weight.cpu().clone()
                            val_loss_history = []
                            print(f"New best binary-acc: {best_metric}")
                    elif 'multiclass-acc' in tv_score:
                        if tv_score['multiclass-acc'] > best_metric:
                            best_metric = tv_score['multiclass-acc']
                            best_weights = self.weight.cpu().clone()
                            val_loss_history = []
                            print(f"New best multiclass-acc: {best_metric}")
                else:
                    if tv_score['mse'] < best_metric:
                        best_metric = tv_score['mse']
                        best_weights = self.weight.cpu().clone()
                        val_loss_history = []
                        print(f"New best mse: {best_metric}")

                if tr_score['mse'] < threshold:
                    break

            initial_epoch = epoch

        self.weight = best_weights.to(self.device)

        if self.kernel_matrix is not None:
            del self.kernel_matrix

        return res
