import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import set_device_config


class RotaryPositionalEncoding(nn.Module):
  def __init__(self, dim: int, base: int = 10_000):
    super().__init__()
    self.dim = dim
    self.base = base
    self.cos_cached = None
    self.sin_cached = None

  def _neg_half(self, x: torch.Tensor):
    half_dim = self.dim // 2
    return torch.cat([-x[..., half_dim:], x[..., :half_dim]], dim=-1)

  def forward(self, x: torch.Tensor):
    *b, l, _, _ = x.shape
    n_b = len(b)
    if self.cos_cached is None or l > self.cos_cached.shape[n_b]:
        # build cache
        theta = 1. / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
        seq_idx = torch.arange(l, device=x.device).float()
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        self.cos_cached = idx_theta2.cos()[:, None, :][(None,) * n_b]
        self.sin_cached = idx_theta2.sin()[:, None, :][(None,) * n_b]

    neg_half_x = self._neg_half(x)
    x_rope = (x * self.cos_cached[..., :x.shape[n_b], :, :]) + (neg_half_x * self.sin_cached[..., :x.shape[n_b], :, :])
    return x_rope


class LinearAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_in=None,
                 num_mem_kv=4, dropout=0.0, rope=False):
        super().__init__()
        d_in = d_model if d_in is None else d_in

        self.n_heads = n_heads
        self.dropout = dropout
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.to_qkv = nn.Linear(d_in, d_model * 3, bias=False)
        self.mem_kv = nn.Parameter(torch.randn(2, n_heads, num_mem_kv, self.d_head))
        self.rope = RotaryPositionalEncoding(self.d_head) if rope else None

    def forward(self, x, mask=None, kv_cache=False):
        """
        x: (..., sequence_length, feature_size)
        """
        if mask is not None:
            mask = self.prepare_mask(mask, x_ndim=x.ndim)
        *b, l, _ = x.shape

        qkv = self.to_qkv(x).view(*b, l, self.n_heads, -1, 3).transpose(-4, -3)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        if self.rope is not None:
            qk = self.rope(torch.stack((q, k), dim=0))
            q, k = qk[0], qk[1]

        out, kv_cache = self.attend(q, k, v, mask=mask, kv_cache=kv_cache)

        out = out.transpose(-3, -2).contiguous().view(*b, l, -1)
        if kv_cache is not None:
            return out, kv_cache
        return out

    def prepare_mask(self, mask, x_ndim=3):
        """
        mask: (..., sequence_length)
            *mask is node-level in linear attention
        """
        # source-to-target
        if mask.ndim < x_ndim - 1:
            mask = mask[(None,)*(x_ndim-1-mask.ndim)]
        return mask[..., None, :, None]

    def update_kv(self, k, v, kv_cache=False):
        """
        kv_cache: True, False, or (k_cache, v_cache)
        """
        if kv_cache is not False:
            if kv_cache is not True:
                k_cache, v_cache = kv_cache
                k, v = map(functools.partial(torch.cat, dim=-2), ((k_cache, k), (v_cache, v)))
            kv_cache = (k, v)
        else:
            kv_cache = None
        mk, mv = map(lambda t: t.repeat(*k.shape[:-t.ndim], *[1]*t.ndim), self.mem_kv)
        k, v = map(functools.partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        return k, v, kv_cache

    def attend(self, q, k, v, mask=None, kv_cache=False):
        if mask is not None:
            k = (k * mask).masked_fill(mask == 0, float('-inf'))

        k, v, kv_cache = self.update_kv(k, v, kv_cache=kv_cache)

        q = q * self.scale
        q = q.softmax(dim=-1)
        q = F.dropout(q, p=self.dropout, training=self.training)
        k = k.softmax(dim=-2)
        context = torch.einsum('...nd, ...ne -> ...de', k, v)
        out = torch.einsum('...ld, ...de -> ...le', q, context)
        if mask is not None:
            out = out * mask

        return out, kv_cache


class FullAttention(LinearAttention):
    def __init__(self, d_model=128, n_heads=4, d_in=None,
                 num_mem_kv=4, dropout=0.0, rope=False, flash=True):
        super().__init__(d_model, n_heads, d_in, num_mem_kv, dropout, rope)
        self.cpu_config, self.cuda_config = set_device_config(flash)
        self.flash = flash

    def prepare_mask(self, mask, x_ndim=3):
        """
        mask: (..., sequence_length, sequence_length)
            *mask is edge-level in full attention
        """
        # target-to-source
        if mask.ndim < x_ndim:
            mask = mask[(None,)*(x_ndim-mask.ndim)]
        return mask[..., None, :, :]

    def attend(self, q, k, v, mask=None, kv_cache=False):
        k, v, kv_cache = self.update_kv(k, v, kv_cache=kv_cache)

        if mask is not None:
            mask = F.pad(mask, (k.shape[-2] - mask.shape[-1], 0), "constant", 1)

        if self.flash:
            q, k, v = map(lambda t: t.contiguous(), (q, k, v))
            # Check if there is a compatible device for flash attention
            config = self.cuda_config if q.is_cuda else self.cpu_config

            with torch.nn.attention.sdpa_kernel(config):
                dropout_p = self.dropout if self.training else 0.0
                out = F.scaled_dot_product_attention(q, k, v,
                    attn_mask=mask, dropout_p=dropout_p, scale=self.scale)
        else:
            attn = torch.einsum(f"...ld, ...md -> ...lm", q, k)
            attn = attn * self.scale
            if mask.dtype == torch.bool:
                attn = attn.masked_fill(mask == 0, float('-inf'))
            else:
                attn = attn + mask
            attn = attn.softmax(dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.einsum(f"...lm, ...me -> ...le", attn, v)

        return out, kv_cache
