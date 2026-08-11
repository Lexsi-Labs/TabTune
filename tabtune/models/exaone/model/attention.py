"""Small tensor-only attention module used by the tabular models."""

from __future__ import annotations

import math
import numbers

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn import functional as F

from ..runtime import FLASH_ATTENTION_BATCH_LIMIT


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

MEM_EFFICIENT_MAX_SEQUENCE = 64
MEM_EFFICIENT_HEAD_WIDTH_MULTIPLE = 8

_SDPA_BACKEND_CHOICES = ("auto", "flash", "mem_efficient")


def _select_sdpa_backend(
    query_length: int, context_length: int, head_width: int, *, on_cuda: bool, policy: str
) -> SDPBackend:
    """Pick the SDPA kernel for one attention call, by its longer sequence.

    Flash is the answer whenever mem-efficient cannot serve the call at all --
    off CUDA, where it has no build, and at a head width it rejects. Both hold
    even under an explicit ``mem_efficient`` pin, because the alternative is not
    a slower kernel but no kernel: SDPA raises outright.
    """
    if policy == "flash" or not on_cuda:
        return SDPBackend.FLASH_ATTENTION
    if head_width % MEM_EFFICIENT_HEAD_WIDTH_MULTIPLE:
        return SDPBackend.FLASH_ATTENTION
    if policy == "mem_efficient":
        return SDPBackend.EFFICIENT_ATTENTION
    if max(query_length, context_length) <= MEM_EFFICIENT_MAX_SEQUENCE:
        return SDPBackend.EFFICIENT_ATTENTION
    return SDPBackend.FLASH_ATTENTION


class TensorAttention(nn.Module):
    """Unmasked multi-head self- and cross-attention."""

    def __init__(
        self,
        embedding_dim: int,
        head_count: int,
        *,
        initial_head_scale: float = 0.43,
        learned_head_scale: bool = True,
        score_scale: float | None = None,
        sdpa_backend: str = "auto",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._check_positive_integer("embedding_dim", embedding_dim)
        self._check_positive_integer("head_count", head_count)
        if embedding_dim % head_count:
            raise ValueError("embedding_dim must be divisible by head_count")
        self._check_real("initial_head_scale", initial_head_scale)
        if not math.isfinite(float(initial_head_scale)):
            raise ValueError("initial_head_scale must be finite")
        if not isinstance(learned_head_scale, bool):
            raise TypeError("learned_head_scale must be bool")
        if score_scale is not None:
            self._check_real("score_scale", score_scale)
            if not math.isfinite(float(score_scale)) or float(score_scale) <= 0:
                raise ValueError("score_scale must be finite and positive")
        if not isinstance(sdpa_backend, str):
            raise TypeError("sdpa_backend must be a string")
        if sdpa_backend not in _SDPA_BACKEND_CHOICES:
            raise ValueError("unsupported sdpa_backend")
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype or None")
        chosen_dtype = torch.get_default_dtype() if dtype is None else dtype
        if chosen_dtype not in _FLOAT_DTYPES:
            raise ValueError("unsupported parameter dtype")

        self._embedding_dim = embedding_dim
        self._head_count = head_count
        self._head_width = embedding_dim // head_count
        self._score_scale = score_scale
        self._sdpa_backend = sdpa_backend

        factory = {"device": device, "dtype": chosen_dtype}
        self.query_weight = nn.Parameter(torch.empty((embedding_dim, embedding_dim), **factory))
        self.key_weight = nn.Parameter(torch.empty((embedding_dim, embedding_dim), **factory))
        self.value_weight = nn.Parameter(torch.empty((embedding_dim, embedding_dim), **factory))
        self.output_weight = nn.Parameter(torch.empty((embedding_dim, embedding_dim), **factory))
        scale = torch.full((head_count,), float(initial_head_scale), **factory)
        if learned_head_scale:
            self.head_scale = nn.Parameter(scale)
        else:
            self.register_buffer("head_scale", scale, persistent=False)
        for weight in (
            self.query_weight,
            self.key_weight,
            self.value_weight,
            self.output_weight,
        ):
            nn.init.xavier_uniform_(weight)

        self.register_buffer("_cached_key", None, persistent=False)
        self.register_buffer("_cached_value", None, persistent=False)
        self.register_buffer("_packed_qkv_weight", None, persistent=False)
        self.register_buffer("_packed_kv_weight", None, persistent=False)
        self.register_buffer("_packed_first_kv_weight", None, persistent=False)
        self._packed_weight_signature: tuple[object, ...] | None = None

    @staticmethod
    def _check_positive_integer(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    @staticmethod
    def _check_real(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a real scalar")

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def head_count(self) -> int:
        return self._head_count

    @property
    def head_width(self) -> int:
        return self._head_width

    @property
    def score_scale(self) -> float | None:
        return self._score_scale

    @property
    def sdpa_backend(self) -> str:
        """Kernel policy: "auto" picks per call, the others pin every call."""
        return self._sdpa_backend

    @property
    def has_cached_context(self) -> bool:
        return self._cached_key is not None and self._cached_value is not None

    def clear_cache(self) -> None:
        self._cached_key = None
        self._cached_value = None

    def _inference_projection_weight(self, kind: str) -> torch.Tensor:
        """Return a lazily packed projection weight for inference GEMMs."""
        signature = tuple(
            (weight._version, weight.data_ptr(), weight.device, weight.dtype)
            for weight in (self.query_weight, self.key_weight, self.value_weight)
        )
        if signature != self._packed_weight_signature:
            self._packed_qkv_weight = None
            self._packed_kv_weight = None
            self._packed_first_kv_weight = None
            self._packed_weight_signature = signature

        if kind == "qkv":
            if self._packed_qkv_weight is None:
                self._packed_qkv_weight = torch.cat(
                    (self.query_weight, self.key_weight, self.value_weight), dim=0
                ).contiguous()
            return self._packed_qkv_weight
        if kind == "kv":
            if self._packed_kv_weight is None:
                self._packed_kv_weight = torch.cat(
                    (self.key_weight, self.value_weight), dim=0
                ).contiguous()
            return self._packed_kv_weight
        if kind == "first_kv":
            if self._packed_first_kv_weight is None:
                self._packed_first_kv_weight = torch.cat(
                    (
                        self.key_weight[: self.head_width],
                        self.value_weight[: self.head_width],
                    ),
                    dim=0,
                ).contiguous()
            return self._packed_first_kv_weight
        raise ValueError("unsupported packed projection kind")

    def _project_context(
        self, context: torch.Tensor, *, first_head_only: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_grad_enabled():
            key_heads = self._as_heads(F.linear(context, self.key_weight))
            value_heads = self._as_heads(F.linear(context, self.value_weight))
            if first_head_only:
                key_heads = key_heads[..., :1, :, :].contiguous()
                value_heads = value_heads[..., :1, :, :].contiguous()
            return key_heads, value_heads

        if first_head_only:
            projected = F.linear(
                context, self._inference_projection_weight("first_kv")
            )
            key, value = projected.split(self.head_width, dim=-1)
            return key.unsqueeze(-3), value.unsqueeze(-3)
        projected = F.linear(context, self._inference_projection_weight("kv"))
        key, value = projected.split(self.embedding_dim, dim=-1)
        return self._as_heads(key), self._as_heads(value)

    def cache_context_projections(
        self, context: torch.Tensor, *, first_head_only: bool = False
    ) -> None:
        """Cache context K/V without evaluating attention query outputs."""
        if not isinstance(first_head_only, bool):
            raise TypeError("first_head_only must be bool")
        context = self._validate_tensor("context", context)
        key_heads, value_heads = self._project_context(
            context, first_head_only=first_head_only
        )
        key_heads.mul_(math.log(key_heads.shape[-2]))
        self._cached_key, self._cached_value = _retain_or_detach(key_heads, value_heads)

    def offload_cached_context(self) -> None:
        """Move a retained context to host memory without changing its values.

        Large tabular supports can make the otherwise small per-layer cache add
        up across the transformer stack. Host-resident caches keep only the
        layer currently being evaluated on the accelerator.
        """
        if not self.has_cached_context:
            raise RuntimeError("no cached context is available")
        assert self._cached_key is not None and self._cached_value is not None
        self._cached_key = self._cached_key.to(device="cpu")
        self._cached_value = self._cached_value.to(device="cpu")

    def reload_cached_context(self, device: torch.device | str) -> None:
        """Move a host-resident context cache back onto ``device`` in place.

        Inverse of ``offload_cached_context``: after a support build offloaded
        each layer's K/V to host memory (keeping the build's working region
        clear), one bulk reload before the query phase restores GPU residency so
        cached cross-attention incurs no per-chunk host transfer. A no-op when
        there is no cache or it already sits on a device of the same type -- so a
        CPU-resident model reloading to its own device does nothing (and never
        reaches for an absent accelerator).
        """
        if self._cached_key is None:
            return
        assert self._cached_value is not None
        if self._cached_key.device.type == torch.device(device).type:
            return
        self._cached_key = self._cached_key.to(device=device)
        self._cached_value = self._cached_value.to(device=device)

    def retain_first_cached_context_head(self) -> None:
        """Keep only the context head used by multi-query cross-attention.

        ``reuse_first_context_head`` deliberately ignores every other cached
        key/value head, so once the support self-attention has completed only
        the first head has to stay cached. Dropping the rest does not change a
        subsequent cached cross-attention call.
        """
        if not self.has_cached_context:
            raise RuntimeError("no cached context is available")
        assert self._cached_key is not None and self._cached_value is not None
        self._cached_key = self._cached_key[..., :1, :, :].contiguous()
        self._cached_value = self._cached_value[..., :1, :, :].contiguous()

    def _validate_tensor(self, name: str, tensor: object) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim < 2:
            raise ValueError(f"{name} must have rank at least two")
        if tensor.shape[-2] <= 0:
            raise ValueError(f"{name} sequence length must be positive")
        if tensor.shape[-1] != self.embedding_dim:
            raise ValueError(f"{name} has the wrong embedding dimension")
        if tensor.dtype != self.query_weight.dtype or tensor.device != self.query_weight.device:
            raise ValueError(f"{name} dtype and device must match the module")
        return tensor

    def _as_heads(self, projected: torch.Tensor) -> torch.Tensor:
        # (..., sequence, embedding) -> (..., heads, sequence, head width)
        return projected.unflatten(-1, (self.head_count, self.head_width)).transpose(-3, -2)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        cache_context: bool = False,
        use_cached_context: bool = False,
        reuse_first_context_head: bool = False,
    ) -> torch.Tensor:
        for name, option in (
            ("cache_context", cache_context),
            ("use_cached_context", use_cached_context),
            ("reuse_first_context_head", reuse_first_context_head),
        ):
            if not isinstance(option, bool):
                raise TypeError(f"{name} must be bool")
        if context is not None and not isinstance(context, torch.Tensor):
            raise TypeError("context must be a torch.Tensor or None")
        if cache_context and use_cached_context:
            raise ValueError("cache_context and use_cached_context are mutually exclusive")
        if use_cached_context and context is not None:
            raise ValueError("cached context cannot be combined with explicit context")
        if reuse_first_context_head and context is None and not use_cached_context:
            raise ValueError("head reuse requires explicit or cached context")

        query = self._validate_tensor("query", query)
        query_batch = query.shape[:-2]
        self_projected_key: torch.Tensor | None = None
        self_projected_value: torch.Tensor | None = None
        if context is None and not use_cached_context and not torch.is_grad_enabled():
            projected = F.linear(
                query, self._inference_projection_weight("qkv")
            )
            projected_query, self_projected_key, self_projected_value = projected.split(
                self.embedding_dim, dim=-1
            )
            query_heads = self._as_heads(projected_query)
        else:
            query_heads = self._as_heads(F.linear(query, self.query_weight))
        query_heads.mul_(
            self.head_scale.reshape(
                *((1,) * (query_heads.ndim - 3)), self.head_count, 1, 1
            )
        )

        if use_cached_context:
            if not self.has_cached_context:
                raise RuntimeError("no cached context is available")
            key_heads = self._cached_key
            value_heads = self._cached_value
            assert key_heads is not None and value_heads is not None
            if key_heads.shape[:-3] != query_batch or key_heads.dtype != query.dtype:
                raise RuntimeError("cached context is incompatible with query")
            if key_heads.device != query.device:
                if key_heads.device.type != "cpu":
                    raise RuntimeError("cached context is on an unsupported device")
                key_heads = key_heads.to(
                    device=query.device, non_blocking=key_heads.is_pinned()
                )
                value_heads = value_heads.to(
                    device=query.device, non_blocking=value_heads.is_pinned()
                )
        else:
            selected = query if context is None else self._validate_tensor("context", context)
            if selected.shape[:-2] != query_batch:
                raise ValueError("query and context batch dimensions must match")
            if self_projected_key is not None and self_projected_value is not None:
                key_heads = self._as_heads(self_projected_key)
                value_heads = self._as_heads(self_projected_value)
            else:
                key_heads, value_heads = self._project_context(
                    selected,
                    first_head_only=reuse_first_context_head and not cache_context,
                )
            key_heads.mul_(math.log(key_heads.shape[-2]))

        # Cache the already context-length-scaled K and unmodified V. The same
        # context length is intrinsic to a retained projection, so later calls
        # can reuse the exact scaled bits instead of allocating another K copy.
        # Head reuse remains a per-call view and does not change cache selection.
        cache_key = key_heads if cache_context else None
        cache_value = value_heads if cache_context else None

        if reuse_first_context_head:
            key_heads = key_heads[..., :1, :, :].expand(*key_heads.shape[:-3], self.head_count, *key_heads.shape[-2:])
            value_heads = value_heads[..., :1, :, :].expand(*value_heads.shape[:-3], self.head_count, *value_heads.shape[-2:])

        context_length = key_heads.shape[-2]
        prefix = query_heads.shape[:-3]
        query_length = query_heads.shape[-2]
        flat_query = query_heads.reshape(
            -1, self.head_count, query_length, self.head_width
        )
        flat_key = key_heads.reshape(
            -1, self.head_count, context_length, self.head_width
        )
        flat_value = value_heads.reshape(
            -1, self.head_count, context_length, self.head_width
        )
        attention_options = {"dropout_p": 0.0}
        if self.score_scale is not None:
            attention_options["scale"] = self.score_scale
        # Chunking splits the flat batch, never the sequence: one choice per call.
        backend = _select_sdpa_backend(
            query_length,
            context_length,
            self.head_width,
            on_cuda=flat_query.is_cuda,
            policy=self._sdpa_backend,
        )
        split_flat_batch = (
            backend is SDPBackend.FLASH_ATTENTION
            and flat_query.shape[0] > FLASH_ATTENTION_BATCH_LIMIT
        )
        with sdpa_kernel([backend]):
            if not split_flat_batch:
                attended = F.scaled_dot_product_attention(
                    flat_query,
                    flat_key,
                    flat_value,
                    **attention_options,
                )
            else:
                attended = torch.cat(
                    [
                        F.scaled_dot_product_attention(
                            flat_query[start : start + FLASH_ATTENTION_BATCH_LIMIT],
                            flat_key[start : start + FLASH_ATTENTION_BATCH_LIMIT],
                            flat_value[start : start + FLASH_ATTENTION_BATCH_LIMIT],
                            **attention_options,
                        )
                        for start in range(
                            0,
                            flat_query.shape[0],
                            FLASH_ATTENTION_BATCH_LIMIT,
                        )
                    ],
                    dim=0,
                )
        attended = attended.reshape(
            *prefix, self.head_count, query_length, self.head_width
        )
        del flat_query, flat_key, flat_value, query_heads
        if not cache_context:
            del key_heads, value_heads
        joined = attended.transpose(-3, -2).flatten(-2)
        output = F.linear(joined, self.output_weight)

        if cache_context:
            # Assign only after all computations succeed, replacing the pair together.
            assert cache_key is not None and cache_value is not None
            self._cached_key, self._cached_value = _retain_or_detach(cache_key, cache_value)
        return output



def _retain_or_detach(
    key: "torch.Tensor", value: "torch.Tensor"
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Detach a retained K/V pair for inference; keep it in the graph under autograd.

    TabTune modification. Upstream detaches unconditionally here. That is correct
    and cheaper for pure inference -- the cache exists so the query rows can
    cross-attend a support context that was computed once -- but it means the
    query rows read the support representation *out of the autograd graph*, in
    every layer, on every forward that has query rows to predict.

    The consequence is not a loud failure. ``d(query logit)/d(support row)`` still
    returns a non-``None`` gradient, because three side paths survive the cut: the
    row-summary tokens, the feature encoder's support-only normalisation
    statistics, and the parameters. The number it returns is simply wrong -- it
    disagrees with a finite-difference check in magnitude and frequently in sign.
    Two attention parameters in the final layer also never receive a gradient at
    all, since ``cache_context_projections`` is their only consumer there.

    Gating on ``torch.is_grad_enabled()`` keeps inference byte-identical to
    upstream (the detach still runs under ``no_grad``/``inference_mode``, which is
    how every prediction path executes) while making the support-side backward
    exact when a caller has deliberately enabled gradients -- which is what
    support-set attribution and support-side fine-tuning need.
    """
    if torch.is_grad_enabled():
        return key, value
    return key.detach(), value.detach()


__all__ = ["TensorAttention", "MEM_EFFICIENT_MAX_SEQUENCE"]
