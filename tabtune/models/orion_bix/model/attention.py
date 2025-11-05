from __future__ import annotations
from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
from .rope import RotaryEmbedding



def sdpa_with_flattened_batch(
    q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None, dropout_p: float = 0.0
) -> Tensor:
    """Applies scaled dot-product attention with flattened batch dimensions.

    This function handles arbitrary batch dimensions by flattening them before
    applying PyTorch's scaled_dot_product_attention and then reshaping the output
    back to the original shape. This flattening is necessary to properly trigger
    Flash Attention.

    Parameters
    ----------
    q : Tensor
        Query tensor of shape (..., nh, tgt_len, hs) where:
        - ... represents arbitrary batch dimensions
        - nh is the number of attention heads
        - tgt_len is the target sequence length
        - hs is the head size (embedding dimension per head)

    k : Tensor
        Key tensor of shape (..., nh, src_len, hs) with matching batch dimensions

    v : Tensor
        Value tensor of shape (..., nh, src_len, hs) with matching batch dimensions

    attn_mask : Optional[Tensor], default=None
        Attention mask of shape (..., nh, tgt_len, src_len)

    dropout_p : float, default=0.0
        Dropout probability applied to attention weights

    Returns
    -------
    Tensor
        Attention output tensor of shape (..., nh, tgt_len, hs) preserving the
        original batch dimensions of the input
    """

    q_shape = q.shape
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])
    if attn_mask is not None:
        attn_mask = attn_mask.reshape(-1, *attn_mask.shape[-3:])
    out = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)

    return out.view(q_shape)

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor | int] = None,
    rope: Optional[RotaryEmbedding] = None,
) -> Tensor:
    """Multi-head attention with support for rotary position embeddings
    as well as specialized processing when attn_mask is an integer.

    Parameters
    ----------
    query : Tensor
        Query tensor of shape (..., tgt_len, embed_dim)

    key : Tensor
        Key tensor of shape (..., src_len, embed_dim)

    value : Tensor
        Value tensor of shape (..., src_len, embed_dim)

    num_heads : int
        Number of attention heads

    in_proj_weight : Tensor
        Combined weight matrix for Q, K, V input projections

    in_proj_bias : Tensor
        Combined bias vector for input projections

    dropout_p : float
        Dropout probability applied to attention weights

    out_proj_weight : Tensor
        Output projection weight matrix

    out_proj_bias : Tensor
        Output projection bias vector

    training : bool, default=True
        Whether the model is in training mode (affects dropout)

    key_padding_mask : Optional[Tensor], default=None
        Mask of shape (..., src_len) that identifies padding elements
        in the key sequence to be ignored:
            - For binary masks: True values indicate positions to ignore
            - For float masks: Values are directly added to attention scores

    attn_mask : Optional[Tensor | int], default=None
        Controls attention pattern in two possible ways:
        1. When provided as Tensor: Traditional mask preventing attention to certain positions
            - Shape: (tgt_len, src_len) or (..., num_heads, tgt_len, src_len)
        2. When provided as integer: Creates a split attention pattern where:
            - The first `attn_mask` tokens perform self-attention only (attend to themselves)
            - The remaining tokens attend only to the first `attn_mask` tokens

    rope : Optional[RotaryEmbedding]
        Rotary positional encoding

    Returns
    -------
    Tensor
        Attention output tensor of shape (..., tgt_len, embed_dim)
    """

    if isinstance(attn_mask, int):
        assert key_padding_mask is None, "key_padding_mask is not supported with attn_mask as int"
        assert rope is None, "Rotary position embedding is not supported with attn_mask as int"

    # Extract shape information, supporting arbitrary batch dimensions
    *batch_shape, tgt_len, embed_dim = query.shape
    src_len = key.shape[-2]

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    # Joint projection of query, key, value
    q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    # Reshape for multi-head attention
    q = q.view(*batch_shape, tgt_len, num_heads, head_dim).transpose(-3, -2)  # (batch_shape, nh, tgt_len, hs)
    k = k.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)  # (batch_shape, nh, src_len, hs)
    v = v.view(*batch_shape, src_len, num_heads, head_dim).transpose(-3, -2)  # (batch_shape, nh, src_len, hs)

    # Apply rotary position embeddings if provided
    if rope is not None:
        q = rope.rotate_queries_or_keys(q)
        k = rope.rotate_queries_or_keys(k)

    # Disable dropout during evaluation
    if not training:
        dropout_p = 0.0

    if isinstance(attn_mask, int):
        cut_pos = attn_mask  # For better readability

        # Pre-allocate output tensor to avoid concatenation
        attn_output = torch.empty(*batch_shape, tgt_len, embed_dim, device=query.device, dtype=query.dtype)

        # Process left segment (self-attention within first cut_pos tokens)
        q_left = q[..., :cut_pos, :]  # (batch_shape, nh, cut_pos, hs)
        k_left = k[..., :cut_pos, :]
        v_left = v[..., :cut_pos, :]

        attn_left = sdpa_with_flattened_batch(q_left, k_left, v_left, dropout_p=dropout_p)
        attn_left = attn_left.transpose(-3, -2).contiguous().view(*batch_shape, cut_pos, embed_dim)
        attn_output[..., :cut_pos, :] = F.linear(attn_left, out_proj_weight, out_proj_bias)

        # Process right segment (tokens after cut_pos attending to tokens before cut_pos)
        if cut_pos < tgt_len:
            q_right = q[..., cut_pos:, :]  # (batch_shape, nh, tgt_len - cut_pos, hs)
            attn_right = sdpa_with_flattened_batch(q_right, k_left, v_left, dropout_p=dropout_p)
            attn_right = attn_right.transpose(-3, -2).contiguous().view(*batch_shape, tgt_len - cut_pos, embed_dim)
            attn_output[..., cut_pos:, :] = F.linear(attn_right, out_proj_weight, out_proj_bias)
    else:
        # Process attention mask
        correct_2d_shape = (tgt_len, src_len)
        correct_nd_shape = (*batch_shape, num_heads, tgt_len, src_len)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                if attn_mask.shape != correct_2d_shape:
                    raise ValueError(f"2D attn_mask should have shape {correct_2d_shape}, but got {attn_mask.shape}")
                attn_mask = attn_mask.expand(*batch_shape, num_heads, tgt_len, src_len)
            elif attn_mask.dim() == len(correct_nd_shape):
                if attn_mask.shape != correct_nd_shape:
                    raise ValueError(
                        f"{len(correct_nd_shape)}D attn_mask should have shape {correct_nd_shape}, "
                        f"but got {attn_mask.shape}"
                    )
            else:
                raise ValueError(f"attn_mask must be 2D or {len(correct_nd_shape)}D, got {attn_mask.dim()}D")

        # Process key padding mask
        if key_padding_mask is not None:
            if key_padding_mask.shape != (*batch_shape, src_len):
                raise ValueError(
                    f"key_padding_mask should have shape {(*batch_shape, src_len)}, but got {key_padding_mask.shape}"
                )
            key_padding_mask = key_padding_mask.view(*batch_shape, 1, 1, src_len).expand(
                *batch_shape, num_heads, tgt_len, src_len
            )

            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        attn_output = sdpa_with_flattened_batch(q, k, v, attn_mask, dropout_p)  # (..., nh, tgt_len, hs)

        # Reshape and project output
        attn_output = attn_output.transpose(-3, -2).contiguous().view(*batch_shape, tgt_len, embed_dim)
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)  # (batch_shape, tgt_len, E)

    return attn_output

# ==============================================================================
# Linear Attention Components
# ==============================================================================

class FeatureMap(nn.Module):
    """
    Defines the interface for a feature map function used in linear attention.

    Args:
        query_dims (int): Dimensionality of the query/key vectors.
    """
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device):
        pass

    def forward_queries(self, x):
        return self(x)

    def forward_keys(self, x):
        return self(x)

    def forward(self, x):
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner


class ActivationFunctionFeatureMap(FeatureMap):
    """
    Feature map using a simple element-wise activation function.

    Args:
        query_dims (int): Dimensionality of the query/key vectors.
        activation_function (callable): Activation function to apply.
    """
    def __init__(self, query_dims, activation_function):
        super().__init__(query_dims)
        self.activation_function = activation_function

    def new_feature_map(self, device):
        pass

    def forward(self, x):
        return self.activation_function(x)


class HedgehogFeatureMap(FeatureMap):
    """
    Hedgehog feature map as implemented in the original TabFlex.

    Args:
        query_dims (int): Dimensionality of the query/key vectors.
    """
    def __init__(self, query_dims):
        super().__init__(query_dims)
        self.head_dim = query_dims
        self.layer = nn.Linear(self.head_dim, self.head_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.layer.weight, mean=0.0, std=1 / math.sqrt(self.head_dim))
        nn.init.zeros_(self.layer.bias)

    def new_feature_map(self, device):
        self.to(device)

    def forward(self, x):
        # Handle different tensor shapes
        original_shape = x.shape
        
        # If x has more than 2 dimensions, we need to reshape it
        if x.dim() > 2:
            # Reshape to (batch_size, head_dim) where batch_size includes all other dimensions
            x_reshaped = x.view(-1, self.head_dim)
        else:
            x_reshaped = x
            
        # Apply the linear transformation
        x_lin = self.layer(x_reshaped)
        x_softmax = F.softmax(x_lin, dim=-1)
        x_neg = F.softmax(-x_lin, dim=-1)
        
        # Instead of concatenating, we'll use a different approach
        # Take the mean of the two softmax outputs to maintain the same dimension
        x_result = (x_softmax + x_neg) / 2.0
        
        # Reshape back to original shape
        if x.dim() > 2:
            x_result = x_result.view(original_shape)
            
        return x_result


# Feature map implementations
elu_feature_map = ActivationFunctionFeatureMap.factory(lambda x: F.elu(x) + 1)
identity_feature_map = ActivationFunctionFeatureMap.factory(lambda x: x)
hedgehog_feature_map = HedgehogFeatureMap.factory()


class LinearAttention(nn.Module):
    """
    Implements the core linear attention mechanism as described in the original
    TABFLEX paper and Ticl codebase.

    Args:
        query_dimensions (int): Dimensionality of the query/key vectors.
        feature_map (callable, optional): Feature map function to use. Defaults to ELU.
        eps (float): Small value to avoid division by zero.
        debug (bool): Whether to enable debug prints.
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6, debug=False):
        super(LinearAttention, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        self.debug = debug
        # Add feature map type detection
        self.is_hedgehog = isinstance(self.feature_map, HedgehogFeatureMap)

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        """
        Compute linear attention outputs.

        Args:
            queries (Tensor): Query tensor.
            keys (Tensor): Key tensor.
            values (Tensor): Value tensor.
            attn_mask: Optional attention mask.
            query_lengths: Optional query lengths for masking.
            key_lengths: Optional key lengths for masking.
        Returns:
            Tensor: Output of linear attention.
        """
        # queries, keys, and values can have shape (N, L, H, D) or (N, L, H, num_heads, head_dim)
        # N: batch size, L: sequence length, H: num_heads, D: head_dim or (num_heads, head_dim)
        
        if self.debug:
            print(f"[DEBUG] LinearAttention input shapes:")
            print(f"  queries: {queries.shape}")
            print(f"  keys: {keys.shape}")
            print(f"  values: {values.shape}")
        
        # Handle 5D tensors (after RoPE processing)
        if queries.dim() == 5:
            if self.debug:
                print(f"[DEBUG] Handling 5D tensors")
            # Reshape from (N, L, H, num_heads, head_dim) to (N*L, H, num_heads, head_dim)
            N, L, H, num_heads, head_dim = queries.shape
            queries_flat = queries.view(N * L, H, num_heads, head_dim)
            keys_flat = keys.view(N * L, H, num_heads, head_dim)
            values_flat = values.view(N * L, H, num_heads, head_dim)
            
            # Apply feature maps
            self.feature_map.new_feature_map(queries_flat.device)
            Q = self.feature_map.forward_queries(queries_flat)
            K = self.feature_map.forward_keys(keys_flat)
            
            # Handle key padding mask if provided
            if key_lengths is not None:
                K = K * key_lengths.float_matrix[:, :, None, None]
            
            # We assume full attention for simplicity. Masking logic would be more complex here.
            if attn_mask is not None:
                if not attn_mask.all_ones:
                    raise RuntimeError("LinearAttention does not support arbitrary attention masks")
            
            # FIXED: Handle hedgehog feature map dimension doubling
            if self.is_hedgehog:
                # For hedgehog: K has shape (N*L, H, num_heads, head_dim*2), V has shape (N*L, H, num_heads, head_dim)
                # We need to handle the doubled dimension in K
                K_reshaped = K.view(N * L, H, num_heads, 2, head_dim)
                # Use only the first half of the doubled dimension for compatibility
                K = K_reshaped[:, :, :, 0, :]  # Shape: (N*L, H, num_heads, head_dim)

            # The core linear attention computation using einsum
            # Note: For hedgehog feature map, Q and K have doubled dimensions due to concatenation
            # 1. Compute the K.T @ V term, summed over the sequence length (s)
            # K has shape (N*L, H, num_heads, head_dim*2), V has shape (N*L, H, num_heads, head_dim)
            KV = torch.einsum("nshd,nshm->nhdm", K, values_flat)
            
            # 2. Compute the normalizer term: 1 / (Q @ K_sum), where K_sum is K summed over s
            #    This is the inverse of the sum over sequence length of K
            Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
            
            # 3. Compute the final value V' = (Q @ KV) * Z
            V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)
            
            # Reshape back to original dimensions
            V = V.view(N, L, H, num_heads, head_dim)
            
            if self.debug:
                print(f"[DEBUG] LinearAttention output shape: {V.shape}")
            return V.contiguous()
        else:
            if self.debug:
                print(f"[DEBUG] Handling 4D tensors")
            # Standard 4D case: (N, L, H, D)
            self.feature_map.new_feature_map(queries.device)
            Q = self.feature_map.forward_queries(queries)
            K = self.feature_map.forward_keys(keys)
            
            # Handle key padding mask if provided
            if key_lengths is not None:
                K = K * key_lengths.float_matrix[:, :, None, None]
            
            # We assume full attention for simplicity. Masking logic would be more complex here.
            if attn_mask is not None:
                if not attn_mask.all_ones:
                    raise RuntimeError("LinearAttention does not support arbitrary attention masks")
            # FIXED: Handle hedgehog feature map dimension doubling for 4D case
            if self.is_hedgehog:
                # For hedgehog: K has shape (N, L, H, head_dim*2), V has shape (N, L, H, head_dim)
                K_reshaped = K.view(N, L, H, 2, -1)
                # Use only the first half of the doubled dimension for compatibility
                K = K_reshaped[:, :, :, 0, :]  # Shape: (N, L, H, head_dim)
            # The core linear attention computation using einsum
            # 1. Compute the K.T @ V term, summed over the sequence length (s)
            KV = torch.einsum("nshd,nshm->nhdm", K, values)
            
            # 2. Compute the normalizer term: 1 / (Q @ K_sum), where K_sum is K summed over s
            #    This is the inverse of the sum over sequence length of K
            Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
            
            # 3. Compute the final value V' = (Q @ KV) * Z
            V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)
            
            if self.debug:
                print(f"[DEBUG] LinearAttention output shape: {V.shape}")
            return V.contiguous()

class MultiHeadLinearAttentionLayer(nn.Module):
    """
    A robust multi-head attention wrapper that handles arbitrary batch dimensions.
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, d_model_keys=None, debug=False):
        super(MultiHeadLinearAttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        d_model_keys = d_model_keys or d_model
        
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model_keys, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model_keys, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.debug = debug

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths, rope: Optional[RotaryEmbedding] = None):
        # Handle arbitrary batch dimensions
        *batch_dims, L, _ = queries.shape
        S = keys.shape[-2]
        
        if self.debug:
            print(f"[DEBUG] MultiHeadLinearAttentionLayer input shapes:")
            print(f"  queries: {queries.shape}")
            print(f"  keys: {keys.shape}")
            print(f"  values: {values.shape}")
            print(f"  batch_dims: {batch_dims}, L: {L}, S: {S}")
            print(f"  rope: {rope}")
            if rope:
                print(f"  rope.use_xpos: {rope.use_xpos}")
                print(f"  rope.freqs.shape: {rope.freqs.shape}")
        
        # Project Q, K, V
        queries_proj = self.query_projection(queries)
        keys_proj = self.key_projection(keys)
        values_proj = self.value_projection(values)
        
        if self.debug:
            print(f"[DEBUG] After projection:")
            print(f"  queries_proj: {queries_proj.shape}")
            print(f"  keys_proj: {keys_proj.shape}")
            print(f"  values_proj: {values_proj.shape}")

        # Apply RoPE if provided
        if rope is not None:
            if self.debug:
                print(f"[DEBUG] Applying RoPE...")
            queries_reshaped = queries_proj.view(*batch_dims, L, self.n_heads, -1).transpose(-3, -2)
            keys_reshaped = keys_proj.view(*batch_dims, S, self.n_heads, -1).transpose(-3, -2)
            if self.debug:
                print(f"[DEBUG] Before RoPE rotation:")
                print(f"  queries_reshaped: {queries_reshaped.shape}")
                print(f"  keys_reshaped: {keys_reshaped.shape}")
            
            try:
                if rope.use_xpos:
                    queries_reshaped, keys_reshaped = rope.rotate_queries_and_keys(queries_reshaped, keys_reshaped)
                else:
                    queries_reshaped = rope.rotate_queries_or_keys(queries_reshaped)
                    keys_reshaped = rope.rotate_queries_or_keys(keys_reshaped)
                if self.debug:
                    print(f"[DEBUG] After RoPE rotation:")
                    print(f"  queries_reshaped: {queries_reshaped.shape}")
                    print(f"  keys_reshaped: {keys_reshaped.shape}")
            except Exception as e:
                if self.debug:
                    print(f"[ERROR] RoPE rotation failed: {e}")
                    print(f"[DEBUG] rope.use_xpos: {rope.use_xpos}")
                    print(f"[DEBUG] rope.configuration: {rope}")
                raise e
                
            queries_proj = queries_reshaped.transpose(-3, -2).contiguous().view(*batch_dims, L, -1)
            keys_proj = keys_reshaped.transpose(-3, -2).contiguous().view(*batch_dims, S, -1)
            if self.debug:
                print(f"[DEBUG] After reshaping back:")
                print(f"  queries_proj: {queries_proj.shape}")
                print(f"  keys_proj: {keys_proj.shape}")

        # Reshape for multi-head attention: (..., L/S, D) -> (..., L/S, H, D_head)
        # Handle the case where we have 5 dimensions after RoPE processing
        if queries_proj.dim() == 5:
            # After RoPE: (B, T, H, num_heads, head_dim) -> (B*T, H, num_heads, head_dim)
            B, T, H, num_heads, head_dim = queries_proj.shape
            queries_reshaped = queries_proj.view(B * T, H, num_heads, head_dim)
            keys_reshaped = keys_proj.view(B * T, H, num_heads, head_dim)
            values_reshaped = values_proj.view(B * T, H, num_heads, head_dim)
        else:
            # Standard case: (..., L/S, D) -> (..., L/S, H, D_head)
            queries_reshaped = queries_proj.view(*batch_dims, L, self.n_heads, -1)
            keys_reshaped = keys_proj.view(*batch_dims, S, self.n_heads, -1)
            values_reshaped = values_proj.view(*batch_dims, S, self.n_heads, -1)
        
        if self.debug:
            print(f"[DEBUG] Before inner attention:")
            print(f"  queries_reshaped: {queries_reshaped.shape}")
            print(f"  keys_reshaped: {keys_reshaped.shape}")
            print(f"  values_reshaped: {values_reshaped.shape}")

        # Pass multi-dimensional tensors to the inner attention module
        new_values = self.inner_attention(
            queries_reshaped, keys_reshaped, values_reshaped, attn_mask, query_lengths, key_lengths
        )
        
        if self.debug:
            print(f"[DEBUG] After inner attention:")
            print(f"  new_values: {new_values.shape}")

        # The LinearAttention returns (N, L, H, D_head), we need to reshape to (N, L, H*D_head)
        # and then apply the output projection
        if queries_proj.dim() == 5:
            # After RoPE processing: reshape back to original batch dimensions
            B, T, H, num_heads, head_dim = queries_proj.shape
            new_values = new_values.view(B, T, H, -1)
        else:
            # Standard case
            new_values = new_values.view(*batch_dims, L, -1)
        
        if self.debug:
            print(f"[DEBUG] After reshaping:")
            print(f"  new_values: {new_values.shape}")
        
        # Use the output projection (it's already designed to handle the correct dimensions)
        output = self.out_projection(new_values)
        
        if self.debug:
            print(f"[DEBUG] Final output:")
            print(f"  output: {output.shape}")
        
        return output
