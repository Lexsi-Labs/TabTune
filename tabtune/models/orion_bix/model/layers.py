
# --- model/layers.py ---
from __future__ import annotations
from typing import List, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .attention import multi_head_attention_forward, elu_feature_map, identity_feature_map, hedgehog_feature_map, LinearAttention, MultiHeadLinearAttentionLayer
from .rope import RotaryEmbedding
from .inference import InferenceManager
from .inference_config import MgrConfig

class ClassNode:
    """Node in the hierarchical classification tree for handling many-class problems.

    Attributes
    ----------
    depth : int
        Current depth level in the hierarchical tree

    is_leaf : bool
        Whether this node handles a small enough subset of classes directly

    classes_ : Tensor
        List of unique class indices this node is responsible for

    child_nodes : list
        Child nodes for non-leaf nodes, each handling a subset of classes

    class_mapping : dict
        Maps original class indices to group indices for internal nodes

    group_indices : Tensor
        Transformed labels after mapping original classes to their group indices

    R : Tensor
        Feature data associated with this node

    y : Tensor
        Target labels associated with this node
    """

    def __init__(self, depth=0):
        self.depth = depth
        self.is_leaf = False
        self.classes_ = None
        self.child_nodes = []
        self.class_mapping = {}
        self.group_indices = None
        self.R = None
        self.y = None

class OneHotAndLinear(nn.Linear):
    """Combines one-hot encoding and linear projection in a single efficient operation
    to convert categorical indices to embeddings.

    Parameters
    ----------
    num_classes : int
        Number of distinct categories for one-hot encoding

    embed_dim : int
        Output embedding dimension
    """

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__(num_classes, embed_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

    def forward(self, src: Tensor) -> Tensor:
        """Transform integer indices to dense embeddings.

        Parameters
        ----------
        src : Tensor
            Integer tensor of shape (batch_size, sequence_length) containing category indices

        Returns
        -------
        Tensor
            Embedded representation of shape (batch_size, sequence_length, embed_dim)
        """
        # Convert indices to one-hot vectors and apply linear projection
        one_hot = F.one_hot(src.long(), self.num_classes).to(src.dtype)
        return F.linear(one_hot, self.weight, self.bias)

class SkippableLinear(nn.Linear):
    """Linear layer that handles inputs where all values equal `skip_value`.

    First applies the linear transformation to all inputs, then replaces outputs for inputs
    where all values equal `skip_value` with the `skip_value`.

    Parameters
    ----------
    in_features : int
        Size of each input sample

    out_features : int
        Size of each output sample

    bias : bool, default=True
        If set to False, the layer will not learn an additive bias

    skip_value : float, default=-100.0
        Value used to mark inputs that should be skipped
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, skip_value: float = -100.0):
        super().__init__(in_features, out_features, bias)
        self.skip_value = skip_value

    def forward(self, src: Tensor) -> Tensor:
        """Forward pass that handles inputs flagged with `skip_value`.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., in_features)

        Returns
        -------
        Tensor
            Output tensor of shape (..., out_features) where rows corresponding
            to skipped inputs are filled with `skip_value`
        """

        out = F.linear(src, self.weight, self.bias)
        skip_mask = (src == self.skip_value).all(dim=-1)
        if skip_mask.any():
            out[skip_mask] = self.skip_value

        return out

class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture.

    Parameters
    ----------
    in_dim : int
        Input feature dimension

    out_dim : Optional[int], default=None
        Output dimension. If None, uses the last hidden dimension

    hidden_dims : List[int], default=[256, 256, 256]
        Dimensions of hidden layers

    activation : str, default='gelu'
        Activation function: 'relu', 'gelu', 'leaky_relu', or 'tanh'

    bias : bool, default=True
        Whether to include bias terms in linear layers
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        hidden_dims: List[int] = [256, 256, 256],
        activation: str = "gelu",
        bias: bool = True,
    ):
        super().__init__()
        # Build network architecture
        act_fn = self.get_activation(activation)
        layers = []

        # Create hidden layers with activations
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=bias))
            layers.append(act_fn())
            prev_dim = hidden_dim

        # Optional output projection
        if out_dim is not None:
            layers.append(nn.Linear(prev_dim, out_dim, bias=bias))

        self.net = nn.Sequential(*layers)

    @staticmethod
    def get_activation(activation: str) -> nn.Module:
        """Get activation function class from string name.

        Parameters
        ----------
        activation : str
            Name of activation function

        Returns
        -------
        class
            PyTorch activation function class
        """

        activation_map = {
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }

        if activation not in activation_map:
            raise ValueError(f"Unknown activation: {activation}. Supported: {list(activation_map.keys())}")

        return activation_map[activation]

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass through the MLP.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (..., in_dim)

        Returns
        -------
        Tensor
            Output tensor of shape (..., out_dim or last_hidden_dim)
        """
        return self.net(X)

class MultiheadAttention(nn.MultiheadAttention):
    """Enhanced multi-head attention with rotary positional embedding support.

    This extends PyTorch's MultiheadAttention to support rotary position embeddings (RoPE)
    and specialized attention masking when `attn_mask` is an integer. The implementation always
    uses `batch_first=True`, meaning all input tensors have shape (..., seq_len, embed_dim).

    Parameters
    ----------
    embed_dim : int
        Model dimension (total size of each attention head combined)

    num_heads : int
        Number of attention heads

    dropout : float, default=0.0
        Dropout probability applied to attention weights

    References
    ----------
    .. [1] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
           https://arxiv.org/abs/2104.09864
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__(embed_dim, num_heads, dropout, batch_first=True)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor | int] = None,
        rope: Optional[RotaryEmbedding] = None,
    ) -> Tensor:
        """Compute multi-head attention with support for rotary positional encoding.

        Parameters
        ----------
        query : Tensor
            Query tensor of shape (..., tgt_len, embed_dim)

        key : Tensor
            Key tensor of shape (..., src_len, embed_dim)

        value : Tensor
            Value tensor of shape (..., src_len, embed_dim)

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
            Attention output of shape (..., tgt_len, embed_dim)
        """

        if isinstance(attn_mask, int):
            assert key_padding_mask is None, "key_padding_mask is not supported with attn_mask as int"
            assert rope is None, "Rotary position embedding is not supported with attn_mask as int"

        return multi_head_attention_forward(
            query,
            key,
            value,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            rope=rope,
        )

class MultiheadAttentionBlock(nn.TransformerEncoderLayer):
    """Attention block supporting rotary positional encoding.

    Parameters
    ----------
    d_model : int
        Model dimension

    nhead : int
        Number of attention heads

    dim_feedforward : int
       Dimension of the feedforward network

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
    ):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, norm_first=norm_first, batch_first=True)
        del self.self_attn
        self.attn = MultiheadAttention(d_model, nhead, dropout)
        self.init_weights()

    def init_weights(self):
        """Initialize projection layers to zero for stable training."""
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor | int] = None,
        rope: Optional[RotaryEmbedding] = None,
    ) -> Tensor:
        """Process input through attention with optional rotary positional encoding.

        Parameters
        ----------
        q : Tensor
            Query tensor of shape (..., tgt_len, d_model)

        k : Tensor
            Key tensor of shape (..., src_len, d_model)
            If None, uses q for self-attention.

        v : Tensor
            Value tensor of shape (..., src_len, d_model)
            If None, uses q for self-attention.

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
            Output tensor of shape (..., tgt_len, d_model)
        """

        if isinstance(attn_mask, int):
            assert key_padding_mask is None, "key_padding_mask is not supported with attn_mask as int"
            assert rope is None, "Rotary position embedding is not supported with attn_mask as int"
        else:
            # Convert masks to correct dtype for compatibility
            key_padding_mask = F._canonical_mask(
                mask=key_padding_mask,
                mask_name="key_padding_mask",
                other_type=F._none_or_dtype(attn_mask),
                other_name="src_mask",
                target_type=q.dtype,
            )
            attn_mask = F._canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=q.dtype,
                check_other=False,
            )

        # Use q as k,v if not provided
        k = q if k is None else k
        v = q if v is None else v

        # Apply layer depending on normalization order
        x = q
        if self.norm_first:
            # Pre-norm: normalize before attention and FFN
            attn = self._attn_block(self.norm1(q), self.norm1(k), self.norm1(v), key_padding_mask, attn_mask, rope)
            x = x + attn
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-norm: normalize after attention and FFN
            attn = self._attn_block(q, k, v, key_padding_mask, attn_mask, rope)
            x = self.norm1(x + attn)
            x = self.norm2(x + self._ff_block(x))

        return x

    def _attn_block(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor | int],
        rope: Optional[RotaryEmbedding],
    ) -> Tensor:
        attn = self.attn(q, k, v, key_padding_mask, attn_mask, rope)
        return self.dropout1(attn)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class InducedSelfAttentionBlock(nn.Module):
    """Induced Self-Attention for efficient O(n) attention on large sets.

    This module implements a bottleneck attention mechanism using a small set of
    learned inducing points that mediate interactions between input elements.
    The complexity is reduced from O(n²) to O(n) by:

    1. Projecting inputs onto inducing points (size m << n)
    2. Propagating information through these inducing points
    3. Projecting back to the original sequence

    Parameters
    ----------
    d_model : int
        Model dimension

    nhead : int
        Number of attention heads

    dim_feedforward : int
        Dimension of the feedforward network

    num_inds : int
        Number of inducing points (controls capacity vs. efficiency)

    dropout : float, default=0.0
        Dropout probability

    activation : str or unary callable, default="gelu"
        The activation function used in the feedforward network, can be
        either string ("relu" or "gelu") or unary callable

    norm_first : bool, default=True
        If True, uses pre-norm architecture (LayerNorm before attention and feedforward)

    skip_value : float, default=-100.0
        Value used to mark inputs that should be skipped

    References
    ----------
    .. [1] Lee et al. "Set Transformer: A Framework for Attention-based
           Permutation-Invariant Neural Networks", ICML 2019
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        skip_value: float = -100.0,
    ):
        super().__init__()
        self.skip_value = skip_value

        # Two-stage attention mechanism
        self.multihead_attn1 = MultiheadAttentionBlock(d_model, nhead, dim_feedforward, dropout, activation, norm_first)
        self.multihead_attn2 = MultiheadAttentionBlock(d_model, nhead, dim_feedforward, dropout, activation, norm_first)

        # Learnable inducing points
        self.num_inds = num_inds
        self.ind_vectors = nn.Parameter(torch.empty(num_inds, d_model))
        nn.init.trunc_normal_(self.ind_vectors, std=0.02)

    def induced_attention(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        """Apply induced self-attention to input sequence.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model)

        train_size : Optional[int], default=None
            Position to split the input into training and test data

        Returns
        -------
        Tensor
            Output tensor with same shape as input
        """

        *batch_shape, _, d_model = src.shape
        ind_vectors = self.ind_vectors.expand(*batch_shape, self.num_inds, d_model)

        if train_size is None:
            hidden = self.multihead_attn1(ind_vectors, src, src)
        else:
            hidden = self.multihead_attn1(ind_vectors, src[..., :train_size, :], src[..., :train_size, :])

        out = self.multihead_attn2(src, hidden, hidden)

        return out

    def forward(self, src: Tensor, train_size: Optional[int] = None) -> Tensor:
        """Apply induced self-attention to input sequence.

        Parameters
        ----------
        src : Tensor
            Input tensor of shape (..., seq_len, d_model)

        train_size : Optional[int], default=None
            Position to split the input into training and test data. When provided,
            inducing points will only attend to training data in the first attention
            stage to prevent information leakage from test data during evaluation.

        Returns
        -------
        Tensor
            Output tensor with same shape as input
        """

        skip_mask = (src == self.skip_value).all(dim=(-2, -1))  # batch shape
        if skip_mask.any():
            if skip_mask.all():
                out = torch.full_like(src, self.skip_value)
            else:
                out = torch.empty_like(src)
                out[~skip_mask] = self.induced_attention(src[~skip_mask], train_size)
                out[skip_mask] = self.skip_value
        else:
            out = self.induced_attention(src, train_size)

        return out


class LinearAttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", debug=False):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
        self.debug = debug

    def forward(self, x, src_mask=None, rope: Optional[RotaryEmbedding] = None):
        # x has shape (B, T, E) where:
        # B = batch size, T = sequence length, E = embedding dim
        
        if self.debug:
            print(f"[DEBUG] LinearAttentionTransformerEncoderLayer input:")
            print(f"  x: {x.shape}")
            print(f"  src_mask: {src_mask}")
            print(f"  rope: {rope}")
            if rope:
                print(f"  rope.use_xpos: {rope.use_xpos}")
                print(f"  rope.freqs.shape: {rope.freqs.shape}")
        
        if isinstance(src_mask, int):
            single_eval_position = src_mask
            if self.debug:
                print(f"[DEBUG] Train/test split mode with eval_pos: {single_eval_position}")
            
            # Split into train and test samples like original TabFlex
            # x has shape (B, T, E) where B=batch, T=sequence, E=embedding
            train_samples = x[:, :single_eval_position, :]
            test_samples = x[:, single_eval_position:, :]
            
            if self.debug:
                print(f"[DEBUG] Split samples:")
                print(f"  train_samples: {train_samples.shape}")
                print(f"  test_samples: {test_samples.shape}")
            
            # Training samples attend only to themselves
            attn_left = self.attention(train_samples, train_samples, train_samples, None, None, None, rope=rope)
            
            # Testing samples attend to training samples
            if test_samples.shape[1] > 0:  # If there are test samples
                attn_right = self.attention(test_samples, train_samples, train_samples, None, None, None, rope=rope)
                attn_output = torch.cat([attn_left, attn_right], dim=1)
            else:
                attn_output = attn_left
        else:
            if self.debug:
                print(f"[DEBUG] Standard attention mode")
            # Standard case - no train/test split
            attn_output = self.attention(x, x, x, None, None, None, rope=rope)
        
        if self.debug:
            print(f"[DEBUG] After attention:")
            print(f"  attn_output: {attn_output.shape}")
        
        x = x + self.dropout(attn_output)
        
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        
        output = self.norm2(x + y)
        
        if self.debug:
            print(f"[DEBUG] Final output:")
            print(f"  output: {output.shape}")
        
        return output


class LinearAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        skip_value: float = -100.0,
        feature_map: str = "elu",  # "elu", "identity", "hedgehog"
        debug: bool = False,
    ):
        super().__init__()
        self.skip_value = skip_value
        self.debug = debug

        # Select feature map based on configuration
        if feature_map == "elu":
            feature_map_func = elu_feature_map
            head_dim = d_model // nhead
        elif feature_map == "identity":
            feature_map_func = identity_feature_map
            head_dim = d_model // nhead
        elif feature_map == "hedgehog":
            feature_map_func = hedgehog_feature_map
            # Hedgehog feature map doubles the dimension
            head_dim = (d_model // nhead) * 2
        else:
            raise ValueError(f"Unknown feature_map: {feature_map}. Must be one of ['elu', 'identity', 'hedgehog']")

        attention_mech = LinearAttention(
            query_dimensions=d_model // nhead,
            feature_map=feature_map_func,
            debug=debug
        )
        attention_layer = MultiHeadLinearAttentionLayer(
            attention=attention_mech,
            d_model=d_model,
            n_heads=nhead,
            d_keys=head_dim,  # Use the actual head dimension
            d_values=head_dim,  # Use the actual head dimension
            debug=debug
        )
        
        self.transformer_layer = LinearAttentionTransformerEncoderLayer(
            attention=attention_layer,
            d_model=d_model,
            d_ff=dim_feedforward,
            dropout=dropout,
            activation=activation,
            debug=debug
        )

    def forward(self, src: Tensor, train_size: Optional[int] = None, rope: Optional[RotaryEmbedding] = None) -> Tensor:
        if self.debug:
            print(f"[DEBUG] LinearAttentionBlock input:")
            print(f"  src: {src.shape}")
            print(f"  train_size: {train_size}")
            print(f"  rope: {rope}")
            if rope:
                print(f"  rope.use_xpos: {rope.use_xpos}")
                print(f"  rope.freqs.shape: {rope.freqs.shape}")

        
        skip_mask = (src == self.skip_value).all(dim=(-2, -1))
        if self.debug:
            print(f"[DEBUG] Skip mask:")
            print(f"  skip_mask: {skip_mask.shape}, any: {skip_mask.any()}, all: {skip_mask.all()}")
        
        if skip_mask.any():
            if skip_mask.all():
                if self.debug:
                    print(f"[DEBUG] All values are skip values")
                out = torch.full_like(src, self.skip_value)
            else:
                if self.debug:
                    print(f"[DEBUG] Partial skip values")
                out = torch.empty_like(src)
                # Pass train_size as src_mask for proper train/test split logic
                out[~skip_mask] = self.transformer_layer(src[~skip_mask], src_mask=train_size, rope=rope)
                out[skip_mask] = self.skip_value
        else:
            if self.debug:
                print(f"[DEBUG] No skip values")
            # Pass train_size as src_mask for proper train/test split logic
            out = self.transformer_layer(src, src_mask=train_size, rope=rope)
        
        if self.debug:
            print(f"[DEBUG] LinearAttentionBlock output:")
            print(f"  out: {out.shape}")
        
        return out

"""
True Bi-Axial Attention for tabular data.

This module implements alternating attention patterns within feature space:
- Standard Cross-Feature Attention
- Grouped Feature Attention
- Standard Cross-Feature Attention
- Hierarchical Feature Attention
- Standard Cross-Feature Attention
- Relational Feature Attention
- Alternating Pattern: Standard → Grouped → Standard → Hierarchical → Standard → Relational
"""

class StandardCrossFeatureAttention(nn.Module):
    """Standard cross-feature attention within each sample."""
    
    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.attention = MultiheadAttentionBlock(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply standard cross-feature attention."""
        return self.attention(q=x, k=x, v=x)


class GroupedFeatureAttention(nn.Module):
    """Grouped feature attention - processes features in groups."""
    
    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        dim_feedforward: int,
        num_groups: int = 4,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.attention = MultiheadAttentionBlock(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        
        # Group projection
        self.group_proj = nn.Linear(embed_dim, embed_dim)
        self.group_ln = nn.LayerNorm(embed_dim) if norm_first else nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply grouped feature attention."""
        batch_size, num_features, embed_dim = x.shape
        
        # Group features
        features_per_group = num_features // self.num_groups
        x_grouped = x[:, :features_per_group * self.num_groups, :].view(
            batch_size, self.num_groups, features_per_group, embed_dim
        )
        
        # Apply attention within groups
        x_grouped = self.group_ln(self.group_proj(x_grouped))
        x_grouped = self.attention(q=x_grouped, k=x_grouped, v=x_grouped)
        
        # Ungroup features
        x_ungrouped = x_grouped.view(batch_size, features_per_group * self.num_groups, embed_dim)
        
        # Handle remaining features
        if num_features > features_per_group * self.num_groups:
            remaining = x[:, features_per_group * self.num_groups:, :]
            x_ungrouped = torch.cat([x_ungrouped, remaining], dim=1)
        
        return x_ungrouped


class HierarchicalFeatureAttention(nn.Module):
    """Hierarchical feature attention - processes features in hierarchical patterns."""
    
    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.attention = MultiheadAttentionBlock(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        
        # Hierarchical projection
        self.hier_proj = nn.Linear(embed_dim, embed_dim)
        self.hier_ln = nn.LayerNorm(embed_dim) if norm_first else nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply hierarchical feature attention."""
        # Create hierarchical structure (first half attends to second half)
        batch_size, num_features, embed_dim = x.shape
        mid_point = num_features // 2
        
        first_half = x[:, :mid_point, :]
        second_half = x[:, mid_point:, :]
        
        # First half attends to second half
        first_half = self.hier_ln(self.hier_proj(first_half))
        first_half = self.attention(q=first_half, k=second_half, v=second_half)
        
        # Second half attends to first half
        second_half = self.hier_ln(self.hier_proj(second_half))
        second_half = self.attention(q=second_half, k=first_half, v=first_half)
        
        return torch.cat([first_half, second_half], dim=1)


class RelationalFeatureAttention(nn.Module):
    """Relational feature attention - processes feature relationships."""
    
    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.attention = MultiheadAttentionBlock(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        
        # Relational projection
        self.rel_proj = nn.Linear(embed_dim, embed_dim)
        self.rel_ln = nn.LayerNorm(embed_dim) if norm_first else nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply relational feature attention."""
        # Create relational structure (every feature attends to every other feature)
        x_rel = self.rel_ln(self.rel_proj(x))
        return self.attention(q=x_rel, k=x_rel, v=x_rel)


class CLSAttention(nn.Module):
    """Attention mechanism for CLS tokens to attend to features."""
    
    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.norm_first = norm_first
        
        # Multi-head attention for CLS tokens
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim) if norm_first else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim) if norm_first else nn.Identity()
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, cls_tokens: Tensor, features: Tensor) -> Tensor:
        """
        CLS tokens attend to features.
        
        Parameters
        ----------
        cls_tokens : Tensor
            CLS tokens of shape (batch_size, num_cls, embed_dim)
        features : Tensor
            Features of shape (batch_size, num_features, embed_dim)
            
        Returns
        -------
        Tensor
            Updated CLS tokens of shape (batch_size, num_cls, embed_dim)
        """
        if self.norm_first:
            # Pre-norm architecture
            cls_norm = self.norm1(cls_tokens)
            features_norm = self.norm1(features)
            
            # CLS tokens attend to features
            attn_out, _ = self.attn(cls_norm, features_norm, features_norm)
            cls_tokens = cls_tokens + self.dropout(attn_out)
            
            # FFN
            cls_norm = self.norm2(cls_tokens)
            ffn_out = self.ffn(cls_norm)
            cls_tokens = cls_tokens + self.dropout(ffn_out)
        else:
            # Post-norm architecture
            attn_out, _ = self.attn(cls_tokens, features, features)
            cls_tokens = self.norm1(cls_tokens + self.dropout(attn_out))
            
            ffn_out = self.ffn(cls_tokens)
            cls_tokens = self.norm2(cls_tokens + self.dropout(ffn_out))
        
        return cls_tokens

class BiAxialAttentionBlock(nn.Module):
    """Single Bi-Axial Attention Block with alternating patterns."""
    
    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        dim_feedforward: int,
        num_cls: int = 4,
        rope_base: float = 100000,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.num_cls = num_cls
        self.norm_first = norm_first
        
        # Alternating attention patterns
        self.standard_attention = StandardCrossFeatureAttention(
            embed_dim=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        
        self.grouped_attention = GroupedFeatureAttention(
            embed_dim=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        
        self.hierarchical_attention = HierarchicalFeatureAttention(
            embed_dim=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        
        self.relational_attention = RelationalFeatureAttention(
            embed_dim=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        
        # CLS tokens for feature aggregation
        self.cls_tokens = nn.Parameter(torch.empty(num_cls, embed_dim))
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)
        
        # CLS attention mechanism
        self.cls_attention = CLSAttention(
            embed_dim=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        OPTIMIZED: Apply 4 attention patterns instead of 6 for better performance.
        Original: Standard → Grouped → Standard → Hierarchical → Standard → Relational
        OPTIMIZED: Standard → Grouped → Hierarchical → Relational
        """
        # OPTIMIZATION: Remove redundant standard attention calls
        x = self.standard_attention(x)      # Standard cross-feature (ONCE)
        x = self.grouped_attention(x)       # Grouped attention
        x = self.hierarchical_attention(x)  # Hierarchical attention
        x = self.relational_attention(x)    # Relational attention
        
        # Add CLS tokens for feature aggregation
        batch_size = x.shape[0]
        cls_expanded = self.cls_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        # CLS tokens attend to features
        cls_output = self.cls_attention(cls_expanded, x)
        
        return cls_output



class BiAxialAttention(nn.Module):

    """True Bi-Axial Attention with alternating patterns for tabular data.
    
    This module replaces the standard RowInteraction with true Bi-Axial attention
    that alternates between different attention patterns within feature space.
    
    Note: This is NOT a complete architecture replacement. It only replaces tf_row.
    The tf_col (Set Transformer) remains unchanged for column embedding.
    
    Parameters
    ----------
    embed_dim : int
        Embedding dimension
        
    num_blocks : int
        Number of Bi-Axial Attention blocks
        
    nhead : int
        Number of attention heads
        
    dim_feedforward : int
        Dimension of the feedforward network
        
    num_cls : int, default=4
        Number of CLS tokens for feature aggregation
        
    rope_base : float, default=100000
        Base scaling factor for rotary position encoding (not used in this implementation)
        
    dropout : float, default=0.0
        Dropout probability
        
    activation : str, default="gelu"
        Activation function
        
    norm_first : bool, default=True
        If True, uses pre-norm architecture
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_cls: int = 4,
        rope_base: float = 100000,
        dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cls = num_cls
        self.num_blocks = num_blocks
        
        # Stack of Bi-Axial Attention blocks
        self.bi_axial_blocks = nn.ModuleList([
            BiAxialAttentionBlock(
                embed_dim=embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_cls=num_cls,
                rope_base=rope_base,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
            ) for _ in range(num_blocks)
        ])
        
        # Final output layer normalization
        self.out_ln = nn.LayerNorm(embed_dim) if norm_first else nn.Identity()
        
        # Inference manager for compatibility
        self.inference_mgr = InferenceManager(
            enc_name="bi_axial", 
            out_dim=embed_dim * num_cls, 
            out_no_seq=True
        )
    
    def _train_forward(self, embeddings: Tensor, d: Optional[Tensor] = None) -> Tensor:
        """
        Transform feature embeddings through Bi-Axial Attention for training.
        
        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings from tf_col (Set Transformer) of shape (batch_size, num_samples, num_features, embed_dim)
            
        d : Optional[Tensor], default=None
            Number of features per dataset (for compatibility)
            
        Returns
        -------
        Tensor
            Row representations of shape (batch_size, num_samples, num_cls * embed_dim)
        """
        batch_size, num_samples, num_features, embed_dim = embeddings.shape
        
        # Reshape to process all samples together: (batch_size * num_samples, num_features, embed_dim)
        x = embeddings.view(batch_size * num_samples, num_features, embed_dim)
        
        # Apply Bi-Axial Attention blocks
        for block in self.bi_axial_blocks:
            x = block(x)
        
        # Reshape back: (batch_size, num_samples, num_cls, embed_dim)
        x = x.view(batch_size, num_samples, self.num_cls, embed_dim)
        
        # Flatten CLS tokens: (batch_size, num_samples, num_cls * embed_dim)
        return x.flatten(-2)
    
    def _inference_forward(self, embeddings: Tensor, mgr_config: MgrConfig = None) -> Tensor:
        """
        Transform feature embeddings through Bi-Axial Attention for inference.
        
        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings from tf_col (Set Transformer) of shape (batch_size, num_samples, num_features, embed_dim)
            
        mgr_config : MgrConfig, optional
            Inference configuration for memory management
            
        Returns
        -------
        Tensor
            Row representations of shape (batch_size, num_samples, num_cls * embed_dim)
        """
        batch_size, num_samples, num_features, embed_dim = embeddings.shape
        
        # Reshape to process all samples together: (batch_size * num_samples, num_features, embed_dim)
        x = embeddings.view(batch_size * num_samples, num_features, embed_dim)
        
        # Apply Bi-Axial Attention blocks with inference management
        for i, block in enumerate(self.bi_axial_blocks):
            if mgr_config is not None and hasattr(mgr_config, 'use_checkpointing') and  mgr_config.use_checkpointing:
                # Use gradient checkpointing for memory efficiency
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(block, x)
            else:
                x = block(x)
        
        # Reshape back: (batch_size, num_samples, num_cls, embed_dim)
        x = x.view(batch_size, num_samples, self.num_cls, embed_dim)
        
        # Flatten CLS tokens: (batch_size, num_samples, num_cls * embed_dim)
        return x.flatten(-2)
    

    def forward(self, embeddings: Tensor, d: Optional[Tensor] = None, mgr_config: MgrConfig = None) -> Tensor:
        """
        Forward pass through Bi-Axial Attention.
        
        Parameters
        ----------
        embeddings : Tensor
            Feature embeddings from tf_col (Set Transformer) of shape (batch_size, num_samples, num_features, embed_dim)
            
        d : Optional[Tensor], default=None
            Number of features per dataset
            
        mgr_config : MgrConfig, optional
            Inference configuration
            
        Returns
        -------
        Tensor
            Row representations of shape (batch_size, num_samples, num_cls * embed_dim)
        """
        if self.training:
            return self._train_forward(embeddings, d)
        else:
            return self._inference_forward(embeddings, mgr_config)