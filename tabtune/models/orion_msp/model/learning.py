# learning.py
from __future__ import annotations

from collections import OrderedDict
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .layers import ClassNode, OneHotAndLinear
from .encoders import Encoder
from .inference import InferenceManager
from .inference_config import MgrConfig


# -------------------- Perceiver-style latent memory -------------------- #

class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention + FFN block using a single Encoder block for attn."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout=0.0, activation="gelu", norm_first=True):
        super().__init__()
        self.norm_first = norm_first
        self.q_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        self.kv_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        # use one Encoder block's attention as a convenient MHA wrapper
        self.attn = Encoder(
            num_blocks=1,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            use_rope=False,
        ).blocks[0]
        self.ffn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation, norm_first=norm_first, batch_first=True
        )

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        residual = q
        qn = self.q_norm(q)
        kvn = self.kv_norm(kv)
        x = self.attn(q=qn, k=kvn, v=kvn)
        x = x + residual
        x = self.ffn(x)
        return x


class PerceiverMemory(nn.Module):
    """Latent memory with train-only write and train+test read (leak-safe)."""
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_latents: int = 0,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        self.num_latents = int(num_latents)
        self.d_model = d_model
        if self.num_latents <= 0:
            self.latents = None
            self.write_layers = nn.ModuleList()
            self.read_layers = nn.ModuleList()
            return
        
        if self.num_latents > 0 and self.num_latents < 4:
            raise ValueError("num_latents must be at least 4 or 0 to disable")
    
        self.latents = nn.Parameter(torch.empty(self.num_latents, d_model))
        nn.init.trunc_normal_(self.latents, std=0.02)

        self.write_layers = nn.ModuleList(
            [CrossAttnBlock(d_model, nhead, dim_feedforward, dropout, activation, norm_first) for _ in range(num_layers)]
        )
        self.read_layers = nn.ModuleList(
            [CrossAttnBlock(d_model, nhead, dim_feedforward, dropout, activation, norm_first) for _ in range(num_layers)]
        )

    @torch.no_grad()
    def has_memory(self) -> bool:
        return self.latents is not None

    def write(self, train_tokens: Tensor) -> Tensor:
        """Write from TRAIN rows (no labels). train_tokens: (B, S_tr, D)"""
        if not self.has_memory():
            return None
        B, _, D = train_tokens.shape
        L = self.latents.expand(B, self.num_latents, D)
        for blk in self.write_layers:
            L = blk(L, train_tokens)
        return L  # (B, L, D)

    def read(self, tokens: Tensor, latents: Tensor) -> Tensor:
        """Read into train+test rows. tokens: (B,S,D), latents: (B,L,D)"""
        if not self.has_memory():
            return tokens
        x = tokens
        for blk in self.read_layers:
            x = blk(x, latents)
        return x


# --------------------------- ICLearning -------------------------------- #

class ICLearning(nn.Module):
    """Dataset-wise in-context learning with optional Perceiver memory."""

    def __init__(
        self,
        max_classes: int,
        d_model: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str | callable = "gelu",
        norm_first: bool = True,
        # NEW
        perc_num_latents: int = 16,
        perc_layers: int = 2,
    ):
        super().__init__()
        self.max_classes = max_classes
        self.norm_first = norm_first

        self.tf_icl = Encoder(
            num_blocks=num_blocks,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )
        self.ln = nn.LayerNorm(d_model) if norm_first else nn.Identity()

        self.y_encoder = OneHotAndLinear(max_classes, d_model)
        self.decoder = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, max_classes))

        # Perceiver memory
        self.memory = PerceiverMemory(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_latents=perc_num_latents,
            num_layers=perc_layers,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
        )

        self.inference_mgr = InferenceManager(enc_name="tf_icl", out_dim=max_classes)

    # --- internals --- #

    def _label_encoding(self, y: Tensor) -> Tensor:
        unique_vals, _ = torch.unique(y, return_inverse=True)
        indices = unique_vals.argsort()
        return indices[torch.searchsorted(unique_vals, y)]

    def _icl_predictions(self, R: Tensor, y_train: Tensor) -> Tensor:
        """Apply (optional) memory write/read, then label-inject and ICL with split mask."""
        B, T, D = R.shape
        train_size = y_train.shape[1]

        # 1) train-only write (no labels yet) + read for all rows
        if self.memory.has_memory():
            latents = self.memory.write(R[:, :train_size, :])
            R = self.memory.read(R, latents)

        # 2) inject labels ONLY to train slice
        R[:, :train_size] = R[:, :train_size] + self.y_encoder(y_train.float())

        # 3) ICL with standard split mask (int train_size indicates split)
        src = self.tf_icl(R, attn_mask=train_size)
        src = self.ln(src)
        out = self.decoder(src)  # (B, T, max_classes)
        return out

    # --- public API --- #
    
    """

    def _predict_standard(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = False,
        softmax_temperature: float = 0.9,
        auto_batch: bool = True,
    ) -> Tensor:
        train_size = y_train.shape[1]
        logits = self._icl_predictions(R, y_train)[:, train_size:]
        if return_logits:
            return logits
        return torch.softmax(logits, dim=-1)
    """
    
    def _predict_standard(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = False,
        softmax_temperature: float = 0.9,
        auto_batch: bool = True,
        n_classes: int | None = None,                 # <--- NEW
        class_indices: torch.Tensor | None = None,    # <--- NEW, 1D LongTensor of columns to keep
    ) -> Tensor:
        train_size = y_train.shape[1]

        out = self.inference_mgr(
            self._icl_predictions,
            inputs=OrderedDict([("R", R), ("y_train", y_train)]),
            auto_batch=auto_batch,
        )  # (B, T, max_classes)

        # Slice to test region, then select K active classes
        out = out[:, train_size:]  # (B, Ttest, max_classes)

        if class_indices is not None:
            out = out.index_select(dim=-1, index=class_indices)  # (B, Ttest, K)
        elif n_classes is not None:
            out = out[..., :n_classes]  # assume first K columns are active
        else:
            # fallback: infer K from training labels (works when train has all K)
            k_infer = len(torch.unique(y_train[0]))
            out = out[..., :k_infer]

        if not return_logits:
            out = torch.softmax(out / softmax_temperature, dim=-1)

        return out

    """
    def _inference_forward(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: MgrConfig | None = None,
    ) -> Tensor:
        if mgr_config is None:
            mgr_config = MgrConfig()
        self.inference_mgr.configure(**mgr_config)
        
        def _inner(R: Tensor, y_train: Tensor) -> Tensor:
            logits = self._icl_predictions(R, y_train)[:, y_train.shape[1] :]
            if return_logits:
                return logits
            return torch.softmax(logits, dim=-1)

        probs = self.inference_mgr(_inner, inputs=OrderedDict([("R", R), ("y_train", y_train)]))
        return probs
    """
    
    def _grouping(self, num_classes: int) -> tuple[torch.Tensor, int]:
        """Divide classes into balanced groups for hierarchical classification."""
        if num_classes <= self.max_classes:
            return torch.zeros(num_classes, dtype=torch.int), 1

        import math
        num_groups = min(math.ceil(num_classes / self.max_classes), self.max_classes)
        group_assignments = torch.zeros(num_classes, dtype=torch.int)
        current_pos = 0

        remaining_classes = num_classes
        remaining_groups = num_groups
        for i in range(num_groups):
            group_size = math.ceil(remaining_classes / remaining_groups)
            group_assignments[current_pos : current_pos + group_size] = i
            current_pos += group_size
            remaining_classes -= group_size
            remaining_groups -= 1

        return group_assignments, num_groups

    def _fit_node(self, node, R: torch.Tensor, y: torch.Tensor, current_depth: int):
        """Recursively build a node in the hierarchical classification tree."""
        unique_classes = torch.unique(y).int()
        node.classes_ = unique_classes

        if len(unique_classes) <= self.max_classes:
            node.is_leaf = True
            node.R = R
            node.y = y
            return

        group_assignments, num_groups = self._grouping(len(unique_classes))
        node.class_mapping = {c.item(): g.item() for c, g in zip(unique_classes, group_assignments)}
        node.group_indices = torch.tensor([node.class_mapping[c.item()] for c in y], dtype=torch.int)
        node.R = R
        node.y = y
        node.is_leaf = False

        for group in range(num_groups):
            mask = node.group_indices == group
            child_node = ClassNode(current_depth + 1)
            self._fit_node(child_node, R[mask], y[mask], current_depth + 1)
            node.child_nodes.append(child_node)

    def _fit_hierarchical(self, R_train: torch.Tensor, y_train: torch.Tensor):
        """Initialize the hierarchical classification tree."""
        self.root = ClassNode(depth=0)
        self._fit_node(self.root, R_train, y_train, current_depth=0)

    def _predict_hierarchical(self, R_test: torch.Tensor, softmax_temperature: float = 0.9) -> torch.Tensor:
        """Generate predictions using the hierarchical classification tree."""
        test_size = R_test.shape[0]
        device = R_test.device
        num_classes = len(self.root.classes_)

        def process_node(node, R_test):
            node_R = torch.cat([node.R.to(device), R_test], dim=0)

            if node.is_leaf:
                node_y = self._label_encoding(node.y.to(device))
                leaf_preds = self._predict_standard(
                    R=node_R.unsqueeze(0),
                    y_train=node_y.unsqueeze(0),
                    softmax_temperature=softmax_temperature,
                    auto_batch=False,
                ).squeeze(0)
                global_preds = torch.zeros((test_size, num_classes), device=device)
                for local_idx, global_idx in enumerate(node.classes_):
                    global_preds[:, global_idx] = leaf_preds[:, local_idx]
                return global_preds

            final_probs = torch.zeros((test_size, num_classes), device=device)
            node_y = node.group_indices.to(device)
            group_probs = self._predict_standard(
                R=node_R.unsqueeze(0),
                y_train=node_y.unsqueeze(0),
                softmax_temperature=softmax_temperature,
                auto_batch=False,
            ).squeeze(0)

            for group_idx, child_node in enumerate(node.child_nodes):
                child_probs = process_node(child_node, R_test)
                final_probs += child_probs * group_probs[:, group_idx : group_idx + 1]

            return final_probs

        return process_node(self.root, R_test)

    def _inference_forward(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: MgrConfig | None = None,
        n_classes: int | None = None,                 # <--- NEW
        class_indices: torch.Tensor | None = None,    # <--- NEW
    ) -> Tensor:
        if mgr_config is None:
            mgr_config = MgrConfig(
                min_batch_size=1,
                safety_factor=0.8,
                offload=False,
                auto_offload_pct=0.5,
                device=None,
                use_amp=True,
                verbose=False,
            )
        self.inference_mgr.configure(**mgr_config)

        # How many classes are in this task?
        # (Used only as a fallback when caller didn't specify n_classes/class_indices)
        num_classes_fallback = len(torch.unique(y_train[0]))

        if num_classes_fallback <= self.max_classes:
            out = self._predict_standard(
                R,
                y_train,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                auto_batch=True,
                n_classes=n_classes,
                class_indices=class_indices,
            )
        else:
            # Hierarchical path unchanged, but we must also slice the final
            # logits/probs to the active set if caller requested it.
            outs = []
            train_size = y_train.shape[1]
            for ri, yi in zip(R, y_train):
                dev = mgr_config.device
                ri = ri.to(dev) if dev else ri
                yi = yi.to(dev) if dev else yi
                self._fit_hierarchical(ri[:train_size], yi)
                probs = self._predict_hierarchical(ri[train_size:], softmax_temperature=softmax_temperature)
                outs.append(probs)
            out = torch.stack(outs, dim=0)  # (B, Ttest, C_all)
            if class_indices is not None:
                out = out.index_select(dim=-1, index=class_indices)
            elif n_classes is not None:
                out = out[..., :n_classes]
            if return_logits:
                out = softmax_temperature * torch.log(out + 1e-6)

        return out
    
    """
    def forward(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: MgrConfig | None = None,
    ) -> Tensor:
        if self.training:
            train_size = y_train.shape[1]
            out = self._icl_predictions(R, y_train)
            return out[:, train_size:]
        else:
            return self._inference_forward(R, y_train, return_logits, softmax_temperature, mgr_config)
    """
    
    def forward(
        self,
        R: Tensor,
        y_train: Tensor,
        return_logits: bool = True,
        softmax_temperature: float = 0.9,
        mgr_config: MgrConfig | None = None,
        n_classes: int | None = None,                # <--- NEW
        class_indices: torch.Tensor | None = None,   # <--- NEW
    ) -> Tensor:
        if self.training:
            train_size = y_train.shape[1]
            out = self._icl_predictions(R, y_train)[:, train_size:]
        else:
            out = self._inference_forward(
                R, y_train,
                return_logits=return_logits,
                softmax_temperature=softmax_temperature,
                mgr_config=mgr_config,
                n_classes=n_classes,                 # <--- NEW
                class_indices=class_indices,         # <--- NEW
            )
        return out
