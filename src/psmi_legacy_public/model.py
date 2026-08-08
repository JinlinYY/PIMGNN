# -*- coding: utf-8 -*-
"""Implement the archived model.py compatibility module."""
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from .utils import atom_feature_dim, bond_feature_dim, global_feature_dim


# ----------------------------
# pooling helpers (no torch_scatter required)
# ----------------------------
def global_pool_mean(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Run the legacy global pool mean operation."""
    if x.numel() == 0:
        return torch.zeros((num_graphs, x.size(-1)), device=x.device, dtype=x.dtype)

    out = torch.zeros((num_graphs, x.size(-1)), device=x.device, dtype=x.dtype)
    out.index_add_(0, batch, x)
    cnt = torch.zeros((num_graphs,), device=x.device, dtype=x.dtype)
    one = torch.ones((batch.size(0),), device=x.device, dtype=x.dtype)
    cnt.index_add_(0, batch, one)
    cnt = cnt.clamp_min(1.0).unsqueeze(-1)
    return out / cnt


def _num_graphs_from_batch(batch: torch.Tensor, fallback: int) -> int:
    """Run the legacy num graphs from batch operation."""
    if batch.numel() == 0:
        return int(fallback)
    return int(batch.max().item()) + 1


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Run the legacy masked mean operation."""
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(dtype=x.dtype)
    if m.dim() == 2:
        m = m.unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (x * m).sum(dim=1) / denom


def s3_equivariant_embedding(mol_embedding: torch.Tensor) -> torch.Tensor:
    """Run the legacy s3 equivariant embedding operation."""
    if isinstance(mol_embedding, (list, tuple)):
        mol_embedding = torch.stack(mol_embedding, dim=1)
    if mol_embedding.ndim != 3 or mol_embedding.shape[1] != 3:
        raise ValueError(f"mol_embedding must be (B,3,H), got {tuple(mol_embedding.shape)}")
    # Legacy public-release workflow step.
    mean = mol_embedding.mean(dim=1, keepdim=True)
    # Legacy public-release workflow step.
    return mol_embedding + mean


def cross_molecular_fg_attention(
    fg1: torch.Tensor,
    fg2: torch.Tensor,
    fg3: torch.Tensor,
    mask1: Optional[torch.Tensor] = None,
    mask2: Optional[torch.Tensor] = None,
    mask3: Optional[torch.Tensor] = None,
    attn: Optional[nn.MultiheadAttention] = None,
    norm: Optional[nn.LayerNorm] = None,
    drop: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the legacy cross molecular fg attention operation."""
    # Legacy public-release workflow step.
    if attn is None:
        return _masked_mean(fg1, mask1), _masked_mean(fg2, mask2), _masked_mean(fg3, mask3)

    # Legacy public-release workflow step.
    def _prep_kv(a: torch.Tensor, b: torch.Tensor, ma: Optional[torch.Tensor], mb: Optional[torch.Tensor]):
        kv = torch.cat([a, b], dim=1)
        if ma is None or mb is None:
            return kv, None
        m = torch.cat([ma, mb], dim=1).to(dtype=torch.bool)
        key_padding_mask = ~m
        # Legacy public-release workflow step.
        if key_padding_mask.all(dim=1).any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[key_padding_mask.all(dim=1), 0] = False
        return kv, key_padding_mask

    # Legacy public-release workflow step.
    def _attend(q: torch.Tensor, kv: torch.Tensor, q_mask: Optional[torch.Tensor], kv_mask: Optional[torch.Tensor]):
        # Legacy public-release workflow step.
        out, _ = attn(q, kv, kv, key_padding_mask=kv_mask, need_weights=False)
        if drop is not None:
            out = drop(out)
        # Legacy public-release workflow step.
        out = q + out
        if norm is not None:
            out = norm(out)
        # Legacy public-release workflow step.
        if q_mask is not None:
            out = out * q_mask.to(dtype=out.dtype).unsqueeze(-1)
        return out

    # Legacy public-release workflow step.
    kv23, m23 = _prep_kv(fg2, fg3, mask2, mask3)  # Legacy public-release workflow step.
    kv13, m13 = _prep_kv(fg1, fg3, mask1, mask3)  # Legacy public-release workflow step.
    kv12, m12 = _prep_kv(fg1, fg2, mask1, mask2)  # Legacy public-release workflow step.

    # Legacy public-release workflow step.
    a1 = _attend(fg1, kv23, mask1, m23)
    a2 = _attend(fg2, kv13, mask2, m13)
    a3 = _attend(fg3, kv12, mask3, m12)

    # Legacy public-release workflow step.
    return _masked_mean(a1, mask1), _masked_mean(a2, mask2), _masked_mean(a3, mask3)


# ----------------------------
# Fingerprint baseline (kept)
# ----------------------------
class LLECurveNet(nn.Module):
    """Represent the legacy LLECurveNet component."""
    def __init__(self, in_dim: int, hidden: int = 1024, dropout: float = 0.15):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_E = nn.Linear(hidden, 3)
        self.head_R = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the legacy forward operation."""
        h = self.backbone(x)
        # Legacy public-release workflow step.
        E = torch.softmax(self.head_E(h), dim=-1)
        R = torch.softmax(self.head_R(h), dim=-1)
        return torch.cat([E, R], dim=-1)


# ----------------------------
# Molecule graph encoder (custom graph dict)
# ----------------------------
class MPNNLayer(nn.Module):
    """Represent the legacy MPNNLayer component."""
    def __init__(
        self,
        hidden: int,
        edge_dim: int,
        dropout: float,
        update_edges: bool = True,
        edge_dropout: float = 0.1,
        edge_scale: float = 0.1,
    ):
        super().__init__()
        self.update_edges = bool(update_edges)
        self.edge_dropout = float(edge_dropout)
        self.edge_scale = float(edge_scale)

        # Legacy public-release workflow step.
        self.msg = nn.Sequential(
            nn.Linear(hidden + edge_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        # Legacy public-release workflow step.
        self.upd = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

        if self.update_edges:
            # Legacy public-release workflow step.
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden * 2 + edge_dim, edge_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(edge_dim, edge_dim),
            )
            # Legacy public-release workflow step.
            self.edge_gate = nn.Sequential(
                nn.Linear(hidden * 2 + edge_dim, 1),
                nn.Sigmoid()
            )
            self.edge_norm = nn.LayerNorm(edge_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        """Run the legacy forward operation."""
        # Legacy public-release workflow step.
        if edge_index.numel() == 0:
            return self.norm(h), edge_attr

        # Legacy public-release workflow step.
        if edge_attr.dtype != h.dtype:
            edge_attr = edge_attr.to(dtype=h.dtype)

        # Legacy public-release workflow step.
        src, dst = edge_index[0], edge_index[1]
        hs = h[src]  # Legacy public-release workflow step.
        hd = h[dst]  # Legacy public-release workflow step.

        E = edge_attr.size(0)
        # Legacy public-release workflow step.
        if self.training and self.edge_dropout > 0:
            keep = (torch.rand(E, device=edge_attr.device) > self.edge_dropout).to(edge_attr.dtype)
        else:
            keep = torch.ones(E, device=edge_attr.device, dtype=edge_attr.dtype)

        # Legacy public-release workflow step.
        if self.update_edges:
            # Legacy public-release workflow step.
            z = torch.cat([hs, hd, edge_attr], dim=-1)  # (E, 2H+De)
            # Legacy public-release workflow step.
            gate = self.edge_gate(z)                    # Legacy public-release workflow step.
            delta = self.edge_mlp(z)                    # Legacy public-release workflow step.
            # Legacy public-release workflow step.
            delta = delta * gate                        # Legacy public-release workflow step.
            edge_attr = self.edge_norm(edge_attr + (self.edge_scale * keep.unsqueeze(-1) * delta))

        # Legacy public-release workflow step.
        # Legacy public-release workflow step.
        m = self.msg(torch.cat([hs, edge_attr], dim=-1))  # (E, hidden)
        # Legacy public-release workflow step.
        m = m * keep.unsqueeze(-1)

        # Legacy public-release workflow step.
        # Legacy public-release workflow step.
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)

        # Legacy public-release workflow step.
        # Legacy public-release workflow step.
        h2 = self.upd(torch.cat([h, agg], dim=-1))
        # Legacy public-release workflow step.
        h = self.norm(h + h2)
        return h, edge_attr


class MPNNEncoder(nn.Module):
    """Represent the legacy MPNNEncoder component."""
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        global_dim: int,
        hidden: int = 256,
        layers: int = 4,
        dropout: float = 0.15,
        pool: str = "mean",
        update_edges: bool = True,
        edge_dropout: float = 0.1,
        edge_scale: float = 0.1,
    ):
        super().__init__()
        self.pool = pool

        # Legacy public-release workflow step.
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Legacy public-release workflow step.
        self.glob_proj = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Legacy public-release workflow step.
        self.layers = nn.ModuleList([
            MPNNLayer(
                hidden=hidden,
                edge_dim=edge_dim,
                dropout=dropout,
                update_edges=update_edges,
                edge_dropout=edge_dropout,
                edge_scale=edge_scale,
            )
            for _ in range(int(layers))
        ])

        # Legacy public-release workflow step.
        self.out = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, g: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the legacy forward operation."""
        x = g["x"]
        edge_index = g["edge_index"]
        edge_attr = g["edge_attr"]
        batch = g["batch"]
        glob = g["g"]

        # Legacy public-release workflow step.
        h = self.node_proj(x)

        # Legacy public-release workflow step.
        if edge_attr.dtype != h.dtype:
            edge_attr = edge_attr.to(dtype=h.dtype)
        if glob.dtype != h.dtype:
            glob = glob.to(dtype=h.dtype)

        # Legacy public-release workflow step.
        for layer in self.layers:
            h, edge_attr = layer(h, edge_index, edge_attr)

        # Legacy public-release workflow step.
        num_graphs = int(glob.shape[0])
        if num_graphs == 0:
            return torch.zeros((0, h.size(-1)), device=h.device, dtype=h.dtype)

        # Legacy public-release workflow step.
        pooled = global_pool_mean(h, batch, num_graphs)
        # Legacy public-release workflow step.
        g_emb = self.glob_proj(glob)
        # Legacy public-release workflow step.
        z = self.out(torch.cat([pooled, g_emb], dim=-1))
        return z


# ----------------------------
# Mixture graph encoder (3 nodes per sample)
# ----------------------------
class MixGraphEncoder(nn.Module):
    """Represent the legacy MixGraphEncoder component."""
    def __init__(
        self,
        hidden: int,
        layers: int = 2,
        dropout: float = 0.15,
        edge_hidden: int = 64,
        update_edges: bool = True,
        edge_dropout: float = 0.1,
        edge_scale: float = 0.1,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.edge_hidden = int(edge_hidden)

        # Legacy public-release workflow step.
        self.edge_in = nn.LazyLinear(self.edge_hidden)
        self.edge_act = nn.GELU()
        self.edge_drop = nn.Dropout(dropout)
        self.edge_norm = nn.LayerNorm(self.edge_hidden)

        # Legacy public-release workflow step.
        self.layers = nn.ModuleList([
            MPNNLayer(
                hidden=self.hidden,
                edge_dim=self.edge_hidden,
                dropout=dropout,
                update_edges=update_edges,
                edge_dropout=edge_dropout,
                edge_scale=edge_scale,
            )
            for _ in range(int(layers))
        ])
        self.out_norm = nn.LayerNorm(self.hidden)

    def forward(
        self,
        node_h: torch.Tensor,
        mix_g: Dict[str, torch.Tensor],
        fallback_num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the legacy forward operation."""
        edge_index = mix_g["edge_index"]
        edge_attr = mix_g["edge_attr"]
        batch = mix_g["batch"]

        # Legacy public-release workflow step.
        if edge_attr.numel() == 0 or edge_index.numel() == 0:
            num_graphs = _num_graphs_from_batch(batch, fallback=fallback_num_graphs)
            # Legacy public-release workflow step.
            node_out = self.out_norm(node_h)
            mix_emb = global_pool_mean(node_out, batch, num_graphs)
            return node_out, mix_emb

        # Legacy public-release workflow step.
        if edge_attr.dtype != node_h.dtype:
            edge_attr = edge_attr.to(dtype=node_h.dtype)

        # Legacy public-release workflow step.
        e = self.edge_in(edge_attr)
        e = self.edge_drop(self.edge_act(e))
        e = self.edge_norm(e)

        # Legacy public-release workflow step.
        h = node_h
        for layer in self.layers:
            h, e = layer(h, edge_index, e)

        # Legacy public-release workflow step.
        h = self.out_norm(h)
        # Legacy public-release workflow step.
        num_graphs = _num_graphs_from_batch(batch, fallback=fallback_num_graphs)
        # Legacy public-release workflow step.
        mix_emb = global_pool_mean(h, batch, num_graphs)
        return h, mix_emb



# ----------------------------
# Token fusion transformer (for multi-scale features)
# ----------------------------
class TokenFusionTransformer(nn.Module):
    """Represent the legacy TokenFusionTransformer component."""
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.10,
        pool: str = "cls",
        max_len: int = 32,
        type_vocab_size: int = 16,
        use_type_embed: bool = True,
        use_pos_embed: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.pool = str(pool).lower()
        self.use_cls = self.pool == "cls"
        self.use_type_embed = bool(use_type_embed)
        self.use_pos_embed = bool(use_pos_embed)

        # Legacy public-release workflow step.
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        else:
            self.cls_token = None

        # Legacy public-release workflow step.
        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, int(max_len) + (1 if self.use_cls else 0), self.d_model))
        else:
            self.pos_embed = None

        # Legacy public-release workflow step.
        if self.use_type_embed:
            self.type_embed = nn.Embedding(int(type_vocab_size), self.d_model)
        else:
            self.type_embed = None

        # Legacy public-release workflow step.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.out_norm = nn.LayerNorm(self.d_model)

        # Legacy public-release workflow step.
        nn.init.normal_(self.cls_token, std=0.02) if self.cls_token is not None else None
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, std=0.02)
        if self.type_embed is not None:
            nn.init.normal_(self.type_embed.weight, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the legacy forward operation."""
        assert tokens.dim() == 3, f"tokens must be (B,L,D), obtained {tokens.shape}"
        B, L, D = tokens.shape
        assert D == self.d_model, f" token dimension mismatch : obtained {D}, expected {self.d_model}"
        device = tokens.device

        if type_ids is not None:
            assert type_ids.shape[:2] == (B, L), f"type_ids must be (B,L), obtained {type_ids.shape}"
            type_ids = type_ids.to(device=device, dtype=torch.long)

        # Legacy public-release workflow step.
        x = tokens
        # Legacy public-release workflow step.
        if self.use_cls:
            cls = self.cls_token.expand(B, 1, D).to(device=device, dtype=x.dtype)
            x = torch.cat([cls, x], dim=1)
            # Legacy public-release workflow step.
            if type_ids is not None:
                cls_type = torch.zeros((B, 1), device=device, dtype=torch.long)
                type_ids = torch.cat([cls_type, type_ids], dim=1)

        # Legacy public-release workflow step.
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, : x.shape[1], :].to(device=device, dtype=x.dtype)

        # Legacy public-release workflow step.
        if (self.type_embed is not None) and (type_ids is not None):
            x = x + self.type_embed(type_ids)

        # Legacy public-release workflow step.
        # Legacy public-release workflow step.
        h = self.encoder(x)
        # Legacy public-release workflow step.
        h = self.out_norm(h)

        # Legacy public-release workflow step.
        if self.use_cls:
            # Legacy public-release workflow step.
            pooled = h[:, 0, :]
        else:
            # Legacy public-release workflow step.
            pooled = h.mean(dim=1)

        return pooled, h

# ----------------------------
# Full graph model (concat / mixture graph)
# ----------------------------

class LLEGraphNet(nn.Module):
    """Represent the legacy LLEGraphNet component."""
    def __init__(
        self,
        gnn_hidden: int = 256,
        gnn_layers: int = 4,
        mlp_hidden: int = 512,
        dropout: float = 0.15,
        pool: str = "mean",
        use_interaction: bool = True,
        # molecule edge update knobs (optional)
        update_edges: bool = True,
        edge_dropout: float = 0.1,
        edge_scale: float = 0.1,
        # mixture graph options (optional; keep backward compatibility)
        use_mix_graph: bool = False,
        mix_layers: int = 2,
        mix_edge_hidden: int = 64,
        mix_update_edges: bool = True,
        mix_edge_dropout: float = 0.1,
        mix_edge_scale: float = 0.1,
        mix_append_global: bool = True,
        # mix head knobs (ignored; kept for backward compatibility with train.py kwargs)
        mix_hidden: Optional[int] = None,
        mix_dropout: float = 0.10,
        # functional-group options (optional)
        use_fg: bool = True,
        fg_vocab_size: int = 0,
        fg_hidden: int = 256,
        fg_out_dim: Optional[int] = None,
        fg_dropout: float = 0.10,
        fg_token_mode: bool = True,
        fg_max_tokens: int = 32,
        fg_cross_attn: bool = True,
        fg_attn_heads: int = 8,
        # fusion options
        fusion_mode: str = "concat",
        tf_dim: Optional[int] = None,
        tf_layers: int = 2,
        tf_heads: int = 8,
        tf_ff: int = 1024,
        tf_dropout: float = 0.10,
        tf_pool: str = "cls",
        tf_max_len: int = 32,
        tf_type_vocab: int = 16,
        s3_equivariant: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.hidden = int(gnn_hidden)
        self.use_interaction = bool(use_interaction)
        self.s3_equivariant = bool(s3_equivariant)

        self.use_mix_graph = bool(use_mix_graph)
        self.mix_append_global = bool(mix_append_global)

        self.fusion_mode = str(fusion_mode).lower().strip()
        if tf_dim is None:
            tf_dim = self.hidden
        self.tf_dim = int(tf_dim)

        # molecule encoder
        self.encoder = MPNNEncoder(
            node_dim=atom_feature_dim(),
            edge_dim=bond_feature_dim(),
            global_dim=global_feature_dim(),
            hidden=self.hidden,
            layers=int(gnn_layers),
            dropout=float(dropout),
            pool=str(pool),
            update_edges=bool(update_edges),
            edge_dropout=float(edge_dropout),
            edge_scale=float(edge_scale),
        )

        # mixture encoder (optional)
        if self.use_mix_graph:
            self.mix_encoder = MixGraphEncoder(
                hidden=self.hidden,
                layers=int(mix_layers),
                dropout=float(dropout),
                edge_hidden=int(mix_edge_hidden),
                update_edges=bool(mix_update_edges),
                edge_dropout=float(mix_edge_dropout),
                edge_scale=float(mix_edge_scale),
            )
        else:
            self.mix_encoder = None

        # FG encoder (multi-hot -> dense) or FG token encoder
        self.use_fg = bool(use_fg) and (int(fg_vocab_size) > 0)
        self.fg_vocab_size = int(fg_vocab_size)
        self.fg_out_dim = int(self.hidden if fg_out_dim is None else fg_out_dim)
        self.fg_token_mode = bool(fg_token_mode) and self.use_fg
        self.fg_max_tokens = int(fg_max_tokens)
        self.fg_cross_attn = bool(fg_cross_attn) and self.use_fg

        if self.use_fg and self.fg_token_mode:
            self.fg_token_dim = int(fg_hidden)
            self.fg_token_embed = nn.Embedding(self.fg_vocab_size + 1, self.fg_token_dim, padding_idx=0)
            self.fg_token_drop = nn.Dropout(float(fg_dropout))
            if self.fg_cross_attn:
                heads = int(fg_attn_heads)
                if heads <= 0 or (self.fg_token_dim % heads) != 0:
                    heads = 1
                self.fg_attn = nn.MultiheadAttention(
                    embed_dim=self.fg_token_dim,
                    num_heads=heads,
                    dropout=float(fg_dropout),
                    batch_first=True,
                )
                self.fg_attn_norm = nn.LayerNorm(self.fg_token_dim)
            else:
                self.fg_attn = None
                self.fg_attn_norm = None
            self.fg_token_proj = nn.Sequential(
                nn.Linear(self.fg_token_dim, self.fg_out_dim),
                nn.GELU(),
                nn.Dropout(float(fg_dropout)),
            )
            self.fg_encoder = None
        elif self.use_fg:
            self.fg_token_dim = 0
            self.fg_token_embed = None
            self.fg_token_drop = None
            self.fg_attn = None
            self.fg_attn_norm = None
            self.fg_token_proj = None
            self.fg_encoder = nn.Sequential(
                nn.Linear(self.fg_vocab_size, int(fg_hidden)),
                nn.GELU(),
                nn.Dropout(float(fg_dropout)),
                nn.Linear(int(fg_hidden), self.fg_out_dim),
                nn.GELU(),
                nn.Dropout(float(fg_dropout)),
            )
        else:
            self.fg_token_dim = 0
            self.fg_token_embed = None
            self.fg_token_drop = None
            self.fg_attn = None
            self.fg_attn_norm = None
            self.fg_token_proj = None
            self.fg_encoder = None

        # --- fusion + head ---
        if self.fusion_mode == "transformer":
            # token projections
            self.proj_mol = nn.Identity() if self.hidden == self.tf_dim else nn.Linear(self.hidden, self.tf_dim)
            self.proj_inter = nn.Identity() if self.hidden == self.tf_dim else nn.Linear(self.hidden, self.tf_dim)
            self.proj_mix = nn.Identity() if self.hidden == self.tf_dim else nn.Linear(self.hidden, self.tf_dim)
            self.proj_fg = nn.Identity() if self.fg_out_dim == self.tf_dim else nn.Linear(self.fg_out_dim, self.tf_dim)
            self.proj_scalar = nn.Linear(2, self.tf_dim)

            self.token_fuser = TokenFusionTransformer(
                d_model=self.tf_dim,
                nhead=int(tf_heads),
                num_layers=int(tf_layers),
                dim_feedforward=int(tf_ff),
                dropout=float(tf_dropout),
                pool=str(tf_pool),
                max_len=int(tf_max_len),
                type_vocab_size=int(tf_type_vocab),
                use_type_embed=True,
                use_pos_embed=True,
            )

            self.backbone = nn.Sequential(
                nn.Linear(self.tf_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.comp_backbone = None
            self.comp_head_E = None
            self.comp_head_R = None
        elif self.fusion_mode == "s3_set":
            comp_dim = self.hidden
            if self.use_interaction:
                comp_dim += 3 * self.hidden
            if self.use_mix_graph and self.mix_append_global:
                comp_dim += self.hidden
            if self.use_fg:
                comp_dim += self.fg_out_dim
            comp_dim += 2

            self.comp_backbone = nn.Sequential(
                nn.Linear(comp_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.comp_head_E = nn.Linear(mlp_hidden, 1)
            self.comp_head_R = nn.Linear(mlp_hidden, 1)

            self.backbone = None
            self.proj_mol = None
            self.proj_inter = None
            self.proj_mix = None
            self.proj_fg = None
            self.proj_scalar = None
            self.token_fuser = None
        else:
            # original concat mode
            base_dim = 3 * self.hidden
            inter_dim = 0
            if self.use_interaction:
                inter_dim = 6 * self.hidden  # abs diffs + products

            mix_dim = self.hidden if (self.use_mix_graph and self.mix_append_global) else 0
            fg_dim = 3 * self.fg_out_dim if self.use_fg else 0
            in_dim = base_dim + inter_dim + mix_dim + fg_dim + 2  # + scalars

            self.backbone = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            self.proj_mol = None
            self.proj_inter = None
            self.proj_mix = None
            self.proj_fg = None
            self.proj_scalar = None
            self.token_fuser = None
            self.comp_backbone = None
            self.comp_head_E = None
            self.comp_head_R = None

        self.head_E = nn.Linear(mlp_hidden, 3)
        self.head_R = nn.Linear(mlp_hidden, 3)
        # Legacy public-release workflow step.
        self.is_binary = kwargs.get("is_binary", False)
        if self.is_binary:
            # Legacy public-release workflow step.
            self.binary_head = nn.Linear(mlp_hidden, 1)
        # =======================================================

    def _encode_fg(self, x: Dict[str, Any], B: int, device: torch.device) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Run the legacy encode fg operation."""
        if not self.use_fg:
            return None

        # Legacy public-release workflow step.
        if self.fg_token_mode and self.fg_token_embed is not None:
            # Legacy public-release workflow step.
            fg_ids = x.get("fg_ids", None)
            fg1_ids = fg2_ids = fg3_ids = None
            if fg_ids is not None:
                fg_ids = fg_ids.to(device=device, dtype=torch.long)
                # Legacy public-release workflow step.
                if fg_ids.dim() == 3 and fg_ids.shape[1] == 3:
                    fg1_ids, fg2_ids, fg3_ids = fg_ids[:, 0, :], fg_ids[:, 1, :], fg_ids[:, 2, :]
            
            # Legacy public-release workflow step.
            if fg1_ids is None:
                fg1_ids = x.get("fg1_ids", None)
                fg2_ids = x.get("fg2_ids", None)
                fg3_ids = x.get("fg3_ids", None)

            L = int(max(1, self.fg_max_tokens))
            # Legacy public-release workflow step.
            if fg1_ids is None:
                fg1_ids = torch.zeros((B, L), device=device, dtype=torch.long)
            else:
                fg1_ids = fg1_ids.to(device=device, dtype=torch.long)
            if fg2_ids is None:
                fg2_ids = torch.zeros((B, L), device=device, dtype=torch.long)
            else:
                fg2_ids = fg2_ids.to(device=device, dtype=torch.long)
            if fg3_ids is None:
                fg3_ids = torch.zeros((B, L), device=device, dtype=torch.long)
            else:
                fg3_ids = fg3_ids.to(device=device, dtype=torch.long)

            # Legacy public-release workflow step.
            fg1_mask = x.get("fg1_mask", None)
            fg2_mask = x.get("fg2_mask", None)
            fg3_mask = x.get("fg3_mask", None)
            if fg1_mask is None:
                fg1_mask = (fg1_ids != 0).to(dtype=torch.float32)
            else:
                fg1_mask = fg1_mask.to(device=device, dtype=torch.float32)
            if fg2_mask is None:
                fg2_mask = (fg2_ids != 0).to(dtype=torch.float32)
            else:
                fg2_mask = fg2_mask.to(device=device, dtype=torch.float32)
            if fg3_mask is None:
                fg3_mask = (fg3_ids != 0).to(dtype=torch.float32)
            else:
                fg3_mask = fg3_mask.to(device=device, dtype=torch.float32)

            # Legacy public-release workflow step.
            f1 = self.fg_token_drop(self.fg_token_embed(fg1_ids))
            f2 = self.fg_token_drop(self.fg_token_embed(fg2_ids))
            f3 = self.fg_token_drop(self.fg_token_embed(fg3_ids))

            # Legacy public-release workflow step.
            if self.fg_cross_attn and (self.fg_attn is not None):
                # Legacy public-release workflow step.
                p1, p2, p3 = cross_molecular_fg_attention(
                    f1, f2, f3,
                    mask1=fg1_mask, mask2=fg2_mask, mask3=fg3_mask,
                    attn=self.fg_attn, norm=self.fg_attn_norm, drop=self.fg_token_drop
                )
            else:
                # Legacy public-release workflow step.
                p1 = _masked_mean(f1, fg1_mask)
                p2 = _masked_mean(f2, fg2_mask)
                p3 = _masked_mean(f3, fg3_mask)

            # Legacy public-release workflow step.
            f1 = self.fg_token_proj(p1)
            f2 = self.fg_token_proj(p2)
            f3 = self.fg_token_proj(p3)
            return f1, f2, f3

        # Legacy public-release workflow step.
        if self.fg_encoder is not None:
            # Legacy public-release workflow step.
            if "fg" in x and x["fg"] is not None:
                fg_all = x["fg"].to(device=device, dtype=torch.float32)
                fg1, fg2, fg3 = fg_all[:, 0, :], fg_all[:, 1, :], fg_all[:, 2, :]
            else:
                fg1 = x.get("fg1", None)
                fg2 = x.get("fg2", None)
                fg3 = x.get("fg3", None)

                # Legacy public-release workflow step.
                if fg1 is None:
                    fg1 = torch.zeros((B, self.fg_vocab_size), device=device, dtype=torch.float32)
                else:
                    fg1 = fg1.to(device=device, dtype=torch.float32)
                if fg2 is None:
                    fg2 = torch.zeros((B, self.fg_vocab_size), device=device, dtype=torch.float32)
                else:
                    fg2 = fg2.to(device=device, dtype=torch.float32)
                if fg3 is None:
                    fg3 = torch.zeros((B, self.fg_vocab_size), device=device, dtype=torch.float32)
                else:
                    fg3 = fg3.to(device=device, dtype=torch.float32)

            # Legacy public-release workflow step.
            f1 = self.fg_encoder(fg1)
            f2 = self.fg_encoder(fg2)
            f3 = self.fg_encoder(fg3)
            return f1, f2, f3

        return None

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        """Run the legacy forward operation."""
        # Legacy public-release workflow step.
        g1 = x["g1"]
        g2 = x["g2"]
        g3 = x["g3"]
        scalars = x["scalars"]  # (B,2) [T_norm, t]

        B = int(scalars.shape[0])
        device = scalars.device

        # Legacy public-release workflow step.
        # Legacy public-release workflow step.
        e1 = self.encoder(g1)  # (B, hidden)
        e2 = self.encoder(g2)  # (B, hidden)
        e3 = self.encoder(g3)  # (B, hidden)

        # Legacy public-release workflow step.
        # Legacy public-release workflow step.
        if self.s3_equivariant:
            e_stack = s3_equivariant_embedding([e1, e2, e3])  # (B, 3, hidden)
            e1, e2, e3 = e_stack[:, 0, :], e_stack[:, 1, :], e_stack[:, 2, :]

        # Legacy public-release workflow step.
        # Legacy public-release workflow step.
        mix_emb = None
        if self.use_mix_graph and (self.mix_encoder is not None):
            mix = x.get("mix", None)
            if mix is not None:
                # Legacy public-release workflow step.
                node_h = torch.cat([e1, e2, e3], dim=0)  # (3*B, hidden)
                _, mix_emb = self.mix_encoder(node_h, mix, fallback_num_graphs=B)  # (B, hidden)

        # Legacy public-release workflow step.
        # Legacy public-release workflow step.
        fg_tuple = self._encode_fg(x, B, device)
        if fg_tuple is not None:
            f1, f2, f3 = fg_tuple  # (B, fg_out_dim)
        else:
            f1 = f2 = f3 = None

        # Legacy public-release workflow step.
        if self.fusion_mode == "transformer":
            # Legacy public-release workflow step.
            tokens = []
            type_ids = []

            def _append(tok: torch.Tensor, t_id: int, proj: nn.Module):
                """Run the legacy append operation."""
                # tok: (B, d_in)
                tok = proj(tok)
                tokens.append(tok)
                type_ids.append(torch.full((B, 1), int(t_id), device=device, dtype=torch.long))

            # Legacy public-release workflow step.
            T_MOL = 1    # Legacy public-release workflow step.
            T_ABS = 2    # Legacy public-release workflow step.
            T_PROD = 3   # Legacy public-release workflow step.
            T_MIX = 4    # Legacy public-release workflow step.
            T_FG = 5     # Legacy public-release workflow step.
            T_SCALAR = 6 # Legacy public-release workflow step.

            # Legacy public-release workflow step.
            _append(e1, T_MOL, self.proj_mol)
            _append(e2, T_MOL, self.proj_mol)
            _append(e3, T_MOL, self.proj_mol)

            # Legacy public-release workflow step.
            if self.use_interaction:
                # Legacy public-release workflow step.
                _append(torch.abs(e1 - e2), T_ABS, self.proj_inter)
                _append(torch.abs(e1 - e3), T_ABS, self.proj_inter)
                _append(torch.abs(e2 - e3), T_ABS, self.proj_inter)
                # Legacy public-release workflow step.
                _append(e1 * e2, T_PROD, self.proj_inter)
                _append(e1 * e3, T_PROD, self.proj_inter)
                _append(e2 * e3, T_PROD, self.proj_inter)

            # Legacy public-release workflow step.
            if self.use_mix_graph and self.mix_append_global:
                if mix_emb is None:
                    mix_emb = (e1 + e2 + e3) / 3.0  # Legacy public-release workflow step.
                _append(mix_emb, T_MIX, self.proj_mix)

            # Legacy public-release workflow step.
            if f1 is not None:
                _append(f1, T_FG, self.proj_fg)
                _append(f2, T_FG, self.proj_fg)
                _append(f3, T_FG, self.proj_fg)

            # Legacy public-release workflow step.
            _append(scalars.to(device=device, dtype=torch.float32), T_SCALAR, self.proj_scalar)

            # Legacy public-release workflow step.
            tok = torch.stack(tokens, dim=1)                 # (B, L, tf_dim)
            tid = torch.cat(type_ids, dim=1)                 # (B, L)
            pooled, _ = self.token_fuser(tok, tid)           # Legacy public-release workflow step.
            # Legacy public-release workflow step.
            h = self.backbone(pooled)  # (B, mlp_hidden)
            if getattr(self, "is_binary", False):
                return self.binary_head(h).squeeze(-1)
            E = torch.softmax(self.head_E(h), dim=-1)
            R = torch.softmax(self.head_R(h), dim=-1)
            return torch.cat([E, R], dim=-1)

        # Legacy public-release workflow step.
        if self.fusion_mode == "s3_set":
            # Legacy public-release workflow step.
            if mix_emb is None and self.use_mix_graph and self.mix_append_global:
                mix_emb = (e1 + e2 + e3) / 3.0

            if self.use_interaction:
                # Legacy public-release workflow step.
                mean_23 = (e2 + e3) / 2.0
                mean_13 = (e1 + e3) / 2.0
                mean_12 = (e1 + e2) / 2.0

                def _comp_feat(ei: torch.Tensor, mean_other: torch.Tensor, fi: Optional[torch.Tensor]) -> torch.Tensor:
                    """Run the legacy comp feat operation."""
                    feats = [ei, mean_other, torch.abs(ei - mean_other), ei * mean_other]
                    if self.use_mix_graph and self.mix_append_global:
                        feats.append(mix_emb)
                    if fi is not None:
                        feats.append(fi)
                    feats.append(scalars.to(device=device, dtype=torch.float32))
                    return torch.cat(feats, dim=-1)

                c1 = _comp_feat(e1, mean_23, f1)
                c2 = _comp_feat(e2, mean_13, f2)
                c3 = _comp_feat(e3, mean_12, f3)
            else:
                def _comp_feat(ei: torch.Tensor, fi: Optional[torch.Tensor]) -> torch.Tensor:
                    """Run the legacy comp feat operation."""
                    feats = [ei]
                    if self.use_mix_graph and self.mix_append_global:
                        feats.append(mix_emb)
                    if fi is not None:
                        feats.append(fi)
                    feats.append(scalars.to(device=device, dtype=torch.float32))
                    return torch.cat(feats, dim=-1)

                c1 = _comp_feat(e1, f1)
                c2 = _comp_feat(e2, f2)
                c3 = _comp_feat(e3, f3)

            # Legacy public-release workflow step.
            comp = torch.stack([c1, c2, c3], dim=1)  # (B, 3, feat_dim)
            h = self.comp_backbone(comp.view(B * 3, -1)).view(B, 3, -1)  # (B, 3, mlp_hidden)
            if getattr(self, "is_binary", False):
                # Legacy public-release workflow step.
                return self.binary_head(h.mean(dim=1)).squeeze(-1)
            # Legacy public-release workflow step.
            E_logits = self.comp_head_E(h).squeeze(-1)  # (B, 3)
            R_logits = self.comp_head_R(h).squeeze(-1)  # (B, 3)
            E = torch.softmax(E_logits, dim=1)
            R = torch.softmax(R_logits, dim=1)
            return torch.cat([E, R], dim=-1)

        # Legacy public-release workflow step.
        # Legacy public-release workflow step.
        feats = [e1, e2, e3]

        # Legacy public-release workflow step.
        if self.use_interaction:
            feats += [
                torch.abs(e1 - e2), torch.abs(e1 - e3), torch.abs(e2 - e3),  # Legacy public-release workflow step.
                e1 * e2, e1 * e3, e2 * e3,  # Legacy public-release workflow step.
            ]

        # Legacy public-release workflow step.
        if self.use_mix_graph and self.mix_append_global:
            if mix_emb is None:
                mix_emb = (e1 + e2 + e3) / 3.0  # Legacy public-release workflow step.
            feats.append(mix_emb)

        # Legacy public-release workflow step.
        if f1 is not None:
            feats.extend([f1, f2, f3])

        # Legacy public-release workflow step.
        feats.append(scalars.to(device=device, dtype=torch.float32))

        # Legacy public-release workflow step.
        h = torch.cat(feats, dim=-1)
        h = self.backbone(h)  # (B, mlp_hidden)
        if getattr(self, "is_binary", False):
            return self.binary_head(h).squeeze(-1)
        # Legacy public-release workflow step.
        E = torch.softmax(self.head_E(h), dim=-1)
        R = torch.softmax(self.head_R(h), dim=-1)
        return torch.cat([E, R], dim=-1)
