# -*- coding: utf-8 -*-
"""
Model zoo for LLE curve point prediction.

All torch models output 6 numbers in [0,1] with:
  [Ex1, Ex2, Ex3, Rx1, Rx2, Rx3]
and each triplet sums to 1 (softmax constraint).

Model names (torch):
  - "mlp" (default): LLECurveNet
  - "ann": deeper MLP
  - "lstm": sequence model over [fp1, fp2, fp3, (T,t)]
  - "transformer": transformer encoder over same 4 tokens
  - "tabknet": lightweight KAN-inspired tabular net (Fourier basis on hidden)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# helpers
# -------------------------
def _phase_softmax(y: torch.Tensor) -> torch.Tensor:
    """Apply softmax separately on E(0:3) and R(3:6)."""
    yE = F.softmax(y[:, :3], dim=-1)
    yR = F.softmax(y[:, 3:], dim=-1)
    return torch.cat([yE, yR], dim=-1)


def _split_fp_and_scalars(x: torch.Tensor, fp_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    x: (B, 3*fp_bits + 2) -> fp: (B, 3, fp_bits), scalars: (B, 2) where scalars=[Tn, t]
    """
    fp = x[:, : 3 * fp_bits].view(x.shape[0], 3, fp_bits)
    scalars = x[:, 3 * fp_bits :]
    return fp, scalars


# -------------------------
# Multilayer-perceptron baseline
# -------------------------
class LLECurveNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return _phase_softmax(y)


# -------------------------
# ANN: deeper MLP
# -------------------------
class LLECurveANN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 768, dropout: float = 0.20):
        super().__init__()
        h1 = hidden
        h2 = max(256, hidden // 2)
        h3 = max(128, hidden // 4)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(h2, h3),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(h3, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _phase_softmax(self.net(x))


# -------------------------
# LSTM over 4 tokens
# -------------------------
class LLECurveLSTM(nn.Module):
    def __init__(
        self,
        fp_bits: int,
        d_model: int = 256,
        n_layers: int = 2,
        dropout: float = 0.15,
        hidden_mlp: int = 256,
    ):
        super().__init__()
        self.fp_bits = int(fp_bits)
        self.fp_proj = nn.Linear(fp_bits, d_model)
        self.num_proj = nn.Linear(2, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mlp, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fp, scalars = _split_fp_and_scalars(x, self.fp_bits)          # (B,3,fp_bits), (B,2)
        tok_fp = self.fp_proj(fp)                                     # (B,3,d)
        tok_num = self.num_proj(scalars).unsqueeze(1)                 # (B,1,d)
        seq = torch.cat([tok_fp, tok_num], dim=1)                     # (B,4,d)
        out, (h, _c) = self.lstm(seq)                                 # h: (n_layers,B,d)
        h_last = h[-1]                                                # (B,d)
        y = self.head(h_last)
        return _phase_softmax(y)


# -------------------------
# Transformer encoder over 4 tokens
# -------------------------
class LLECurveTransformer(nn.Module):
    def __init__(
        self,
        fp_bits: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.15,
        mlp_ratio: int = 2,
    ):
        super().__init__()
        self.fp_bits = int(fp_bits)
        self.fp_proj = nn.Linear(fp_bits, d_model)
        self.num_proj = nn.Linear(2, d_model)

        self.pos = nn.Parameter(torch.zeros(1, 4, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 6),
        )

        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fp, scalars = _split_fp_and_scalars(x, self.fp_bits)
        tok_fp = self.fp_proj(fp)                        # (B,3,d)
        tok_num = self.num_proj(scalars).unsqueeze(1)    # (B,1,d)
        seq = torch.cat([tok_fp, tok_num], dim=1)        # (B,4,d)
        seq = seq + self.pos
        h = self.enc(seq)                                # (B,4,d)
        pooled = h.mean(dim=1)                           # (B,d)
        y = self.head(pooled)
        return _phase_softmax(y)


# -------------------------
# TabKNet: KAN-inspired Fourier basis on hidden features
# (Practical interpretation of "TabKNet/TabKAN-style" for tabular data.)
# -------------------------
class _FourierKANBlock(nn.Module):
    def __init__(self, dim: int, k: int = 3, dropout: float = 0.10):
        super().__init__()
        self.dim = int(dim)
        self.k = int(k)
        in_dim = dim * (1 + 2 * k)  # [x, sin(kx), cos(kx)]
        self.proj = nn.Linear(in_dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for i in range(1, self.k + 1):
            feats.append(torch.sin(i * x))
            feats.append(torch.cos(i * x))
        z = torch.cat(feats, dim=-1)
        z = self.proj(z)
        z = self.drop(z)
        return self.ln(x + z)


class LLECurveTabKNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 512,
        k: int = 3,
        n_blocks: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        blocks = []
        for _ in range(n_blocks):
            blocks.append(_FourierKANBlock(hidden, k=k, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        y = self.head(h)
        return _phase_softmax(y)


# -------------------------
# SMILES RNN (no fingerprints)
# -------------------------
class SmilesRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int = 0,
        emb_dim: int = 256,
        hidden: int = 384,
        num_layers: int = 2,
        dropout: float = 0.15,
        scalar_dim: int = 2,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.scalar_proj = nn.Linear(scalar_dim, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 6),
        )

    def forward(self, tokens: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        tok = self.emb(tokens)                              # (B,L,d)
        scalar_tok = self.scalar_proj(scalars).unsqueeze(1) # (B,1,d)
        seq = torch.cat([tok, scalar_tok], dim=1)           # (B,L+1,d)
        out, (h, _c) = self.lstm(seq)
        h_last = h[-1]
        y = self.head(h_last)
        return _phase_softmax(y)


def _normalize_adjacency(adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply masked symmetric adjacency normalization with self-loops."""
    batch_size, atom_count, _ = adjacency.shape
    identity = torch.eye(atom_count, device=adjacency.device).expand(batch_size, -1, -1)
    pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
    adjacency = (adjacency + identity) * pair_mask
    degree_inverse_sqrt = (adjacency.sum(dim=-1) + 1e-6).pow(-0.5)
    return (
        degree_inverse_sqrt.unsqueeze(-1)
        * adjacency
        * degree_inverse_sqrt.unsqueeze(1)
    )


class _GCNLayer(nn.Module):
    """One dense graph-convolution layer for a padded molecular batch."""

    def __init__(self, dimension: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Attribute names retain compatibility with the published checkpoints.
        self.lin = nn.Linear(dimension, dimension)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        nodes: torch.Tensor,
        normalized_adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = torch.bmm(normalized_adjacency, nodes)
        hidden = self.drop(F.relu(self.lin(hidden)))
        return hidden * mask.unsqueeze(-1)


class GraphEncoder(nn.Module):
    """Encode one molecular graph with stacked GCN layers and mean pooling."""

    def __init__(
        self,
        node_dim: int,
        hidden: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList(
            [_GCNLayer(hidden, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(
        self,
        nodes: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = F.relu(self.input(nodes)) * mask.unsqueeze(-1)
        normalized_adjacency = _normalize_adjacency(adjacency, mask)
        for layer in self.layers:
            hidden = layer(hidden, normalized_adjacency, mask)
        denominator = mask.sum(dim=1, keepdim=True) + 1e-6
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denominator


class LLECurveGNN(nn.Module):
    """Predict LLE compositions from three independently encoded molecules."""

    is_gnn = True

    def __init__(
        self,
        node_dim: int,
        hidden: int = 256,
        n_layers: int = 3,
        dropout: float = 0.10,
        mlp_hidden: int = 256,
        scalar_dim: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = GraphEncoder(node_dim, hidden, n_layers, dropout)
        input_dim = hidden * 3 + scalar_dim
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 6),
        )

    def forward(self, graph1, graph2, graph3, scalars: torch.Tensor) -> torch.Tensor:
        embeddings = [self.encoder(*graph) for graph in (graph1, graph2, graph3)]
        prediction = self.head(torch.cat([*embeddings, scalars], dim=-1))
        return _phase_softmax(prediction)


# -------------------------
# Model factory
# -------------------------
TORCH_MODEL_ZOO: Dict[str, str] = {
    "mlp": "mlp",
    "lle_curve_net": "mlp",
    "ann": "ann",
    "lstm": "lstm",
    "transformer": "transformer",
    "tabknet": "tabknet",
    "tabkanet": "tabknet",   # common user typo / alias
    "tabkan": "tabknet",     # common user typo / alias
    "smiles_rnn": "smiles_rnn",
    "gnn": "gnn",
}

def build_torch_model(model_name: str, *, in_dim: int, fp_bits: int, hidden: int, dropout: float, **kwargs) -> nn.Module:
    name = TORCH_MODEL_ZOO.get(model_name.lower(), model_name.lower())

    if name == "mlp":
        return LLECurveNet(in_dim=in_dim, hidden=hidden, dropout=dropout)

    if name == "ann":
        ann_hidden = kwargs.get("ANN_HIDDEN", max(512, hidden))
        ann_dropout = kwargs.get("ANN_DROPOUT", dropout)
        return LLECurveANN(in_dim=in_dim, hidden=ann_hidden, dropout=ann_dropout)

    if name == "lstm":
        return LLECurveLSTM(
            fp_bits=fp_bits,
            d_model=kwargs.get("LSTM_D_MODEL", 256),
            n_layers=kwargs.get("LSTM_LAYERS", 2),
            dropout=kwargs.get("LSTM_DROPOUT", dropout),
            hidden_mlp=kwargs.get("LSTM_MLP", 256),
        )

    if name == "transformer":
        return LLECurveTransformer(
            fp_bits=fp_bits,
            d_model=kwargs.get("TR_D_MODEL", 256),
            n_heads=kwargs.get("TR_HEADS", 8),
            n_layers=kwargs.get("TR_LAYERS", 3),
            dropout=kwargs.get("TR_DROPOUT", dropout),
            mlp_ratio=kwargs.get("TR_MLP_RATIO", 2),
        )

    if name == "tabknet":
        return LLECurveTabKNet(
            in_dim=in_dim,
            hidden=kwargs.get("TABK_HIDDEN", hidden),
            k=kwargs.get("TABK_K", 3),
            n_blocks=kwargs.get("TABK_BLOCKS", 2),
            dropout=kwargs.get("TABK_DROPOUT", dropout),
        )

    if name == "smiles_rnn":
        vocab_size = kwargs.get("vocab_size")
        pad_idx = kwargs.get("pad_idx", 0)
        scalar_dim = kwargs.get("scalar_dim", 2)
        if vocab_size is None:
            raise ValueError("smiles_rnn requires vocab_size kwarg")
        return SmilesRNN(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            emb_dim=kwargs.get("SMILES_EMB_DIM", 256),
            hidden=kwargs.get("SMILES_HIDDEN", 384),
            num_layers=kwargs.get("SMILES_LAYERS", 2),
            dropout=kwargs.get("SMILES_DROPOUT", 0.15),
            scalar_dim=scalar_dim,
        )

    if name == "gnn":
        return LLECurveGNN(
            node_dim=kwargs.get("GNN_NODE_DIM", 11),
            hidden=kwargs.get("GNN_HIDDEN", 256),
            n_layers=kwargs.get("GNN_LAYERS", 3),
            dropout=kwargs.get("GNN_DROPOUT", 0.10),
            mlp_hidden=kwargs.get("GNN_MLP", 256),
            scalar_dim=kwargs.get("GNN_SCALAR_DIM", 2),
        )

    raise ValueError(f"Unknown torch model_name={model_name}. Available: {sorted(set(TORCH_MODEL_ZOO))}")
