# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def _split_fp_and_scalars(x: torch.Tensor, fp_bits: int) -> tuple:
    fp = x[:, : 2 * fp_bits].view(x.shape[0], 2, fp_bits)
    scalars = x[:, 2 * fp_bits :]
    return fp, scalars


class SolubilityMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),  # Configure the output artifacts.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # (B, 1) -> (B,)


class SolubilityANN(nn.Module):
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
            nn.Linear(h3, 1),  # Configure the output artifacts.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SolubilityLSTM(nn.Module):
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
        self.num_proj = nn.Linear(1, d_model)
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
            nn.Linear(hidden_mlp, 1),  # Configure the output artifacts.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fp, scalars = _split_fp_and_scalars(x, self.fp_bits)  # (B,2,fp_bits), (B,1)
        tok_fp = self.fp_proj(fp)  # (B,2,d)
        tok_num = self.num_proj(scalars).unsqueeze(1)  # (B,1,d)
        seq = torch.cat([tok_fp, tok_num], dim=1)  # (B,3,d)
        out, (h, _c) = self.lstm(seq)
        h_last = h[-1]  # (B,d)
        return self.head(h_last).squeeze(-1)


class SolubilityTransformer(nn.Module):
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
        self.num_proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 3, d_model))
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
            nn.Linear(d_model, 1),  # Configure the output artifacts.
        )
        
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fp, scalars = _split_fp_and_scalars(x, self.fp_bits)
        tok_fp = self.fp_proj(fp)  # (B,2,d)
        tok_num = self.num_proj(scalars).unsqueeze(1)  # (B,1,d)
        seq = torch.cat([tok_fp, tok_num], dim=1)  # (B,3,d)
        seq = seq + self.pos
        h = self.enc(seq)  # (B,3,d)
        pooled = h.mean(dim=1)  # (B,d)
        return self.head(pooled).squeeze(-1)


class SolubilityTabKNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 512,
        k: int = 3,
        n_blocks: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        
        class _FourierKANBlock(nn.Module):
            def __init__(self, dim: int, k: int = 3, dropout: float = 0.10):
                super().__init__()
                self.dim = int(dim)
                self.k = int(k)
                in_dim = dim * (1 + 2 * k)
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
            nn.Linear(hidden // 2, 1),  # Configure the output artifacts.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        return self.head(h).squeeze(-1)


def build_solubility_model(
    model_name: str,
    *,
    in_dim: int,
    fp_bits: int,
    hidden: int,
    dropout: float,
    **kwargs
) -> nn.Module:
    name = model_name.lower()
    
    if name == "mlp":
        return SolubilityMLP(in_dim=in_dim, hidden=hidden, dropout=dropout)
    
    if name == "ann":
        ann_hidden = kwargs.get("ANN_HIDDEN", max(512, hidden))
        ann_dropout = kwargs.get("ANN_DROPOUT", dropout)
        return SolubilityANN(in_dim=in_dim, hidden=ann_hidden, dropout=ann_dropout)
    
    if name == "lstm":
        return SolubilityLSTM(
            fp_bits=fp_bits,
            d_model=kwargs.get("LSTM_D_MODEL", 256),
            n_layers=kwargs.get("LSTM_LAYERS", 2),
            dropout=kwargs.get("LSTM_DROPOUT", dropout),
            hidden_mlp=kwargs.get("LSTM_MLP", 256),
        )
    
    if name == "transformer":
        return SolubilityTransformer(
            fp_bits=fp_bits,
            d_model=kwargs.get("TR_D_MODEL", 256),
            n_heads=kwargs.get("TR_HEADS", 8),
            n_layers=kwargs.get("TR_LAYERS", 3),
            dropout=kwargs.get("TR_DROPOUT", dropout),
            mlp_ratio=kwargs.get("TR_MLP_RATIO", 2),
        )
    
    if name == "tabknet":
        return SolubilityTabKNet(
            in_dim=in_dim,
            hidden=kwargs.get("TABK_HIDDEN", hidden),
            k=kwargs.get("TABK_K", 3),
            n_blocks=kwargs.get("TABK_BLOCKS", 2),
            dropout=kwargs.get("TABK_DROPOUT", dropout),
        )
    
    raise ValueError(f"Unknown model_name={model_name}. Available: mlp, ann, lstm, transformer, tabknet")

