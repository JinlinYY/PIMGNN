# -*- coding: utf-8 -*-
import os
import re
import json
import argparse
from typing import Optional, Tuple

# Avoid OpenMP duplicate runtime crash on Windows (libomp vs libiomp5md).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import config as C
from train import build_model
from metrics import compute_metrics, print_metrics
from utils import (
    Scaler,
    canonicalize_smiles,
    batch_graphs,
    batch_to_device,
    morgan_fp,
    safe_group_apply_t,
)
from data import GraphCache, MixGraphCache, FunctionalGroupCache


def _norm_col(c: object) -> str:
    s = str(c).strip().replace("\n", " ").replace("\r", " ")
    return " ".join(s.split())


def _find_col(available_cols, candidates) -> Optional[str]:
    norm_cols = {_norm_col(c).lower(): c for c in available_cols}
    for cand in candidates:
        k = _norm_col(cand).lower()
        if k in norm_cols:
            return norm_cols[k]
    return None


def _require_col(df: pd.DataFrame, name: str, candidates) -> str:
    col = _find_col(df.columns.tolist(), candidates)
    if col is None:
        raise KeyError(
            f"Cannot find column for '{name}'. Tried candidates={candidates}\n"
            f"Available columns ({len(df.columns)}):\n{list(df.columns)}"
        )
    return col


def _try_get_col(df: pd.DataFrame, candidates) -> Optional[str]:
    return _find_col(df.columns.tolist(), candidates)


def _load_scaler_from_ckpt_dir(ckpt_dir: str, df_T: np.ndarray) -> Scaler:
    p_last = os.path.join(ckpt_dir, "last_model.pt")
    if os.path.isfile(p_last):
        ck = torch.load(p_last, map_location="cpu")
        if "T_mean" in ck and "T_std" in ck:
            return Scaler(mean=float(ck["T_mean"]), std=float(ck["T_std"]))
    return Scaler.fit(df_T.astype(np.float32))


def _build_fg_cache_for_infer(ckpt_dir: str, vocab_size: int) -> Optional[FunctionalGroupCache]:
    p = os.path.join(ckpt_dir, "fg_corpus.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        fg_cache = FunctionalGroupCache(
            corpus=corpus,
            vocab_size=int(vocab_size),
            min_freq=int(getattr(C, "FG_MIN_FREQ", 3)),
        )
        fg_cache.set_corpus(list(corpus))
        return fg_cache
    except Exception:
        return None


@torch.no_grad()
def _predict_df_graph(model: torch.nn.Module, T_scaler: Scaler, df: pd.DataFrame, ckpt_dir: str) -> np.ndarray:
    device = getattr(C, "DEVICE", "cpu")

    smiles_all = df[["smiles1", "smiles2", "smiles3"]].values.reshape(-1).tolist()
    smiles_all = [canonicalize_smiles(s) for s in smiles_all if isinstance(s, str)]
    smiles_all = [s for s in smiles_all if s]

    g_cache = GraphCache(
        add_hs=getattr(C, "GRAPH_ADD_HS", False),
        add_3d=getattr(C, "GRAPH_ADD_3D", False),
        use_gasteiger=getattr(C, "GRAPH_USE_GASTEIGER", True),
        max_atoms=getattr(C, "GRAPH_MAX_ATOMS", 256),
    )
    if smiles_all:
        g_cache.build_from_smiles(smiles_all)

    use_fg = bool(getattr(C, "USE_FG", False))
    fg_cache = None
    if use_fg:
        fg_cache = _build_fg_cache_for_infer(ckpt_dir, vocab_size=int(getattr(C, "FG_TOPK", 0)))
        if fg_cache is None:
            use_fg = False

    use_mix = bool(getattr(C, "USE_MIX_GRAPH", False))
    mix_cache = MixGraphCache(C) if use_mix else None

    bs = int(getattr(C, "PRED_BATCH_SIZE_GRAPH", 128))
    preds = []
    n = len(df)

    for i0 in range(0, n, bs):
        sub = df.iloc[i0:i0 + bs]
        g1s = [g_cache.get(s) for s in sub["smiles1"].tolist()]
        g2s = [g_cache.get(s) for s in sub["smiles2"].tolist()]
        g3s = [g_cache.get(s) for s in sub["smiles3"].tolist()]

        x = {
            "g1": batch_graphs(g1s),
            "g2": batch_graphs(g2s),
            "g3": batch_graphs(g3s),
        }

        Tn = T_scaler.transform(sub["T"].to_numpy(dtype=np.float32))
        t = sub["t"].to_numpy(dtype=np.float32)
        x["scalars"] = torch.from_numpy(np.stack([Tn, t], axis=1)).to(dtype=torch.float32)

        if use_fg and fg_cache is not None:
            if bool(getattr(C, "FG_TOKEN_MODE", False)):
                L = int(getattr(C, "FG_MAX_TOKENS", 32))
                ids1 = np.zeros((len(sub), L), dtype=np.int64)
                ids2 = np.zeros((len(sub), L), dtype=np.int64)
                ids3 = np.zeros((len(sub), L), dtype=np.int64)
                m1 = np.zeros((len(sub), L), dtype=np.float32)
                m2 = np.zeros((len(sub), L), dtype=np.float32)
                m3 = np.zeros((len(sub), L), dtype=np.float32)
                for j, r in enumerate(sub.itertuples(index=False)):
                    a1, b1 = fg_cache.get_token_ids(getattr(r, "smiles1"), L)
                    a2, b2 = fg_cache.get_token_ids(getattr(r, "smiles2"), L)
                    a3, b3 = fg_cache.get_token_ids(getattr(r, "smiles3"), L)
                    ids1[j, :], ids2[j, :], ids3[j, :] = a1, a2, a3
                    m1[j, :], m2[j, :], m3[j, :] = b1, b2, b3
                x["fg1_ids"] = torch.from_numpy(ids1).to(dtype=torch.long)
                x["fg2_ids"] = torch.from_numpy(ids2).to(dtype=torch.long)
                x["fg3_ids"] = torch.from_numpy(ids3).to(dtype=torch.long)
                x["fg1_mask"] = torch.from_numpy(m1).to(dtype=torch.float32)
                x["fg2_mask"] = torch.from_numpy(m2).to(dtype=torch.float32)
                x["fg3_mask"] = torch.from_numpy(m3).to(dtype=torch.float32)
            else:
                fg1 = np.stack([fg_cache.get(s) for s in sub["smiles1"].tolist()], axis=0).astype(np.float32)
                fg2 = np.stack([fg_cache.get(s) for s in sub["smiles2"].tolist()], axis=0).astype(np.float32)
                fg3 = np.stack([fg_cache.get(s) for s in sub["smiles3"].tolist()], axis=0).astype(np.float32)
                x["fg1"] = torch.from_numpy(fg1).to(dtype=torch.float32)
                x["fg2"] = torch.from_numpy(fg2).to(dtype=torch.float32)
                x["fg3"] = torch.from_numpy(fg3).to(dtype=torch.float32)

        if use_mix and mix_cache is not None:
            from utils import batch_mixture_graphs
            mix_graphs = []
            for r in sub.itertuples(index=False):
                T_raw = float(getattr(r, "T"))
                T_norm = float(T_scaler.transform(np.array([T_raw], dtype=np.float32))[0])
                mix_graphs.append(
                    mix_cache.build(getattr(r, "smiles1"), getattr(r, "smiles2"), getattr(r, "smiles3"), T_norm, T_raw)
                )
            x["mix"] = batch_mixture_graphs(mix_graphs)

        x = batch_to_device(x, device)
        y = model(x).detach().cpu().numpy()
        preds.append(y)

    return np.concatenate(preds, axis=0)


@torch.no_grad()
def _predict_df_fp(model: torch.nn.Module, T_scaler: Scaler, df: pd.DataFrame) -> np.ndarray:
    device = getattr(C, "DEVICE", "cpu")
    bs = int(getattr(C, "BATCH_SIZE", 1024))
    preds = []
    n = len(df)
    for i0 in range(0, n, bs):
        sub = df.iloc[i0:i0 + bs]
        X = []
        for r in sub.itertuples(index=False):
            s1 = canonicalize_smiles(getattr(r, "smiles1"))
            s2 = canonicalize_smiles(getattr(r, "smiles2"))
            s3 = canonicalize_smiles(getattr(r, "smiles3"))
            fp1 = morgan_fp(s1, radius=getattr(C, "FP_RADIUS", 2), n_bits=getattr(C, "FP_BITS", 2048))
            fp2 = morgan_fp(s2, radius=getattr(C, "FP_RADIUS", 2), n_bits=getattr(C, "FP_BITS", 2048))
            fp3 = morgan_fp(s3, radius=getattr(C, "FP_RADIUS", 2), n_bits=getattr(C, "FP_BITS", 2048))
            Tn = T_scaler.transform(np.array([float(getattr(r, "T"))], dtype=np.float32))[0].astype(np.float32)
            t = float(getattr(r, "t"))
            feat = np.concatenate([fp1, fp2, fp3, np.array([Tn, t], dtype=np.float32)], axis=0).astype(np.float32)
            X.append(feat)
        X = torch.from_numpy(np.stack(X, axis=0)).to(device=device, dtype=torch.float32)
        y = model(X).detach().cpu().numpy()
        preds.append(y)
    return np.concatenate(preds, axis=0)


def _renorm3_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(3,)
    x = np.clip(x, 0.0, None)
    s = float(np.sum(x))
    if s < eps:
        return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=np.float64)
    return x / s


def _ternary_to_xy(x1: float, x2: float, x3: float) -> Tuple[float, float]:
    s = x1 + x2 + x3
    if abs(s - 1.0) > 1e-6 and s > 1e-12:
        x1, x2, x3 = x1 / s, x2 / s, x3 / s
    X = x2 + 0.5 * x3
    Y = (np.sqrt(3.0) / 2.0) * x3
    return float(X), float(Y)


def _draw_ternary_axes(ax, labels=("Comp1", "Comp2", "Comp3")) -> None:
    A = (0.0, 0.0)
    B = (1.0, 0.0)
    C_ = (0.5, np.sqrt(3.0) / 2.0)
    ax.plot([A[0], B[0]], [A[1], B[1]], color="black", linewidth=1.3)
    ax.plot([B[0], C_[0]], [B[1], C_[1]], color="black", linewidth=1.3)
    ax.plot([C_[0], A[0]], [C_[1], A[1]], color="black", linewidth=1.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3.0) / 2.0 + 0.08)
    ax.axis("off")
    ax.text(A[0] - 0.02, A[1] - 0.035, labels[0], ha="right", va="top", color="black")
    ax.text(B[0] + 0.02, B[1] - 0.035, labels[1], ha="left", va="top", color="black")
    ax.text(C_[0], C_[1] + 0.04, labels[2], ha="center", va="bottom", color="black")


def _get_component_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    c1 = _find_col(df.columns.tolist(), ["component_1", "component 1", "component1", "comp1", "component"])
    c2 = _find_col(df.columns.tolist(), ["component_2", "component 2", "component2", "comp2", "component.1"])
    c3 = _find_col(df.columns.tolist(), ["component_3", "component 3", "component3", "comp3", "component.2"])
    return c1, c2, c3


def _get_labels(g: pd.DataFrame) -> Tuple[str, str, str]:
    c1, c2, c3 = _get_component_cols(g)
    if c1 and c2 and c3:
        l1 = _first_non_empty(g[c1])
        l2 = _first_non_empty(g[c2])
        l3 = _first_non_empty(g[c3])
        if l1 or l2 or l3:
            return (l1 or "Comp1"), (l2 or "Comp2"), (l3 or "Comp3")
    s1 = _first_non_empty(g["smiles1"]) if "smiles1" in g.columns else "Comp1"
    s2 = _first_non_empty(g["smiles2"]) if "smiles2" in g.columns else "Comp2"
    s3 = _first_non_empty(g["smiles3"]) if "smiles3" in g.columns else "Comp3"
    return s1, s2, s3


def _plot_one_group(g: pd.DataFrame, sid: str, T: Optional[float], save_path: str, tie_max: int = 14) -> None:
    y_true_E = g[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=np.float64)
    y_true_R = g[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float64)
    y_pred_E = g[["pred_Ex1", "pred_Ex2", "pred_Ex3"]].to_numpy(dtype=np.float64)
    y_pred_R = g[["pred_Rx1", "pred_Rx2", "pred_Rx3"]].to_numpy(dtype=np.float64)

    y_true_E = np.stack([_renorm3_np(r) for r in y_true_E], axis=0)
    y_true_R = np.stack([_renorm3_np(r) for r in y_true_R], axis=0)
    y_pred_E = np.stack([_renorm3_np(r) for r in y_pred_E], axis=0)
    y_pred_R = np.stack([_renorm3_np(r) for r in y_pred_R], axis=0)

    Exy_true = np.array([_ternary_to_xy(*p) for p in y_true_E])
    Rxy_true = np.array([_ternary_to_xy(*p) for p in y_true_R])
    Exy_pred = np.array([_ternary_to_xy(*p) for p in y_pred_E])
    Rxy_pred = np.array([_ternary_to_xy(*p) for p in y_pred_R])

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    _draw_ternary_axes(ax, labels=_get_labels(g))

    ax.scatter(Exy_true[:, 0], Exy_true[:, 1], s=22, marker="o", label="True E")
    ax.scatter(Rxy_true[:, 0], Rxy_true[:, 1], s=22, marker="x", label="True R")
    ax.scatter(Exy_pred[:, 0], Exy_pred[:, 1], s=20, marker="^", label="Pred E")
    ax.scatter(Rxy_pred[:, 0], Rxy_pred[:, 1], s=20, marker="v", label="Pred R")

    n = len(g)
    step = max(1, n // int(max(1, tie_max)))
    for i in range(0, n, step):
        ax.plot([Exy_true[i, 0], Rxy_true[i, 0]], [Exy_true[i, 1], Rxy_true[i, 1]], linewidth=1.0)
        ax.plot([Exy_pred[i, 0], Rxy_pred[i, 0]], [Exy_pred[i, 1], Rxy_pred[i, 1]], linewidth=1.0, linestyle="--")

    title = f"System {sid}"
    if T is not None and not np.isnan(T):
        title += f" | T={T:.2f} K"
    title += f" | n={n}"
    ax.set_title(title)
    ax.legend(loc="upper left")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def _coerce_num(v) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip()
    if not s or s.lower() in {"nan", "none"}:
        return float("nan")
    if s.startswith("<"):
        s = s[1:].strip()
    s = s.replace("×", "x").replace("X", "x").replace(" ", "")
    try:
        return float(s)
    except Exception:
        pass
    if "^-" in s:
        try:
            base, exp = s.split("^-", 1)
            base = float(base)
            exp = int("-" + exp)
            return base * (10.0 ** exp)
        except Exception:
            return float("nan")
    if "10^" in s:
        try:
            a, b = s.split("10^", 1)
            a = a.rstrip("x")
            base = float(a) if a else 1.0
            exp = int(b)
            return base * (10.0 ** exp)
        except Exception:
            return float("nan")
    return float("nan")


def _first_non_empty(series: pd.Series) -> str:
    if series is None:
        return ""
    try:
        s = series.astype(str)
    except Exception:
        return ""
    for v in s.tolist():
        v = str(v).strip()
        if v and v.lower() not in {"nan", "none"}:
            return v
    return ""


def _prepare_case_df(excel_path: str) -> Tuple[pd.DataFrame, bool]:
    df = pd.read_excel(excel_path)
    df.columns = [_norm_col(c) for c in df.columns]

    col_sid = _try_get_col(df, [
        "LLE system NO.", "LLE system NO", "LLE system No.", "LLE system No",
        "LLE system number", "LLE system#", "LLE system #",
        "system_id", "system", "System", "System ID", "System_ID",
        "System NO.", "System No.", "System NO", "System No", "System NO"
    ])
    col_T = _require_col(df, "T", ["T/K", "T / K", "T (K)", "T", "Temp", "Temperature", "Temperature/K", "Temperature (K)"])

    col_s1 = _require_col(df, "smiles1", [
        "smiles1", "SMILES1", "SMILES 1",
        "Component 1 SMILES", "Comp 1 SMILES",
        "Component1-SMILES",
        "IL (Component 1) SMILES", "IL (Component 1) full name SMILES"
    ])
    col_s2 = _require_col(df, "smiles2", [
        "smiles2", "SMILES2", "SMILES 2",
        "Component 2 SMILES", "Comp 2 SMILES",
        "Component2-SMILES"
    ])
    col_s3 = _require_col(df, "smiles3", [
        "smiles3", "SMILES3", "SMILES 3",
        "Component 3 SMILES", "Comp 3 SMILES",
        "Component3-SMILES"
    ])

    df = df.rename(columns={
        col_T: "T",
        col_s1: "smiles1",
        col_s2: "smiles2",
        col_s3: "smiles3",
    })
    if col_sid is not None:
        df = df.rename(columns={col_sid: "system_id"})
    else:
        df["system_id"] = np.arange(len(df), dtype=np.int64).astype(str)

    col_c1 = _try_get_col(df, ["Component 1", "Component1", "component 1", "component1", "component_1"])
    col_c2 = _try_get_col(df, ["Component 2", "Component2", "component 2", "component2", "component_2"])
    col_c3 = _try_get_col(df, ["Component 3", "Component3", "component 3", "component3", "component_3"])

    if col_c1 is not None and col_c1 != "component_1":
        df = df.rename(columns={col_c1: "component_1"})
        col_c1 = "component_1"
    if col_c2 is not None and col_c2 != "component_2":
        df = df.rename(columns={col_c2: "component_2"})
        col_c2 = "component_2"
    if col_c3 is not None and col_c3 != "component_3":
        df = df.rename(columns={col_c3: "component_3"})
        col_c3 = "component_3"

    def _fix_water_smiles(comp_col: Optional[str], smi_col: str) -> None:
        if comp_col is None:
            return
        comp = df[comp_col].astype(str)
        smi = df[smi_col].astype(str).str.strip()
        m = smi.isin(["0", "0.0"]) & comp.str.lower().str.contains("water", na=False)
        if m.any():
            df.loc[m, smi_col] = "O"

    _fix_water_smiles(col_c1, "smiles1")
    _fix_water_smiles(col_c2, "smiles2")
    _fix_water_smiles(col_c3, "smiles3")

    for c in ["smiles1", "smiles2", "smiles3"]:
        df[c] = df[c].astype(str).map(canonicalize_smiles)
    df = df[(df["smiles1"] != "") & (df["smiles2"] != "") & (df["smiles3"] != "")].copy()

    df["T"] = pd.to_numeric(df["T"], errors="coerce")
    df = df.dropna(subset=["T"]).copy()

    col_t = "t" if "t" in df.columns else _try_get_col(df, ["T_point", "t_point", "curve_t", "tau"])
    if col_t is not None and col_t != "t":
        df = df.rename(columns={col_t: "t"})

    cols_y = ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
    has_labels = all(k in df.columns for k in cols_y)
    if has_labels:
        for c in cols_y:
            df[c] = df[c].map(_coerce_num)
        df = df.dropna(subset=cols_y).copy()

    if "t" not in df.columns:
        if has_labels:
            df["system_id"] = df["system_id"].astype(str)
            df = safe_group_apply_t(df)
        else:
            df["t"] = 0.5

    df["t"] = pd.to_numeric(df["t"], errors="coerce").fillna(0.5).astype(np.float32)
    df = df[(df["t"] >= 0.0) & (df["t"] <= 1.0)].copy()
    return df, has_labels


def _load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        ckpt = {"state_dict": ckpt}

    if "use_fg" in ckpt:
        setattr(C, "USE_FG", bool(ckpt["use_fg"]))
    if "fg_topk" in ckpt:
        setattr(C, "FG_TOPK", int(ckpt["fg_topk"]))

    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain valid model weights")

    keys = list(state.keys())
    enc_layers = []
    for k in keys:
        m = re.match(r"^encoder\.layers\.(\d+)\.", k)
        if m is not None:
            enc_layers.append(int(m.group(1)))
    if enc_layers:
        setattr(C, "GNN_LAYERS", int(max(enc_layers)) + 1)

    use_mix_graph = any(k.startswith("mix_encoder.") for k in keys)
    setattr(C, "USE_MIX_GRAPH", bool(use_mix_graph))
    if use_mix_graph:
        mix_layers = []
        for k in keys:
            m = re.match(r"^mix_encoder\.layers\.(\d+)\.", k)
            if m is not None:
                mix_layers.append(int(m.group(1)))
        if mix_layers:
            setattr(C, "MIX_LAYERS", int(max(mix_layers)) + 1)

    model = build_model().to(getattr(C, "DEVICE", "cpu"))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(f"[WARN] load_state_dict strict=True failed: {e}")
        incomp = model.load_state_dict(state, strict=False)
        print(f"[WARN] Loaded with strict=False. missing={len(incomp.missing_keys)} unexpected={len(incomp.unexpected_keys)}")

    return model


def _save_metrics_and_predictions(out: pd.DataFrame, out_dir: str, has_labels: bool) -> str:
    p_csv = os.path.join(out_dir, "application_case_predictions.csv")
    out.to_csv(p_csv, index=False, encoding="utf-8-sig")

    p_xlsx = os.path.join(out_dir, "application_case_predictions.xlsx")
    out.to_excel(p_xlsx, index=False)

    print(f"[OK] Saved: {p_csv}")
    print(f"[OK] Saved: {p_xlsx}")

    if has_labels:
        y_true = out[["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float64)
        y_pred = out[["pred_Ex1", "pred_Ex2", "pred_Ex3", "pred_Rx1", "pred_Rx2", "pred_Rx3"]].to_numpy(dtype=np.float64)
        m_all = compute_metrics(y_true, y_pred)
        print_metrics("[Case Metrics]", m_all)

        metrics_sys = []
        for sid, g in out.groupby("system_id"):
            yt = g[["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]].to_numpy(dtype=np.float64)
            yp = g[["pred_Ex1", "pred_Ex2", "pred_Ex3", "pred_Rx1", "pred_Rx2", "pred_Rx3"]].to_numpy(dtype=np.float64)
            m = compute_metrics(yt, yp)
            row = {
                "system_id": str(sid),
                "n": int(len(g)),
                "component_1": _first_non_empty(g["component_1"]) if "component_1" in g.columns else "",
                "component_2": _first_non_empty(g["component_2"]) if "component_2" in g.columns else "",
                "component_3": _first_non_empty(g["component_3"]) if "component_3" in g.columns else "",
                "smiles1": _first_non_empty(g["smiles1"]) if "smiles1" in g.columns else "",
                "smiles2": _first_non_empty(g["smiles2"]) if "smiles2" in g.columns else "",
                "smiles3": _first_non_empty(g["smiles3"]) if "smiles3" in g.columns else "",
            }
            row.update(m)
            metrics_sys.append(row)

        df_metrics_sys = pd.DataFrame(metrics_sys)
        if len(df_metrics_sys) > 0:
            df_metrics_sys = df_metrics_sys.sort_values(["system_id"]).reset_index(drop=True)

        p_metrics = os.path.join(out_dir, "application_case_metrics.json")
        with open(p_metrics, "w", encoding="utf-8") as f:
            json.dump({"overall": m_all}, f, ensure_ascii=False, indent=2)

        p_metrics_txt = os.path.join(out_dir, "application_case_metrics.txt")
        with open(p_metrics_txt, "w", encoding="utf-8") as f:
            for k, v in m_all.items():
                f.write(f"{k}: {v}\n")

        p_sys_csv = os.path.join(out_dir, "application_case_metrics_by_system.csv")
        df_metrics_sys.to_csv(p_sys_csv, index=False, encoding="utf-8-sig")

        p_sys_xlsx = os.path.join(out_dir, "application_case_metrics_by_system.xlsx")
        df_metrics_sys.to_excel(p_sys_xlsx, index=False)

        p_sys_json = os.path.join(out_dir, "application_case_metrics_by_system.json")
        with open(p_sys_json, "w", encoding="utf-8") as f:
            json.dump(metrics_sys, f, ensure_ascii=False, indent=2)

        print(f"[OK] Saved: {p_metrics}")

    print(f"[OK] Rows: {len(out)}")
    return p_csv


def generate_ternary_overlay(csv_path: str, out_dir: str, tie_max: int = 14) -> Tuple[str, str]:
    df = pd.read_csv(csv_path)
    df.columns = [_norm_col(c) for c in df.columns]

    col_sid = _require_col(df, "system_id", ["system_id", "system", "System", "LLE system NO.", "System NO."])
    df = df.rename(columns={col_sid: "system_id"})
    df["system_id"] = df["system_id"].astype(str).str.strip()

    if "T" in df.columns:
        df["T"] = df["T"].map(_coerce_num)
        use_T = df["T"].notna().any()
    else:
        use_T = False

    required = [
        "smiles1", "smiles2", "smiles3",
        "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3",
        "pred_Ex1", "pred_Ex2", "pred_Ex3", "pred_Rx1", "pred_Rx2", "pred_Rx3",
    ]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise KeyError(f"Missing required columns for draw: {miss}")

    num_cols = ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3", "pred_Ex1", "pred_Ex2", "pred_Ex3", "pred_Rx1", "pred_Rx2", "pred_Rx3"]
    for c in num_cols:
        df[c] = df[c].map(_coerce_num)
    df = df.dropna(subset=num_cols).copy()

    if "t" in df.columns:
        df["t"] = df["t"].map(_coerce_num)

    ternary_dir = os.path.join(out_dir, "ternary_overlay")
    png_dir = os.path.join(ternary_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
    pdf_path = os.path.join(ternary_dir, "ternary_overlay_all.pdf")

    groups = df.groupby(["system_id", "T"]) if use_T else df.groupby(["system_id"])

    with PdfPages(pdf_path) as pdf:
        for key, g in groups:
            if use_T:
                sid, T = key
                T = float(T) if (T is not None and not np.isnan(T)) else float("nan")
                name = f"system_{sid}_T_{T:.2f}.png" if not np.isnan(T) else f"system_{sid}_T_nan.png"
            else:
                sid = key[0] if isinstance(key, tuple) else key
                T = None
                name = f"system_{sid}.png"

            if "t" in g.columns and g["t"].notna().any():
                g = g.sort_values("t")

            fig_path = os.path.join(png_dir, name)
            _plot_one_group(g, sid=str(sid), T=T, save_path=fig_path, tie_max=tie_max)

            img = plt.imread(fig_path)
            fig = plt.figure(figsize=(7.2, 6.2))
            plt.imshow(img)
            plt.axis("off")
            ttl = f"System {sid}" + (f" | T={T:.2f} K" if (T is not None and not np.isnan(T)) else "")
            plt.title(ttl, fontsize=10)
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
            plt.close(fig)

    return png_dir, pdf_path


def run_predict_and_draw(ckpt_path: str, excel_path: str, out_dir: Optional[str], tie_max: int = 14, skip_draw: bool = False) -> None:
    ckpt_dir = os.path.dirname(ckpt_path)
    out_dir = out_dir or os.path.join(ckpt_dir, "pred_case_outputs")
    os.makedirs(out_dir, exist_ok=True)

    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass

    df, has_labels = _prepare_case_df(excel_path)
    model = _load_model(ckpt_path)
    T_scaler = _load_scaler_from_ckpt_dir(ckpt_dir, df["T"].to_numpy(dtype=np.float32))

    if bool(getattr(C, "USE_GRAPH", False)):
        preds = _predict_df_graph(model, T_scaler, df, ckpt_dir=ckpt_dir)
    else:
        preds = _predict_df_fp(model, T_scaler, df)

    out = df.copy()
    out["pred_Ex1"] = preds[:, 0]
    out["pred_Ex2"] = preds[:, 1]
    out["pred_Ex3"] = preds[:, 2]
    out["pred_Rx1"] = preds[:, 3]
    out["pred_Rx2"] = preds[:, 4]
    out["pred_Rx3"] = preds[:, 5]

    p_csv = _save_metrics_and_predictions(out, out_dir=out_dir, has_labels=has_labels)

    if not skip_draw:
        png_dir, pdf_path = generate_ternary_overlay(csv_path=p_csv, out_dir=out_dir, tie_max=tie_max)
        print(f"[OK] Ternary PNG dir: {png_dir}")
        print(f"[OK] Ternary PDF: {pdf_path}")


def _pick_existing_path(candidates) -> str:
    for p in candidates:
        if not p:
            continue
        if os.path.isfile(p):
            return p
    return ""


def main():
    parser = argparse.ArgumentParser(description="Application case pipeline: predict + metrics + ternary overlay")
    parser.add_argument("--ckpt", help="Path to model checkpoint (.pt)")
    parser.add_argument("--excel", help="Path to application-case Excel")
    parser.add_argument("--out_dir", default="", help="Output directory (default: <ckpt_dir>/pred_case_outputs)")
    parser.add_argument("--csv", default="", help="Existing prediction CSV for draw-only mode")
    parser.add_argument("--draw_only", action="store_true", help="Only draw ternary overlay from --csv")
    parser.add_argument("--skip_draw", action="store_true", help="Skip ternary drawing after prediction")
    parser.add_argument("--tie_max", type=int, default=14, help="Maximum number of tie-lines to draw per group")
    args = parser.parse_args()

    if args.draw_only:
        if not args.csv:
            raise ValueError("--draw_only requires --csv")
        out_dir = args.out_dir or os.path.dirname(args.csv)
        png_dir, pdf_path = generate_ternary_overlay(csv_path=args.csv, out_dir=out_dir, tie_max=args.tie_max)
        print(f"[OK] Ternary PNG dir: {png_dir}")
        print(f"[OK] Ternary PDF: {pdf_path}")
        return

    ckpt_path = _pick_existing_path([
        args.ckpt,
        getattr(C, "LOAD_CKPT_PATH", ""),
        r".\lle_run_literature\best_model.pt",
        r".\lle_run_aichej\best_model.pt",
        r".\lle_run_aichej1\best_model.pt",
    ])
    excel_path = _pick_existing_path([
        args.excel,
        r".\data_update\case1&2.xlsx",
        getattr(C, "EXCEL_PATH", ""),
    ])

    if not ckpt_path or not excel_path:
        raise ValueError(
            "Missing input files. Provide --ckpt and --excel, or ensure default files exist.\n"
            f"Resolved ckpt: {ckpt_path or 'NOT FOUND'}\n"
            f"Resolved excel: {excel_path or 'NOT FOUND'}"
        )

    if not args.ckpt:
        print(f"[INFO] Auto ckpt: {ckpt_path}")
    if not args.excel:
        print(f"[INFO] Auto excel: {excel_path}")

    run_predict_and_draw(
        ckpt_path=ckpt_path,
        excel_path=excel_path,
        out_dir=(args.out_dir or None),
        tie_max=args.tie_max,
        skip_draw=bool(args.skip_draw),
    )


if __name__ == "__main__":
    main()
