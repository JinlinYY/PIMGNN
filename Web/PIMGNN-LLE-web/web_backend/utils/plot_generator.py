# web_backend/utils/plot_generator.py
import base64
import json
import math
from io import BytesIO
from typing import List, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import importlib.util
import os
import sys

import config as WEB_CONFIG


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        raise ImportError(f"Failed to load module: {file_path}")
    loader.exec_module(module)
    return module


def _load_project_modules():
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    config_path = os.path.join(src_dir, "config.py")
    utils_path = os.path.join(src_dir, "utils.py")
    data_path = os.path.join(src_dir, "data.py")

    project_config = _load_module("project_config", config_path)
    project_utils = _load_module("project_utils", utils_path)

    prev_config = sys.modules.get("config")
    prev_utils = sys.modules.get("utils")

    try:
        sys.modules["config"] = project_config
        sys.modules["utils"] = project_utils
        project_data = _load_module("project_data", data_path)
    finally:
        if prev_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = prev_config
        if prev_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = prev_utils

    return project_config, project_utils, project_data


def apply_nature_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


def ternary_to_xy(x1: float, x2: float, x3: float) -> Tuple[float, float]:
    s = x1 + x2 + x3
    if abs(s - 1.0) > 1e-6 and s > 1e-12:
        x1, x2, x3 = x1 / s, x2 / s, x3 / s
    X = x2 + 0.5 * x3
    Y = (math.sqrt(3) / 2.0) * x3
    return float(X), float(Y)


def draw_ternary_axes(ax, labels=("Comp1(IL)", "Comp2", "Comp3")) -> None:
    A = (0.0, 0.0)
    B = (1.0, 0.0)
    C = (0.5, math.sqrt(3) / 2.0)

    ax.plot([A[0], B[0]], [A[1], B[1]], color="black", linewidth=1.3)
    ax.plot([B[0], C[0]], [B[1], C[1]], color="black", linewidth=1.3)
    ax.plot([C[0], A[0]], [C[1], A[1]], color="black", linewidth=1.3)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, math.sqrt(3) / 2 + 0.08)
    ax.axis("off")

    ax.text(A[0] - 0.02, A[1] - 0.035, labels[0], ha="right", va="top", color="black", fontsize=10)
    ax.text(B[0] + 0.02, B[1] - 0.035, labels[1], ha="left", va="top", color="black", fontsize=10)
    ax.text(C[0], C[1] + 0.04, labels[2], ha="center", va="bottom", color="black", fontsize=10)


def draw_ternary_ticks(ax, ticks=(0.2, 0.4, 0.6, 0.8)) -> None:
    """Draw tick marks and labels along triangle edges."""
    A = (0.0, 0.0)
    B = (1.0, 0.0)
    C = (0.5, math.sqrt(3) / 2.0)

    tick_color = "#555555"
    tick_size = 10
    font_size = 8

    for t in ticks:
        t = float(t)
        # Edge AB (bottom) -> x2 = t
        x, y = t, 0.0
        ax.scatter(x, y, s=tick_size, color=tick_color, zorder=3)
        ax.text(x, y - 0.03, f"{t:.1f}", ha="center", va="top", fontsize=font_size, color=tick_color)

        # Edge AC (left) -> x3 = t
        x = 0.5 * t
        y = (math.sqrt(3) / 2.0) * t
        ax.scatter(x, y, s=tick_size, color=tick_color, zorder=3)
        ax.text(x - 0.02, y, f"{t:.1f}", ha="right", va="center", fontsize=font_size, color=tick_color)

        # Edge BC (right) -> x1 = t
        x = 1.0 - 0.5 * t
        y = (math.sqrt(3) / 2.0) * t
        ax.scatter(x, y, s=tick_size, color=tick_color, zorder=3)
        ax.text(x + 0.02, y, f"{t:.1f}", ha="left", va="center", fontsize=font_size, color=tick_color)


def predict_curve_sweep(model, T_scaler,
                        smiles1: str, smiles2: str, smiles3: str, T: float,
                        n_sweep: int,
                        config_module,
                        utils_module,
                        data_module) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    s1 = utils_module.canonicalize_smiles(smiles1)
    s2 = utils_module.canonicalize_smiles(smiles2)
    s3 = utils_module.canonicalize_smiles(smiles3)
    if not (s1 and s2 and s3):
        raise ValueError("Invalid SMILES.")

    t_grid = np.linspace(0.0, 1.0, n_sweep, dtype=np.float32)
    Tn = T_scaler.transform(np.array([T], dtype=np.float32))[0].astype(np.float32)
    try:
        device = next(model.parameters()).device
    except Exception:
        device = getattr(config_module, "DEVICE", "cpu")
    use_graph = getattr(config_module, "USE_GRAPH", False)

    if use_graph:
        g_cache = data_module.GraphCache(
            add_hs=getattr(config_module, "GRAPH_ADD_HS", False),
            add_3d=getattr(config_module, "GRAPH_ADD_3D", False),
            use_gasteiger=getattr(config_module, "GRAPH_USE_GASTEIGER", True),
            max_atoms=getattr(config_module, "GRAPH_MAX_ATOMS", 256),
        )
        g_cache.build_from_smiles([s1, s2, s3])

        g1 = g_cache.get(s1)
        g2 = g_cache.get(s2)
        g3 = g_cache.get(s3)

        bg1 = utils_module.batch_graphs([g1] * n_sweep)
        bg2 = utils_module.batch_graphs([g2] * n_sweep)
        bg3 = utils_module.batch_graphs([g3] * n_sweep)

        scalars = torch.from_numpy(
            np.stack([np.array([Tn, t], dtype=np.float32) for t in t_grid], axis=0)
        )
        x = {"g1": bg1, "g2": bg2, "g3": bg3, "scalars": scalars}

        if getattr(config_module, "USE_FG", False):
            fg_path = os.path.join(getattr(WEB_CONFIG, "MODEL_DIR", ""), "fg_corpus.json")
            if os.path.isfile(fg_path):
                with open(fg_path, "r", encoding="utf-8") as f:
                    corpus = json.load(f)
                fg_cache = data_module.FunctionalGroupCache(
                    corpus=corpus,
                    vocab_size=int(getattr(config_module, "FG_TOPK", len(corpus))),
                    min_freq=int(getattr(config_module, "FG_MIN_FREQ", 3)),
                )
                fg_cache.set_corpus(list(corpus))
                if bool(getattr(config_module, "FG_TOKEN_MODE", False)):
                    L = int(getattr(config_module, "FG_MAX_TOKENS", 32))
                    ids1, m1 = fg_cache.get_token_ids(s1, L)
                    ids2, m2 = fg_cache.get_token_ids(s2, L)
                    ids3, m3 = fg_cache.get_token_ids(s3, L)
                    x["fg1_ids"] = torch.tensor(np.repeat(ids1[None, :], n_sweep, axis=0), dtype=torch.long)
                    x["fg2_ids"] = torch.tensor(np.repeat(ids2[None, :], n_sweep, axis=0), dtype=torch.long)
                    x["fg3_ids"] = torch.tensor(np.repeat(ids3[None, :], n_sweep, axis=0), dtype=torch.long)
                    x["fg1_mask"] = torch.tensor(np.repeat(m1[None, :], n_sweep, axis=0), dtype=torch.float32)
                    x["fg2_mask"] = torch.tensor(np.repeat(m2[None, :], n_sweep, axis=0), dtype=torch.float32)
                    x["fg3_mask"] = torch.tensor(np.repeat(m3[None, :], n_sweep, axis=0), dtype=torch.float32)
                else:
                    x["fg1"] = torch.tensor(np.repeat(fg_cache.get(s1)[None, :], n_sweep, axis=0), dtype=torch.float32)
                    x["fg2"] = torch.tensor(np.repeat(fg_cache.get(s2)[None, :], n_sweep, axis=0), dtype=torch.float32)
                    x["fg3"] = torch.tensor(np.repeat(fg_cache.get(s3)[None, :], n_sweep, axis=0), dtype=torch.float32)

        if getattr(config_module, "USE_MIX_GRAPH", False):
            mix_cache = data_module.MixGraphCache(config_module)
            mix_graphs = [mix_cache.build(s1, s2, s3, float(Tn), float(T)) for _ in range(n_sweep)]
            x["mix"] = utils_module.batch_mixture_graphs(mix_graphs)

        x = utils_module.batch_to_device(x, device)
        pred = model(x).detach().cpu().numpy()
    else:
        fp_bits = getattr(config_module, "FP_BITS", 2048)
        fp_radius = getattr(config_module, "FP_RADIUS", 2)
        fp1 = utils_module.morgan_fp(s1, radius=fp_radius, n_bits=fp_bits)
        fp2 = utils_module.morgan_fp(s2, radius=fp_radius, n_bits=fp_bits)
        fp3 = utils_module.morgan_fp(s3, radius=fp_radius, n_bits=fp_bits)

        X = []
        for t in t_grid:
            feat = np.concatenate([fp1, fp2, fp3, np.array([Tn, t], dtype=np.float32)], axis=0)
            X.append(feat)
        X = torch.from_numpy(np.stack(X, axis=0)).to(device)
        pred = model(X).detach().cpu().numpy()

    E = np.vstack([utils_module.renorm3(p[:3]) for p in pred])
    R = np.vstack([utils_module.renorm3(p[3:]) for p in pred])
    return t_grid, E, R


def generate_ternary_plot(
    model,
    T_scaler,
    smiles_list: List[str],
    temperature: float,
    e_compositions: List[float],
    r_compositions: List[float],
    tie_lines_count: int = 14,
) -> str:
    """生成三元相图并返回 base64 编码（复刻 viz.py 核心绘图逻辑）"""
    config_module, utils_module, data_module = _load_project_modules()
    apply_nature_style()

    s1 = utils_module.canonicalize_smiles(smiles_list[0])
    s2 = utils_module.canonicalize_smiles(smiles_list[1])
    s3 = utils_module.canonicalize_smiles(smiles_list[2])
    labels = ("Component 1", "Component 2", "Component 3")

    e_norm = utils_module.renorm3(np.array(e_compositions, dtype=np.float32))
    r_norm = utils_module.renorm3(np.array(r_compositions, dtype=np.float32))

    if model is None:
        class _DummyModel:
            def eval(self):
                return self

            def __call__(self, x):
                if isinstance(x, dict) and "scalars" in x:
                    batch = int(x["scalars"].shape[0])
                    device = x["scalars"].device
                elif torch.is_tensor(x):
                    batch = int(x.shape[0])
                    device = x.device
                else:
                    batch = 1
                    device = torch.device("cpu")
                return torch.zeros((batch, 6), dtype=torch.float32, device=device)

        model = _DummyModel()

    t_grid, E_pred, R_pred = predict_curve_sweep(
        model,
        T_scaler,
        s1,
        s2,
        s3,
        float(temperature),
        n_sweep=getattr(config_module, "N_SWEEP", 80),
        config_module=config_module,
        utils_module=utils_module,
        data_module=data_module,
    )

    Exy_pred = np.array([ternary_to_xy(*p) for p in E_pred])
    Rxy_pred = np.array([ternary_to_xy(*p) for p in R_pred])

    e_xy = ternary_to_xy(*e_norm)
    r_xy = ternary_to_xy(*r_norm)

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    draw_ternary_axes(ax, labels=labels)
    draw_ternary_ticks(ax)

    ax.plot(Exy_pred[:, 0], Exy_pred[:, 1], linewidth=2.0, label="Pred E (curve)")
    ax.plot(Rxy_pred[:, 0], Rxy_pred[:, 1], linewidth=2.0, label="Pred R (curve)")

    # No input points (match requested minimal style)

    draw_max = int(tie_lines_count) if tie_lines_count is not None else int(getattr(config_module, "DRAW_TIELINES_MAX", 14))
    draw_max = max(1, min(draw_max, len(t_grid)))
    idxs = np.linspace(0, len(t_grid) - 1, draw_max, dtype=int)

    # Pred points along curves (triangles), count matches tie-lines
    ax.scatter(Exy_pred[idxs, 0], Exy_pred[idxs, 1], s=16, marker="^", label="Pred E (points)")
    ax.scatter(Rxy_pred[idxs, 0], Rxy_pred[idxs, 1], s=16, marker="v", label="Pred R (points)")

    first = True
    for i in idxs:
        ax.plot(
            [Exy_pred[i, 0], Rxy_pred[i, 0]],
            [Exy_pred[i, 1], Rxy_pred[i, 1]],
            linewidth=1.0,
            linestyle="--",
            label="Pred tie-lines" if first else None,
        )
        first = False

    ax.set_title(f"Prediction | T={float(temperature):.2f} K")
    ax.legend(loc="upper left", fontsize=9)

    fig.subplots_adjust(left=0.06, right=0.94, top=0.9, bottom=0.1)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=260)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"
