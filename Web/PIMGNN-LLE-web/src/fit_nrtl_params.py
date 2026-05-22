# -*- coding: utf-8 -*-
"""
fit_nrtl_params.py (v2)

Fix: your data.load_and_prepare_excel() does NOT accept the keyword argument "seed".
So we remove that argument and keep the rest unchanged.

Fit per-system NRTL interaction energies g_ij (J/mol) from tie-line endpoints by minimizing:
    mean_i [ ln(xE_i*gammaE_i) - ln(xR_i*gammaR_i) ]^2

Output:
  <out_dir>/nrtl_params_train.json

Usage (Windows):
  python fit_nrtl_params.py --excel_path "D:/GGNN/YXFL/data_update/update-LLE-all-with-smiles_min3.xlsx"
"""
from __future__ import annotations

import os
import json
import argparse
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch

import config as C
from data import load_and_prepare_excel, split_by_system
from loss import renorm3_torch, nrtl_mu_residual


REQ_COLS = ["T", "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]


def fit_one_system(
    df_sys: pd.DataFrame,
    alpha: float = 0.30,
    steps: int = 3000,
    lr: float = 5e-2,
    g_max: float = 8000.0,
    device: str = "cpu",
) -> np.ndarray:
    """
    Fit g_ij (3x3) for one system_id by minimizing mean squared mu residual over all rows.

    Parameterization:
      g_ij = g_max * tanh(p_ij)  (keeps g bounded)
    """
    # keep only numeric & finite rows
    df_sys = df_sys.copy()
    df_sys = df_sys[REQ_COLS].apply(pd.to_numeric, errors="coerce")
    df_sys = df_sys.dropna(axis=0, how="any")
    if len(df_sys) < 3:
        raise ValueError("Too few valid rows after dropna.")

    T = torch.from_numpy(df_sys["T"].to_numpy(dtype=np.float32)).to(device=device).view(-1)
    xE = torch.from_numpy(df_sys[["Ex1","Ex2","Ex3"]].to_numpy(dtype=np.float32)).to(device=device)
    xR = torch.from_numpy(df_sys[["Rx1","Rx2","Rx3"]].to_numpy(dtype=np.float32)).to(device=device)

    # force 2D for safety
    if xE.dim() == 1:
        xE = xE.view(1, 3)
    if xR.dim() == 1:
        xR = xR.view(1, 3)

    xE = renorm3_torch(xE)
    xR = renorm3_torch(xR)

    # p matrix (3x3) with zero diagonal
    p = torch.zeros((3, 3), device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([p], lr=lr)
    eye = torch.eye(3, device=device)

    for it in range(steps):
        opt.zero_grad(set_to_none=True)

        g = g_max * torch.tanh(p)          # (3,3)
        g = g * (1.0 - eye)                # diag=0
        g_batch = g.unsqueeze(0).expand(xE.size(0), 3, 3).contiguous()

        res = nrtl_mu_residual(xE, xR, T, g_batch, alpha=alpha, R=8.314462618)  # (N,3)
        loss = (res ** 2).mean()

        # mild regularization to avoid extreme params when data is sparse
        reg = 1e-4 * (p ** 2).mean()
        total = loss + reg

        total.backward()
        opt.step()

        if (it + 1) % 500 == 0:
            if float(loss.detach().cpu()) < 1e-6:
                break

    g = (g_max * torch.tanh(p)).detach().cpu().numpy()
    np.fill_diagonal(g, 0.0)
    return g.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel_path", type=str, default=getattr(C, "EXCEL_PATH", ""))
    ap.add_argument("--out_dir", type=str, default=getattr(C, "OUT_DIR", "./model_out"))
    ap.add_argument("--alpha", type=float, default=0.30)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-2)
    ap.add_argument("--g_max", type=float, default=8000.0)
    ap.add_argument("--max_systems", type=int, default=0, help="0 means fit all train systems")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # IMPORTANT: your load_and_prepare_excel() signature is:
    #   (path, min_points_per_group=..., permute_23_aug=...)
    df_raw, _df_aug = load_and_prepare_excel(
        args.excel_path,
        min_points_per_group=int(getattr(C, "MIN_POINTS_PER_GROUP", 3)),
        permute_23_aug=bool(getattr(C, "PERMUTE_23_AUG", True)),
    )

    # Fit NRTL parameters for ALL systems (not just train split)
    # This ensures consistent parameters across train/val/test when using physics loss
    print("[fit_nrtl] Using ALL systems (train+val+test) for parameter fitting")
    all_systems_df = df_raw

    sys_ids = sorted(all_systems_df["system_id"].unique().tolist())
    if args.max_systems and args.max_systems > 0:
        sys_ids = sys_ids[: int(args.max_systems)]

    print(f"[fit_nrtl] Total systems to fit: {len(sys_ids)}")

    params: Dict[str, Any] = {}
    n_ok = 0
    for n, sid in enumerate(sys_ids, 1):
        df_sys = all_systems_df[all_systems_df["system_id"] == sid].copy()
        df_sys = df_sys.dropna(subset=REQ_COLS, how="any")
        if len(df_sys) < 3:
            continue

        try:
            g = fit_one_system(
                df_sys,
                alpha=float(args.alpha),
                steps=int(args.steps),
                lr=float(args.lr),
                g_max=float(args.g_max),
                device="cpu",
            )
        except Exception:
            continue

        params[str(int(sid))] = g.tolist()
        n_ok += 1

        if n % 50 == 0:
            print(f"[fit_nrtl] processed {n}/{len(sys_ids)} | fitted: {n_ok}")

    out_path = os.path.join(args.out_dir, "nrtl_params_all.json")
    obj = {
        "meta": {
            "alpha": float(args.alpha),
            "R": 8.314462618,
            "g_max": float(args.g_max),
            "note": "g_ij fitted on ALL systems (train+val+test); units: J/mol; diag set to 0",
        },
        "params": params,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    print(f"[fit_nrtl] saved: {out_path} | systems with params: {len(params)}")


if __name__ == "__main__":
    main()
