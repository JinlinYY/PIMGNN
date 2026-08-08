# -*- coding: utf-8 -*-
"""Render ternary LLE phase diagrams and tie-line families."""

from __future__ import annotations
import argparse
import math
import re
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



_SQRT3_2 = math.sqrt(3.0) / 2.0

def normalize_simplex(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = x.sum(axis=1, keepdims=True)
    s = np.where(np.abs(s) < eps, 1.0, s)
    x = x / s
    x = np.clip(x, 0.0, 1.0)
    s2 = x.sum(axis=1, keepdims=True)
    s2 = np.where(np.abs(s2) < eps, 1.0, s2)
    return x / s2

def ternary_to_xy(x123: np.ndarray) -> np.ndarray:
    x123 = normalize_simplex(x123)
    x1, x2, x3 = x123[:, 0], x123[:, 1], x123[:, 2]
    X = x2 + 0.5 * x3
    Y = _SQRT3_2 * x3
    return np.stack([X, Y], axis=1)

def pca_order(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    if len(xy) <= 2:
        return np.arange(len(xy))
    m = xy.mean(axis=0, keepdims=True)
    xc = xy - m
    C = (xc.T @ xc) / max(len(xy) - 1, 1)
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, np.argmax(vals)]
    proj = xc @ v
    return np.argsort(proj)



def draw_triangle(ax: plt.Axes) -> None:
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, _SQRT3_2], [0.0, 0.0]])
    ax.plot(tri[:, 0], tri[:, 1], linewidth=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, _SQRT3_2 + 0.05)
    ax.axis("off")

def add_corner_labels(ax: plt.Axes, labels: Tuple[str, str, str]) -> None:
    l1, l2, l3 = labels
    ax.text(-0.02, -0.03, l1, ha="right", va="top", fontsize=9)
    ax.text(1.02, -0.03, l2, ha="left", va="top", fontsize=9)
    ax.text(0.50, _SQRT3_2 + 0.03, l3, ha="center", va="bottom", fontsize=9)

def infer_corner_labels(sub: pd.DataFrame) -> Tuple[str, str, str]:
    l1 = str(sub.iloc[0].get("IL abbreviation", "Comp1")).strip()
    l2 = str(sub.iloc[0].get("Component 2", "Comp2")).strip()
    l3 = str(sub.iloc[0].get("Component 3", "Comp3")).strip()

    def shrink(s: str, n: int = 18) -> str:
        s = s.replace("\n", " ")
        return s if len(s) <= n else (s[: n - 1] + "…")

    return (shrink(l1), shrink(l2), shrink(l3))

def plot_one_system(
    sub: pd.DataFrame,
    out_png: Path,
    title: str,
    corner_labels: Tuple[str, str, str],
    draw_curve: bool = True,
) -> None:
    ex = normalize_simplex(sub[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=float))
    rx = normalize_simplex(sub[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=float))
    ex_xy = ternary_to_xy(ex)
    rx_xy = ternary_to_xy(rx)

    fig = plt.figure(figsize=(4.2, 4.2))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])

    draw_triangle(ax)
    add_corner_labels(ax, corner_labels)

    
    for i in range(len(ex_xy)):
        ax.plot([ex_xy[i, 0], rx_xy[i, 0]], [ex_xy[i, 1], rx_xy[i, 1]], linewidth=0.6, alpha=0.7)

    ax.scatter(ex_xy[:, 0], ex_xy[:, 1], s=18, marker="o", label="Extract (E)")
    ax.scatter(rx_xy[:, 0], rx_xy[:, 1], s=18, marker="^", label="Raffinate (R)")

    if draw_curve and len(ex_xy) >= 3:
        ax.plot(ex_xy[pca_order(ex_xy), 0], ex_xy[pca_order(ex_xy), 1], linewidth=1.0, alpha=0.9)
        ax.plot(rx_xy[pca_order(rx_xy), 0], rx_xy[pca_order(rx_xy), 1], linewidth=1.0, alpha=0.9)

    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(title, fontsize=10, pad=6)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)



def sanitize_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = s.replace(" ", "")
    return s

def format_temp(t: float) -> str:
    # 330.0 -> 330 ; 330.5 -> 330p5
    if abs(t - round(t)) < 1e-8:
        return str(int(round(t)))
    return str(t).replace(".", "p")

def main():
    ap = argparse.ArgumentParser("Plot all LLE systems (name by LLE system NO.)")
    ap.add_argument("--excel_path", type=str, required=True, help=" input Excel path ")
    ap.add_argument("--out_dir", type=str, required=True, help=" output directory ")
    ap.add_argument("--group_by_temp", action="store_true",
                    help=" whether put ( system identifier + temperature ) Treat as Different system ( Referral On , Avoid the same identifier multiple temperature Override )")
    ap.add_argument("--draw_curve", action="store_true", help=" whether to plot Ex/Rx boundary curve (PCA ordered polyline )")
    ap.add_argument("--make_pdf", action="store_true", help=" whether to export a multipage document PDF")
    ap.add_argument("--max_systems", type=int, default=-1, help=" maximum number of systems to export (-1= all )")
    args = ap.parse_args()

    df = pd.read_excel(Path(args.excel_path))

    
    rename_map = {
        "System NO.": "LLE system NO.",  
        "R1": "Rx1",                     
    }
    df = df.rename(columns=rename_map)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    
    need_cols = ["LLE system NO.", "T/K", "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f" missing columns :{c}, verify You Excel column First Name Already update is Ex*/Rx*")

    keys = ["LLE system NO.", "T/K"] if args.group_by_temp else ["LLE system NO."]
    grouped = df.groupby(keys, dropna=False, sort=False)
    items = list(grouped)
    total = len(items)
    print(f"Detected systems = {total} using keys={keys}")

    if args.max_systems is not None and args.max_systems > 0:
        items = items[: args.max_systems]
        print(f"Will plot first {len(items)} systems (max_systems={args.max_systems})")

    pdf = None
    if args.make_pdf:
        pdf_path = out_dir / "all_systems.pdf"
        pdf = PdfPages(str(pdf_path))
        print(f"PDF enabled -> {pdf_path}")

    records: List[Dict[str, Any]] = []

    for idx, (k, sub) in enumerate(items, start=1):
        
        sysno = sub["LLE system NO."].iloc[0]
        sysno_str = sanitize_filename(str(sysno))

        
        temp = float(sub["T/K"].iloc[0]) if "T/K" in sub.columns and pd.notna(sub["T/K"].iloc[0]) else None
        temp_str = format_temp(temp) if temp is not None else "NA"

        
        png_name = f"system_{sysno_str}_T{temp_str}K.png"
        out_png = out_dir / "png" / png_name

        corner_labels = infer_corner_labels(sub)
        title = f"System {sysno} | T={temp} K | n={len(sub)}" if temp is not None else f"System {sysno} | n={len(sub)}"

        plot_one_system(
            sub=sub,
            out_png=out_png,
            title=title,
            corner_labels=corner_labels,
            draw_curve=args.draw_curve,
        )

        if pdf is not None:
            
            fig = plt.figure(figsize=(4.2, 4.2))
            ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
            draw_triangle(ax)
            add_corner_labels(ax, corner_labels)

            ex = normalize_simplex(sub[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=float))
            rx = normalize_simplex(sub[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=float))
            ex_xy = ternary_to_xy(ex)
            rx_xy = ternary_to_xy(rx)
            for i in range(len(ex_xy)):
                ax.plot([ex_xy[i, 0], rx_xy[i, 0]], [ex_xy[i, 1], rx_xy[i, 1]], linewidth=0.6, alpha=0.7)
            ax.scatter(ex_xy[:, 0], ex_xy[:, 1], s=18, marker="o", label="Extract (E)")
            ax.scatter(rx_xy[:, 0], rx_xy[:, 1], s=18, marker="^", label="Raffinate (R)")
            if args.draw_curve and len(ex_xy) >= 3:
                ax.plot(ex_xy[pca_order(ex_xy), 0], ex_xy[pca_order(ex_xy), 1], linewidth=1.0, alpha=0.9)
                ax.plot(rx_xy[pca_order(rx_xy), 0], rx_xy[pca_order(rx_xy), 1], linewidth=1.0, alpha=0.9)
            ax.legend(loc="upper right", fontsize=8, frameon=False)
            ax.set_title(title, fontsize=10, pad=6)

            pdf.savefig(fig, dpi=140, bbox_inches="tight")
            plt.close(fig)

        records.append({
            "idx": idx,
            "system_no": sysno,
            "T_K": temp,
            "n_points": len(sub),
            "png": str(out_png.relative_to(out_dir)),
        })

        if idx % 50 == 0 or idx == len(items):
            print(f"[{idx}/{len(items)}] plotted")

    index_path = out_dir / "index.csv"
    pd.DataFrame(records).to_csv(index_path, index=False, encoding="utf-8-sig")
    print(f"[OK] index saved: {index_path}")

    if pdf is not None:
        pdf.close()
        print("[OK] pdf saved")

if __name__ == "__main__":
    main()
