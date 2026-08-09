# -*- coding: utf-8 -*-
"""Compare experimental and predicted application-case phase diagrams."""

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
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial"]
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=[
    "#4E79A7",  # muted blue
    "#F28E2B",  # orange
    "#59A14F",  # green
    "#E15759",  # red
    "#76B7B2",  # teal
    "#EDC948",  # yellow
    "#B07AA1",  # purple
    "#FF9DA7",  # pink
    "#9C755F",  # brown
    "#BAB0AC",  # gray
])
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
    l1 = str(sub.iloc[0].get("Component 1", "Comp1")).strip()
    l2 = str(sub.iloc[0].get("Component 2", "Comp2")).strip()
    l3 = str(sub.iloc[0].get("Component 3", "Comp3")).strip()

    def shrink(s: str, n: int = 18) -> str:
        s = s.replace("\n", " ")
        return s if len(s) <= n else (s[: n - 1] + "…")

    return (shrink(l1), shrink(l2), shrink(l3))


def sanitize_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = s.replace(" ", "")
    return s


def format_temp(t: float) -> str:
    if abs(t - round(t)) < 1e-8:
        return str(int(round(t)))
    return str(t).replace(".", "p")


def plot_one_system_compare(
    sub: pd.DataFrame,
    out_png: Path,
    title: str,
    corner_labels: Tuple[str, str, str],
    draw_curve: bool = False,
    draw_tielines: bool = False,
) -> None:
    fig = plt.figure(figsize=(4.6, 4.6))
    ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])

    draw_triangle(ax)
    add_corner_labels(ax, corner_labels)

    models = list(pd.unique(sub["Model"]))
    colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]

    for i, m in enumerate(models):
        sm = sub[sub["Model"] == m]
        ex = normalize_simplex(sm[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=float))
        rx = normalize_simplex(sm[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=float))
        ex_xy = ternary_to_xy(ex)
        rx_xy = ternary_to_xy(rx)

        color = colors[i % len(colors)]
        ax.scatter(ex_xy[:, 0], ex_xy[:, 1], s=22, marker="o",
               label=f"{m} - E", color=color, edgecolor="black", linewidth=0.4, alpha=0.9)
        ax.scatter(rx_xy[:, 0], rx_xy[:, 1], s=22, marker="s",
               label=f"{m} - R", color=color, edgecolor="black", linewidth=0.4, alpha=0.9)

    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=1)
    ax.set_title(title, fontsize=10, pad=6)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser("Plot application-case LLE phase diagram comparison")
    ap.add_argument("--excel_path", type=str, required=True, help=" input Excel path ")
    ap.add_argument("--out_dir", type=str, required=True, help=" output directory ")
    ap.add_argument("--group_by_temp", action="store_true",
                    help="Treat each system-temperature pair as a separate phase diagram.")
    ap.add_argument("--draw_curve", action="store_true", help="Draw PCA-ordered extract and raffinate boundary curves.")
    ap.add_argument("--draw_tielines", action="store_true", help="Connect paired extract and raffinate compositions.")
    ap.add_argument("--make_pdf", action="store_true", help="Export all systems as a multipage PDF.")
    ap.add_argument("--max_systems", type=int, default=-1, help=" maximum number of systems to export (-1= all )")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    need_cols = ["Model", "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]

    items: List[Dict[str, Any]] = []
    excel_path = Path(args.excel_path)
    with pd.ExcelFile(excel_path) as xls:
        sheet_names = xls.sheet_names
        if len(sheet_names) > 1:
            
            for sname in sheet_names:
                sub = pd.read_excel(xls, sheet_name=sname)
                if sub.empty:
                    continue
                if not all(c in sub.columns for c in need_cols):
                    continue
                items.append({"key": sname, "sub": sub})
        else:
            
            df = pd.read_excel(xls, sheet_name=sheet_names[0])
            base_cols = ["LLE system NO.", "T/K"] + need_cols
            for c in base_cols:
                if c not in df.columns:
                    raise ValueError(f" missing columns :{c}")
            keys = ["LLE system NO.", "T/K"] if args.group_by_temp else ["LLE system NO."]
            grouped = df.groupby(keys, dropna=False, sort=False)
            for k, sub in grouped:
                sysno = sub["LLE system NO."].iloc[0]
                temp = float(sub["T/K"].iloc[0]) if "T/K" in sub.columns and pd.notna(sub["T/K"].iloc[0]) else None
                temp_str = format_temp(temp) if temp is not None else "NA"
                key = f"system_{sysno}_T{temp_str}K"
                items.append({"key": key, "sub": sub})

    if args.max_systems is not None and args.max_systems > 0:
        items = items[: args.max_systems]

    pdf = None
    if args.make_pdf:
        pdf_path = out_dir / "all_systems_compare.pdf"
        pdf = PdfPages(str(pdf_path))

    records: List[Dict[str, Any]] = []

    for idx, item in enumerate(items, start=1):
        sub = item["sub"]
        key = sanitize_filename(str(item["key"]))

        sysno = sub["LLE system NO."].iloc[0] if "LLE system NO." in sub.columns else key
        temp = float(sub["T/K"].iloc[0]) if "T/K" in sub.columns and pd.notna(sub["T/K"].iloc[0]) else None
        temp_str = format_temp(temp) if temp is not None else "NA"

        png_name = f"{key}_compare.png"
        out_png = out_dir / "png" / png_name

        corner_labels = infer_corner_labels(sub)
        title = f"System {sysno} | T={temp} K | n={len(sub)}" if temp is not None else f"System {sysno} | n={len(sub)}"

        plot_one_system_compare(
            sub=sub,
            out_png=out_png,
            title=title,
            corner_labels=corner_labels,
            draw_curve=args.draw_curve,
            draw_tielines=args.draw_tielines,
        )

        if pdf is not None:
            fig = plt.figure(figsize=(4.6, 4.6))
            ax = fig.add_axes([0.08, 0.08, 0.84, 0.84])
            draw_triangle(ax)
            add_corner_labels(ax, corner_labels)

            models = list(pd.unique(sub["Model"]))
            colors = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]
            for i, m in enumerate(models):
                sm = sub[sub["Model"] == m]
                ex = normalize_simplex(sm[["Ex1", "Ex2", "Ex3"]].to_numpy(dtype=float))
                rx = normalize_simplex(sm[["Rx1", "Rx2", "Rx3"]].to_numpy(dtype=float))
                ex_xy = ternary_to_xy(ex)
                rx_xy = ternary_to_xy(rx)
                color = colors[i % len(colors)]
                ax.scatter(ex_xy[:, 0], ex_xy[:, 1], s=22, marker="o",
                           label=f"{m} - E", color=color, edgecolor="black", linewidth=0.4, alpha=0.9)
                ax.scatter(rx_xy[:, 0], rx_xy[:, 1], s=22, marker="s",
                           label=f"{m} - R", color=color, edgecolor="black", linewidth=0.4, alpha=0.9)

            ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=1)
            ax.set_title(title, fontsize=10, pad=6)
            pdf.savefig(fig, dpi=160, bbox_inches="tight")
            plt.close(fig)

        records.append({
            "idx": idx,
            "system_key": item["key"],
            "system_no": sysno,
            "T_K": temp,
            "n_points": len(sub),
            "png": str(out_png.relative_to(out_dir)),
        })

    index_path = out_dir / "index.csv"
    pd.DataFrame(records).to_csv(index_path, index=False, encoding="utf-8-sig")

    if pdf is not None:
        pdf.close()


if __name__ == "__main__":
    main()
