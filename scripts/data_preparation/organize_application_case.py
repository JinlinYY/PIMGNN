# -*- coding: utf-8 -*-
"""Normalize and group application-case workbooks for analysis."""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def sanitize_sheet_name(name: str) -> str:
    name = re.sub(r"[\[\]\*\?/\\]", "_", str(name))
    name = name.replace(":", "-")
    return name[:31]  # Excel sheet name limit


def format_temp(t: float) -> str:
    if abs(t - round(t)) < 1e-8:
        return str(int(round(t)))
    return str(t).replace(".", "p")


def main() -> None:
    ap = argparse.ArgumentParser("Organize application-case results by system+temperature")
    ap.add_argument("--excel_path", type=str, required=True, help=" input Excel path ")
    ap.add_argument("--out_dir", type=str, required=True, help=" output directory ")
    ap.add_argument("--group_by_temp", action="store_true", help="Group records by system identifier and temperature.")
    args = ap.parse_args()

    df = pd.read_excel(Path(args.excel_path))

    need_cols = [
        "LLE system NO.", "Model", "Component 1", "Component 2", "Component 3",
        "T/K", "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3",
    ]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f" missing columns :{c}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = ["LLE system NO.", "T/K"] if args.group_by_temp else ["LLE system NO."]
    grouped = df.groupby(keys, dropna=False, sort=False)

    out_xlsx = out_dir / "application_case_grouped.xlsx"
    records: List[Dict[str, Any]] = []

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        for idx, (k, sub) in enumerate(grouped, start=1):
            sysno = sub["LLE system NO."].iloc[0]
            temp = sub["T/K"].iloc[0] if "T/K" in sub.columns else None
            temp_str = format_temp(float(temp)) if pd.notna(temp) else "NA"

            sheet_name = sanitize_sheet_name(f"system_{sysno}_T{temp_str}K")

            cols = [
                "LLE system NO.", "T/K", "Model",
                "Component 1", "Component 2", "Component 3",
                "Component 1 SMILES", "Component 2 SMILES", "Component 3 SMILES",
                "Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3",
            ]
            cols = [c for c in cols if c in sub.columns]
            sub_out = sub[cols].copy()

            sub_out.to_excel(writer, sheet_name=sheet_name, index=False)

            models = list(pd.unique(sub["Model"]))
            records.append({
                "idx": idx,
                "system_no": sysno,
                "T_K": temp,
                "n_rows": len(sub),
                "models": "; ".join(map(str, models)),
                "sheet": sheet_name,
            })

    index_path = out_dir / "index.csv"
    pd.DataFrame(records).to_csv(index_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
