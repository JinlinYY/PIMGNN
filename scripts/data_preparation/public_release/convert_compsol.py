"""Convert a CompSol workbook to the legacy pseudo-ternary schema."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CompSol conversion options."""
    parser = argparse.ArgumentParser(description="Convert a CompSol workbook.")
    parser.add_argument("--input-glob", default="datasets/external/compsol/*.xlsx")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/external/compsol/CompSol_Ready.xlsx"),
    )
    return parser.parse_args()


def main() -> None:
    """Find, convert, and save the first matching CompSol source workbook."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Legacy public-release workflow step.
    possible_files = glob.glob(args.input_glob)
    source_file = None
    for f in possible_files:
        if "compsol" in f.lower() and "ready" not in f.lower():
            source_file = f
            break

    if not source_file:
        print("❌ 没找到 CompSol Excel 文件！请确认文件在当前文件夹里。")
        return

    print(f"🚀 找到源文件: {source_file}")
    save_path = str(args.output)

    # Legacy public-release workflow step.
    # Legacy public-release workflow step.
    smiles_db = {
        # Legacy public-release workflow step.
        "water": "O",
        "methanol": "CO",
        "ethanol": "CCO",
        "1-propanol": "CCCO",
        "propan-1-ol": "CCCO",
        "2-propanol": "CC(C)O",
        "isopropanol": "CC(C)O",
        "1-butanol": "CCCCO",
        "butanol": "CCCCO",
        "2-butanol": "CCC(C)O",
        "2-methyl-1-propanol": "CC(C)CO",
        "isobutanol": "CC(C)CO",
        "2-methyl-2-propanol": "CC(C)(C)O",
        "tert-butanol": "CC(C)(C)O",
        "1-pentanol": "CCCCCO",
        "3-methyl-1-butanol": "CC(C)CCO",
        "1-octanol": "CCCCCCCCO",
        "1_2-ethanediol": "OCCO",
        "ethylene glycol": "OCCO",
        "2-propanone": "CC(=O)C",
        "acetone": "CC(=O)C",
        "2-butanone": "CCC(C)=O",
        "mehyl ethyl ketone": "CCC(C)=O",
        "tetrahydrofuran": "C1CCOC1",
        "thf": "C1CCOC1",
        "1_4-dioxane": "C1COCCO1",
        "dioxane": "C1COCCO1",
        "2_2'-oxybisethanol": "OCCOCCO",
        "diethylene glycol": "OCCOCCO",
        "2_2'-oxybispropane": "CC(C)OC(C)C",
        "diisopropyl ether": "CC(C)OC(C)C",
        
        # Legacy public-release workflow step.
        "hexane": "CCCCCC",
        "n-hexane": "CCCCCC",
        "heptane": "CCCCCCC",
        "octane": "CCCCCCCC",
        "nonane": "CCCCCCCCC",
        "decane": "CCCCCCCCCC",
        "pentane": "CCCCC",
        "butane": "CCCC",
        "propane": "CCC",
        "ethane": "CC",
        "methane": "C",
        "cyclohexane": "C1CCCCC1",
        "methylcyclohexane": "CC1CCCCC1",
        "ethylcyclohexane": "CCC1CCCCC1",
        "2-methylpentane": "CCCC(C)C",
        "3-methylpentane": "CCC(C)CC",
        "2_2_4-trimethylpentane": "CC(C)CC(C)(C)C",
        "isooctane": "CC(C)CC(C)(C)C",
        "hexadecane": "CCCCCCCCCCCCCCCC",
        "octadecane": "CCCCCCCCCCCCCCCCCC",
        "eicosane": "CCCCCCCCCCCCCCCCCCCC",
        "tetracosane": "CCCCCCCCCCCCCCCCCCCCCCCC",
        "octacosane": "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",

        # Legacy public-release workflow step.
        "benzene": "c1ccccc1",
        "methylbenzene": "Cc1ccccc1",
        "toluene": "Cc1ccccc1",
        "ethylbenzene": "CCc1ccccc1",
        "1_2-dimethylbenzene": "Cc1ccccc1C",
        "o-xylene": "Cc1ccccc1C",
        "1_3-dimethylbenzene": "Cc1cccc(C)c1",
        "m-xylene": "Cc1cccc(C)c1",
        "1_4-dimethylbenzene": "Cc1ccc(C)cc1",
        "p-xylene": "Cc1ccc(C)cc1",
        "phenol": "Oc1ccccc1",
        "benzenamine": "Nc1ccccc1",
        "aniline": "Nc1ccccc1",
        "chlorobenzene": "Clc1ccccc1",
        "nitrobenzene": "O=[N+]([O-])c1ccccc1",
        "naphthalene": "c1ccc2ccccc2c1",
        
        # Legacy public-release workflow step.
        "trichloromethane": "ClC(Cl)Cl",
        "chloroform": "ClC(Cl)Cl",
        "tetrachloromethane": "ClC(Cl)(Cl)Cl",
        "carbon tetrachloride": "ClC(Cl)(Cl)Cl",
        "dichloromethane": "ClCCl",
        "1_2-dichloroethane": "ClCCCl",
        "1-chlorobutane": "CCCCCl",
        
        # Legacy public-release workflow step.
        "acetic_acid_ethyl_ester": "CCOC(C)=O",
        "ethyl acetate": "CCOC(C)=O",
        "acetic_acid_methyl_ester": "COC(C)=O",
        "methyl acetate": "COC(C)=O",
        "acetic_acid_butyl_ester": "CCCCOC(C)=O",
        "butyl acetate": "CCCCOC(C)=O",
        "acetic_acid": "CC(=O)O",
        "formic_acid": "O=CO",
        "acetonitrile": "CC#N",
        "nitromethane": "C[N+](=O)[O-]",
        "N_N-dimethylformamide": "CN(C)C=O",
        "dmf": "CN(C)C=O",
        "sulfinylbismethane": "CS(=O)C",
        "dmso": "CS(=O)C",
        "1-methyl-2-pyrrolidinone": "CN1CCCC1=O",
        "nmp": "CN1CCCC1=O",
        "tetrahydrothiophene_1_1-dioxide": "O=S1(=O)CCCC1",
        "sulfolane": "O=S1(=O)CCCC1",
        "carbon_dioxide": "O=C=O",
        "nitrogen": "N#N",
        "oxygen": "O=O",
        "hydrogen": "[H][H]",
    }

    # Legacy public-release workflow step.
    print("📖 正在读取 Excel (Sheet='Binary mixtures')...")
    try:
        # Legacy public-release workflow step.
        df = pd.read_excel(source_file, sheet_name="Binary mixtures", skiprows=1, engine="openpyxl")
    except:
        print("⚠️ 读取 Binary mixtures 失败，尝试读取默认 Sheet...")
        df = pd.read_excel(source_file, skiprows=1, engine="openpyxl")

    print(f"📊 原始数据: {len(df)} 行")
    print("⚡ 开始离线匹配 (Matches -> SMILES)...")

    new_data = []
    match_count = 0

    for index, row in df.iterrows():
        try:
            # Legacy public-release workflow step.
            s_name = str(row['solvent']).lower().strip()
            u_name = str(row['solute']).lower().strip()
            val = row['DG_solv[kcal.mol-1]']
            
            # Legacy public-release workflow step.
            smi_solvent = smiles_db.get(s_name)
            smi_solute = smiles_db.get(u_name)
            
            # Legacy public-release workflow step.
            if smi_solvent and smi_solute and pd.notna(val):
                new_data.append({
                    'system_id': 20000 + match_count,
                    'smiles1': smi_solute,   # Legacy public-release workflow step.
                    'smiles2': smi_solvent,  # Legacy public-release workflow step.
                    'smiles3': 'O',          # Legacy public-release workflow step.
                    'T': 298.15,             # Legacy public-release workflow step.
                    'Ex1': 0.01, 'Ex2': 0.99, 'Ex3': 0.0,
                    'Rx1': 0.01, 'Rx2': 0.99, 'Rx3': 0.0,
                    'value': float(val),
                    # Run the training step.
                    'split': 'train' if match_count % 10 < 8 else 'test'
                })
                match_count += 1
                
        except Exception as e:
            continue

    # Legacy public-release workflow step.
    if new_data:
        df_out = pd.DataFrame(new_data)
        df_out.to_excel(save_path, index=False)
        print("\n" + "="*40)
        print(f"🎉 成功生成！无需联网！")
        print(f"📊 成功匹配数据: {len(new_data)} 条")
        print(f"   (这比之前的几百条多多了，而且速度极快)")
        print(f"📂 保存路径: {os.path.abspath(save_path)}")
        print("="*40)
        print("👉 现在的下一步：")
        print(f"1. 确认 config.py 里 EXCEL_PATH = '{save_path}'")
        print("2. 运行 python main.py")
    else:
        print("❌ 匹配数量为 0。请检查 Excel 里的名字是否真的很奇怪。")


if __name__ == "__main__":
    main()
