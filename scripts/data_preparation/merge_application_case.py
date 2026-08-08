"""Merge experimental and model application-case predictions."""

import pandas as pd
import os
from pathlib import Path

# Define file paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
csv_path = PROJECT_ROOT / "experiments" / "09_application_cases" / "runs" / "reproduction" / "application_case_plot_data_formatted.csv"
xlsx_path = PROJECT_ROOT / "datasets" / "raw" / "应用案例-all.xlsx"
output_path = PROJECT_ROOT / "datasets" / "processed" / "应用案例-all_merged.xlsx"

def merge_data():
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    if not os.path.exists(xlsx_path):
        print(f"Error: Excel file not found at {xlsx_path}")
        return

    print("Loading files...")
    df_csv = pd.read_csv(csv_path)
    df_xlsx = pd.read_excel(xlsx_path)

    print(f"XLSX shape: {df_xlsx.shape}")
    print(f"CSV shape: {df_csv.shape}")

    # Inspect models
    print(f"Models in XLSX: {df_xlsx['Model'].unique()}")
    print(f"Models in CSV: {df_csv['Model'].unique()}")

    # Filter CSV to get only PSMI (assuming Experiment is already in XLSX or we want to avoid duplicates)
    # We should perform a check. If 'Experiment' is in both, we trust the one in XLSX (master) or assume they are identical.
    # To be safe and avoid duplicates, we only take 'PSMI' from the generated CSV.
    df_new_models = df_csv[df_csv['Model'] == 'PSMI']
    
    if df_new_models.empty:
        print("Warning: No 'PSMI' rows found in CSV.")
        # Fallback: maybe the user wants everything from CSV merged?
        # But usually 'Experiment' is duplicated.
        # Let's check if 'PSMI' is indeed the model name in CSV.
        pass
    else:
        print(f"Found {len(df_new_models)} rows for PSMI.")

    # combine
    # Common columns
    common_cols = list(df_xlsx.columns)
    
    # Ensure CSV has these columns
    df_new_models = df_new_models[common_cols]

    # Concatenate
    df_final = pd.concat([df_xlsx, df_new_models], axis=0, ignore_index=True)

    # Sort
    # Define custom sort order for Model: Experiment, PSMI, then others
    model_order = ['Experiment', 'PSMI', 'COSMO-rs', 'NRTL', 'UNIFAC']
    # Create a categorical type for sorting
    df_final['Model'] = pd.Categorical(df_final['Model'], categories=model_order, ordered=True)
    
    df_final.sort_values(by=['LLE system NO.', 'T/K', 'Model'], inplace=True)

    print(f"Final shape: {df_final.shape}")
    print(f"Final models: {df_final['Model'].unique()}")

    # Save
    print(f"Saving to {output_path}...")
    df_final.to_excel(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    merge_data()
