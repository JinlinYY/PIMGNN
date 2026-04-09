# Commands & Script Overview

This project is intended to be run from the project root.


PowerShell:

```powershell
cd <project_root>
```

---

## 0) One-time setup

### Configure paths in `src\config.py`

- Required:

  - `EXCEL_PATH`: path to your Excel dataset
  - `OUT_DIR`: output directory, either relative or absolute
- Common dataset files:

  - `D:\GGNN\YXFL-github\data_update\update-LLE-all-with-smiles_min3.xlsx`: dataset from Sun et al. literature
  - `D:\GGNN\YXFL-github\data_update\LLE-literature-data-boosted.xlsx`: literature-collected LLE dataset
  - `D:\GGNN\YXFL-github\data_update\case12.xlsx`: two application-case datasets

---

## 1) Main pipeline

### `src\main.py`

- Purpose: end-to-end pipeline for loading Excel data, splitting train/val/test sets, training or loading a model, evaluating it, running pointwise prediction, and generating parity and ternary plots.
- Run:

```powershell
python .\src\main.py
```

- Outputs: written to `OUT_DIR`, such as prediction CSV files, parity PNGs, ternary PDFs/PNGs, and checkpoints.

---

## 2) Fit NRTL parameters for mechanistic / physics loss

### `src\fit_nrtl_params.py`

- Purpose: fit per-system NRTL interaction parameters and save them as JSON.
- Run:

```powershell
python .\src\fit_nrtl_params.py --excel_path <excel_path> --out_dir ".\nrtl_param"
```

- Output: `.\nrtl_param\nrtl_params_all.json`
- Next step: set `NRTL_PARAMS_PATH` in `src\config.py` to the generated JSON, or continue using `src\nrtl_params_train.json`.

---

## 3) Evaluation and explainability

### `src\eval_explain.py`

- Purpose: evaluate the model and run explanation methods such as saliency, Integrated Gradients, GraphExplainer, and SHAP-FG, depending on installed dependencies.
- Important: this script reads the dataset path from `src\config.py` via `EXCEL_PATH`.

Run in overall test mode:

```powershell
python .\src\eval_explain.py --mode test --ckpt auto --out_dir ".\eval_output" --explain saliency --objective loss --target ALL --max_explain_samples 256
```

Run for a single system:

```powershell
python .\src\eval_explain.py --mode system --system_id 123 --ckpt auto --out_dir ".\eval_output" --explain saliency --objective loss --target ALL
```

If `--ckpt auto` cannot find a checkpoint, use an explicit checkpoint path:

```powershell
python .\src\eval_explain.py --mode test --ckpt <ckpt_path> --out_dir ".\eval_output" --explain saliency --objective loss --target ALL --max_explain_samples 256
```

---

## 4) Visualize from an existing prediction CSV

These scripts do not require editing `EXCEL_PATH`.

### `src\plot_test_viz_from_csv.py`

- Purpose: generate parity and ternary plots from a prediction CSV, grouped by `system_id` and `T`.
- Run:

```powershell
python .\src\plot_test_viz_from_csv.py --csv <prediction_csv> --out_dir ".\eval_output\viz_from_csv"
```

### `src\plot_test_viz_from_csv_extra.py`

- Purpose: generate enhanced plots, including combined parity plots, error statistics, and optional ternary PNGs.
- Run:

```powershell
python .\src\plot_test_viz_from_csv_extra.py --csv <prediction_csv> --out_dir ".\eval_output\viz_from_csv_extra"
```

- Skip ternary plots for faster execution:

```powershell
python .\src\plot_test_viz_from_csv_extra.py --csv <prediction_csv> --out_dir ".\eval_output\viz_from_csv_extra" --skip_ternary
```

---

## 5) One-off utility scripts

### `src\case_predict_draw.py` (recommended)

- Purpose: unified application-case workflow for prediction, metrics export, and ternary overlay plotting.
- This script replaces the typical two-step usage of `testcase.py` + `draw.py`.

Predict + metrics + ternary plots:

```powershell
python .\src\case_predict_draw.py --ckpt <ckpt_path> --excel <excel_path>
```

Predict + metrics only (skip ternary plots):

```powershell
python .\src\case_predict_draw.py --ckpt <ckpt_path> --excel <excel_path> --skip_draw
```

Draw-only mode from an existing prediction CSV:

```powershell
python .\src\case_predict_draw.py --draw_only --csv <prediction_csv>
```

- Main outputs (under `out_dir`):
  - `application_case_predictions.csv/.xlsx`
  - `application_case_metrics.json/.txt`
  - `application_case_metrics_by_system.csv/.xlsx/.json`
  - `ternary_overlay/png/*.png`
  - `ternary_overlay/ternary_overlay_all.pdf`

---

## Module files

These files are usually imported by other scripts rather than run directly.

- `src\config.py`: configuration, including paths, hyperparameters, and feature switches
- `src\data.py`: Excel loading, preprocessing, dataset splitting, caches, and datasets
- `src\utils.py`: utilities such as RDKit featurization, batching, and scalers
- `src\model.py`: model architectures
- `src\train.py`: training loop, checkpointing, and training-curve plotting
- `src\predict.py`: inference helpers used by the main pipeline
- `src\metrics.py`: evaluation metrics and physics-consistency metrics
- `src\loss.py`: mechanistic NRTL-based physics loss
- `src\viz.py` / `src\viz_advanced.py`: visualization utilities

---

## Quick start

1. Edit `EXCEL_PATH` and `OUT_DIR` in `src\config.py`.
2. Run:

```powershell
python .\src\main.py
```

3. Outputs will be written to `OUT_DIR`.
