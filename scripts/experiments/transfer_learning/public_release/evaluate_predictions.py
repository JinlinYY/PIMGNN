import argparse
import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    residual = y_true - y_pred
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual**2))
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - np.sum(residual**2) / denominator if denominator > 0 else np.nan
    return {"n": len(y_true), "mae": mae, "rmse": rmse, "r2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--true-column", default="y_true")
    parser.add_argument("--pred-column", default="y_pred")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv_path)
    metrics = regression_metrics(frame[args.true_column], frame[args.pred_column])
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
