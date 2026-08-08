#!/usr/bin/env python
"""Cluster-bootstrap and plot the PSMI temperature-encoding experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TRUE = ["Ex1", "Ex2", "Ex3", "Rx1", "Rx2", "Rx3"]
PRED = [f"pred_{name}" for name in TRUE]
GROUP = ["system_id", "T"]
ENCODINGS = ["linear_quadratic", "inverse"]
ENCODING_LABELS = {
    "linear_quadratic": r"Polynomial ($T$, $T^2$)",
    "inverse": r"Reciprocal ($1/T$, $1/T^2$)",
}
DISTANCE_LABELS = ["0-5 K", "5-10 K", "10-20 K", ">20 K"]
COLORS = {"linear_quadratic": "#0072B2", "inverse": "#D55E00"}


def add_row_mae(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["row_mae"] = np.abs(
        out[PRED].to_numpy(float) - out[TRUE].to_numpy(float)
    ).mean(axis=1)
    return out


def group_sufficient_statistics(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(GROUP, sort=True)["row_mae"]
        .agg(loss_sum="sum", n_rows="size")
        .reset_index()
    )


def cluster_bootstrap(
    df: pd.DataFrame, rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float]:
    stats = group_sufficient_statistics(df)
    loss = stats["loss_sum"].to_numpy(float)
    count = stats["n_rows"].to_numpy(float)
    draws = rng.integers(0, len(stats), size=(n_boot, len(stats)))
    values = loss[draws].sum(axis=1) / count[draws].sum(axis=1)
    estimate = float(loss.sum() / count.sum())
    low, high = np.quantile(values, [0.025, 0.975])
    return estimate, float(low), float(high)


def paired_cluster_bootstrap(
    polynomial: pd.DataFrame,
    inverse: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float, float]:
    left = group_sufficient_statistics(polynomial).rename(
        columns={"loss_sum": "loss_poly", "n_rows": "n_poly"}
    )
    right = group_sufficient_statistics(inverse).rename(
        columns={"loss_sum": "loss_inverse", "n_rows": "n_inverse"}
    )
    paired = left.merge(right, on=GROUP, validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise RuntimeError("Temperature-encoding predictions are not group aligned")
    if not np.array_equal(paired["n_poly"], paired["n_inverse"]):
        raise RuntimeError("Paired groups have different row counts")
    difference = (paired["loss_inverse"] - paired["loss_poly"]).to_numpy(float)
    count = paired["n_poly"].to_numpy(float)
    draws = rng.integers(0, len(paired), size=(n_boot, len(paired)))
    values = difference[draws].sum(axis=1) / count[draws].sum(axis=1)
    estimate = float(difference.sum() / count.sum())
    low, high = np.quantile(values, [0.025, 0.975])
    return estimate, float(low), float(high)


def distance_label(distance: pd.Series) -> pd.Series:
    return pd.cut(
        distance,
        bins=[-1e-12, 5.0, 10.0, 20.0, np.inf],
        labels=DISTANCE_LABELS,
        include_lowest=True,
    )


def reciprocal_quadratic_approximation(low_k: float, high_k: float) -> dict[str, object]:
    temperature = np.linspace(low_k, high_k, 10001)
    target = 1.0 / temperature
    coefficients = np.polyfit(temperature, target, deg=2)
    fitted = np.polyval(coefficients, temperature)
    relative = np.abs(fitted - target) / target
    return {
        "temperature_interval_k": [low_k, high_k],
        "quadratic_coefficients_descending": coefficients.tolist(),
        "maximum_relative_error_percent": float(100.0 * relative.max()),
        "rms_relative_error_percent": float(100.0 * np.sqrt(np.mean(relative ** 2))),
    }


def load_predictions(root: Path) -> dict[tuple[str, str], pd.DataFrame]:
    frames = {}
    for encoding in ENCODINGS:
        run = root / "runs" / "seed42" / encoding
        for subset, filename in [
            ("Interpolation", "interpolation_predictions.csv"),
            ("Extrapolation", "extrapolation_predictions.csv"),
        ]:
            frame = add_row_mae(pd.read_csv(run / filename))
            if subset == "Extrapolation":
                frame["distance_label"] = distance_label(
                    frame["distance_from_training_range_k"]
                ).astype(str)
            frames[(encoding, subset)] = frame
    return frames


def analyze(
    frames: dict[tuple[str, str], pd.DataFrame], n_boot: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    overall_rows = []
    distance_rows = []
    paired_rows = []
    for subset in ["Interpolation", "Extrapolation"]:
        for encoding in ENCODINGS:
            frame = frames[(encoding, subset)]
            estimate, low, high = cluster_bootstrap(frame, rng, n_boot)
            overall_rows.append({
                "subset": subset,
                "encoding": encoding,
                "n_tielines": len(frame),
                "n_groups": frame.groupby(GROUP).ngroups,
                "mae": estimate,
                "ci_low": low,
                "ci_high": high,
            })
        estimate, low, high = paired_cluster_bootstrap(
            frames[("linear_quadratic", subset)],
            frames[("inverse", subset)],
            rng,
            n_boot,
        )
        paired_rows.append({
            "subset": subset,
            "inverse_minus_linear_quadratic_mae": estimate,
            "ci_low": low,
            "ci_high": high,
        })

    for label in DISTANCE_LABELS:
        selected = {}
        for encoding in ENCODINGS:
            frame = frames[(encoding, "Extrapolation")]
            selected[encoding] = frame[frame["distance_label"] == label]
            estimate, low, high = cluster_bootstrap(selected[encoding], rng, n_boot)
            distance_rows.append({
                "distance_bin": label,
                "encoding": encoding,
                "n_tielines": len(selected[encoding]),
                "n_groups": selected[encoding].groupby(GROUP).ngroups,
                "mae": estimate,
                "ci_low": low,
                "ci_high": high,
            })
        estimate, low, high = paired_cluster_bootstrap(
            selected["linear_quadratic"], selected["inverse"], rng, n_boot
        )
        paired_rows.append({
            "subset": label,
            "inverse_minus_linear_quadratic_mae": estimate,
            "ci_low": low,
            "ci_high": high,
        })
    return pd.DataFrame(overall_rows), pd.DataFrame(distance_rows), pd.DataFrame(paired_rows)


def closed_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)


def plot(
    overall: pd.DataFrame,
    distance: pd.DataFrame,
    paired: pd.DataFrame,
    dataset: Path,
    output_dir: Path,
    uncertainty_label: str = "95% group-bootstrap CI",
) -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11.5,
        "axes.titlesize": 13.2,
        "axes.titleweight": "bold",
        "axes.titlepad": 5.0,
        "axes.labelsize": 12.0,
        "legend.fontsize": 9.8,
        "legend.frameon": False,
        "xtick.labelsize": 10.8,
        "ytick.labelsize": 10.8,
        "axes.linewidth": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.65), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, wspace=0.035, hspace=0.035)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    raw = pd.read_excel(dataset, usecols=["T/K"])
    temperatures = raw["T/K"].to_numpy(float)
    bins = np.arange(np.floor(temperatures.min()) - 0.5, np.ceil(temperatures.max()) + 2.5, 2.5)
    ax_a.hist(temperatures, bins=bins, color="#999999", edgecolor="white", linewidth=0.35)
    ax_a.axvspan(293.15, 323.20, color="#56B4E9", alpha=0.25, label="Training interval")
    ax_a.axvline(293.15, color="#0072B2", linestyle="--", linewidth=1.2)
    ax_a.axvline(323.20, color="#0072B2", linestyle="--", linewidth=1.2)
    ax_a.text(321.8, ax_a.get_ylim()[1] * 0.80, "293.15-323.20 K", ha="right",
              color="#005A8D", fontsize=11.0)
    ax_a.set_xlabel("Temperature (K)")
    ax_a.set_ylabel("Number of tie-lines")
    ax_a.set_title("A  Temperature coverage", loc="left")
    ax_a.legend(loc="upper left")

    x = np.arange(2)
    offsets = [-0.10, 0.10]
    for offset, encoding in zip(offsets, ENCODINGS):
        part = overall[overall["encoding"] == encoding].set_index("subset").loc[["Interpolation", "Extrapolation"]]
        y = part["mae"].to_numpy()
        err = np.vstack([y - part["ci_low"].to_numpy(), part["ci_high"].to_numpy() - y])
        ax_b.errorbar(x + offset, y, yerr=err, fmt="o", color=COLORS[encoding],
                      markersize=7.2, capsize=3.5, capthick=1.5, linewidth=1.9,
                      label=ENCODING_LABELS[encoding])
    ax_b.set_xticks(x, ["Inside interval", "Outside interval"])
    ax_b.set_ylabel("Composition MAE")
    ax_b.set_title("B  Encoding comparison", loc="left")
    ax_b.legend(loc="upper right")
    ax_b.text(0.025, 0.045, uncertainty_label, transform=ax_b.transAxes,
              color="#555555", fontsize=9.2)

    x = np.arange(len(DISTANCE_LABELS))
    for encoding, marker in zip(ENCODINGS, ["o", "s"]):
        part = distance[distance["encoding"] == encoding].set_index("distance_bin").loc[DISTANCE_LABELS]
        y = part["mae"].to_numpy()
        err = np.vstack([y - part["ci_low"].to_numpy(), part["ci_high"].to_numpy() - y])
        ax_c.errorbar(x, y, yerr=err, marker=marker, color=COLORS[encoding],
                      markersize=7.0, capsize=3.5, capthick=1.5, linewidth=1.9,
                      label=ENCODING_LABELS[encoding])
    ax_c.set_xticks(x, DISTANCE_LABELS)
    ax_c.set_xlabel("Distance from training interval")
    ax_c.set_ylabel("Composition MAE")
    ax_c.set_title("C  Error vs extrapolation distance", loc="left")
    ax_c.legend(loc="upper left")

    order = ["Interpolation", "Extrapolation", *DISTANCE_LABELS]
    part = paired.set_index("subset").loc[order]
    y = 1000.0 * part["inverse_minus_linear_quadratic_mae"].to_numpy()
    low = 1000.0 * part["ci_low"].to_numpy()
    high = 1000.0 * part["ci_high"].to_numpy()
    ax_d.axhline(0, color="#444444", linewidth=1.0)
    ax_d.errorbar(np.arange(len(order)), y, yerr=np.vstack([y - low, high - y]),
                  fmt="o", color="#009E73", markersize=7.0, capsize=3.5,
                  capthick=1.5, linewidth=1.9)
    ax_d.set_xticks(np.arange(len(order)), ["Inside", "Outside", "0-5", "5-10", "10-20", ">20"], rotation=20)
    ax_d.set_ylabel(r"Paired MAE difference ($10^{-3}$)")
    ax_d.set_xlabel("Subset / distance from interval (K)")
    ax_d.set_title("D  Reciprocal minus polynomial", loc="left")
    ax_d.text(0.02, 0.96, "< 0 favors reciprocal", transform=ax_d.transAxes,
              va="top", color="#555555", fontsize=9.2)

    for ax in axes.ravel():
        closed_axes(ax)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "temperature_encoding_sensitivity.pdf")
    fig.savefig(output_dir / "temperature_encoding_sensitivity.png", dpi=600)
    plt.close(fig)


def aggregate_across_seeds(
    roots: list[Path], reference_frames: dict[tuple[str, str], pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_rows = []
    raw_distance_rows = []
    for root in roots:
        manifest = json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))
        seed = int(manifest["training"]["seed"])
        table = pd.read_csv(root / "encoding_metrics.csv")
        for _, row in table.iterrows():
            for subset, prefix in [("Interpolation", "interpolation"), ("Extrapolation", "extrapolation")]:
                raw_rows.append({"seed": seed, "subset": subset, "encoding": row["encoding"],
                                 "mae": row[f"{prefix}_mae"]})
        distance_table = pd.read_csv(root / "distance_metrics.csv")
        distance_table["distance_bin"] = distance_table["distance_bin"].astype(str).str.replace("\u2013", "-", regex=False)
        for _, row in distance_table.iterrows():
            raw_distance_rows.append({"seed": seed, "distance_bin": row["distance_bin"],
                                      "encoding": row["encoding"], "mae": row["mae"]})

    raw = pd.DataFrame(raw_rows)
    raw_distance = pd.DataFrame(raw_distance_rows)
    overall_rows = []
    paired_rows = []
    for subset in ["Interpolation", "Extrapolation"]:
        for encoding in ENCODINGS:
            values = raw[(raw.subset == subset) & (raw.encoding == encoding)]["mae"]
            frame = reference_frames[(encoding, subset)]
            mean, sd = float(values.mean()), float(values.std(ddof=1))
            overall_rows.append({"subset": subset, "encoding": encoding,
                                 "n_tielines": len(frame), "n_groups": frame.groupby(GROUP).ngroups,
                                 "mae": mean, "ci_low": mean - sd, "ci_high": mean + sd})
        pivot = raw[raw.subset == subset].pivot(index="seed", columns="encoding", values="mae")
        delta = pivot["inverse"] - pivot["linear_quadratic"]
        mean, sd = float(delta.mean()), float(delta.std(ddof=1))
        paired_rows.append({"subset": subset, "inverse_minus_linear_quadratic_mae": mean,
                            "ci_low": mean - sd, "ci_high": mean + sd})

    distance_rows = []
    for label in DISTANCE_LABELS:
        for encoding in ENCODINGS:
            values = raw_distance[(raw_distance.distance_bin == label) & (raw_distance.encoding == encoding)]["mae"]
            frame = reference_frames[(encoding, "Extrapolation")]
            frame = frame[frame["distance_label"] == label]
            mean, sd = float(values.mean()), float(values.std(ddof=1))
            distance_rows.append({"distance_bin": label, "encoding": encoding,
                                  "n_tielines": len(frame), "n_groups": frame.groupby(GROUP).ngroups,
                                  "mae": mean, "ci_low": mean - sd, "ci_high": mean + sd})
        pivot = raw_distance[raw_distance.distance_bin == label].pivot(index="seed", columns="encoding", values="mae")
        delta = pivot["inverse"] - pivot["linear_quadratic"]
        mean, sd = float(delta.mean()), float(delta.std(ddof=1))
        paired_rows.append({"subset": label, "inverse_minus_linear_quadratic_mae": mean,
                            "ci_low": mean - sd, "ci_high": mean + sd})
    return pd.DataFrame(overall_rows), pd.DataFrame(distance_rows), pd.DataFrame(paired_rows), raw.merge(
        raw_distance, on=["seed", "encoding"], how="outer", suffixes=("_overall", "_distance")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=ROOT / "experiments" / "12_temperature_encoding")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles_no-missing-smiles.xlsx")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "12_temperature_encoding")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--additional-experiment-roots", type=Path, nargs="*", default=[])
    args = parser.parse_args()

    frames = load_predictions(args.experiment_root)
    overall, distance, paired = analyze(frames, args.bootstrap_replicates, args.seed)
    overall.to_csv(args.experiment_root / "encoding_metrics_with_ci.csv", index=False, encoding="utf-8-sig")
    distance.to_csv(args.experiment_root / "distance_metrics_with_ci.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(args.experiment_root / "paired_differences_with_ci.csv", index=False, encoding="utf-8-sig")
    approximation = {
        "original_seed42_training_range": reciprocal_quadratic_approximation(283.15, 353.20),
        "controlled_central_training_range": reciprocal_quadratic_approximation(293.15, 323.20),
        "interpretation": "Numerical approximation only; this does not make the learned polynomial channel a thermodynamic law.",
    }
    (args.experiment_root / "quadratic_approximation_of_inverse_temperature.json").write_text(
        json.dumps(approximation, indent=2), encoding="utf-8"
    )
    plot(overall, distance, paired, args.dataset, args.output_dir)
    if args.additional_experiment_roots:
        roots = [args.experiment_root, *args.additional_experiment_roots]
        multi_overall, multi_distance, multi_paired, multi_raw = aggregate_across_seeds(roots, frames)
        multi_overall.to_csv(args.experiment_root / "multi_seed_encoding_metrics.csv", index=False, encoding="utf-8-sig")
        multi_distance.to_csv(args.experiment_root / "multi_seed_distance_metrics.csv", index=False, encoding="utf-8-sig")
        multi_paired.to_csv(args.experiment_root / "multi_seed_paired_differences.csv", index=False, encoding="utf-8-sig")
        multi_raw.to_csv(args.experiment_root / "multi_seed_raw_metrics.csv", index=False, encoding="utf-8-sig")
        plot(multi_overall, multi_distance, multi_paired, args.dataset, args.output_dir,
             uncertainty_label=rf"Mean $\pm$ 1 SD across {len(roots)} seeds")
        print("\nMulti-seed summary")
        print(multi_overall.to_string(index=False))
        print(multi_distance.to_string(index=False))
        print(multi_paired.to_string(index=False))
    print(overall.to_string(index=False))
    print(distance.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
