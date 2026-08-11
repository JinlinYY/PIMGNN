# Controlled Temperature-Extrapolation Experiment

This experiment evaluates PSMI outside a deliberately restricted training
temperature interval and compares two temperature representations under an
otherwise matched protocol. It maps to SI Section S3.3.2, Tables S3-S4, and
Figure S5.

## Scientific question

The experiment asks whether changing the temperature channels from a
polynomial representation to a reciprocal representation materially changes
interpolation or extrapolation accuracy. The two alternatives are:

- polynomial: scalar `z-score(T)` and edge features `[T/T_ref, (T/T_ref)^2]`;
- reciprocal: scalar `z-score(T_ref/T)` and edge features
  `[T_ref/T, (T_ref/T)^2]`.

Both alternatives use the same model dimensionality, optimization settings,
data partitioning, and supervised objective. They are independently trained
controlled models; they are not evaluations of the Figure 2a checkpoint.

## Dataset partition

The central interval is 293.15-323.20 K. Systems with observations outside
this interval form the extrapolation partition. Systems retained inside the
interval are divided into training, validation, and interpolation partitions
at the chemical-system level. The archived manifests report zero system
overlap between all partitions.

| Seed | Interpolation tie lines / groups | Extrapolation tie lines / groups |
| ---: | ---: | ---: |
| 7 | 709 / 70 | 712 / 72 |
| 42 | 698 / 70 | 712 / 72 |
| 2024 | 705 / 70 | 712 / 72 |

The interpolation systems change with the random split seed, so their tie-line
counts range from 698 to 709 even though every run contains 70
system-temperature groups. The outer-temperature systems are fixed by the
temperature criterion and contain 712 tie lines in all three runs.

The extrapolation partition contains 191, 364, 52, and 105 tie lines in the
0-5 K, 5-10 K, 10-20 K, and greater-than-20 K distance bins, respectively.
The corresponding group counts are 20, 38, 6, and 8. These unequal and sparse
outer bins should be considered when interpreting distance-specific trends.

## Archived results

The three completed split/training seeds are 7, 42, and 2024. The table below
reports the arithmetic mean and sample standard deviation across these seeds.

| Evaluation | Temperature representation | MAE, mean +/- SD |
| --- | --- | ---: |
| Interpolation | Polynomial | 0.08809 +/- 0.00253 |
| Interpolation | Reciprocal | 0.08840 +/- 0.00259 |
| Extrapolation | Polynomial | 0.08637 +/- 0.00186 |
| Extrapolation | Reciprocal | 0.08795 +/- 0.00075 |

For extrapolation, the paired reciprocal-minus-polynomial MAE difference is
0.00159 +/- 0.00259 across the three seeds. This is a descriptive mean and
sample standard deviation, not a confidence interval. The result does not
support a robust accuracy advantage for either representation.

For seed 42, a separate paired bootstrap over complete system-temperature
groups gives an extrapolation MAE difference of 0.000204 with a 95% percentile
interval from -0.000108 to 0.000495. The bootstrap interval quantifies sampling
variation within that completed run; the three-seed standard deviation
quantifies run-to-run dispersion. The two uncertainty summaries are therefore
stored and labeled separately.

## Directory structure

```text
02_temperature_extrapolation/
|-- README.md
|-- figures/
|   |-- temperature_extrapolation_robustness.pdf
|   `-- temperature_extrapolation_robustness.png
`-- results/
    |-- aggregate/                  # Cross-seed and bootstrap summaries
    `-- runs/
        |-- seed_7/
        |-- seed_42/
        `-- seed_2024/
```

Each seed directory contains its split/training manifest, summary tables, and
the pointwise interpolation and extrapolation predictions for both encodings
under `encodings/`. The aggregate directory contains explicitly labeled
three-seed mean/SD tables, seed-42 group-bootstrap tables, and an analysis
manifest.

## Reproduce the archived analysis

Run the following command from the repository root in the documented
`ggnn39` environment:

```bash
python scripts/plot_temperature_encoding_sensitivity.py \
  --results-root experiments/supporting_information/s3_additional_evaluation_and_validation/s3_3_temperature_robustness/02_temperature_extrapolation/results \
  --analysis-output-dir outputs/temperature_extrapolation/results \
  --figure-output-dir outputs/temperature_extrapolation/figures \
  --completed-seeds 7 42 2024 \
  --reference-seed 42 \
  --bootstrap-replicates 10000 \
  --bootstrap-seed 20260806
```

This command reads the archived pointwise predictions, recomputes all
statistics, and rebuilds the figure. It does not train a model.

To execute a new controlled run for a selected seed, use:

```bash
python scripts/run_temperature_encoding_sensitivity.py --seed 42
```

The default output is
`outputs/temperature_extrapolation/runs/seed_42`. New training runs are kept
outside the archived evidence directory unless they are deliberately promoted
after validation.

## Evidence files

| File | Contents |
| --- | --- |
| `results/aggregate/three_seed_encoding_metrics.csv` | Interpolation and extrapolation MAE mean/SD across three seeds |
| `results/aggregate/three_seed_distance_metrics.csv` | Distance-binned MAE mean/SD across three seeds |
| `results/aggregate/three_seed_paired_differences.csv` | Paired representation differences across seeds |
| `results/aggregate/seed_42_system_temperature_bootstrap_*.csv` | Seed-42 95% group-bootstrap intervals |
| `results/aggregate/per_seed_*.csv` | Long-form source values used in cross-seed aggregation |
| `results/aggregate/reciprocal_temperature_approximation.json` | Numerical approximation diagnostics for reciprocal temperature; not a trained-model metric |
| `results/aggregate/analysis_manifest.json` | Seeds, partitions, uncertainty conventions, and figure provenance |
| `results/runs/seed_*/experiment_manifest.json` | Dataset split, training configuration, runtime, and system-overlap audit |
| `results/runs/seed_*/encodings/*/*_predictions.csv` | Pointwise predictions used to recompute the metrics |
