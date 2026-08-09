# PSMI: Physics-Informed Prediction of Ternary Liquid-Liquid Equilibria

PSMI predicts tie-line compositions for ternary liquid-liquid equilibrium (LLE) systems from molecular graphs and operating conditions. This repository provides the model implementation, thermodynamic regularization, curated datasets, fixed system-level splits, trained checkpoints, baseline comparisons, paper-aligned experiments, reference results, and a browser-based prediction interface.

## Scientific scope

PSMI is designed for data-driven prediction of coexisting extract- and raffinate-phase compositions. Each sample contains three molecular components, temperature, pressure where applicable, and a phase-path coordinate. The repository supports:

- ternary LLE prediction on chemical systems excluded from training;
- NRTL-based chemical-potential regularization;
- architecture and thermodynamic-loss ablations;
- comparison with molecular graph and language-model baselines;
- temperature, data-splitting, tie-line-density, and system-generalization analyses;
- expanded-LLE adaptation and industrial extraction case studies.

The [scientific model contract](docs/architecture/scientific_model_contract.md) defines the executable architecture, thermodynamic objective, output constraints, and checkpoint conventions.

## Model architecture

The maintained `sample_major` architecture contains five principal stages:

1. a shared message-passing encoder converts each molecular graph into atom- and molecule-level representations;
2. cross-molecular functional-group attention captures interactions among the three components;
3. a three-node mixture graph combines molecular embeddings with normalized operating variables;
4. multi-scale features are fused across molecular and mixture representations;
5. separate heads predict the three-component extract and raffinate compositions.

The physics-informed training stage evaluates phase activity coefficients with an NRTL excess-Gibbs-energy model and penalizes chemical-potential mismatch between the predicted phases. The NRTL formulation supplies the thermodynamically consistent Gibbs-energy representation; the neural objective does not introduce a separate Gibbs-Duhem residual term.

Paper checkpoints use the `component_major` node layout and are evaluated through explicit checkpoint registries. Maintained training configurations use `sample_major`. Layouts must not be mixed within a metric comparison.

## Repository layout

```text
PSMI-public/
|- configs/                 Dataset, model, training, and checkpoint registries
|- datasets/                Processed workbooks, fixed splits, and NRTL parameters
|- src/psmi/                Maintained PSMI implementation
|- src/psmi_baselines/      Baseline implementations used in the comparison study
|- scripts/                 Training, evaluation, analysis, and plotting commands
|- experiments/             Experiments organized by paper section
|- results/                 Figure 2a and reference physics-objective result packages
|- Web/PSMI-LLE-web/        FastAPI backend and Vue frontend
|- tests/                   Unit, regression, interface, and release-hygiene tests
`- docs/                    Architecture, result, and usage documentation
```

## Installation

The reference environment is named `ggnn39` and uses Python 3.9.

```bash
conda env create -f environment.yml
conda activate ggnn39
```

For CPU-only evaluation and analysis:

```bash
conda env create -f environment-cpu.yml
conda activate psmi-cpu
```

An existing compatible environment can install the package and its declared dependencies with:

```bash
python -m pip install -e .
```

RDKit and PyTorch Geometric binary compatibility depends on the selected PyTorch/CUDA combination. Conda installation is recommended when building a new GPU environment.

## Quick start

List the distributed checkpoint entries:

```bash
python scripts/evaluate_checkpoint_registry.py \
  --registry configs/reproduction/published_checkpoint_registry.json \
  --list
```

Evaluate the Figure 2a checkpoint on the fixed test systems:

```bash
python scripts/evaluate_checkpoint_registry.py \
  --registry configs/reproduction/published_checkpoint_registry.json \
  --only figure2a_psmi \
  --device cpu \
  --no-plots
```

Export the canonical Figure 2a result bundle:

```bash
python scripts/analysis/export_figure_2a_results.py
```

Training entry points are configuration driven:

```bash
python scripts/train.py --config configs/experiments/main_benchmark_stage1.yaml
python scripts/train.py --config configs/experiments/main_benchmark_stage2.yaml
python scripts/train.py --config configs/experiments/expanded_lle_finetune.yaml
```

## Data contract

The main workbook contains 8,343 experimental records from 860 ternary systems. Applying the fixed minimum density of six records per `(system_id, temperature)` group produces 7,683 modeling records from 765 systems. The expanded workbook contains 7,134 raw records and 6,709 filtered records.

Splits are disjoint at the `system_id` level. Component-permutation augmentation is applied only to training examples after splitting; it does not change the reported number of experimental records. File identities, exact partition counts, preprocessing rules, field aliases, and reuse guidance are documented in the [Dataset Card](datasets/DATASET_CARD.md).

## Experiments and reference results

The [experiment index](experiments/README.md) maps each main-text and Supporting Information section to its implementation, command, checkpoint, source data, table, or figure. It includes the main benchmark, baseline comparison, ablation studies, molecular-interaction analysis, solubility transfer, industrial extraction cases, expanded-LLE adaptation, and all supplementary robustness analyses.

The [results directory](results/README.md) contains the canonical Figure 2a and two reference result packages for the data-driven and chemical-potential-regularized models. Each package includes its validation-selected checkpoint, pointwise test predictions, metrics, figures, and SHA-256 artifact manifest.

Run the test suite from the repository root:

```bash
python -m pytest -q
```

## Web application

The web application provides molecule entry, operating-condition input, checkpoint-backed inference, and ternary composition visualization. Windows launch scripts are included:

```powershell
Web\PSMI-LLE-web\scripts\run_backend.ps1
Web\PSMI-LLE-web\scripts\run_frontend.ps1
```

After startup, open `http://localhost:3000`; the FastAPI schema is available at `http://localhost:8000/docs`. See the [Web deployment guide](Web/PSMI-LLE-web/README.md) for Windows, Linux, and macOS setup, GPU configuration, environment variables, API verification, and troubleshooting.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Please cite the PSMI article and the original experimental sources when reusing the datasets. Baseline-method references and dependency notices are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Original PSMI source code is released under the [MIT License](LICENSE). Dataset measurements and third-party components retain their respective attribution and reuse requirements. Contributions are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md).
