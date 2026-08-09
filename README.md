# PSMI: Physics-Informed Prediction of Ternary Liquid-Liquid Equilibria

PSMI is a graph-neural-network framework for predicting ternary liquid-liquid equilibrium (LLE) tie lines. The repository provides curated datasets, fixed system-level splits, model implementations, thermodynamic regularization, baseline comparisons, trained checkpoints, paper-aligned experiments, result tables, figures, and a web application.

## Repository layout

```text
PSMI/
|- configs/                 Data, model, training, and reproduction configurations
|- datasets/                Processed datasets, split manifests, and thermodynamic parameters
|- src/psmi/                Maintained PSMI implementation
|- src/psmi_baselines/      Baseline model implementations
|- scripts/                 Training, evaluation, analysis, and visualization entry points
|- experiments/             Paper-aligned experiment index and reference artifacts
|- models/                  Published checkpoints and transfer-learning models
|- results/                 Paper tables, pointwise predictions, figures, and audit manifests
|- Web/PSMI-LLE-web/        FastAPI and Vue web application
|- tests/                   Unit, regression, and interface tests
`- docs/                    Architecture, results, and usage documentation
```

## Environment

```powershell
conda env create -f environment.yml
conda activate ggnn39
```

An existing Python environment can also use `python -m pip install -r requirements.txt`.

## Checkpoint-based reproduction

The following commands evaluate published checkpoints and organize the outputs into paper-aligned tables and figures:

```powershell
python scripts/reproduce_current_weights.py --device cuda
python scripts/reproduce_current_weights.py `
  --registry configs/reproduction/historical_paper_weight_registry.json `
  --output-root results/paper_reproduction/historical_weight_inference `
  --device cuda
python scripts/analysis/build_paper_reproduction_bundle.py
```

The current corrected protocol and the paper's historical protocol are represented by separate registries because they use different mixture-node layouts. See `results/paper_reproduction/README.md` for numerical alignment and provenance details.

## Training

```powershell
python scripts/train.py --config configs/experiments/main_benchmark_stage1.yaml
python scripts/train.py --config configs/experiments/main_benchmark_stage2.yaml
python scripts/train.py --config configs/experiments/expanded_lle_finetune.yaml
```

## Experiments

The `experiments/` index maps every main-text and Supporting Information experiment to its implementation, command-line entry point, data products, checkpoints, tables, and figures.

## Web application

```powershell
Web/PSMI-LLE-web/scripts/run_backend.ps1
Web/PSMI-LLE-web/scripts/run_frontend.ps1
```

Open `http://localhost:3000` for the interface and `http://localhost:8000/docs` for the API schema.

See `Web/PSMI-LLE-web/README.md` for the complete local deployment guide, environment variables, API checks, GPU configuration, and troubleshooting.

## Scientific scope

The main physics-regularized configuration uses an NRTL excess-Gibbs-energy model to evaluate phase-wise activity coefficients and penalize chemical-potential mismatch. The current configuration does not add a separate Gibbs-Duhem residual term. The distinction between thermodynamic model consistency and an explicit neural-network loss term is documented in `docs/architecture/scientific_model_contract.md`.

## Testing

```powershell
$env:PYTHONPATH='src'
python -m pytest -q
```
