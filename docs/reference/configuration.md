# Configuration Reference

## Configuration layers

PSMI uses Python defaults plus layered YAML profiles:

```text
src/psmi/config.py                    Runtime defaults
configs/data/                         Dataset and split profiles
configs/model/                        Architecture profiles
configs/training/                     Optimization-stage profiles
configs/experiments/                  Composed experiment profiles
command line --set KEY=VALUE          Final runtime overrides
```

An experiment YAML usually includes one file from each lower-level group. For
example, `main_benchmark_stage1.yaml` includes the primary dataset, maintained
model, and supervised-training profiles.

## Include semantics

```yaml
include:
  - ../data/primary_lle.yaml
  - ../model/psmi_sample_major.yaml
  - ../training/stage1_supervised.yaml

SEED: 42
OUT_DIR: results/main_benchmark/multiseed_reference/seed42/stage1_supervised
```

Included files are resolved relative to the YAML file that contains the
`include`. Values are merged in order; later includes and then keys in the
current file override earlier values. Cyclic includes are rejected.

All runtime keys must be uppercase and must already exist in
`src/psmi/config.py`. Unknown or lowercase keys raise an error instead of being
silently ignored.

## Path resolution

The loader treats the following keys as project paths:

- `EXCEL_PATH`;
- `FINE_TUNE_EXCEL_PATH`;
- `LOAD_CKPT_PATH`;
- `PRETRAINED_MODEL_PATH`;
- `NRTL_TRAIN_PARAMS_PATH`;
- `NRTL_EVAL_PARAMS_PATH`;
- `SPLIT_MANIFEST_PATH`;
- `OUT_DIR`.

Relative values are resolved against the repository root, not the current YAML
directory. This makes commands independent of the shell's path once they are
started from the repository checkout.

## Command-line overrides

`--set` accepts `KEY=VALUE`. The value is parsed as YAML, so numbers, booleans,
strings, and lists retain their types:

```bash
python scripts/train.py \
  --config configs/experiments/main_benchmark_stage1.yaml \
  --set SEED=43 \
  --set LR=0.00003 \
  --set USE_AMP=false \
  --set OUT_DIR=outputs/main_seed43
```

Repeat `--config` or `--set` when multiple values are needed. Later values take
precedence. Misspelled keys fail validation.

## Dataset settings

| Key | Meaning | Public main setting |
| --- | --- | --- |
| `EXCEL_PATH` | Input workbook | Main processed workbook |
| `MIN_POINTS_PER_GROUP` | Minimum rows per `(system_id, T)` | 6 |
| `PERMUTE_23_AUG` | Training component-2/3 augmentation | `true` |
| `SPLIT_STRATEGY` | `manifest`, `random`, or `stratified` | `manifest` |
| `SPLIT_MANIFEST_PATH` | Fixed system partition | Main manifest |
| `SCALAR_DIM` | Number of condition scalars | 2 main; 3 expanded |
| `TRAIN_RATIO` | Random/stratified train fraction | Profile-dependent |
| `VAL_RATIO` | Random/stratified validation fraction | Profile-dependent |

For published comparisons, retain `manifest` splitting. Random or stratified
splits define new protocols even if they use the same seed.

## Model settings

| Key | Meaning | Public maintained setting |
| --- | --- | --- |
| `USE_GRAPH` | Use molecular graph encoder | `true` |
| `USE_MIX_GRAPH` | Use three-node mixture graph | `true` |
| `USE_FG` | Use functional-group features | `true` |
| `FG_TOKEN_MODE` | Use functional-group token ids | `true` |
| `FG_CROSS_ATTN` | Cross-component token attention | `true` |
| `MIXTURE_NODE_LAYOUT` | Batch node ordering | `sample_major` |
| `USE_S3_COMPONENT_EMBEDDING` | S3-aware component representation | `true` |
| `FUSION_MODE` | Multi-scale feature fusion | `concat` |

The published profile uses `component_major` and `S3_EQUIVARIANT`. Select it
only through a published checkpoint registry entry or a deliberate compatibility
experiment.

## Training settings

| Key | Meaning |
| --- | --- |
| `EPOCHS` | Maximum training epochs |
| `LR` | Initial learning rate |
| `WEIGHT_DECAY` | Optimizer weight decay |
| `BATCH_SIZE_GRAPH` | Graph batch size |
| `USE_AMP` | Automatic mixed precision |
| `GRAD_CLIP` | Gradient-norm clipping threshold |
| `USE_EARLY_STOP` | Enable validation-based stopping |
| `EARLY_STOP_METRIC` | Stage-1 selection metric |
| `EARLY_STOP_PATIENCE` | Stage-1 patience |
| `FREEZE_BACKBONE` | Freeze non-output model layers |
| `EVALUATE_TEST_DURING_TRAINING` | Test-set access during optimization |
| `GENERATE_PHASE_DIAGRAMS` | Render expensive system plots during a run |

The public profiles set `EVALUATE_TEST_DURING_TRAINING: false`. Preserve this
for unbiased test evaluation.

## Physics settings

| Key | Meaning | Public stage-2 value |
| --- | --- | ---: |
| `USE_MECH_LOSS` | Enable thermodynamic penalty | `true` |
| `GE_MODEL` | Excess-Gibbs-energy model | `nrtl` |
| `LAMBDA_PHY` | Target physics multiplier | 0.001 |
| `MECH_W_EQ` | Chemical-potential equilibrium weight | 1.0 |
| `MECH_W_GD` | Gibbs-Duhem auxiliary-loss weight | 0.0 |
| `MECH_W_STAB` | TPD auxiliary-loss weight | 0.0 |
| `ROBUST_DELTA` | Huber residual transition | 5.0 |
| `WARMUP_EPOCHS` | Epochs before physics weight begins | 0 by default |
| `RAMP_EPOCHS` | Linear ramp duration | 5 by default |

The Gibbs-Duhem and TPD metrics can still be computed when their training
weights are zero. Do not infer the objective from exported diagnostic columns.

## Public experiment profiles

### Supervised main benchmark

```bash
python scripts/train.py --config configs/experiments/main_benchmark_stage1.yaml
```

Uses the fixed main split, sample-major architecture, 300-epoch maximum,
validation early stopping, and no mechanistic loss.

### Physics-informed stage

```bash
python scripts/train.py --config configs/experiments/main_benchmark_stage2.yaml
```

Loads the stage-1 path declared by the profile, freezes the backbone, reduces
the learning rate, and enables the chemical-potential residual.

### Expanded-LLE adaptation

```bash
python scripts/train.py --config configs/experiments/expanded_lle_finetune.yaml
```

Uses a three-scalar input, lower learning rate, full-network updates, and
supervised validation selection.

## Creating a new profile

1. Copy the closest experiment YAML to a new filename.
2. Keep shared data, model, and training settings in included files.
3. Change `OUT_DIR` to a new, descriptive path.
4. Declare a new split manifest for a new data protocol.
5. Set `LOAD_CKPT_PATH` only to a layout-compatible checkpoint.
6. Record the seed and every CLI override.
7. Add a README beside the experiment results.

Do not edit the frozen manuscript registry to point at exploratory weights.
Create a separate registry for a new release artifact.

## Diagnosing the effective configuration

If a run behaves unexpectedly, inspect values in this order:

1. `src/psmi/config.py` defaults;
2. each included YAML in order;
3. keys in the top-level experiment YAML;
4. command-line `--set` assignments;
5. checkpoint metadata and compatibility adaptations printed at load time.

Path errors usually indicate that a renamed artifact was not updated in a YAML
profile. Architecture errors usually indicate a checkpoint/layout mismatch.
