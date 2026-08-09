# Data Pipeline and Split Policy

## Overview

The data pipeline converts experimental ternary liquid-liquid-equilibrium rows
into graph-model samples while preserving a strict separation between raw
measurements, filtered records, training augmentation, and model partitions.

```text
Excel workbook
  -> column alias resolution
  -> numeric conversion and SMILES canonicalization
  -> phase-composition normalization
  -> minimum-density filtering per (system_id, T)
  -> phase-path coordinate assignment
  -> fixed system-level split
  -> training-only component permutation
  -> training-partition feature scaling
  -> graph and functional-group caches
```

## Distributed datasets

| Dataset | Purpose | Workbook | Scalar dimension |
| --- | --- | --- | ---: |
| Main benchmark | Fixed ternary LLE evaluation | `datasets/processed/update-LLE-all-with-smiles.xlsx` | 2 |
| Expanded LLE | Pressure-aware adaptation | `datasets/processed/LLE-literature-data-boosted.xlsx` | 3 |

Frozen identities and exact counts are recorded in the
[dataset card](../../datasets/DATASET_CARD.md).

## Required scientific fields

The loader accepts several historical column aliases but normalizes them to:

| Canonical field | Meaning | Unit or domain |
| --- | --- | --- |
| `system_id` | Ternary chemical-system identifier | Integer-like identifier |
| `smiles1` | Component 1 structure | Canonicalizable SMILES |
| `smiles2` | Component 2 structure | Canonicalizable SMILES |
| `smiles3` | Component 3 structure | Canonicalizable SMILES |
| `T` | Temperature | K in distributed data |
| `P` | Pressure | kPa in expanded-data profile |
| `Ex1`-`Ex3` | Extract composition | Mole fractions |
| `Rx1`-`Rx3` | Raffinate composition | Mole fractions |

If pressure is absent, the generic loader inserts `101.325`. This fallback does
not make a two-scalar checkpoint pressure-aware; scalar dimensionality remains
an explicit model-profile choice.

## SMILES processing

Each SMILES is parsed and canonicalized with RDKit. Rows with an invalid or
empty component SMILES are removed. Canonicalization reduces superficial string
variation but does not establish stereochemical or tautomeric equivalence
beyond RDKit's canonical representation.

For a new dataset, audit the number of records removed by SMILES parsing before
training. A silent change in molecule validity changes the system distribution
and may invalidate a frozen split manifest.

## Composition normalization

The extract and raffinate triplets are processed independently:

1. negative entries are clipped to zero;
2. each triplet is divided by its sum;
3. an all-zero triplet falls back to `[1/3, 1/3, 1/3]`.

The distributed data should already represent valid compositions. The
normalization step is a numerical safeguard and should not replace a source-data
quality audit.

## Minimum tie-line density

Filtering is performed per `(system_id, T)` group, not per chemical system
alone. A group is retained when it contains at least six records:

```text
n_records(system_id, T) >= 6
```

The public setting is `MIN_POINTS_PER_GROUP: 6`. Groups below the threshold are
removed before phase-path assignment and before the system split. The threshold
supports a stable within-group ordering and enough points to represent a phase
path, but it is still a modeling choice. Sensitivity results for alternative
thresholds are archived under SI Section S3.5.

## Phase-path construction

For each retained `(system_id, T)` group, the six composition coordinates are
centered. Singular-value decomposition supplies the first principal direction,
and rows are ordered by their projection. Ordered records receive evenly spaced
`t` values from 0 to 1.

Consequences:

- `t` is group-relative rather than globally physical;
- changing group membership can change assigned `t` values;
- filtering must occur before `t` assignment;
- an isolated point would receive `t = 0.5`, although the public density rule
  prevents such groups from entering the benchmark.

## Fixed system-level manifests

The benchmark uses JSON manifests under `datasets/splits/`. The split loader
checks that:

- train, validation, and test system identifiers are mutually disjoint;
- every filtered system occurs in exactly one partition;
- the manifest does not contain identifiers absent from the filtered dataset.

This prevents records from the same chemical system appearing in both training
and evaluation partitions.

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Main benchmark | 612 systems / 6,092 rows | 75 / 788 | 78 / 803 |
| Expanded LLE | 575 systems / 5,370 rows | 72 / 707 | 72 / 632 |

All counts are unaugmented.

## Component-permutation augmentation

After splitting, training rows are duplicated with components 2 and 3 swapped:

```text
(smiles2, Ex2, Rx2) <-> (smiles3, Ex3, Rx3)
```

The model input, target, and system-specific thermodynamic interaction matrix
are permuted together. Validation and test rows are never augmented. The
`aug_swap23` flag records orientation so NRTL parameter lookup can apply the
same permutation.

Augmentation changes the number of optimization samples, not the number of
experimental records or systems.

## Feature scalers

Temperature and optional pressure scalers are fitted from the unaugmented
training partition. The same scalers are applied to validation and test data.
Maintained checkpoints store scaler statistics as provenance. Published
checkpoints may require a named compatibility fallback declared in the
checkpoint registry.

## NRTL parameter stores

The main benchmark distributes:

```text
datasets/parameters/main_benchmark/
|- nrtl_params_train.json
|- nrtl_params_all.json
`- nrtl_split_manifest.json
```

`nrtl_params_train.json` is the only store permitted in the physics training
loss. `nrtl_params_all.json` supports post-selection diagnostics on all
partitions. The split record documents how the parameter-fitting scope was
constructed.

To fit parameters for a new dataset, inspect the complete interface first:

```bash
python scripts/fit_nrtl.py --help
```

Parameter fitting is an optimization workflow and is distinct from evaluating
the neural checkpoint.

## Rebuilding or auditing split manifests

The canonical main-benchmark builder locks the reference test systems to a
prediction table and requires explicit output:

```bash
python scripts/data_preparation/build_canonical_split.py --help
```

The expanded-data builder creates the documented 575/72/72-system partition:

```bash
python scripts/data_preparation/build_expanded_split.py --help
```

Do not overwrite the distributed manifests when exploring a new split. Write to
a new file and use a separate experiment configuration.

## Rebuilding distribution summaries

```bash
python scripts/analysis/analyze_dataset_distribution.py \
  --results-dir outputs/dataset_audit/tables \
  --figures-dir outputs/dataset_audit/figures \
  --min-points-per-group 6
```

The analysis intentionally reports raw rows, filtered rows, and augmentation
separately. Preserve that distinction in publications and derivative datasets.

## Leakage-prevention checklist

Before reporting a result on a modified dataset:

1. confirm system-level disjointness;
2. fit scalers on training systems only;
3. apply augmentation only after splitting and only to training rows;
4. keep training and all-system thermodynamic parameter stores separate;
5. select checkpoints on validation metrics only;
6. evaluate the test partition only after selection;
7. record workbook, manifest, and checkpoint hashes.
