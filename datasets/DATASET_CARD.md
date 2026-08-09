# PSMI Dataset Card

## Scope

The repository distributes two ternary liquid-liquid-equilibrium (LLE)
workbooks. Each row represents one experimental tie-line record: a paired
extract-phase and raffinate-phase composition for a ternary system at a stated
temperature and pressure.

## Record schema

The data loaders resolve the following scientific fields from explicit column
aliases:

- `system_id`: ternary chemical-system identifier;
- component names and `smiles1`, `smiles2`, `smiles3`;
- `T` in kelvin and `P` in kilopascals;
- extract-phase mole fractions `Ex1`, `Ex2`, `Ex3`;
- raffinate-phase mole fractions `Rx1`, `Rx2`, `Rx3`.

Preprocessing canonicalizes SMILES, normalizes each three-component phase
composition, and assigns a continuous phase-path coordinate within each
`(system_id, T)` group.

## Counting convention

Dataset size is reported at three distinct stages:

| Dataset | Raw workbook records | Raw systems | Raw `(system_id, T)` groups | Filtered records | Filtered systems | Filtered groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main benchmark | 8,343 | 860 | 872 | 7,683 | 765 | 766 |
| Expanded LLE | 7,134 | 830 | 883 | 6,709 | 719 | 764 |

The filtered stage retains only `(system_id, T)` groups containing at least six
tie-line records. This threshold is encoded in both split manifests. Counts in
the table exclude component-permutation augmentation.

The component-2/component-3 swap is applied only after the system-level split
and only to the training partition. It doubles eligible training examples but
does not create new experimental measurements and is never included in the
reported dataset size.

## Fixed partitions

| Dataset | Train systems / records | Validation systems / records | Test systems / records |
| --- | ---: | ---: | ---: |
| Main benchmark | 612 / 6,092 | 75 / 788 | 78 / 803 |
| Expanded LLE | 575 / 5,370 | 72 / 707 | 72 / 632 |

All partitions are disjoint at the `system_id` level. Checkpoint selection uses
validation systems only; the test partition is evaluated after selection.

## Frozen file identities

| File | SHA-256 |
| --- | --- |
| `processed/update-LLE-all-with-smiles.xlsx` | `76812d7a9ba8c6e6660e1acbf15817ff0725664d7fcf69b0fb0a84e133529f06` |
| `processed/LLE-literature-data-boosted.xlsx` | `24b54761e6509c2a4a28a3cb026c191f5f058127f908a90ab63a8751a4fce9da` |
| `splits/main_benchmark_system_split.json` | `83b354d7aa91c0f94044f461d489d9a39ecf3729bcdfabc1c6e77899a18ede9a` |
| `splits/expanded_lle_system_split.json` | `023bf37f005d5ddcd414802fa791c6e9186f40e3d7c30c9addfa2fb6dd579f2c` |

Recompute the coverage tables and figures with:

```bash
python scripts/analysis/analyze_dataset_distribution.py
```

The generated report is stored with SI Section S5 under
`experiments/supporting_information/s5_dataset_construction_and_distribution/`.

## Data reuse

The software license at the repository root covers original PSMI source code.
It does not replace the attribution or reuse requirements of the publications
from which experimental measurements were collected. Users should cite the
associated PSMI article and the underlying experimental sources when reusing
the curated measurements.
