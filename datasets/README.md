# Datasets

The public data package contains the ternary LLE benchmark, the expanded
literature dataset, fixed system-level partitions, baseline-model input tables,
and thermodynamic parameter files.

- `processed/update-LLE-all-with-smiles.xlsx`: main ternary LLE benchmark.
- `processed/LLE-literature-data-boosted.xlsx`: expanded LLE dataset used for
  adaptation and transfer evaluation.
- `splits/main_benchmark_system_split.json`: fixed 612/75/78-system benchmark
  partition.
- `splits/expanded_lle_system_split.json`: fixed 575/72/72-system expanded-data
  partition.
- `parameters/main_benchmark/`: NRTL parameters and the parameter-fitting split
  record.

See [DATASET_CARD.md](DATASET_CARD.md) for the schema, counting convention,
filtering rule, hashes, partition sizes, and data-reuse notes.
