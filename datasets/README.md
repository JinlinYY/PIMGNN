# Datasets

The public data package contains the primary ternary LLE dataset, the expanded LLE dataset, fixed system-level split manifests, baseline-comparison partitions, and NRTL parameter files.

`update-LLE-all-with-smiles.xlsx` is the primary benchmark dataset. `LLE-literature-data-boosted.xlsx` is the expanded dataset used for transfer adaptation. Component names, SMILES strings, temperature, pressure, extract-phase compositions, and raffinate-phase compositions are stored in explicit columns.

The main benchmark uses a fixed 612/75/78 train/validation/test system split. Component-2/component-3 permutation augmentation is restricted to the training partition.
