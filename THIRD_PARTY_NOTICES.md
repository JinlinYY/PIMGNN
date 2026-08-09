# Third-Party Methods, Data, and Dependencies

The root `LICENSE` covers original PSMI software in this repository. Scientific
method names, external datasets, and third-party dependencies remain subject to
their respective attribution and licensing terms.

## Baseline methods

The modules under `src/psmi_baselines/` are benchmark-oriented adaptations to
the common PSMI data and metric protocol. They are not presented as verbatim
copies of upstream repositories. Researchers using these baselines should cite
the corresponding method papers:

- CIGIN: *Learning Atomic Interactions through Solvation Free Energy Prediction
  Using Graph Neural Networks*, DOI
  [10.1021/acs.jcim.0c01413](https://doi.org/10.1021/acs.jcim.0c01413).
- CGIB: *Conditional Graph Information Bottleneck for Molecular Relational
  Learning*, Proceedings of Machine Learning Research 202 (2023),
  [paper](https://proceedings.mlr.press/v202/lee23e.html).
- MMGNN: *MMGNN: A Molecular Merged Graph Neural Network for Explainable
  Solvation Free Energy Prediction*, DOI
  [10.24963/ijcai.2024/642](https://doi.org/10.24963/ijcai.2024/642).
- SolvBERT: *SolvBERT for Solvation Free Energy and Solubility Prediction*, DOI
  [10.1039/D2DD00107A](https://doi.org/10.1039/D2DD00107A).

`GLAM` denotes the local graph-learning ensemble baseline defined in this
repository. Classical regression baselines use scikit-learn estimators. These
labels should not be interpreted as redistribution of an identically named
external software package.

## External data

The curated LLE measurements and binary-solubility resources originate from
scientific data sources. The MIT software license does not grant additional
rights over those measurements. See `datasets/DATASET_CARD.md` and cite the
associated article and underlying data publications when reusing data.

## Runtime libraries

PyTorch, RDKit, NumPy, pandas, SciPy, scikit-learn, Matplotlib, FastAPI, Vue,
and other dependencies are distributed under their own licenses. Dependency
versions are recorded in `requirements.txt`, `environment.yml`, and the Web
application package files.
