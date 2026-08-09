# Contributing to PSMI

Contributions that improve reproducibility, thermodynamic validation, model
interfaces, documentation, or platform support are welcome.

## Development setup

Create the project environment and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate ggnn39
python -m pip install -e .
```

Use a CPU-compatible PyTorch installation when CUDA 12.6 is unavailable. The
remaining dependencies can be installed from `requirements.txt`.

## Change requirements

- Keep public code, paths, command output, and documentation in English.
- Preserve system-level train/validation/test isolation.
- Do not change frozen dataset or checkpoint files without updating their
  SHA-256 identities and every dependent manifest.
- Store experiment outputs in the directory matching the relevant paper or SI
  section.
- Add focused tests for changes to data contracts, checkpoint loading,
  thermodynamic losses, metrics, or command-line interfaces.
- Do not add manuscripts, review correspondence, private paths, credentials,
  or local run logs.

## Verification

Run the repository tests before submitting a change:

```bash
python -m compileall -q src scripts tests
python -m pytest -q
```

For Web changes, also run the backend tests and build the frontend as described
in `Web/PSMI-LLE-web/README.md`.
