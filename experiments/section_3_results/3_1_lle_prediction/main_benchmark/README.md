# PSMI Main Benchmark

## Paper mapping

- Main text Section 3.1
- Figures 2a and 2d
- Table 1

## Scientific scope

The fixed-split benchmark, multi-seed summaries, Figure 2 source predictions, figures, and the corresponding checkpoint are available.

## Code entry points

- `scripts/train.py`
- `src/psmi/predict.py`
- `scripts/experiments/run_multiseed_benchmark.py`
- `scripts/visualization/build_paper_figure2_assets.py`

## Representative commands

```bash
python scripts/evaluate_checkpoint_registry.py --registry configs/reproduction/published_checkpoint_registry.json --device cuda
python scripts/visualization/build_paper_figure2_assets.py
```
