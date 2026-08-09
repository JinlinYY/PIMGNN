# Main-Text Result Mapping

This document follows the final main-text numbering. Reported values are kept
separate from auxiliary or newly recomputed metrics.

## Section 3.1: LLE prediction

### Figure 2a: extract and raffinate parity plots

Status: **Complete**

The canonical image and its section-aligned copy are byte-identical:

- [`results/figure_2a.png`](../../results/figure_2a.png)
- [`figure_2a_parity.png`](../../experiments/section_3_results/3_1_lle_prediction/main_benchmark/figures/figure_2a_parity.png)

The values displayed in the manuscript panel are:

| Phase | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| Extract | 0.0371 | 0.0566 | 0.9671 |
| Raffinate | 0.0318 | 0.0545 | 0.9784 |

Source evidence:

- [`figure_2a_predictions.csv`](../../experiments/section_3_results/3_1_lle_prediction/main_benchmark/data/figure_2a_predictions.csv)
- [`best_model.pt`](../../experiments/section_3_results/3_1_lle_prediction/main_benchmark/models/figure_2a_psmi/best_model.pt)
- [`figure_2a_fg_corpus.json`](../../experiments/section_3_results/3_1_lle_prediction/main_benchmark/artifacts/figure_2a_fg_corpus.json)
- registry identifier `figure2a_psmi` in [`published_checkpoint_registry.json`](../../configs/reproduction/published_checkpoint_registry.json)

This is a single validation-selected checkpoint. It is not the five-seed PSMI
aggregate reported in Table 1.

### Table 1: comparison with baseline methods

Status: **Partial**

The manuscript reports mean values with sample standard deviations in
parentheses over five independent runs. The PSMI row is:

| Phase | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| Extract | 0.0355 (0.0006) | 0.0577 (0.0010) | 0.9654 (0.0014) |
| Raffinate | 0.0356 (0.0025) | 0.0801 (0.0146) | 0.9516 (0.0153) |
| Overall | 0.0356 (0.0011) | 0.0700 (0.0084) | 0.9585 (0.0083) |

Exact five-seed summaries for the common fingerprint, sequence, tabular, and
GNN baselines are distributed in:

- [`multiple_seeds_summary_formatted.csv`](../../experiments/section_3_results/3_1_lle_prediction/3_1_1_baseline_comparison/results/reference_five_seed/fingerprint_and_sequence_models/multiple_seeds_summary_formatted.csv)
- [`gnn_extension/multiple_seeds_summary.csv`](../../experiments/section_3_results/3_1_lle_prediction/3_1_1_baseline_comparison/results/reference_five_seed/gnn_extension/multiple_seeds_summary.csv)
- [`all_models_overall.csv`](../../experiments/section_3_results/3_1_lle_prediction/3_1_1_baseline_comparison/results/reference_five_seed/all_models_overall.csv)

The archived files reproduce the manuscript rows for MLP, LSTM, XGBoost,
Transformer, TabNet, KAN/TabKNet, SMILES-RNN, ANN, random forest, and the common
GNN/UALF-GNN interface. Implementations for MMGNN, CIGIN, SolvBERT, CGIB, and
GLAM are present, but their complete five-seed manuscript summary and the PSMI
five-seed aggregate are not distributed as one machine-readable result table.
Do not reconstruct missing per-seed identities from folder names.

The controlled comparison protocol is documented in
[Baseline Comparison Protocol](../guides/baseline_comparison.md).

### Table 2: architecture ablation

Status: **Partial**

The table contains twelve manuscript rows. Eight rows have an exact matching
`best_metrics.json` in the public architecture-ablation tree:

| Manuscript row | Overall MAE/RMSE/R2 | Public metric record |
| --- | --- | --- |
| No graph, no FG, permutation, concatenation | 0.0463 / 0.0900 / 0.9309 | Exact record not identified |
| Single graph, no FG, permutation, concatenation | 0.0405 / 0.0722 / 0.9556 | Exact record not identified |
| Mixture graph, no FG, permutation, concatenation | 0.0365 / 0.0658 / 0.9631 | Exact record not identified |
| Mixture, top-k, permutation, concatenation | 0.0350 / 0.0608 / 0.9685 | [`concatenation_fusion`](../../experiments/section_3_results/3_1_lle_prediction/3_1_2_ablation_analysis/architecture_ablation/results/multiscale_fusion/concatenation_fusion/metrics/best_metrics.json) |
| Mixture, top-k, permutation, FiLM | 0.0374 / 0.0674 / 0.9613 | Exact record not identified |
| Mixture, top-k, permutation, Transformer | 0.0357 / 0.0622 / 0.9670 | [`topk_permutation_transformer`](../../experiments/section_3_results/3_1_lle_prediction/3_1_2_ablation_analysis/architecture_ablation/results/architecture_variants/topk_permutation_transformer/metrics/best_metrics.json) |
| Mixture, cross-attention, permutation, concatenation | 0.0329 / 0.0561 / 0.9732 | [`cross_attention_permutation_concat`](../../experiments/section_3_results/3_1_lle_prediction/3_1_2_ablation_analysis/architecture_ablation/results/architecture_variants/cross_attention_permutation_concat/metrics/best_metrics.json) |
| Mixture, cross-attention, permutation, Hadamard | 0.0317 / 0.0552 / 0.9740 | [`cross_attention_permutation_hadamard`](../../experiments/section_3_results/3_1_lle_prediction/3_1_2_ablation_analysis/architecture_ablation/results/architecture_variants/cross_attention_permutation_hadamard/metrics/best_metrics.json) |
| Mixture, cross-attention, permutation, bilinear | 0.0319 / 0.0552 / 0.9741 | [`cross_attention_permutation_bilinear`](../../experiments/section_3_results/3_1_lle_prediction/3_1_2_ablation_analysis/architecture_ablation/results/architecture_variants/cross_attention_permutation_bilinear/metrics/best_metrics.json) |
| Mixture, cross-attention, permutation, Transformer | 0.0335 / 0.0564 / 0.9729 | [`cross_permutation_transformer`](../../experiments/section_3_results/3_1_lle_prediction/3_1_2_ablation_analysis/architecture_ablation/results/architecture_variants/cross_permutation_transformer/metrics/best_metrics.json) |
| Mixture, cross-attention, S3, S3-Set | 0.0335 / 0.0586 / 0.9707 | [`lle_run_s3_set`](../../experiments/section_3_results/3_1_lle_prediction/3_1_2_ablation_analysis/architecture_ablation/results/architecture_variants/lle_run_s3_set/metrics/best_metrics.json) |
| Mixture, cross-attention, S3, Transformer | 0.0330 / 0.0550 / 0.9742 | [`cross_s3_transformer`](../../experiments/section_3_results/3_1_lle_prediction/3_1_2_ablation_analysis/architecture_ablation/results/architecture_variants/cross_s3_transformer/metrics/best_metrics.json) |

Other metric folders in the same directory are additional variants and must
not be assigned to a manuscript row solely because their scores are similar.

### Table 3: chemical-potential regularization

Status: **Confirmation required for the data-driven row; complete for the
physics-informed row**

The manuscript table reports:

| Model | Overall MAE | Overall RMSE | Overall R2 | mu-MAE | mu-RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Data-driven | 0.0330 | 0.0550 | 0.9742 | 1.771093 | 2.612110 |
| Physics-informed | 0.0349 | 0.0544 | 0.9748 | 0.541074 | 0.788265 |

For the physics-informed row, the rounded composition and chemical-potential
values are jointly present in
[`results/chemical_potential_regularized/metrics/best_metrics.json`](../../results/chemical_potential_regularized/metrics/best_metrics.json).

For the data-driven row, the composition values are present in the Table 2
S3/Transformer metric record at epoch 125, whereas the exact chemical-potential
values are present in
[`results/data_driven/metrics/best_metrics.json`](../../results/data_driven/metrics/best_metrics.json),
whose stored composition metrics are 0.033925/0.055169/0.974053. The registered
Table 3 checkpoint and reference prediction table are:

- [`physics_regularization/models/data_driven/best_model.pt`](../../experiments/section_3_results/3_1_lle_prediction/3_1_2_ablation_analysis/physics_regularization/models/data_driven/best_model.pt)
- [`data_driven_predictions.csv`](../../experiments/section_3_results/3_1_lle_prediction/3_1_2_ablation_analysis/physics_regularization/data/data_driven_predictions.csv)

No single distributed data-driven metric JSON contains all five manuscript
Table 3 values. This provenance boundary must be confirmed before claiming
that one archived package is the complete numerical source of that row.

## Section 3.2: molecular interaction mechanisms

### Figure 2b

Status: **Complete**

- manuscript asset: [`figure_2b_mixture_edge_importance.png`](../../experiments/section_3_results/3_2_molecular_interaction_mechanisms/figures/figure_2b_mixture_edge_importance.png)
- source attribution tables: [`results/global_saliency/target_ALL`](../../experiments/section_3_results/3_2_molecular_interaction_mechanisms/results/global_saliency/target_ALL)

### Figure 2c

Status: **Complete**

- manuscript asset: [`figure_2c_feature_rank_heatmap.png`](../../experiments/section_3_results/3_2_molecular_interaction_mechanisms/figures/figure_2c_feature_rank_heatmap.png)
- atom, bond, and global feature tables: [`data/global_saliency`](../../experiments/section_3_results/3_2_molecular_interaction_mechanisms/data/global_saliency)

### Figure 2d

Status: **Complete**

The representative phase reconstructions for Systems 826 and 22 are stored
with their source prediction tables under
[`main_benchmark`](../../experiments/section_3_results/3_1_lle_prediction/main_benchmark).

### Figure 2e

Status: **Partial**

System-22 saliency summaries, functional-group importance, mixture-node and
mixture-edge plots, and the phase diagram are available under
[`3_2_molecular_interaction_mechanisms`](../../experiments/section_3_results/3_2_molecular_interaction_mechanisms).
The exact final atom-, bond-, and functional-group composite shown as Figure 2e
is not distributed as a separately identifiable image.

## Section 3.3: binary solubility validation

### Figure 2f and Supplementary Table S2

Status: **Partial**

The public release contains the base ternary, CompSol, pretrained BigSolDB, and
binary-fine-tuned checkpoints under
[`3_3_binary_solubility_validation/models`](../../experiments/section_3_results/3_3_binary_solubility_validation/models)
and the transfer-learning implementations under
[`scripts/experiments/transfer_learning/public_release`](../../scripts/experiments/transfer_learning/public_release).

The exact Figure 2f image and a machine-readable table containing the final
FreeSolv, CompSol, Abraham, and CombiSolv-Exp values are not currently
distributed. The weights and code alone should not be described as a complete
reproduction package for Figure 2f or Table S2.

## Section 3.4: industrial extraction design

### Figure 3

Status: **Partial for the complete four-panel figure; complete for panels 3c-3d**

The final equilibrium comparison for the sulfolane and diethoxymethane cases is
distributed in both PNG and PDF form:

- [`figure3cd_industrial_extraction_validation.png`](../../experiments/section_3_results/3_4_industrial_extraction_design/3_4_1_sulfolane_aromatic_extraction/figures/figure3cd_industrial_extraction_validation.png)
- [`figure3cd_industrial_extraction_validation.pdf`](../../experiments/section_3_results/3_4_industrial_extraction_design/3_4_1_sulfolane_aromatic_extraction/figures/figure3cd_industrial_extraction_validation.pdf)

The two standardized source tables are:

- [`aromatic_extraction_lle_data.csv`](../../experiments/section_3_results/3_4_industrial_extraction_design/3_4_1_sulfolane_aromatic_extraction/data/aromatic_extraction_lle_data.csv)
- [`dem_recovery_lle_data.csv`](../../experiments/section_3_results/3_4_industrial_extraction_design/3_4_2_diethoxymethane_recovery/data/dem_recovery_lle_data.csv)

The process schematics used as panels 3a and 3b are not included as separate
public assets, so the repository reproduces the scientific equilibrium panels
3c and 3d rather than the assembled four-panel manuscript figure.
