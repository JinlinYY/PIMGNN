# Artifact Status and Result Discrepancies

This document records boundaries discovered by matching the final manuscript
against the public files. It prevents a partial artifact from being described
as a complete manuscript reproduction.

## Items requiring provenance confirmation

### Table 3 data-driven row

The manuscript combines composition metrics of
`0.0330/0.0550/0.9742` with chemical-potential residuals of
`1.771093/2.612110`.

- The composition metrics match the epoch-125 S3/Transformer metric record in
  the architecture ablation.
- The residual values match the top-level data-driven package, whose stored
  composition metrics are `0.033925/0.055169/0.974053`.
- The published registry points to a separate Table 3 data-driven checkpoint
  and reference prediction table.

No single public JSON contains the complete manuscript row. The public release
therefore preserves each artifact and does not overwrite one set of metrics
with another. The original run provenance should be confirmed before a future
release labels one package as the sole Table 3 data-driven source.

## Resolved counting boundary

### Table S15 before-filter counts

The workbook-ingestion and molecular-validation stages are now reported
separately. The curated workbook has 8,343 candidate rows; canonical-SMILES and
required-field validation retains 7,953 records from 830 systems and 842
system-temperature groups, exactly matching the curated `Before filtering`
entry in Table S15. The density filter then retains 7,683 records from 765
systems and 766 groups.

The expanded Table S15 `Before filtering` entry uses the workbook count of
7,134 rows, 830 systems, and 883 groups; its validated pre-density stage is
7,125 records, 829 systems, and 882 groups. The manuscript-specific selection
is encoded in [`table_s15_counts.csv`](../../experiments/supporting_information/s5_dataset_construction_and_distribution/results/table_s15_counts.csv),
while [`dataset_overview.csv`](../../experiments/supporting_information/s5_dataset_construction_and_distribution/results/dataset_overview.csv)
retains all three stages. This resolves the earlier apparent 8,343-versus-7,953
conflict without deleting source rows or altering the model input pipeline.

## Partial manuscript result packages

| Paper item | Available | Not currently identifiable |
| --- | --- | --- |
| Table 1 | Common five-seed baseline summaries and all baseline implementations | One complete machine-readable table containing specialized baselines and the PSMI five-seed aggregate; exact PSMI seed-to-run mapping |
| Table 2 | Exact metric JSONs for eight of twelve rows | First three graph-representation controls and top-k/FiLM metric records |
| Figure 2e | System-22 saliency tables and several component plots | Exact final atom/bond/functional-group composite image |
| Figure 2f / Table S2 | Transfer code and several checkpoints | Final Figure 2f asset and machine-readable four-dataset result table |
| Figure 3 | Scientific equilibrium panels 3c-3d and source data | Separate process-schematic panels 3a-3b and the assembled four-panel image |
| Figure S4 | Sensitivity implementation | Archived final figure and numerical output package |
| Table S5 | Split code and split manifests | Final aggregate performance table |
| Figure S7 | Complete deployable Web application | Exact manuscript screenshot |

## Auxiliary results that must remain separate

The following public results are valid but are not replacements for manuscript
table values:

- sample-major main-stage summaries for seeds 42, 43, and 44;
- expanded-LLE sample-major summaries for seeds 42, 43, and 44;
- checkpoint metrics recomputed under a corrected node-layout protocol;
- metrics recomputed from a later pointwise export when a training-time best
  metric record already exists.

These results belong in [Auxiliary Multi-Seed Benchmarks](multiseed_benchmark.md)
or a new output directory with an explicit protocol identifier.

## Resolution criteria

A partial item can be upgraded to complete only when the added artifact has:

1. a paper item and panel/table identifier;
2. source checkpoint or deterministic source table;
3. dataset and split identity;
4. seed list and aggregation rule;
5. code entry point or a clear statement that the asset is an assembled
   illustration;
6. a stable repository-relative path;
7. a checksum or artifact manifest for frozen binary assets.

Do not infer missing provenance from similar metric values. Small differences
can arise from checkpoint selection, tensor layout, prediction export timing,
postprocessing, or aggregation, and each case must remain distinguishable.
