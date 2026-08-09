# Paper-Aligned Results Index

This directory maps the final manuscript results to the public repository. It
is the entry point for deciding which file supports a figure, table, or
numerical statement.

## Result documents

| Document | Scope |
| --- | --- |
| [Main-text results](main_text_results.md) | Figure 2, Tables 1-3, and Figure 3 |
| [Supporting Information results](supporting_information_results.md) | Figures S1-S8, Tables S2-S17, and the efficiency benchmark |
| [Evaluation metrics](evaluation_metrics.md) | Mathematical definitions and aggregation conventions |
| [Auxiliary multi-seed benchmarks](multiseed_benchmark.md) | Maintained sample-major runs that are not manuscript table values |
| [Artifact status and discrepancies](artifact_status_and_discrepancies.md) | Missing composite assets, split provenance, and values requiring confirmation |

## Evidence-status vocabulary

Every paper item is assigned one of four statuses:

| Status | Meaning |
| --- | --- |
| Complete | The manuscript-aligned figure or table source, numerical evidence, and relevant implementation are distributed. |
| Partial | Some evidence is available, but the exact final panel, aggregate table, checkpoint, or per-seed provenance is incomplete. |
| Confirmation required | Two public artifacts imply different provenance or values; neither is silently selected as authoritative. |
| Auxiliary | The result is scientifically useful but is not a value reported in the final manuscript. |

The status describes the public evidence package, not the scientific importance
of the experiment.

## Identity rules

Results are identified by the complete protocol rather than by a model name
alone. At minimum, record:

1. manuscript figure, table, or SI section;
2. dataset and minimum tie-line-density filter;
3. system split and random seeds;
4. `component_major` or `sample_major` node layout;
5. checkpoint and functional-group corpus;
6. prediction table or training-time metric record;
7. metric aggregation convention;
8. thermodynamic parameter source when physics diagnostics are reported.

A single-checkpoint Figure 2a metric must not be substituted for the five-seed
PSMI row in Table 1. Likewise, the auxiliary seeds 42/43/44 sample-major
summary is not the numerical source for Tables 1-3.

## Authoritative repository locations

Paper-aligned evidence is organized under:

```text
experiments/section_3_results/
experiments/supporting_information/
```

The top-level `results/` directory contains the canonical Figure 2a image and
two self-contained checkpoint packages. These packages are useful evidence,
but their stored metrics must be matched to a paper item explicitly. Directory
proximity alone does not establish manuscript provenance.

## Dataset-count alignment

Table S15 is reproduced by
[`table_s15_counts.csv`](../../experiments/supporting_information/s5_dataset_construction_and_distribution/results/table_s15_counts.csv).
The accompanying
[`dataset_overview.csv`](../../experiments/supporting_information/s5_dataset_construction_and_distribution/results/dataset_overview.csv)
separates workbook ingestion, molecular-record validation, and tie-line-density
filtering. This distinction explains the curated benchmark's 8,343 workbook
rows versus the manuscript's 7,953 valid pre-density records.

## How to cite a repository result

When reporting a value, cite the paper item and the repository artifact. For
example:

```text
Figure 2a, component-major manuscript checkpoint,
experiments/section_3_results/3_1_lle_prediction/main_benchmark/
```

For a rerun, additionally report the commit, environment, output directory,
checkpoint SHA-256 digest, and whether the value came from a stored metric
record or was recomputed from pointwise predictions.
