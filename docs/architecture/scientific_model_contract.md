# PSMI Scientific Model Contract

## Purpose and authority

This contract defines the scientific meaning shared by the PSMI code,
configurations, datasets, checkpoints, and reported metrics. It prevents
results from being compared across incompatible tensor layouts, data splits,
or thermodynamic objectives.

When documentation and implementation differ, use the following authority
order:

1. checkpoint tensor structure and embedded metadata;
2. executable forward pass in `src/psmi/model.py`;
3. data construction in `src/psmi/data.py` and `src/psmi/utils.py`;
4. layered YAML configuration and checkpoint registry;
5. this explanatory document.

The repository does not silently alter executable behavior to match a paper
description. A scientific discrepancy should be recorded and resolved before
claiming strict reproduction.

## Prediction task

One sample represents a ternary chemical system at an operating condition and
a position along its phase path. The model receives:

- three molecular graphs derived from canonical SMILES;
- normalized temperature;
- a continuous phase-path coordinate `t` in `[0, 1]`;
- normalized pressure when the selected profile declares `SCALAR_DIM: 3`;
- a system identifier used for system-level splitting and optional
  thermodynamic parameter lookup.

The output is ordered as:

```text
[Ex1, Ex2, Ex3, Rx1, Rx2, Rx3]
```

`E` denotes the extract phase and `R` the raffinate phase. Both output heads
apply a three-class softmax, so each predicted phase is non-negative and sums
to one up to floating-point precision.

## Scalar profiles

| Profile | Scalar vector | Intended use |
| --- | --- | --- |
| Main benchmark | `[T_normalized, t]` | Fixed ternary LLE benchmark |
| Expanded LLE | `[T_normalized, t, P_normalized]` | Pressure-aware adaptation data |

Temperature and pressure scalers are fitted from the training partition only.
Checkpoint evaluation should use embedded scalers whenever they are available.
A scaler derived at evaluation time is a compatibility fallback and must be
explicitly authorized by the checkpoint registry.

## Phase-path coordinate

Within each `(system_id, T)` group, the six measured composition coordinates
are centered and projected on the first principal direction. Records are
ordered by that projection and assigned evenly spaced `t` values from 0 to 1.
This coordinate indexes position along an observed tie-line family; it is not a
thermodynamic state variable independent of the dataset construction.

Because `t` is derived from the observed group geometry, an application input
must either provide `t` directly or provide enough measured phase-composition
points for the application workflow to construct it.

## Molecular and mixture representation

The maintained graph model contains five scientific stages:

1. a shared message-passing neural network encodes each molecular graph;
2. functional-group tokens and cross-molecular attention represent recurring
   local chemical motifs and interactions among components;
3. a three-node mixture graph combines component embeddings with operating
   variables and intermolecular edge features;
4. molecular, interaction, and mixture-scale features are fused;
5. separate extract and raffinate heads predict the two compositions.

The same molecular encoder is reused for all three components. Component
identity is carried by tensor position and, in the maintained profile,
S3-aware component embedding and permutation handling.

## Mixture-node layout compatibility

Two batch layouts exist because the published checkpoints and the maintained
training implementation were produced at different stages of the codebase:

### `sample_major`

Nodes are ordered by sample, with the three components for one sample kept
together. This is the maintained layout used by
`configs/model/psmi_sample_major.yaml` and should be used for new work.

### `component_major`

Nodes are grouped by component position across the batch. Published Figure 2a
and Table 3 checkpoints use this layout and are evaluated through
`configs/reproduction/published_checkpoint_registry.json`.

These layouts change tensor indexing even when layer dimensions are identical.
A checkpoint from one layout must not be loaded as if it belonged to the other.
Use registry-declared overrides and `src/psmi_checkpoint_compat/` for archived
weights.

## Component-permutation contract

The implemented augmentation creates exactly one additional orientation for
each eligible training record:

```text
smiles2 <-> smiles3
Ex2     <-> Ex3
Rx2     <-> Rx3
```

The original and swapped records are labeled with `aug_swap23 = 0` and `1`.
When thermodynamic parameters are used, their component-indexed interaction
terms are permuted consistently. Augmentation is applied after the system-level
split and only to training data. Validation and test records retain their
original orientation.

This doubles eligible training examples but does not double the number of
experimental measurements. Dataset counts therefore report unaugmented rows.

## Supervised objective

The supervised loss is the mean squared error over all six predicted mole
fractions:

```text
L_sup = mean((y_pred - y_true)^2)
```

Validation checkpoint selection uses a configured predictive metric. Test data
are not used for optimizer updates or checkpoint selection.

## Chemical-potential regularization

For phase equilibrium, component `i` should have equal chemical potential in
the two phases. After canceling the common standard-state term, the implemented
dimensionless residual is:

```text
r_i = ln(x_i^E * gamma_i^E) - ln(x_i^R * gamma_i^R)
```

Activity coefficients are computed from a selected excess-Gibbs-energy model.
The main physics profile uses system-specific NRTL parameters and a robust
Huber penalty on the residual. The total objective is:

```text
L_total = L_sup + lambda(epoch) * L_phy
L_phy   = w_eq * L_eq + w_gd * L_gd + w_stab * L_stab
```

In the distributed `stage2_physics.yaml` profile:

- `w_eq = 1`;
- `w_gd = 0`;
- `w_stab = 0`;
- the target physics multiplier is `lambda = 0.001`;
- the non-output backbone is frozen;
- the learning rate is reduced relative to supervised training.

Therefore, the maintained physics-training objective penalizes
chemical-potential mismatch but does not add a separate Gibbs-Duhem or TPD loss.

## Gibbs-Duhem interpretation

NRTL defines activity coefficients through an excess Gibbs-energy model, which
provides the underlying thermodynamic structure. The code also computes a
finite-difference Gibbs-Duhem diagnostic along simplex directions:

```text
sum_i x_i * d(ln gamma_i) = 0
```

This diagnostic is reported independently of the stage-2 objective because
`MECH_W_GD` is zero in the public profile. A low diagnostic value must not be
described as the result of a separately optimized Gibbs-Duhem loss.

## TPD interpretation

The tangent-plane-distance diagnostic evaluates local stability using sampled
trial compositions. `tpd_viol_rate` is the fraction of evaluated samples with
a positive implemented violation penalty, and `tpd_viol_mean` averages the
positive penalty values. The public stage-2 profile reports this diagnostic but
does not optimize it because `MECH_W_STAB` is zero.

## Thermodynamic parameter separation

Two NRTL parameter stores serve different purposes:

- `nrtl_params_train.json` is permitted during optimization and contains
  training-system parameters;
- `nrtl_params_all.json` is used for post-selection diagnostics where parameter
  coverage on validation or test systems is needed.

This separation prevents all-system thermodynamic fits from entering the
training loss. `param_cov` reports the fraction of evaluated samples for which
the selected diagnostic store contains parameters.

## Checkpoint provenance

Maintained checkpoints can record:

- architecture and node layout;
- scalar dimension and feature settings;
- dataset and split-manifest hashes;
- training seed and optimizer contract;
- temperature and pressure scalers;
- functional-group vocabulary.

Published checkpoints created before the complete metadata contract are bound
to explicit registry records. Flags such as `allow_input_hash_mismatch` and
`allow_derived_scalers` are compatibility declarations for named artifacts,
not general recommendations to disable provenance validation.

## Comparison boundary

Two metrics may be compared as one controlled experiment only when all of the
following match or are intentionally varied and disclosed:

- source dataset and filtering threshold;
- fixed system-level partition;
- augmentation policy;
- node layout and architecture;
- scalar definition and scaler provenance;
- checkpoint-selection rule;
- metric aggregation convention;
- thermodynamic parameter coverage for physics diagnostics.

When any boundary differs, report the comparison as a separate protocol rather
than attributing the difference to a single model choice.
