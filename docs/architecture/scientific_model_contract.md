# PSMI Scientific Model Contract

## Purpose

This document defines the architecture and terminology shared by the code, configurations, checkpoints, and scientific reports. The executable forward pass and checkpoint tensor structure are the authoritative implementation references.

## Corrected model

The maintained `corrected_v2` model combines a shared molecular message-passing encoder, cross-molecular functional-group attention, a three-node mixture graph, multi-scale feature concatenation, and independent extract- and raffinate-phase composition heads.

The main benchmark uses normalized temperature and phase-path coordinates `[T, s]`. The expanded-LLE model adds normalized pressure and therefore uses `[T, s, P]`. Pressure statistics are fitted only on the expanded training partition.

## Mixture-node layout

`corrected_v2` uses sample-major node ordering. Historical paper checkpoints use the legacy component-major batch layout. The two layouts have separate configuration files and checkpoint registries and must not be combined within one metric comparison.

## Thermodynamic regularization

The physics-regularized stage evaluates activity coefficients with an NRTL excess-Gibbs-energy model and penalizes phase-wise chemical-potential mismatch. NRTL provides an internally consistent Gibbs-energy representation. The current neural objective does not contain a separate Gibbs-Duhem residual loss.

## Composition contract

Both output heads return three-component phase compositions. Composition closure and non-negativity are evaluated during model diagnostics. Thermodynamic residuals are reported separately from pointwise composition errors.

## Checkpoint provenance

Corrected checkpoints record the model architecture, scalar dimension, dataset digest, split-manifest digest, optimizer contract, and feature scalers. Historical checkpoints without provenance metadata are evaluated through an explicit compatibility registry.
