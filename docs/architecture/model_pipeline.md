# Model Pipeline

## End-to-end view

```text
SMILES x 3
  -> canonicalization and molecular graphs
  -> shared molecular message passing
  -> functional-group tokenization and cross-component attention
  -> three-node mixture graph with operating conditions
  -> multi-scale feature fusion
  -> extract head -> [Ex1, Ex2, Ex3]
  -> raffinate head -> [Rx1, Rx2, Rx3]
```

The implementation is configuration-driven. Most scientific switches are
defined in `src/psmi/config.py` and overridden by layered YAML files under
`configs/`.

## 1. Molecular graph construction

Each component SMILES is canonicalized with RDKit. Invalid or empty SMILES are
removed during dataset preparation. The graph builder can include atom,
bond, charge, and optional geometry-related features according to the runtime
configuration.

Important defaults include:

| Setting | Default meaning |
| --- | --- |
| `GRAPH_ADD_HS` | Do not add explicit hydrogen nodes |
| `GRAPH_ADD_3D` | Do not require 3D conformers |
| `GRAPH_USE_GASTEIGER` | Include computed Gasteiger charge information |
| `GRAPH_MAX_ATOMS` | Reject or constrain molecules beyond 256 atoms |

Graph construction is cached so repeated molecules are featurized once per
run. The shared encoder ensures that the same chemical substructure is treated
by the same learned parameters regardless of component position.

## 2. Molecular message passing

`MPNNEncoder` maps atom and bond features to a molecular embedding. The public
defaults use four message-passing layers, a hidden size of 256, and mean graph
pooling. Interaction-aware updates can be enabled through `GNN_INTERACTION`.

The encoder produces both local representations used by attribution methods
and pooled molecular features used by later fusion stages.

## 3. Functional-group representation

Functional-group tokens are derived from recurring molecular fragments and a
checkpoint-associated vocabulary. `FG_TOKEN_MODE` and `FG_CROSS_ATTN` control
whether token-level representations and cross-molecular attention are active.

For archived checkpoints, the matching functional-group corpus must be loaded.
Changing the corpus can change token indices even if the network dimensions
remain valid, so the corpus is part of checkpoint provenance rather than an
optional visualization artifact.

## 4. Cross-molecular attention

The three molecular representations are allowed to exchange information before
the final prediction heads. This stage represents the fact that extraction and
phase separation depend on interactions among components, not on isolated
molecular properties alone.

Attribution outputs under
`experiments/section_3_results/3_2_molecular_interaction_mechanisms/` expose
node-, bond-, functional-group-, and mixture-edge importance generated from
this interaction-aware model.

## 5. Three-node mixture graph

Each sample contributes three mixture nodes, one per component. Mixture edges
encode pairwise relationships and operating-condition features. Configurable
heuristics include hydrogen-bond, halogen-bond, pi-interaction, electrostatic,
and packing-related terms.

The mixture encoder combines these pairwise features with component embeddings.
`MIXTURE_NODE_LAYOUT` determines how the three-node graphs are flattened into a
batch; this is the principal compatibility difference between maintained and
published checkpoints.

## 6. Operating-condition features

The main benchmark passes normalized temperature and phase-path coordinate to
the network. Expanded-LLE adaptation adds pressure. Temperature can also enter
mixture-edge features through a two-term basis controlled by
`TEMPERATURE_ENCODING`:

- `linear_quadratic` uses scaled `T` and `T^2` terms;
- `inverse` uses a scaled reciprocal-temperature term and its square.

The alternative encoding is an ablation. It should not be mixed with a
checkpoint trained using the default basis.

## 7. Multi-scale fusion

The architecture can combine molecular, functional-group, and mixture-graph
features using concatenation or alternative experimental fusion modes.
`FUSION_MODE: concat` is used by the public model profiles. Architecture
ablation records are stored under:

```text
experiments/section_3_results/3_1_lle_prediction/
  3_1_2_ablation_analysis/architecture_ablation/
```

The archived ablation metrics should be interpreted with their recorded
configuration rather than as interchangeable checkpoints.

## 8. Composition heads

Separate neural heads emit extract and raffinate logits. A softmax is applied
within each phase:

```text
x_i = exp(z_i) / sum_j exp(z_j)
```

This makes the predicted mole fractions non-negative and compositionally
closed. Closure diagnostics remain useful for verifying exported predictions,
checkpoint compatibility, and postprocessing behavior.

## 9. Training stages

### Stage 1: supervised training

The full network is optimized using six-output composition MSE. Early stopping
uses validation performance, and test evaluation is disabled during training
in the public profile.

### Stage 2: physics-informed fine-tuning

The stage-1 checkpoint initializes a lower-learning-rate run. The public
profile freezes the non-output backbone and adds a chemical-potential residual
penalty based on training-system NRTL parameters. Gibbs-Duhem and TPD terms are
diagnostics in this profile, not optimized losses.

### Expanded-LLE adaptation

The expanded profile initializes from a prior model, uses a smaller learning
rate, enables pressure as a third scalar, and updates the full network because
`FREEZE_BACKBONE` is false. The distributed expanded profile is supervised
(`USE_MECH_LOSS: false`) and selects by validation RMSE.

## 10. Evaluation path

Checkpoint evaluation reconstructs the dataset, filtering, fixed manifest
split, scalers, graph caches, model configuration, and optional thermodynamic
parameter store. It then computes predictions without optimizer updates.

For published weights, registry-based evaluation is preferred because it also
loads the matching functional-group corpus and component-major override. Use
the [canonical quick-start command](../getting_started/quickstart.md#3-evaluate-the-figure-2a-checkpoint)
instead of reconstructing the checkpoint contract manually.

## 11. Extension checklist

Before adding a new model block or chemical feature:

1. declare its configuration key in `src/psmi/config.py`;
2. add the key to a named YAML profile;
3. preserve component permutation alignment;
4. record any new tensor-layout or vocabulary dependency in checkpoints;
5. evaluate on the same fixed split before comparing with existing metrics;
6. report predictive and thermodynamic diagnostics separately;
7. add an experiment README explaining inputs, outputs, and evidence status.
