# Historical Checkpoint Compatibility

This package isolates the model, data, feature, and path utilities required by early public checkpoints. It is intended for historical checkpoint evaluation and is separate from the maintained `src/psmi/` implementation.

The compatibility layer preserves the two-scalar input contract and legacy batching behavior. New experiments should use the corrected configuration unless a historical checkpoint is explicitly selected.
