# Checkpoint Compatibility

This package isolates the model, data, feature, and path utilities required by the distributed two-scalar checkpoints. It is separate from the maintained `src/psmi/` implementation so that checkpoint loading cannot silently change the scientific model contract.

The compatibility layer preserves the `[temperature, phase_path]` scalar input and component-major batching contract. New experiments should use `configs/model/psmi_sample_major.yaml` unless a checkpoint explicitly declares the component-major profile.
