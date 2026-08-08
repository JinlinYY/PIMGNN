"""Checkpoint extraction and backward-compatible state-dict loading."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Tuple

import torch
from torch.nn.parameter import is_lazy


_STATE_DICT_KEYS = ("state_dict", "model", "model_state_dict")
_PREFIXES = ("module.", "_orig_mod.")
_PRESSURE_INPUT_WEIGHTS = {
    "proj_scalar.weight",
    "backbone.0.weight",
    "comp_backbone.0.weight",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checkpoint_provenance(
    model: torch.nn.Module,
    *,
    dataset_path: Path | str | None = None,
    split_manifest_path: Path | str | None = None,
    source_checkpoint_path: Path | str | None = None,
    training_contract: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Describe the numerical architecture and hashed data inputs of a checkpoint."""
    scalar_dim = int(getattr(model, "scalar_dim", 0))
    scalar_features = {
        2: ["temperature", "phase_path"],
        3: ["temperature", "phase_path", "pressure"],
    }.get(scalar_dim, [])
    result: MutableMapping[str, Any] = OrderedDict(
        model_class=type(model).__name__,
        architecture={
            "fusion_mode": getattr(model, "fusion_mode", None),
            "mixture_node_layout": getattr(model, "mixture_node_layout", None),
            "scalar_dim": scalar_dim,
            "scalar_features": scalar_features,
            "uses_mixture_graph": bool(getattr(model, "use_mix_graph", False)),
            "uses_functional_groups": bool(getattr(model, "use_fg", False)),
            "uses_s3_embedding": bool(getattr(model, "s3_equivariant", False)),
        },
    )
    for label, raw_path in (
        ("dataset", dataset_path),
        ("split_manifest", split_manifest_path),
        ("source_checkpoint", source_checkpoint_path),
    ):
        if raw_path is None or not str(raw_path).strip():
            result[f"{label}_path"] = None
            result[f"{label}_sha256"] = None
            continue
        path = Path(raw_path).resolve()
        result[f"{label}_path"] = str(path)
        result[f"{label}_sha256"] = _sha256_file(path) if path.is_file() else None
    result["training_contract"] = dict(training_contract or {})
    return result


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Return model weights from supported historical checkpoint layouts."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint).__name__}")

    candidate: Any = checkpoint
    for key in _STATE_DICT_KEYS:
        if key in checkpoint:
            candidate = checkpoint[key]
            break
    if isinstance(candidate, Mapping) and "model" in candidate:
        candidate = candidate["model"]
    if not isinstance(candidate, Mapping):
        raise TypeError("Checkpoint does not contain a state dictionary")
    return candidate


def _strip_training_prefix(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def adapt_state_dict_to_model(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> Tuple[MutableMapping[str, torch.Tensor], Tuple[str, ...]]:
    """Adapt known historical changes while leaving unknown mismatches strict.

    PSMI added normalized pressure as the third scalar after temperature and
    curve position. Historical linear weights therefore need one zero-valued
    input column so their original predictions remain unchanged.
    """
    target = model.state_dict()
    adapted: MutableMapping[str, torch.Tensor] = OrderedDict()
    changes = []

    for raw_name, value in state_dict.items():
        name = _strip_training_prefix(raw_name)
        target_value = target.get(name)
        if (
            target_value is not None
            and not is_lazy(target_value)
            and value.shape != target_value.shape
        ):
            can_append_scalar = (
                name in _PRESSURE_INPUT_WEIGHTS
                and
                value.ndim == 2
                and target_value.ndim == 2
                and value.shape[0] == target_value.shape[0]
                and value.shape[1] + 1 == target_value.shape[1]
            )
            if can_append_scalar:
                zeros = value.new_zeros((value.shape[0], 1))
                value = torch.cat((value, zeros), dim=1)
                changes.append(
                    f"{name}: appended zero input column "
                    f"({target_value.shape[1] - 1}->{target_value.shape[1]})"
                )
        adapted[name] = value

    return adapted, tuple(changes)


def load_state_dict_compat(
    model: torch.nn.Module,
    checkpoint_or_state_dict: Any,
    *,
    strict: bool = True,
) -> Tuple[str, ...]:
    """Load weights and return a description of compatibility adaptations."""
    state_dict = extract_state_dict(checkpoint_or_state_dict)
    adapted, changes = adapt_state_dict_to_model(model, state_dict)
    model.load_state_dict(adapted, strict=strict)
    return changes
