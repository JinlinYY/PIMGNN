"""Load layered YAML profiles into the existing PSMI runtime configuration."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, Mapping, Optional, Set

import yaml

from . import config as default_config


_PATH_KEYS = {
    "EXCEL_PATH",
    "FINE_TUNE_EXCEL_PATH",
    "LOAD_CKPT_PATH",
    "PRETRAINED_MODEL_PATH",
    "NRTL_TRAIN_PARAMS_PATH",
    "NRTL_EVAL_PARAMS_PATH",
    "SPLIT_MANIFEST_PATH",
    "OUT_DIR",
}


def _resolve_value(key: str, value: Any, project_root: Path) -> Any:
    if key not in _PATH_KEYS or not isinstance(value, str) or not value.strip():
        return value
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def load_config_file(
    path: Path | str,
    *,
    project_root: Optional[Path] = None,
    _seen: Optional[Set[Path]] = None,
) -> Dict[str, Any]:
    """Load one YAML profile and its relative ``include`` files."""
    profile_path = Path(path).resolve()
    root = Path(project_root or default_config.PROJECT_ROOT).resolve()
    seen = set() if _seen is None else _seen
    if profile_path in seen:
        raise ValueError(f"Cyclic configuration include detected at {profile_path}")
    seen.add(profile_path)

    with profile_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"Configuration must be a mapping: {profile_path}")

    values: Dict[str, Any] = {}
    includes = payload.get("include", [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list):
        raise TypeError(f"'include' must be a path or list in {profile_path}")
    for include in includes:
        include_path = Path(str(include))
        if not include_path.is_absolute():
            include_path = profile_path.parent / include_path
        values.update(
            load_config_file(include_path, project_root=root, _seen=seen)
        )

    for raw_key, value in payload.items():
        if raw_key == "include":
            continue
        key = str(raw_key)
        if not key.isupper():
            raise ValueError(
                f"Runtime configuration keys must be uppercase, got {key!r} in {profile_path}"
            )
        values[key] = _resolve_value(key, value, root)
    seen.remove(profile_path)
    return values


def apply_config_files(
    paths: Iterable[Path | str],
    *,
    config_module: ModuleType = default_config,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Apply YAML profiles in order and return the effective overrides."""
    effective: Dict[str, Any] = {}
    for path in paths:
        effective.update(load_config_file(path, project_root=project_root))
    unknown = sorted(key for key in effective if not hasattr(config_module, key))
    if unknown:
        raise KeyError(f"Unknown runtime configuration keys: {', '.join(unknown)}")
    for key, value in effective.items():
        setattr(config_module, key, value)
    return effective


def apply_config_overrides(
    assignments: Iterable[str],
    *,
    config_module: ModuleType = default_config,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Apply command-line ``KEY=YAML_VALUE`` overrides with config validation."""
    root = Path(project_root or default_config.PROJECT_ROOT).resolve()
    effective: Dict[str, Any] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"Configuration override must use KEY=VALUE: {assignment!r}")
        raw_key, raw_value = assignment.split("=", 1)
        key = raw_key.strip()
        if not key or not key.isupper():
            raise ValueError(f"Runtime configuration key must be uppercase: {key!r}")
        if not hasattr(config_module, key):
            raise KeyError(f"Unknown runtime configuration key: {key}")
        value = yaml.safe_load(raw_value)
        effective[key] = _resolve_value(key, value, root)
    for key, value in effective.items():
        setattr(config_module, key, value)
    return effective
