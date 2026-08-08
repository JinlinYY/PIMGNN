"""Load a PSMI checkpoint and run ternary LLE curve inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from backend import config as web_config
from psmi import config as model_config
from psmi.checkpoints import load_state_dict_compat
from psmi.data import (
    FunctionalGroupCache,
    GraphCache,
    MixGraphCache,
    condition_scalar_values,
)
from psmi.train import build_model
from psmi.utils import (
    Scaler,
    batch_graphs,
    batch_mixture_graphs,
    batch_to_device,
    canonicalize_smiles,
    renorm3,
    temperature_scalar_value,
)


class ModelPredictor:
    """Own the model, feature caches, scalers, and inference operations."""

    def __init__(self, model_path: Path | str | None = None, device: str | None = None) -> None:
        self.model_path = Path(model_path or web_config.MODEL_PATH).resolve()
        self.model_dir = self.model_path.parent
        self.device = torch.device(device or web_config.DEVICE)
        self._best_checkpoint = self._load_checkpoint(self.model_path)
        self._apply_checkpoint_contract(self._best_checkpoint)
        self.model = build_model()
        self.temperature_scaler: Scaler
        self.pressure_scaler: Scaler | None = None
        self.compatibility_notes: Tuple[str, ...] = ()

        self.use_graph = bool(getattr(model_config, "USE_GRAPH", False))
        self.use_mix_graph = bool(getattr(model_config, "USE_MIX_GRAPH", False))
        self.use_fg = bool(getattr(model_config, "USE_FG", False))
        self.fg_token_mode = bool(getattr(model_config, "FG_TOKEN_MODE", False))
        self.fg_max_tokens = int(getattr(model_config, "FG_MAX_TOKENS", 32))

        self.graph_cache = self._build_graph_cache() if self.use_graph else None
        self.mix_cache = MixGraphCache(model_config) if self.use_mix_graph else None
        self.fg_cache = self._build_fg_cache() if self.use_fg else None
        self._load_model()

    @staticmethod
    def _checkpoint_architecture(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Return architecture metadata when it exists in a corrected checkpoint."""
        provenance = checkpoint.get("provenance", {})
        if not isinstance(provenance, dict):
            return {}
        architecture = provenance.get("architecture", {})
        return architecture if isinstance(architecture, dict) else {}

    def _apply_checkpoint_contract(self, checkpoint: Dict[str, Any]) -> None:
        """Configure the runtime before model construction.

        Corrected checkpoints are self-describing. The bundled historical Web
        checkpoint predates provenance metadata and is therefore assigned its
        audited two-scalar, concatenation, legacy-batch contract explicitly.
        """
        architecture = self._checkpoint_architecture(checkpoint)
        if architecture:
            setattr(model_config, "SCALAR_DIM", int(architecture.get("scalar_dim", 2)))
            setattr(
                model_config,
                "FUSION_MODE",
                str(architecture.get("fusion_mode", "concat")),
            )
            setattr(
                model_config,
                "MIXTURE_NODE_LAYOUT",
                str(architecture.get("mixture_node_layout", "sample_major")),
            )
            setattr(
                model_config,
                "USE_MIX_GRAPH",
                bool(architecture.get("uses_mixture_graph", True)),
            )
            setattr(
                model_config,
                "USE_FG",
                bool(architecture.get("uses_functional_groups", True)),
            )
        else:
            setattr(model_config, "SCALAR_DIM", 2)
            setattr(model_config, "FUSION_MODE", "concat")
            setattr(model_config, "MIXTURE_NODE_LAYOUT", "legacy_component_major")

    @property
    def pressure_supported(self) -> bool:
        """Whether pressure is an input feature of the loaded checkpoint."""
        return int(getattr(self.model, "scalar_dim", 2)) == 3

    @staticmethod
    def _load_checkpoint(path: Path) -> Dict[str, Any]:
        """Load a local checkpoint using PyTorch's restricted loader when available."""
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch versions before weights_only support.
            checkpoint = torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")
        return checkpoint

    @staticmethod
    def _scaler_from_metadata(
        checkpoints: Sequence[Dict[str, Any]],
        prefix: str,
    ) -> Scaler | None:
        mean_key = f"{prefix}_mean"
        std_key = f"{prefix}_std"
        for checkpoint in checkpoints:
            if mean_key in checkpoint and std_key in checkpoint:
                return Scaler(
                    mean=float(checkpoint[mean_key]),
                    std=max(float(checkpoint[std_key]), 1e-12),
                )
        return None

    def _build_graph_cache(self) -> GraphCache:
        return GraphCache(
            add_hs=bool(getattr(model_config, "GRAPH_ADD_HS", False)),
            add_3d=bool(getattr(model_config, "GRAPH_ADD_3D", False)),
            use_gasteiger=bool(getattr(model_config, "GRAPH_USE_GASTEIGER", True)),
            max_atoms=int(getattr(model_config, "GRAPH_MAX_ATOMS", 256)),
        )

    def _build_fg_cache(self) -> FunctionalGroupCache:
        corpus_path = self.model_dir / "fg_corpus.json"
        if not corpus_path.is_file():
            raise FileNotFoundError(f"Functional-group corpus not found: {corpus_path}")
        with corpus_path.open("r", encoding="utf-8") as handle:
            corpus = json.load(handle)
        cache = FunctionalGroupCache(
            corpus=corpus,
            vocab_size=int(getattr(model_config, "FG_TOPK", 0)),
            min_freq=int(getattr(model_config, "FG_MIN_FREQ", 1)),
        )
        cache.set_corpus(corpus)
        return cache

    def _load_model(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"PSMI checkpoint not found: {self.model_path}")

        best_checkpoint = self._best_checkpoint
        metadata_sources = [best_checkpoint]
        last_path = self.model_dir / "last_model.pt"
        if last_path.is_file() and last_path != self.model_path:
            metadata_sources.append(self._load_checkpoint(last_path))

        self.compatibility_notes = load_state_dict_compat(self.model, best_checkpoint)
        temperature_scaler = self._scaler_from_metadata(metadata_sources, "T")
        if temperature_scaler is None:
            raise KeyError("T_mean and T_std are missing from the Web checkpoints")
        self.temperature_scaler = temperature_scaler
        self.pressure_scaler = self._scaler_from_metadata(metadata_sources, "P")

        self.model.to(self.device)
        self.model.eval()

    def _canonicalize_components(self, smiles_list: Sequence[str]) -> Tuple[str, str, str]:
        if len(smiles_list) != 3:
            raise ValueError("Exactly three components are required.")
        canonical = tuple(canonicalize_smiles(smiles) for smiles in smiles_list)
        if not all(canonical):
            raise ValueError("At least one component has an invalid SMILES string.")
        return canonical  # type: ignore[return-value]

    def _normalized_temperature(self, temperature: float) -> np.float32:
        feature = temperature_scalar_value(
            np.array([temperature], dtype=np.float32),
            mode=str(getattr(model_config, "TEMPERATURE_ENCODING", "linear_quadratic")),
            reference_k=float(getattr(model_config, "TEMPERATURE_REFERENCE_K", 500.0)),
        )
        return np.float32(self.temperature_scaler.transform(feature)[0])

    def _normalized_pressure(self, pressure: float) -> np.float32:
        if self.pressure_scaler is None:
            # Historical Web checkpoints predate the pressure feature. Their
            # compatibility column is zero, so zero preserves old predictions.
            return np.float32(0.0)
        value = self.pressure_scaler.transform(np.array([pressure], dtype=np.float32))[0]
        return np.float32(value)

    def _add_functional_groups(
        self,
        batch: Dict[str, Any],
        smiles: Tuple[str, str, str],
        batch_size: int,
    ) -> None:
        if self.fg_cache is None:
            return
        for index, component in enumerate(smiles, start=1):
            if self.fg_token_mode:
                token_ids, mask = self.fg_cache.get_token_ids(component, self.fg_max_tokens)
                batch[f"fg{index}_ids"] = torch.tensor(
                    np.repeat(np.asarray(token_ids)[None, :], batch_size, axis=0),
                    dtype=torch.long,
                )
                batch[f"fg{index}_mask"] = torch.tensor(
                    np.repeat(np.asarray(mask)[None, :], batch_size, axis=0),
                    dtype=torch.float32,
                )
            else:
                features = np.asarray(self.fg_cache.get(component), dtype=np.float32)
                batch[f"fg{index}"] = torch.tensor(
                    np.repeat(features[None, :], batch_size, axis=0),
                    dtype=torch.float32,
                )

    def _build_graph_batch(
        self,
        smiles: Tuple[str, str, str],
        temperature: float,
        pressure: float,
        t_values: np.ndarray,
    ) -> Dict[str, Any]:
        if self.graph_cache is None:
            raise RuntimeError("Graph cache is not initialized.")
        self.graph_cache.build_from_smiles(list(smiles))
        batch_size = len(t_values)
        batch: Dict[str, Any] = {
            f"g{index}": batch_graphs([self.graph_cache.get(component)] * batch_size)
            for index, component in enumerate(smiles, start=1)
        }

        temperature_norm = self._normalized_temperature(temperature)
        pressure_norm = self._normalized_pressure(pressure)
        scalar_dim = int(getattr(self.model, "scalar_dim", 2))
        batch["scalars"] = torch.from_numpy(
            np.stack(
                [
                    condition_scalar_values(
                        temperature_norm,
                        t,
                        pressure_norm,
                        scalar_dim=scalar_dim,
                    )
                    for t in t_values
                ],
                axis=0,
            )
        )
        self._add_functional_groups(batch, smiles, batch_size)

        if self.mix_cache is not None:
            mixture = self.mix_cache.build(
                smiles[0], smiles[1], smiles[2], float(temperature_norm), float(temperature)
            )
            batch["mix"] = batch_mixture_graphs([mixture] * batch_size)
        return batch

    @torch.no_grad()
    def predict_curve(
        self,
        smiles_list: Sequence[str],
        temperature: float,
        pressure: float = web_config.DEFAULT_PRESSURE_KPA,
        n_sweep: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict extract and raffinate compositions across the curve parameter."""
        if not self.use_graph:
            raise NotImplementedError("The packaged PSMI Web checkpoint requires graph mode.")
        smiles = self._canonicalize_components(smiles_list)
        count = int(n_sweep or getattr(model_config, "N_SWEEP", 80))
        if count < 2:
            raise ValueError("n_sweep must be at least 2.")
        t_grid = np.linspace(0.0, 1.0, count, dtype=np.float32)
        batch = self._build_graph_batch(smiles, float(temperature), float(pressure), t_grid)
        prediction = self.model(batch_to_device(batch, self.device)).detach().cpu().numpy()
        extract = np.vstack([renorm3(row[:3]) for row in prediction])
        raffinate = np.vstack([renorm3(row[3:]) for row in prediction])
        return t_grid, extract, raffinate

    def predict_from_smiles(
        self,
        smiles_list: Sequence[str],
        temperature: float,
        pressure: float = web_config.DEFAULT_PRESSURE_KPA,
        t: float = 0.5,
    ) -> Tuple[List[float], List[float]]:
        """Predict one tie-line at a specified curve coordinate."""
        smiles = self._canonicalize_components(smiles_list)
        t_values = np.array([float(t)], dtype=np.float32)
        batch = self._build_graph_batch(smiles, float(temperature), float(pressure), t_values)
        with torch.no_grad():
            row = self.model(batch_to_device(batch, self.device)).detach().cpu().numpy()[0]
        return renorm3(row[:3]).tolist(), renorm3(row[3:]).tolist()
