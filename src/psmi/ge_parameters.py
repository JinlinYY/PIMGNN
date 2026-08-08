"""Parameter stores for excess-Gibbs-energy activity models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import torch


class GEParameterStore:
    """Load system-specific GE interaction energies and mask unknown systems."""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        meta = payload.get("meta", {})
        self.model = str(meta.get("model", "nrtl")).lower()
        self.alpha = float(meta.get("alpha", 0.3))
        self.R = float(meta.get("R", 8.314462618))
        self.device = device
        self.dtype = dtype
        self.params = {
            int(system_id): torch.tensor(values, dtype=dtype)
            for system_id, values in payload.get("params", {}).items()
        }

    def get_batch(
        self,
        system_ids: torch.Tensor,
        *,
        swap23: Optional[torch.Tensor] = None,
        device: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_device = device or self.device
        ids = system_ids.detach().cpu().reshape(-1).tolist()
        batch = torch.zeros((len(ids), 3, 3), dtype=self.dtype)
        mask = torch.zeros(len(ids), dtype=torch.bool)
        for index, raw_system_id in enumerate(ids):
            system_id = int(raw_system_id)
            if system_id in self.params:
                batch[index] = self.params[system_id]
                mask[index] = True

        if swap23 is not None:
            swaps = swap23.detach().cpu().reshape(-1).bool()
            permutation = torch.tensor([0, 2, 1])
            for index in torch.nonzero(mask & swaps, as_tuple=False).reshape(-1).tolist():
                batch[index] = batch[index][permutation][:, permutation]

        return batch.to(target_device), mask.to(target_device)

