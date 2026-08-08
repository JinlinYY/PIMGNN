"""Regression tests for the organized public code-and-weights archive."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from psmi_legacy_public.model import LLEGraphNet
from psmi_legacy_public.paths import (
    BASE_TERNARY_CHECKPOINT,
    BIGSOLVDB_FINETUNED_CHECKPOINT,
    BIGSOLVDB_TEMP_CHECKPOINT,
    COMPSOL_CHECKPOINT,
)
from scripts.data_preparation.public_release.build_freesolv_example import build_example
from scripts.data_preparation.public_release.convert_abraham import (
    load_abraham_csv_as_pseudo_ternary,
)
from scripts.experiments.transfer_learning.public_release.finetune_bigsoldb_degenerate import (
    compatible_backbone_weights,
)


EXPECTED_HASHES = {
    BASE_TERNARY_CHECKPOINT: "1BE55A3BFD4F953064B0FD46CFF0C7CF791919F20982CA14C38F72130DFB80BF",
    COMPSOL_CHECKPOINT: "AB509735FC2A2858F2D80782D47816C250C64730F5EEAB9A80B4C81E4B34A067",
    BIGSOLVDB_TEMP_CHECKPOINT: "66345720CADA46470F97420D853A86C293669D22B1B03C5FF99AA2D20221AB92",
    BIGSOLVDB_FINETUNED_CHECKPOINT: "4A418C968A33E5E448CC17C1CDAAD7E6FC7090E2BEDFB07949AEDB07EEBDCA55",
}


def checkpoint_state(path: Path):
    """Load a trusted local checkpoint through the restricted weights loader."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
        return checkpoint["model"]
    return checkpoint


class PublicCheckpointTest(unittest.TestCase):
    """Verify byte preservation and exact historical architecture compatibility."""

    def test_checkpoint_hashes_match_the_source_archive(self) -> None:
        for path, expected in EXPECTED_HASHES.items():
            digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
            self.assertEqual(digest, expected, path.name)

    def test_all_checkpoint_layouts_load_strictly(self) -> None:
        base = LLEGraphNet(use_mix_graph=True, use_fg=True, fg_vocab_size=512)
        base.load_state_dict(checkpoint_state(BASE_TERNARY_CHECKPOINT), strict=True)

        for path in (COMPSOL_CHECKPOINT, BIGSOLVDB_TEMP_CHECKPOINT):
            binary = LLEGraphNet(is_binary=True)
            binary.load_state_dict(checkpoint_state(path), strict=True)

        frozen_head = LLEGraphNet(use_mix_graph=True, use_fg=True, fg_vocab_size=512)
        frozen_head.load_state_dict(checkpoint_state(BASE_TERNARY_CHECKPOINT), strict=True)
        frozen_head.binary_head = nn.Sequential(
            nn.Linear(3330, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        frozen_head.load_state_dict(
            checkpoint_state(BIGSOLVDB_FINETUNED_CHECKPOINT), strict=True
        )

    def test_base_checkpoint_is_accepted_for_degenerate_transfer(self) -> None:
        checkpoint = torch.load(
            BASE_TERNARY_CHECKPOINT, map_location="cpu", weights_only=True
        )
        binary = LLEGraphNet(is_binary=True)
        transfer = compatible_backbone_weights(checkpoint, binary.state_dict())

        self.assertGreater(len(transfer), 50)
        self.assertFalse(any("head" in key for key in transfer))


class PublicDataConverterTest(unittest.TestCase):
    """Check the deterministic external-data adapters without network access."""

    def test_freesolv_example_has_documented_size_and_split(self) -> None:
        frame = build_example()
        self.assertEqual(len(frame), 10)
        self.assertEqual((frame["split"] == "train").sum(), 6)
        self.assertEqual((frame["split"] == "test").sum(), 4)

    def test_abraham_converter_filters_missing_targets(self) -> None:
        source = pd.DataFrame(
            {
                "system_id": ["a", "b", "c"],
                "smiles1": ["CC", "CCC", "CO"],
                "smiles2": ["O", "O", "O"],
                "smiles3": ["N", "N", "N"],
                "T": [298.15, 298.15, 310.0],
                "L": [1.0, -123.0, 2.0],
            }
        )
        with tempfile.TemporaryDirectory(prefix="psmi-abraham-") as directory:
            temporary = Path(directory) / "test_abraham_public_release.csv"
            source.to_csv(temporary, index=False)
            converted, mapping = load_abraham_csv_as_pseudo_ternary(temporary)

        self.assertEqual(len(converted), 2)
        self.assertEqual(set(mapping), {"a", "c"})
        self.assertEqual(list(converted["y"]), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
