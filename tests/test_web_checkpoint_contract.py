"""Verify that the Web backend follows the loaded checkpoint contract."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
WEB_ROOT = PROJECT_ROOT / "Web" / "PSMI-LLE-web"
for candidate in (str(SRC_ROOT), str(WEB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


def test_published_web_checkpoint_uses_audited_contract() -> None:
    """The bundled checkpoint is two-scalar and keeps its component-major layout."""
    from backend.models.predictor import ModelPredictor
    from psmi import config as model_config

    keys = ["SCALAR_DIM", "FUSION_MODE", "MIXTURE_NODE_LAYOUT", "USE_MIX_GRAPH", "USE_FG"]
    previous = {key: getattr(model_config, key) for key in keys}
    try:
        predictor = ModelPredictor(device="cpu")
        assert predictor.model.scalar_dim == 2
        assert predictor.model.fusion_mode == "concat"
        assert predictor.model.mixture_node_layout == "component_major"
        assert predictor.pressure_supported is False
    finally:
        for key, value in previous.items():
            setattr(model_config, key, value)
