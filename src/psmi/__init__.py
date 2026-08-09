"""PSMI: physics-informed separation and mixture-interaction modeling."""

__all__ = ["PSMI", "LLEGraphNet"]
__version__ = "0.1.0"


def __getattr__(name: str):
    """Load the torch-backed model only when callers request it."""
    if name in {"PSMI", "LLEGraphNet"}:
        from .model import LLEGraphNet

        # Public model name. The internal class name is retained so published
        # checkpoint state dictionaries remain loadable.
        return LLEGraphNet
    raise AttributeError(name)
