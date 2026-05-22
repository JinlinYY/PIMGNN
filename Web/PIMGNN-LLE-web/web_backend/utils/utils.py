# web_backend/utils/utils.py
import importlib.util
import os


def _load_project_utils():
    utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "utils.py"))
    spec = importlib.util.spec_from_file_location("project_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        raise ImportError(f"Failed to load src/utils.py from {utils_path}")
    loader.exec_module(module)
    return module


def renorm3(a, eps: float = 1e-12):
    project_utils = _load_project_utils()
    return project_utils.renorm3(a, eps=eps)


def canonicalize_smiles(smi: str) -> str:
    project_utils = _load_project_utils()
    return project_utils.canonicalize_smiles(smi)
