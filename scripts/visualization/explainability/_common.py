"""Shared rendering utilities for PSMI molecular importance maps."""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXCEL = (
    PROJECT_ROOT / "datasets" / "processed" / "update-LLE-all-with-smiles_min3.xlsx"
)
IMPORTANCE_DIR = PROJECT_ROOT / "experiments" / "08_interpretability" / "importance_tables"
FIGURE_DIR = PROJECT_ROOT / "figures" / "08_interpretability" / "current"

KIND_CONFIG = {
    "bond": {
        "csv": IMPORTANCE_DIR / "all_systems_bond_importance.csv",
        "output": FIGURE_DIR / "bond_importance",
        "label_column": "bond_label",
        "title": "Bond Importance",
        "suffix": "bond_importance",
        "colors": ((0.60, 0.80, 1.00, 0.60), (0.00, 0.40, 0.80, 1.00)),
    },
    "functional_group": {
        "csv": IMPORTANCE_DIR / "all_systems_fg_importance_nonzero.csv",
        "output": FIGURE_DIR / "functional_group_importance",
        "label_column": "functional_group",
        "title": "Functional Group Importance",
        "suffix": "functional_group_importance",
        "colors": ((0.00, 0.90, 0.00, 0.60), (0.00, 0.50, 0.00, 1.00)),
    },
    "node": {
        "csv": IMPORTANCE_DIR / "all_systems_node_importance.csv",
        "output": FIGURE_DIR / "node_importance",
        "label_column": "node_label",
        "title": "Node Importance",
        "suffix": "node_importance",
        "colors": ((0.80, 0.60, 1.00, 0.60), (0.50, 0.00, 0.50, 1.00)),
    },
}

MOLECULE_NAMES = ("Solute", "Solvent 1", "Solvent 2")
SMILES_COLUMNS = ("smiles1", "smiles2", "smiles3")

DATASET_COLUMN_ALIASES = {
    "system_id": ("system_id", "system id", "LLE system NO.", "LLE system NO"),
    "smiles1": (
        "smiles1",
        "Component 1 SMILES",
        "Component1 SMILES",
        "IL (Component 1) full name SMILES",
        "IL (Component 1) SMILES",
    ),
    "smiles2": ("smiles2", "Component 2 SMILES", "Component2 SMILES"),
    "smiles3": ("smiles3", "Component 3 SMILES", "Component3 SMILES"),
}


def _normalize_system_id(value: object) -> str:
    """Return a stable string representation for a system identifier."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _system_sort_key(value: str) -> Tuple[int, object]:
    """Sort numeric identifiers before free-text identifiers."""
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def parse_node_importance(rows: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    """Map node labels to per-component atom weights."""
    parsed: Dict[int, Dict[int, float]] = {1: {}, 2: {}, 3: {}}
    for _, row in rows.iterrows():
        match = re.search(r"g(\d+).*?:(\d+)", str(row["node_label"]))
        if not match:
            continue
        component, atom = int(match.group(1)), int(match.group(2))
        if component in parsed:
            parsed[component][atom] = float(row["importance"])
    return parsed


def parse_bond_importance(rows: pd.DataFrame) -> Dict[int, Dict[int, float]]:
    """Distribute each bond score equally to its two endpoint atoms."""
    parsed: Dict[int, Dict[int, float]] = {1: {}, 2: {}, 3: {}}
    for _, row in rows.iterrows():
        match = re.search(r"g(\d+).*?:(\d+)-.*?:(\d+)", str(row["bond_label"]))
        if not match:
            continue
        component = int(match.group(1))
        atom_a, atom_b = int(match.group(2)), int(match.group(3))
        if component not in parsed:
            continue
        score = float(row["importance"])
        parsed[component][atom_a] = parsed[component].get(atom_a, 0.0) + score
        parsed[component][atom_b] = parsed[component].get(atom_b, 0.0) + score
    return parsed


def atom_weights_from_mapping(mol: Chem.Mol, weights: Mapping[int, float]) -> List[float]:
    """Create an atom-aligned non-negative weight vector."""
    output = np.zeros(mol.GetNumAtoms(), dtype=float)
    for atom_index, score in weights.items():
        if 0 <= atom_index < len(output):
            output[atom_index] = abs(float(score))
    return output.tolist()


def functional_group_weights(
    mol: Chem.Mol,
    functional_groups: Iterable[Tuple[object, object]],
) -> List[float]:
    """Accumulate functional-group importance over matching atoms."""
    output = np.zeros(mol.GetNumAtoms(), dtype=float)
    for pattern_text, score in functional_groups:
        pattern = Chem.MolFromSmiles(str(pattern_text))
        if pattern is None:
            pattern = Chem.MolFromSmarts(str(pattern_text))
        if pattern is None:
            continue
        for match in mol.GetSubstructMatches(pattern):
            for atom_index in match:
                output[atom_index] += abs(float(score))
    return output.tolist()


def normalize_weights(weights: Sequence[float]) -> List[float]:
    """Scale a non-negative weight vector to the interval [0, 1]."""
    maximum = max(weights, default=0.0)
    if maximum <= 0.0:
        return [0.0 for _ in weights]
    return [float(value) / maximum for value in weights]


def _make_colormaps(kind: str) -> Tuple[LinearSegmentedColormap, LinearSegmentedColormap]:
    """Build transparent and opaque maps for one importance type."""
    light, dark = KIND_CONFIG[kind]["colors"]
    transparent = LinearSegmentedColormap.from_list(
        f"{kind}_transparent",
        [(0.0, (1, 1, 1, 0)), (0.8, (1, 1, 1, 0)), (0.9, light), (1.0, dark)],
    )
    opaque = LinearSegmentedColormap.from_list(
        f"{kind}_opaque",
        [(1, 1, 1, 1), light[:3] + (1.0,), dark],
    )
    return transparent, opaque


def _draw_heatmap(
    mol: Chem.Mol,
    weights: Sequence[float],
    colormap: LinearSegmentedColormap,
    pixels: int,
) -> Image.Image:
    """Render an RDKit similarity-map heatmap."""
    drawer = Draw.MolDraw2DCairo(pixels, pixels)
    options = drawer.drawOptions()
    options.clearBackground = True
    options.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        list(weights),
        colorMap=colormap,
        contourLines=0,
        alpha=0.6,
        sigma=0.4,
        draw2d=drawer,
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")


def _draw_highlights(
    mol: Chem.Mol,
    weights: Sequence[float],
    colormap: LinearSegmentedColormap,
    pixels: int,
) -> Image.Image:
    """Render atom-centered solid highlights."""
    drawer = Draw.MolDraw2DCairo(pixels, pixels)
    options = drawer.drawOptions()
    options.clearBackground = True
    options.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
    options.atomHighlightsAreCircles = True
    options.fillHighlights = True

    atoms = [index for index, weight in enumerate(weights) if weight > 0.001]
    colors = {index: colormap(weights[index]) for index in atoms}
    radii = {index: 0.4 for index in atoms}
    drawer.DrawMolecule(mol, atoms, [], colors, None, radii, -1, "")
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")


def _draw_molecule(
    mol: Chem.Mol,
    weights: Sequence[float],
    style: str,
    transparent_map: LinearSegmentedColormap,
    opaque_map: LinearSegmentedColormap,
    pixels: int,
) -> Image.Image:
    """Render a weighted molecule and fall back for very small molecules."""
    try:
        if style == "heatmap":
            return _draw_heatmap(mol, weights, transparent_map, pixels)
        return _draw_highlights(mol, weights, opaque_map, pixels)
    except Exception as exc:
        if "too few atoms" not in str(exc).lower():
            raise
        return Draw.MolToImage(mol, size=(pixels, pixels)).convert("RGB")


def _validate_columns(frame: pd.DataFrame, required: Iterable[str], source: Path) -> None:
    """Raise a clear error when an input table lacks required columns."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {source}: {', '.join(missing)}")


def _canonicalize_dataset_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map supported workbook headers to the internal visualization schema."""
    normalized = {" ".join(str(column).split()).lower(): column for column in frame.columns}
    rename = {}
    for target, aliases in DATASET_COLUMN_ALIASES.items():
        for alias in aliases:
            source = normalized.get(" ".join(alias.split()).lower())
            if source is not None:
                rename[source] = target
                break
    return frame.rename(columns=rename)


def render_importance_maps(
    *,
    kind: str,
    excel_path: Path,
    importance_path: Path,
    output_dir: Path,
    style: str,
    colorbar: bool,
    dpi: int,
    pixels: int,
    requested_systems: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> int:
    """Render one three-component figure per selected system."""
    if kind not in KIND_CONFIG:
        raise ValueError(f"Unsupported importance kind: {kind}")
    if not excel_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {excel_path}")
    if not importance_path.is_file():
        raise FileNotFoundError(f"Importance table not found: {importance_path}")

    data = _canonicalize_dataset_columns(pd.read_excel(excel_path))
    importance = pd.read_csv(importance_path)
    label_column = str(KIND_CONFIG[kind]["label_column"])
    _validate_columns(data, ("system_id", *SMILES_COLUMNS), excel_path)
    _validate_columns(importance, ("system_id", label_column, "importance"), importance_path)

    data = data.copy()
    importance = importance.copy()
    data["_system_key"] = data["system_id"].map(_normalize_system_id)
    importance["_system_key"] = importance["system_id"].map(_normalize_system_id)
    grouped = importance.groupby("_system_key", sort=False)

    systems = sorted(importance["_system_key"].dropna().unique(), key=_system_sort_key)
    if requested_systems:
        requested = {_normalize_system_id(value) for value in requested_systems}
        systems = [system for system in systems if system in requested]
    if limit is not None:
        systems = systems[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    transparent_map, opaque_map = _make_colormaps(kind)
    RDLogger.DisableLog("rdApp.*")

    written = 0
    for system_id in systems:
        system_rows = data[data["_system_key"] == system_id]
        if system_rows.empty:
            print(f"[WARN] System {system_id} was not found in the workbook.")
            continue
        importance_rows = grouped.get_group(system_id)
        row = system_rows.iloc[0]

        if kind == "node":
            component_weights = parse_node_importance(importance_rows)
        elif kind == "bond":
            component_weights = parse_bond_importance(importance_rows)
        else:
            component_weights = None
            group_scores = list(
                zip(importance_rows["functional_group"], importance_rows["importance"])
            )

        if colorbar:
            figure = plt.figure(figsize=(20, 6))
            grid = figure.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.1)
            axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
            color_axis = figure.add_subplot(grid[0, 3])
        else:
            figure, axes = plt.subplots(1, 3, figsize=(18, 6))
            color_axis = None

        figure.suptitle(
            f"System {system_id} {KIND_CONFIG[kind]['title']}",
            fontsize=20,
            fontweight="bold",
        )

        for component_index, (axis, column, name) in enumerate(
            zip(axes, SMILES_COLUMNS, MOLECULE_NAMES), start=1
        ):
            axis.set_title(name, fontsize=14)
            smiles = row[column]
            if pd.isna(smiles) or not str(smiles).strip():
                axis.text(0.5, 0.5, "No data", ha="center", va="center")
                axis.axis("off")
                continue
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                axis.text(0.5, 0.5, "Invalid SMILES", ha="center", va="center")
                axis.axis("off")
                continue

            if kind == "functional_group":
                weights = functional_group_weights(mol, group_scores)
            else:
                weights = atom_weights_from_mapping(
                    mol,
                    component_weights.get(component_index, {}),
                )
            weights = normalize_weights(weights)
            try:
                image = _draw_molecule(
                    mol,
                    weights,
                    style,
                    transparent_map,
                    opaque_map,
                    pixels,
                )
                axis.imshow(image, interpolation="bicubic")
            except Exception as exc:
                print(f"[WARN] System {system_id}, component {component_index}: {exc}")
                axis.text(0.5, 0.5, "Plot error", ha="center", va="center")
            axis.axis("off")

        if color_axis is not None:
            norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
            bar = mpl.colorbar.ColorbarBase(
                color_axis,
                cmap=opaque_map,
                norm=norm,
                orientation="vertical",
            )
            bar.set_label("Relative importance", fontsize=12)
            bar.set_ticks([0.0, 1.0])

        suffix = str(KIND_CONFIG[kind]["suffix"])
        output_path = output_dir / f"system_{system_id}_{suffix}.png"
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        written += 1
        if written % 10 == 0:
            print(f"Rendered {written} systems.")

    print(f"Saved {written} figure(s) to: {output_dir}")
    return written


def build_parser(kind: str, *, style: str, colorbar: bool) -> argparse.ArgumentParser:
    """Create a consistent CLI for one visualization entry point."""
    config = KIND_CONFIG[kind]
    parser = argparse.ArgumentParser(
        description=f"Render PSMI {config['title'].lower()} maps."
    )
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--importance-csv", type=Path, default=config["csv"])
    default_output = config["output"]
    if kind == "functional_group" and not colorbar:
        default_output = FIGURE_DIR / "functional_group_importance_heatmap"
    parser.add_argument("--out-dir", type=Path, default=default_output)
    parser.add_argument("--system-id", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Render only the first N systems.")
    parser.add_argument("--dpi", type=int, default=300 if colorbar else 150)
    parser.add_argument("--pixels", type=int, default=1200 if style == "highlight" else 400)
    parser.set_defaults(kind=kind, style=style, colorbar=colorbar)
    return parser


def run_cli(kind: str, *, style: str, colorbar: bool) -> None:
    """Parse arguments and render one importance-map variant."""
    args = build_parser(kind, style=style, colorbar=colorbar).parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if args.dpi < 1 or args.pixels < 1:
        raise ValueError("--dpi and --pixels must be positive.")
    render_importance_maps(
        kind=args.kind,
        excel_path=args.excel,
        importance_path=args.importance_csv,
        output_dir=args.out_dir,
        style=args.style,
        colorbar=args.colorbar,
        dpi=args.dpi,
        pixels=args.pixels,
        requested_systems=args.system_id,
        limit=args.limit,
    )
