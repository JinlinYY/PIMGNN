from __future__ import annotations

from typing import List

from fastapi import HTTPException
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdDepictor, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


def _mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {smiles}")
    return mol


def canonicalize_smiles(smiles: str) -> str:
    mol = _mol_from_smiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True)


def molecule_svg(smiles: str, width: int = 240, height: int = 170) -> str:
    mol = _mol_from_smiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.padding = 0.08
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("svg:", "")


def descriptors_for_smiles(smiles: str) -> List[dict]:
    mol = _mol_from_smiles(smiles)
    descriptors = [
        ("Molecular weight", Descriptors.MolWt(mol), "g/mol"),
        ("LogP", Crippen.MolLogP(mol), None),
        ("TPSA", rdMolDescriptors.CalcTPSA(mol), "A^2"),
        ("H-bond donors", float(Lipinski.NumHDonors(mol)), None),
        ("H-bond acceptors", float(Lipinski.NumHAcceptors(mol)), None),
        ("Rotatable bonds", float(Lipinski.NumRotatableBonds(mol)), None),
        ("Aromatic rings", float(rdMolDescriptors.CalcNumAromaticRings(mol)), None),
        ("Fraction SP3", rdMolDescriptors.CalcFractionCSP3(mol), None),
    ]
    return [
        {
            "name": name,
            "value": round(float(value), 4),
            "unit": unit,
        }
        for name, value, unit in descriptors
    ]


def validate_smiles(smiles: str) -> dict:
    canonical = canonicalize_smiles(smiles)
    mol = _mol_from_smiles(canonical)
    return {
        "input_smiles": smiles,
        "canonical_smiles": canonical,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "svg": molecule_svg(canonical),
        "descriptors": descriptors_for_smiles(canonical),
    }
