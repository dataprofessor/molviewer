"""
Standalone molecule utilities — no dependency on molfp.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors


_conformer_cache = {}


def ensure_mol(smiles_or_mol):
    """Convert SMILES to RDKit Mol if needed."""
    if isinstance(smiles_or_mol, Chem.rdchem.Mol):
        return smiles_or_mol
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles_or_mol}")
        return mol
    raise TypeError(f"Expected SMILES string or RDKit Mol, got {type(smiles_or_mol)}")


def extract_smiles(input_data, column=None):
    """Extract a list of SMILES from various input types.

    Returns (smiles_list, is_batch) where is_batch=False means single molecule.
    """
    if isinstance(input_data, str):
        return input_data, False

    if isinstance(input_data, Chem.rdchem.Mol):
        return input_data, False

    try:
        import pandas as pd
        if isinstance(input_data, pd.DataFrame):
            if column is not None:
                return input_data[column].tolist(), True
            for col in ["SMILES", "smiles", "Smiles", "canonical_smiles", "CANONICAL_SMILES",
                        "mol", "Mol", "MOL", "molecule", "smi"]:
                if col in input_data.columns:
                    return input_data[col].tolist(), True
            raise ValueError(
                "DataFrame has no recognized SMILES column. "
                "Pass column= or use a column named 'SMILES', 'smiles', or 'canonical_smiles'."
            )
        if isinstance(input_data, pd.Series):
            return input_data.tolist(), True
    except ImportError:
        pass

    if hasattr(input_data, '__iter__') and not isinstance(input_data, (str, bytes)):
        return list(input_data), True

    return input_data, False


# ── Conformer generation ──────────────────────────────────────────────────────

_METAL_ATOMIC_NUMS = (
    set(range(3, 5))
    | set(range(11, 14))
    | set(range(19, 31))
    | set(range(37, 49))
    | set(range(55, 81))
    | set(range(87, 104))
)

_LARGE_MOL_THRESHOLD = 80


def _classify_molecule(mol):
    has_metal = any(a.GetAtomicNum() in _METAL_ATOMIC_NUMS for a in mol.GetAtoms())
    is_large = mol.GetNumHeavyAtoms() > _LARGE_MOL_THRESHOLD
    if has_metal and is_large:
        return "metal_large"
    if has_metal:
        return "metal"
    if is_large:
        return "large"
    return "standard"


def _estimate_max_iters(mol):
    """Estimate MMFF iterations from molecular descriptors (MLR on 1,420 FDA drugs)."""
    n_atoms = mol.GetNumHeavyAtoms()
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_rings = rdMolDescriptors.CalcNumRings(mol)
    fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    raw = -10 + 9.9 * n_atoms + 6.8 * n_rot - 12.4 * n_rings + 38.3 * fsp3
    return max(200, min(2000, int(1.5 * raw)))


def _ensure_conformer_impl(mol):
    """Core conformer generation logic."""
    if mol.GetNumConformers() > 0:
        return mol

    mol_class = _classify_molecule(mol)
    mol = Chem.AddHs(mol)

    if mol_class in ("large", "metal_large"):
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol, params)
        if status == -1:
            params.useRandomCoords = True
            status = AllChem.EmbedMolecule(mol, params)
    else:
        status = AllChem.EmbedMolecule(mol, randomSeed=42)
        if status == -1:
            status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

    if status == -1:
        return mol

    estimated = _estimate_max_iters(mol)

    if mol_class in ("metal", "metal_large"):
        try:
            for multiplier in [1, 2, 4]:
                max_iters = min(estimated * multiplier, 4000)
                result = AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
                if result == 0:
                    return mol
        except Exception:
            pass
        return mol

    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is not None:
        for multiplier in [1, 2, 4]:
            max_iters = min(estimated * multiplier, 4000)
            result = AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
            if result == 0:
                return mol
        return mol

    try:
        for multiplier in [1, 2, 4]:
            max_iters = min(estimated * multiplier, 4000)
            result = AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
            if result == 0:
                return mol
    except Exception:
        pass

    return mol


def ensure_conformer(mol):
    """Add optimized 3D conformer with caching."""
    try:
        smi = Chem.MolToSmiles(mol)
        if smi in _conformer_cache:
            return _conformer_cache[smi]
    except Exception:
        return _ensure_conformer_impl(mol)

    result = _ensure_conformer_impl(mol)
    _conformer_cache[smi] = result
    return result
