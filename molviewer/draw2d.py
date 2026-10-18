"""
2D molecular drawing for Jupyter, Colab, and Streamlit.
"""

from __future__ import annotations

from typing import Union

from rdkit import Chem
from rdkit.Chem import Draw

from ._mol import ensure_mol, extract_smiles


def draw(input_data, *, highlight_bits=None, fp_type=None, legends=None,
         size=(300, 300), mols_per_row=3, column=None):
    """Draw 2D molecular structure(s).

    Works in Jupyter, Colab, and Streamlit (pass returned image to st.image()).

    Args:
        input_data: SMILES string, list of SMILES, RDKit Mol, pd.Series, or pd.DataFrame.
        highlight_bits: list of fingerprint bit indices to highlight (requires fp_type and molfp).
        fp_type: fingerprint type for bit highlighting (e.g. "ecfp4", "maccs").
        legends: list of labels for each molecule in grid mode.
        size: (width, height) per molecule in pixels.
        mols_per_row: molecules per row in grid mode.
        column: column name for DataFrame input.

    Returns:
        PIL.Image.Image
    """
    data, is_batch = extract_smiles(input_data, column=column)

    if not is_batch:
        mol = ensure_mol(data)
        highlight_atoms = _get_highlight_atoms(mol, highlight_bits, fp_type)
        img = Draw.MolToImage(mol, size=size, highlightAtoms=highlight_atoms)
        return img

    smiles_list = data
    mols = []
    for s in smiles_list:
        try:
            mols.append(ensure_mol(s))
        except (ValueError, TypeError):
            mols.append(None)

    if legends is None:
        legends = [s if isinstance(s, str) else "" for s in smiles_list]

    if highlight_bits and fp_type:
        highlight_atom_lists = []
        for mol in mols:
            if mol is not None:
                highlight_atom_lists.append(
                    _get_highlight_atoms(mol, highlight_bits, fp_type)
                )
            else:
                highlight_atom_lists.append([])
    else:
        highlight_atom_lists = None

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=size,
        legends=legends,
        highlightAtomLists=highlight_atom_lists,
    )
    return img


def _get_highlight_atoms(mol, highlight_bits, fp_type):
    """Get atom indices to highlight for given fingerprint bits."""
    if not highlight_bits or not fp_type:
        return []

    try:
        from molfp.explain import explain
    except ImportError:
        return []

    smiles = Chem.MolToSmiles(mol)
    bit_info = explain(smiles, fp_type)

    atoms = set()
    for bit in highlight_bits:
        if bit in bit_info:
            for atom_info in bit_info[bit]:
                if isinstance(atom_info, (list, tuple)):
                    for a in atom_info:
                        if isinstance(a, int):
                            atoms.add(a)
                elif isinstance(atom_info, int):
                    atoms.add(atom_info)
    return list(atoms)
