"""
2D structural attribution maps — requires molfp for SHAP/explain.

Green = active-driving atoms, Red = inactive-driving atoms.
Two styles: 'highlight' (solid atom/bond circles) and 'contour' (gaussian contour map).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ._mol import ensure_mol


def _bit_shap_to_atom_weights(mol, smiles, fp_type, shap_per_bit):
    """Aggregate bit-level SHAP values to per-atom weights via explain()."""
    try:
        from molfp.explain import explain
    except ImportError:
        raise ImportError(
            "molfp is required for draw_attribution(). "
            "Install it with: pip install molfp"
        )

    bit_atoms = explain(smiles, fp_type)
    n_atoms = mol.GetNumAtoms()
    weights = np.zeros(n_atoms)
    for bit, shap_val in shap_per_bit.items():
        if bit not in bit_atoms:
            continue
        for atom_info in bit_atoms[bit]:
            if isinstance(atom_info, (list, tuple)):
                atom_indices = [a for a in atom_info if isinstance(a, int)]
            elif isinstance(atom_info, int):
                atom_indices = [atom_info]
            else:
                continue
            if atom_indices:
                share = shap_val / len(atom_indices)
                for idx in atom_indices:
                    if 0 <= idx < n_atoms:
                        weights[idx] += share
    return weights


def draw_attribution(smiles, interpret_result, *,
                     class_active="active", class_inactive="inactive",
                     size=(400, 400), alpha=0.5, style="highlight", save=None):
    """Draw 2D structure colored by class-specific SHAP attribution.

    Green regions drive the active prediction, red regions drive inactive.
    Intensity is proportional to SHAP magnitude. Neutral atoms are uncolored.

    Requires molfp: pip install molfp

    Args:
        smiles: SMILES string of the molecule to visualize.
        interpret_result: dict returned by molfp.interpret().
        class_active: name of the active class in interpret_result.
        class_inactive: name of the inactive class in interpret_result.
        size: (width, height) in pixels.
        alpha: transparency of the heatmap overlay.
        style: "highlight" (solid atom/bond circles) or "contour" (gaussian contour map).
        save: file path to save the image. If None, image is not saved.

    Returns:
        PIL.Image.Image
    """
    try:
        from molfp.fingerprints import FINGERPRINTS
    except ImportError:
        raise ImportError(
            "molfp is required for draw_attribution(). "
            "Install it with: pip install molfp"
        )

    mol = ensure_mol(smiles)
    AllChem.Compute2DCoords(mol)
    fp_type = interpret_result["fp_type"]
    model = interpret_result["model"]
    label_set = interpret_result["label_set"]

    fn = FINGERPRINTS[fp_type][0]
    fp = np.array([fn(smiles)], dtype=np.int32)

    active_idx = label_set.index(class_active) if class_active in label_set else 0
    inactive_idx = label_set.index(class_inactive) if class_inactive in label_set else -1

    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(fp)

    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        mol_shap_active = sv[0, :, active_idx]
        mol_shap_inactive = sv[0, :, inactive_idx] if inactive_idx >= 0 else -sv[0, :, active_idx]
    elif isinstance(sv, list):
        mol_shap_active = sv[active_idx][0]
        mol_shap_inactive = sv[inactive_idx][0] if inactive_idx >= 0 else -sv[active_idx][0]
    else:
        mol_shap_active = sv[0]
        mol_shap_inactive = -sv[0]

    fp_vec = fn(smiles)
    on_bits = [i for i, v in enumerate(fp_vec) if v > 0]

    active_bit_shap = {b: float(mol_shap_active[b]) for b in on_bits}
    inactive_bit_shap = {b: float(mol_shap_inactive[b]) for b in on_bits}

    active_weights = _bit_shap_to_atom_weights(mol, smiles, fp_type, active_bit_shap)
    inactive_weights = _bit_shap_to_atom_weights(mol, smiles, fp_type, inactive_bit_shap)

    # ── Contour style ─────────────────────────────────────────────────────
    def _render_contour(weights, cmap):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
        from PIL import Image as PILImage
        from io import BytesIO

        max_w = max(abs(weights.max()), abs(weights.min()), 1e-8)
        normalized = list(weights / max_w)

        try:
            d2d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        except Exception:
            d2d = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        d2d.drawOptions().bondLineWidth = 2.0

        result_d2d = SimilarityMaps.GetSimilarityMapFromWeights(
            mol, normalized, colorMap=cmap, alpha=alpha, size=size,
            draw2d=d2d, contourLines=10
        )
        result_d2d.FinishDrawing()
        png = result_d2d.GetDrawingText()
        return PILImage.open(BytesIO(png)).copy()

    # ── Highlight style ───────────────────────────────────────────────────
    def _render_highlight(weights, color_pos, color_neg):
        from rdkit.Chem.Draw import rdMolDraw2D
        from PIL import Image as PILImage
        from io import BytesIO

        max_w = max(abs(weights.max()), abs(weights.min()), 1e-8)
        norm = weights / max_w

        def _color_for_weight(w):
            if abs(w) < 0.05:
                return None
            color = color_pos if w > 0 else color_neg
            intensity = min(abs(w), 1.0)
            r = color[0] + (1 - color[0]) * (1 - intensity)
            g = color[1] + (1 - color[1]) * (1 - intensity)
            b = color[2] + (1 - color[2]) * (1 - intensity)
            return (r, g, b)

        highlight_atoms = []
        highlight_colors = {}
        for i, w in enumerate(norm):
            c = _color_for_weight(w)
            if c is not None:
                highlight_atoms.append(i)
                highlight_colors[i] = c

        highlight_bonds = []
        highlight_bond_colors = {}
        highlight_atom_set = set(highlight_atoms)
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if a1 in highlight_atom_set or a2 in highlight_atom_set:
                bond_idx = bond.GetIdx()
                highlight_bonds.append(bond_idx)
                avg_w = (norm[a1] + norm[a2]) / 2
                c = _color_for_weight(avg_w)
                if c is None:
                    c = (0.95, 0.95, 0.95)
                highlight_bond_colors[bond_idx] = c

        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        except Exception:
            drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        drawer.drawOptions().clearBackground = True

        highlight_radii = {i: 0.4 for i in highlight_atoms}
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors,
            highlightAtomRadii=highlight_radii,
            highlightBonds=highlight_bonds,
            highlightBondColors=highlight_bond_colors,
        )
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
        return PILImage.open(BytesIO(png)).copy()

    green = (0.13, 0.55, 0.13)
    red = (0.8, 0.1, 0.1)

    active_pos = np.clip(active_weights, 0, None)
    inactive_pos = np.clip(-inactive_weights, 0, None)
    dual_weights = active_pos - inactive_pos

    if style == "contour":
        from matplotlib.colors import LinearSegmentedColormap
        cmap_dual = LinearSegmentedColormap.from_list("dual", [red, (1, 1, 1), green])
        img = _render_contour(dual_weights, cmap_dual)
    elif style == "highlight":
        img = _render_highlight(dual_weights, green, red)
    else:
        raise ValueError(f"Unknown style: {style}. Use 'highlight' or 'contour'.")

    if save:
        img.save(save)
    return img
