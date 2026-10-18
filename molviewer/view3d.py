"""
3D molecular viewing for Jupyter, Colab, and Streamlit.

Supports py3Dmol styles (stick, sphere, line, cartoon, ball_and_stick)
and Speck ambient-occlusion renderer with presets.
"""

from __future__ import annotations

import sys

from rdkit import Chem
from rdkit.Chem import AllChem

from ._mol import ensure_mol, ensure_conformer


def _detect_env():
    """Detect runtime environment: 'colab', 'jupyter', 'streamlit', or 'script'."""
    if "streamlit" in sys.modules:
        return "streamlit"
    try:
        if "google.colab" in sys.modules:
            return "colab"
        shell = get_ipython().__class__.__name__
        if shell in ("ZMQInteractiveShell", "Shell"):
            return "jupyter"
    except NameError:
        pass
    return "script"


def view3d(input_data, *, style="stick", size=(600, 400), spin=False,
           background="white", preset=None,
           atom_scale=None, bond_scale=None, ao=None, brightness=None,
           outline=None, bonds=None):
    """Interactive 3D molecular viewer.

    Auto-detects environment:
      - Jupyter/Colab: renders inline interactive widget
      - Streamlit: renders via st.components.v1.html()
      - Script: returns HTML string (py3Dmol) or widget (speck)

    Args:
        input_data: SMILES string or RDKit Mol.
        style: py3Dmol styles ("stick", "sphere", "line", "cartoon", "ball_and_stick")
               or "speck" for the Speck ambient-occlusion renderer.
        size: (width, height) in pixels.
        spin: if True, auto-rotate the molecule.
        background: background color (default "white", py3Dmol only).
        preset: Speck preset. molviewer presets: "ball_and_stick" (default look),
                "toon" (flat shading + outlines), "licorice" (stick model),
                "houkmol" (matte + outlines, Houk group style approximation).
                ipyspeck presets: "default", "matte", "glossy", "cover",
                "metal", "glass", "goodsell".
        atom_scale: Speck atom radius (0.0-1.0).
        bond_scale: Speck bond radius (0.0-1.0).
        ao: Speck ambient occlusion strength (0.0-1.0).
        brightness: Speck brightness (0.0-1.0).
        outline: Speck depth-aware outline strength (0.0-1.0).
        bonds: Speck show bonds (True/False).

    Returns:
        In script mode: HTML string (py3Dmol) or Speck widget.
        In notebook/Streamlit: displays inline.
    """
    if style == "speck":
        env = _detect_env()
        if env == "colab":
            import warnings
            warnings.warn(
                "Speck WebGL widget is not supported in Google Colab. "
                "Falling back to py3Dmol ball_and_stick style.",
                stacklevel=2,
            )
            style = "ball_and_stick"
        else:
            return _view3d_speck(input_data, size=size, spin=spin, preset=preset,
                                 atom_scale=atom_scale, bond_scale=bond_scale,
                                 ao=ao, brightness=brightness, outline=outline,
                                 bonds=bonds)

    try:
        import py3Dmol
    except ImportError:
        raise ImportError(
            "py3Dmol is required for 3D viewing. "
            "Install it with: pip install py3Dmol"
        )

    mol = ensure_mol(input_data)
    mol = ensure_conformer(mol)
    mol_block = Chem.MolToMolBlock(mol)

    viewer = py3Dmol.view(width=size[0], height=size[1])
    viewer.addModel(mol_block, "mol")

    style_spec = _build_style(style)
    viewer.setStyle(style_spec)
    viewer.setBackgroundColor(background)
    if spin:
        viewer.spin(True)
    viewer.zoomTo()

    env = _detect_env()

    if env in ("jupyter", "colab"):
        return viewer.show()
    elif env == "streamlit":
        import streamlit.components.v1 as components
        html = viewer._make_html()
        components.html(html, height=size[1] + 20, width=size[0] + 20)
    else:
        return viewer._make_html()


# ── Speck presets ─────────────────────────────────────────────────────────────

_MOLVIEWER_PRESETS = {
    "ball_and_stick": {
        "atomScale": 0.24, "relativeAtomScale": 0.64, "atomShade": 0.5,
        "bonds": True, "bondScale": 0.5, "bondShade": 0.5,
        "ao": 0.75, "brightness": 0.5, "outline": 0.0,
        "dofStrength": 0.0,
    },
    "toon": {
        "atomScale": 0.24, "relativeAtomScale": 0.64, "atomShade": 0.5,
        "bonds": True, "bondScale": 0.5, "bondShade": 0.5,
        "ao": 0.0, "brightness": 0.5, "outline": 1.0,
        "dofStrength": 0.0,
    },
    "licorice": {
        "atomScale": 0.1, "relativeAtomScale": 0.0, "atomShade": 0.5,
        "bonds": True, "bondScale": 1.0, "bondShade": 0.5,
        "ao": 0.75, "brightness": 0.5, "outline": 0.0,
        "dofStrength": 0.0,
    },
    "houkmol": {
        "atomScale": 0.24, "relativeAtomScale": 0.64, "atomShade": 0.5,
        "bonds": True, "bondScale": 0.5, "bondShade": 0.1,
        "ao": 0.5, "brightness": 0.65, "outline": 1.0,
        "outlineColor": "#000000", "outlineWidth": 2.5,
        "outlineMode": "depth",
        "gloss": 0.0, "metallic": 0.0, "specular": 0.1,
        "saturation": 0.7, "dofStrength": 0.0, "fog": 0.0,
        "fxaa": 1, "shadows": 0.0, "rim": 0.0,
    },
}


def _view3d_speck(input_data, *, size=(600, 400), spin=False, preset=None,
                  atom_scale=None, bond_scale=None, ao=None, brightness=None,
                  outline=None, bonds=None):
    """Render molecule with the Speck ambient-occlusion renderer."""
    try:
        from ipyspeck import Speck
    except ImportError:
        raise ImportError(
            "ipyspeck is required for style='speck'. "
            "Install it with: pip install ipyspeck"
        )

    mol = ensure_mol(input_data)
    mol = Chem.AddHs(mol)
    mol = ensure_conformer(mol)

    w = Speck.from_rdkit(mol)
    w.layout.width = f"{size[0]}px"
    w.layout.height = f"{size[1]}px"

    # Apply preset (default to ball_and_stick if none specified)
    active_preset = preset if preset else "ball_and_stick"
    if active_preset in _MOLVIEWER_PRESETS:
        for key, value in _MOLVIEWER_PRESETS[active_preset].items():
            setattr(w, key, value)
    else:
        try:
            w.apply_preset(active_preset)
        except (ValueError, Exception):
            pass

    # Apply user-specified parameters (override preset/defaults)
    if atom_scale is not None:
        w.atomScale = atom_scale
    if bond_scale is not None:
        w.bondScale = bond_scale
    if ao is not None:
        w.ao = ao
    if brightness is not None:
        w.brightness = brightness
    if outline is not None:
        w.outline = outline
    if bonds is not None:
        w.bonds = bonds
    if spin:
        w.autoRotate = True

    env = _detect_env()
    if env == "streamlit":
        try:
            import stspeck
            mol_block = Chem.MolToMolBlock(mol)
            kwargs = {"data": mol_block}
            if preset:
                kwargs["preset"] = preset
            stspeck.speck(**kwargs, height=size[1])
            return
        except ImportError:
            pass

    return w


def _build_style(style):
    """Convert style name to py3Dmol style dict."""
    styles = {
        "stick": {"stick": {}},
        "sphere": {"sphere": {"radius": 0.4}},
        "line": {"line": {}},
        "cartoon": {"cartoon": {}},
        "ball_and_stick": {"stick": {}, "sphere": {"radius": 0.3}},
    }
    if style in styles:
        return styles[style]
    return {"stick": {}}
