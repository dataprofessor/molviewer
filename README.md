# molviewer

**2D and 3D molecular visualization for Jupyter, Colab, and Streamlit.**

## Install

```bash
pip install molviewer
```

## Quick Start

```python
import molviewer

# 2D drawing
molviewer.draw("CC(=O)Oc1ccccc1C(=O)O")
molviewer.draw(["CCO", "c1ccccc1"], legends=["Ethanol", "Benzene"])

# 3D viewing — py3Dmol
molviewer.view3d("CC(=O)Oc1ccccc1C(=O)O", style="stick")
molviewer.view3d("CC(=O)Oc1ccccc1C(=O)O", style="ball_and_stick", spin=True)

# 3D viewing — Speck (ambient occlusion)
molviewer.view3d("CC(=O)Oc1ccccc1C(=O)O", style="speck")
molviewer.view3d("CC(=O)Oc1ccccc1C(=O)O", style="speck", preset="houkmol")
molviewer.view3d("CC(=O)Oc1ccccc1C(=O)O", style="speck", preset="toon")
molviewer.view3d("CC(=O)Oc1ccccc1C(=O)O", style="speck", preset="licorice")
```

## Speck Presets

| Preset | Description |
|--------|-------------|
| `ball_and_stick` | Default — atoms + bonds with ambient occlusion |
| `toon` | Flat shading with depth-aware outlines |
| `licorice` | Thick bonds, small atoms |
| `houkmol` | Houk group style — matte, outlines, bright atoms |
| `metal` | Metallic material (ipyspeck) |
| `glass` | Transparent glass (ipyspeck) |
| `glossy` | Shiny surface (ipyspeck) |
| `goodsell` | David Goodsell style (ipyspeck) |

## Custom Parameters

```python
molviewer.view3d("CCO", style="speck", atom_scale=0.6, ao=0.9, outline=1.0)
```

Parameters: `atom_scale`, `bond_scale`, `ao`, `brightness`, `outline`, `bonds`.

## Attribution Maps (requires molfp)

```bash
pip install molviewer[attribution]
```

```python
import molfp

result = molfp.interpret(smiles, activities, fp_type="maccs")
molviewer.draw_attribution("CC(=O)Oc1ccccc1C(=O)O", result, style="highlight", save="attr.png")
molviewer.draw_attribution("CC(=O)Oc1ccccc1C(=O)O", result, style="contour", save="attr.png")
```

## License

MIT
