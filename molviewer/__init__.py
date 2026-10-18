"""
molviewer — 2D and 3D molecular visualization for Jupyter, Colab, and Streamlit.

2D drawing with RDKit, 3D viewing with py3Dmol and Speck (ambient occlusion).
"""

__version__ = "0.1.1"

from .draw2d import draw
from .view3d import view3d
from .attribution import draw_attribution
