"""Tests for molviewer."""

import pytest
import molviewer


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL = "CCO"
BENZENE = "c1ccccc1"
SMILES_SET = [ETHANOL, BENZENE, ASPIRIN]


class TestDraw2D:

    def test_single_molecule(self):
        img = molviewer.draw(ASPIRIN)
        assert img is not None
        assert hasattr(img, 'size')

    def test_batch(self):
        img = molviewer.draw(SMILES_SET)
        assert img is not None

    def test_with_legends(self):
        img = molviewer.draw(SMILES_SET, legends=["Ethanol", "Benzene", "Aspirin"])
        assert img is not None

    def test_custom_size(self):
        img = molviewer.draw(ASPIRIN, size=(500, 500))
        assert img is not None

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError, match="Invalid SMILES"):
            molviewer.draw("NOT_A_SMILES")


class TestView3D:

    def test_py3dmol_stick(self):
        result = molviewer.view3d(ASPIRIN, style="stick")
        assert result is not None

    def test_py3dmol_ball_and_stick(self):
        result = molviewer.view3d(ASPIRIN, style="ball_and_stick")
        assert result is not None

    def test_speck_default(self):
        w = molviewer.view3d(ASPIRIN, style="speck")
        assert w is not None
        assert w.atomScale == 0.24
        assert w.bonds is True

    def test_speck_ball_and_stick(self):
        w = molviewer.view3d(ASPIRIN, style="speck", preset="ball_and_stick")
        assert w.ao == 0.75

    def test_speck_toon(self):
        w = molviewer.view3d(ASPIRIN, style="speck", preset="toon")
        assert w.outline == 1.0
        assert w.ao == 0.0

    def test_speck_licorice(self):
        w = molviewer.view3d(ASPIRIN, style="speck", preset="licorice")
        assert w.atomScale == 0.1
        assert w.bondScale == 1.0

    def test_speck_houkmol(self):
        w = molviewer.view3d(ASPIRIN, style="speck", preset="houkmol")
        assert w.outline == 1.0
        assert w.saturation == 0.7

    def test_speck_custom_params(self):
        w = molviewer.view3d(ASPIRIN, style="speck", atom_scale=0.5, ao=0.9)
        assert w.atomScale == 0.5
        assert w.ao == 0.9

    def test_speck_spin(self):
        w = molviewer.view3d(ASPIRIN, style="speck", spin=True)
        assert w.autoRotate is True

    def test_speck_preset_override(self):
        w = molviewer.view3d(ASPIRIN, style="speck", preset="toon", atom_scale=0.4)
        assert w.atomScale == 0.4
        assert w.outline == 1.0


class TestMolUtilities:

    def test_ensure_mol_from_smiles(self):
        from molviewer._mol import ensure_mol
        mol = ensure_mol("CCO")
        assert mol is not None
        assert mol.GetNumAtoms() == 3

    def test_ensure_mol_from_mol(self):
        from rdkit import Chem
        from molviewer._mol import ensure_mol
        mol = Chem.MolFromSmiles("CCO")
        result = ensure_mol(mol)
        assert result is mol

    def test_ensure_mol_invalid(self):
        from molviewer._mol import ensure_mol
        with pytest.raises(ValueError):
            ensure_mol("INVALID")

    def test_ensure_conformer(self):
        from molviewer._mol import ensure_mol, ensure_conformer
        mol = ensure_mol("CCO")
        mol = ensure_conformer(mol)
        assert mol.GetNumConformers() > 0

    def test_extract_smiles_string(self):
        from molviewer._mol import extract_smiles
        data, is_batch = extract_smiles("CCO")
        assert data == "CCO"
        assert is_batch is False

    def test_extract_smiles_list(self):
        from molviewer._mol import extract_smiles
        data, is_batch = extract_smiles(["CCO", "c1ccccc1"])
        assert len(data) == 2
        assert is_batch is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
