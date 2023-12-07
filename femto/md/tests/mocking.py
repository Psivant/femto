"""Utilities for mocking common objects and data"""
import collections
import tempfile

import parmed
from rdkit import Chem
from rdkit.Chem import AllChem


def build_mock_structure(smiles: list[str]) -> parmed.Structure:
    """Build a mock structure from a list of SMILES patterns

    Notes:
        * A conformer is generated for each molecule.

    Args:
        smiles: A list of SMILES patterns.

    Returns:
        The mock structure.
    """
    molecules = [Chem.MolFromSmiles(pattern) for pattern in smiles]

    for molecule, pattern in zip(molecules, smiles, strict=True):
        assert molecule is not None, f"{pattern} is not a valid SMILES pattern"

    complex = Chem.Mol()

    for i, molecule in enumerate(molecules):
        molecule = Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule)

        is_water = Chem.MolToSmiles(Chem.RemoveHs(molecule)) == "O"

        residue_name = (
            "WAT"
            if is_water
            else (
                f"{molecule.GetAtomWithIdx(0).GetSymbol()}"
                if molecule.GetNumAtoms() == 1
                else "UNK"
            )
        )
        symbol_count = collections.defaultdict(int)

        for atom in molecule.GetAtoms():
            atom_name = f"{atom.GetSymbol()}{symbol_count[atom.GetSymbol()] + 1}"
            atom_info = Chem.AtomPDBResidueInfo(
                atom_name.ljust(4, " "), atom.GetIdx(), "", residue_name, i
            )
            atom.SetMonomerInfo(atom_info)

            symbol_count[atom.GetSymbol()] += 1

        complex = Chem.CombineMols(complex, molecule)

    with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp_file:
        Chem.MolToPDBFile(complex, tmp_file.name)
        structure = parmed.load_file(tmp_file.name, structure=True)

    return structure
