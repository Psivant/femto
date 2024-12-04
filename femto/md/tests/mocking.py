"""Utilities for mocking common objects and data"""

import mdtop
from rdkit import Chem
from rdkit.Chem import AllChem


def build_mock_structure(smiles: list[str]) -> mdtop.Topology:
    """Build a mock structure from a list of SMILES patterns

    Notes:
        * A conformer is generated for each molecule.

    Args:
        smiles: A list of SMILES patterns.

    Returns:
        The mock structure.
    """
    molecules = [Chem.MolFromSmiles(pattern) for pattern in smiles]
    topologies = []

    for molecule, pattern in zip(molecules, smiles, strict=True):
        assert molecule is not None, f"{pattern} is not a valid SMILES pattern"

        molecule = Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule)

        is_water = Chem.MolToSmiles(Chem.RemoveHs(molecule)) == "O"

        residue_name = (
            "WAT"
            if is_water
            else (
                f"{molecule.GetAtomWithIdx(0).GetSymbol().upper()}"
                if molecule.GetNumAtoms() == 1
                else "UNK"
            )
        )
        topologies.append(mdtop.Topology.from_rdkit(molecule, residue_name))

    return mdtop.Topology.merge(*topologies)
