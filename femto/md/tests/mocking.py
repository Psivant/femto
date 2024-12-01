"""Utilities for mocking common objects and data"""

from rdkit import Chem
from rdkit.Chem import AllChem

import femto.top


def build_mock_structure(smiles: list[str]) -> femto.top.Topology:
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
        topologies.append(femto.top.Topology.from_rdkit(molecule, residue_name))

    return femto.top.Topology.merge(*topologies)
