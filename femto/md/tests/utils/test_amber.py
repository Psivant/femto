import numpy
import openmm.app
import parmed.openmm
import pytest

from femto.md.tests.mocking import build_mock_structure
from femto.md.utils.amber import extract_water_and_ions_mask, parameterize_structure


def test_extract_water_and_ions_mask(tmp_cwd):
    structure = build_mock_structure(["O", "O", "CC", "[K+]", "[Cl-]", "O", "O", "O"])

    solvent_mask = extract_water_and_ions_mask(structure)
    assert len(solvent_mask) == len(structure.atoms)

    n_ethanol_atoms = 8
    expected_mask = (
        [True] * 3
        + [True] * 3
        + [False] * n_ethanol_atoms
        + [True] * 1
        + [True] * 1
        + [True] * 3
        + [True] * 3
        + [True] * 3
    )

    assert solvent_mask == expected_mask


def test_parameterize_empty_structure(tmp_cwd):
    structure = parameterize_structure(parmed.Structure(), [])
    assert isinstance(structure, parmed.Structure)


def test_parameterize_structure_fail(tmp_cwd):
    topology = openmm.app.Topology()

    residue = topology.addResidue("X", topology.addChain())
    topology.addAtom("X", openmm.app.Element.getBySymbol("Na"), residue)

    with pytest.raises(RuntimeError, match="TLeap failed - "):
        parameterize_structure(
            parmed.openmm.load_topology(topology, xyz=numpy.zeros((1, 3))), []
        )
