import numpy
import openmm.app
import openmm.unit
import scipy.spatial.distance

import femto.md.config
from femto.md.solvate import _compute_box_size, solvate_system
from femto.md.tests.mocking import build_mock_structure
from femto.md.utils import openmm as openmm_utils


def test_compute_box_size():
    box_size = _compute_box_size(
        None,
        build_mock_structure(["[Ar]"]),
        build_mock_structure(["[Ar]"]),
        10.0 * openmm.unit.angstrom,
        numpy.array([0.0, 10.0, 0.0]) * openmm.unit.angstrom,
        numpy.array([0.0, 0.0, 20.0]) * openmm.unit.angstrom,
    )

    assert openmm_utils.all_close(
        box_size, numpy.array([20.0, 30.0, 40.0]) * openmm.unit.angstrom
    )


def test_solvate_system(mocker):
    receptor = build_mock_structure(["O", "CC"])
    receptor.residues[-1].name = "RECEPTOR"

    ligand_1 = build_mock_structure(["CO"])
    ligand_1.residues[-1].name = "L1"
    ligand_2 = build_mock_structure(["CCl"])
    ligand_2.residues[-1].name = "L2"

    box_size = 2.0

    topology_input = ligand_1 + ligand_2 + receptor
    topology_input.box_vectors = box_size * numpy.eye(3) * openmm.unit.angstrom

    mock_modeller = mocker.patch("openmm.app.Modeller", autospec=True)
    mock_modeller.return_value.topology = topology_input.topology
    mock_modeller.return_value.positions = topology_input.positions

    mock_parameterize = mocker.patch(
        "femto.md.utils.amber.parameterize_structure",
        side_effect=lambda structure, *_: structure,
        autospec=True,
    )

    topology = solvate_system(receptor, ligand_1, ligand_2, femto.md.config.Solvent())
    assert len(topology.atoms) == len(topology_input.atoms)

    # waters should be grouped at the end
    expected_residues = ["L1", "L2", "RECEPTOR", "WAT"]
    assert [r.name for r in topology.residues] == expected_residues

    com = numpy.array(topology.coordinates).mean(axis=0)
    assert numpy.allclose(com, numpy.ones(3) * box_size * 0.5, atol=0.1)

    mock_parameterize.assert_called_once()
    mock_modeller.return_value.addSolvent.assert_called_once()


def test_solvate_system_with_cavities(mocker):
    # define the mock system with well define 'rmin' values which are used to determine
    # radius of our dummy ligands.
    r_min = 10.0
    sigma = r_min / (2.0 ** (1.0 / 6.0)) * openmm.unit.angstrom

    mock_system = openmm.System()
    mock_system.addParticle(1.0)
    mock_system.addParticle(1.0)
    mock_system.addParticle(1.0)
    mock_system.addParticle(1.0)
    mock_force = openmm.NonbondedForce()
    mock_force.addParticle(0.0, sigma, 1.0)
    mock_force.addParticle(0.0, sigma, 1.0)
    mock_force.addParticle(0.0, sigma, 1.0)
    mock_force.addParticle(0.0, sigma, 1.0)
    mock_system.addForce(mock_force)

    mocker.patch(
        "femto.md.solvate._MockForceField.createSystem",
        return_value=mock_system,
    )

    ligand_1 = build_mock_structure(["ClCl"])
    ligand_1.coordinates = numpy.array([[-10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
    ligand_2 = build_mock_structure(["ClCl"])
    ligand_2.coordinates = numpy.array([[+10.0, 0.0, 0.0], [+10.0, 0.0, 0.0]])

    topology = solvate_system(
        None, ligand_1, None, femto.md.config.Solvent(), None, None, [ligand_2]
    )
    assert len(topology["!:WAT"].residues) == 1  # no cavity ligand

    com_offset = topology["@1"].coordinates[0, :] - ligand_1["@1"].coordinates[0, :]

    cavity_1_center = ligand_1["@1"].coordinates[0, :] + com_offset
    cavity_2_center = ligand_2["@1"].coordinates[0, :] + com_offset

    water_distances_1 = scipy.spatial.distance.cdist(
        cavity_1_center.reshape(1, -1), topology[":WAT"].coordinates
    )
    water_distances_2 = scipy.spatial.distance.cdist(
        cavity_2_center.reshape(1, -1), topology[":WAT"].coordinates
    )

    min_water_distance_1 = water_distances_1.min()
    min_water_distance_2 = water_distances_2.min()

    assert r_min * 0.5 + 1.5 > min_water_distance_1 > r_min * 0.5
    assert r_min * 0.5 + 1.5 > min_water_distance_2 > r_min * 0.5
