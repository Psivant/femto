import numpy
import openmm
import openmm.app
import openmm.unit
import scipy.spatial.distance

import femto.md.config
import femto.top
from femto.fe.tests.systems import CDK2_SYSTEM, TEMOA_SYSTEM
from femto.md.constants import LIGAND_1_RESIDUE_NAME, LIGAND_2_RESIDUE_NAME
from femto.md.prepare import (
    _compute_box_size,
    apply_hmr,
    load_ligand,
    load_ligands,
    load_receptor,
    prepare_system,
)
from femto.md.tests.mocking import build_mock_structure
from femto.md.utils import openmm as openmm_utils
from femto.md.utils.openmm import is_close


def test_hmr():
    topology = build_mock_structure(["CC"])

    system = openmm.System()
    system.addParticle(12.0 * openmm.unit.amu)
    system.addParticle(12.0 * openmm.unit.amu)
    system.addParticle(1.0 * openmm.unit.amu)
    system.addParticle(1.0 * openmm.unit.amu)
    system.addParticle(1.0 * openmm.unit.amu)
    system.addParticle(1.0 * openmm.unit.amu)
    system.addParticle(1.0 * openmm.unit.amu)
    system.addParticle(1.0 * openmm.unit.amu)

    original_mass = sum(
        [system.getParticleMass(i) for i in range(system.getNumParticles())],
        0.0 * openmm.unit.amu,
    )

    expected_h_mass = 1.5 * openmm.unit.amu
    apply_hmr(system, topology, hydrogen_mass=expected_h_mass)

    new_masses = [system.getParticleMass(i) for i in range(system.getNumParticles())]
    new_mass = sum(new_masses, 0.0 * openmm.unit.amu)

    assert is_close(new_mass, original_mass)

    expected_masses = [
        (12.0 - 0.5 * 3.0) * openmm.unit.amu,
        (12.0 - 0.5 * 3.0) * openmm.unit.amu,
    ] + ([expected_h_mass] * 6)

    assert all(
        is_close(new_mass, expected_mass)
        for new_mass, expected_mass in zip(new_masses, expected_masses, strict=True)
    )


def test_hmr_water():
    """HMR should not modify water molecules."""
    topology = build_mock_structure(["O"])

    expected_masses = [16.0, 1.0, 1.0] * openmm.unit.amu

    system = openmm.System()

    for mass in expected_masses:
        system.addParticle(mass)

    apply_hmr(system, topology)

    new_masses = [system.getParticleMass(i) for i in range(system.getNumParticles())]

    assert all(
        is_close(new_mass, expected_mass)
        for new_mass, expected_mass in zip(new_masses, expected_masses, strict=True)
    )


def test_load_ligand():
    reside_name = "ABC"

    ligand = load_ligand(TEMOA_SYSTEM.ligand_1_coords, reside_name)
    assert ligand[f"r. {reside_name}"].n_atoms == ligand.n_atoms


def test_load_ligands():
    coord_path = CDK2_SYSTEM.ligand_1_coords

    ligand_1, ligand_2 = load_ligands(coord_path, None)
    assert ligand_2 is None

    assert ligand_1.residues[0].name == LIGAND_1_RESIDUE_NAME

    ligand_1, ligand_2 = load_ligands(coord_path, coord_path)
    assert isinstance(ligand_2, femto.top.Topology)

    assert ligand_1.residues[0].name == LIGAND_1_RESIDUE_NAME
    assert ligand_2.residues[0].name == LIGAND_2_RESIDUE_NAME


def test_load_receptor():
    receptor = load_receptor(CDK2_SYSTEM.receptor_coords)
    assert isinstance(receptor, femto.top.Topology)
    assert receptor.residues[0].name == "ACE"


def test_compute_box_size():
    box_size = _compute_box_size(
        None,
        build_mock_structure(["[Ar]"]),
        build_mock_structure(["[Ar]"]),
        [],
        10.0 * openmm.unit.angstrom,
        numpy.array([0.0, 10.0, 0.0]) * openmm.unit.angstrom,
        numpy.array([0.0, 0.0, 20.0]) * openmm.unit.angstrom,
    )

    assert openmm_utils.all_close(
        box_size, numpy.array([23.76, 33.76, 43.76]) * openmm.unit.angstrom
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
    topology_input.box = box_size * numpy.eye(3) * openmm.unit.angstrom

    mock_modeller = mocker.patch("openmm.app.Modeller", autospec=True)
    mock_modeller.return_value.topology = topology_input.to_openmm()
    mock_modeller.return_value.positions = topology_input.xyz

    mock_create_system = mocker.patch(
        "openmm.app.ForceField.createSystem",
        autospec=True,
        return_value=openmm.System(),
    )

    topology, system = prepare_system(
        receptor,
        ligand_1,
        ligand_2,
        femto.md.config.Solvent(default_ligand_ff="openff-2.0.0.offxml"),
    )
    assert isinstance(system, openmm.System)

    assert topology.n_atoms == topology_input.n_atoms

    # waters should be grouped at the end
    expected_residues = ["L1", "L2", "WAT", "RECEPTOR"]
    assert [r.name for r in topology.residues] == expected_residues

    com = topology.xyz.value_in_unit(openmm.unit.angstrom).mean(axis=0)
    assert numpy.allclose(com, numpy.ones(3) * box_size * 0.5, atol=0.1)

    mock_modeller.return_value.addSolvent.assert_called_once()
    mock_create_system.assert_called_once()


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
        "openmm.app.ForceField.createSystem",
        return_value=mock_system,
    )
    mocker.patch("openmm.app.Modeller.addExtraParticles", autospec=True)

    ligand_1 = build_mock_structure(["ClCl"])
    ligand_1.xyz = (
        numpy.array([[-10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]) * openmm.unit.angstrom
    )
    ligand_2 = build_mock_structure(["ClCl"])
    ligand_2.xyz = (
        numpy.array([[+10.0, 0.0, 0.0], [+10.0, 0.0, 0.0]]) * openmm.unit.angstrom
    )

    topology, _ = prepare_system(
        None, ligand_1, None, femto.md.config.Solvent(), [], None, None, [ligand_2]
    )
    assert topology["! r. HOH"].n_residues == 1  # no cavity ligand

    com_offset = topology["index 1"].xyz[0, :] - ligand_1["index 1"].xyz[0, :]

    cavity_1_center = (ligand_1[0].xyz[0, :] + com_offset).value_in_unit(
        openmm.unit.angstrom
    )
    cavity_2_center = (ligand_2[0].xyz[0, :] + com_offset).value_in_unit(
        openmm.unit.angstrom
    )

    water_distances_1 = scipy.spatial.distance.cdist(
        cavity_1_center.reshape(1, -1),
        topology[":HOH"].xyz.value_in_unit(openmm.unit.angstrom),
    )
    water_distances_2 = scipy.spatial.distance.cdist(
        cavity_2_center.reshape(1, -1),
        topology[":HOH"].xyz.value_in_unit(openmm.unit.angstrom),
    )

    min_water_distance_1 = water_distances_1.min()
    min_water_distance_2 = water_distances_2.min()

    assert r_min * 0.5 + 1.5 > min_water_distance_1 > r_min * 0.5
    assert r_min * 0.5 + 1.5 > min_water_distance_2 > r_min * 0.5
