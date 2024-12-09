import collections

import mdtop
import numpy
import openmm.app
import openmm.unit
import pytest

import femto.fe.atm
import femto.md.config
import femto.md.prepare
from femto.fe.atm._setup import _offset_ligand, select_displacement, setup_system
from femto.fe.tests.systems import TEMOA_SYSTEM
from femto.md.constants import (
    LIGAND_1_RESIDUE_NAME,
    LIGAND_2_RESIDUE_NAME,
    OpenMMForceGroup,
)
from femto.md.tests.mocking import build_mock_structure
from femto.md.utils.openmm import all_close


@pytest.fixture
def mock_setup_config() -> femto.fe.atm.ATMSetupStage:
    return femto.fe.atm.ATMSetupStage(
        box_padding=10.0 * openmm.unit.angstrom,
        cation="Na+",
        displacement=38.0 * openmm.unit.angstrom,
        restraints=femto.fe.atm.ATMRestraints(receptor_query=":1"),
    )


@pytest.fixture
def temoa_ligand_1() -> mdtop.Topology:
    return femto.md.prepare.load_ligand(
        TEMOA_SYSTEM.ligand_1_coords, LIGAND_1_RESIDUE_NAME
    )


@pytest.fixture
def temoa_ligand_2() -> mdtop.Topology:
    return femto.md.prepare.load_ligand(
        TEMOA_SYSTEM.ligand_2_coords, LIGAND_2_RESIDUE_NAME
    )


@pytest.fixture
def temoa_receptor() -> mdtop.Topology:
    return femto.md.prepare.load_receptor(TEMOA_SYSTEM.receptor_coords)


def test_select_displacement():
    ligand_1 = build_mock_structure(["[Ar]"])

    receptor = build_mock_structure(["CC"])
    receptor.xyz = (
        numpy.array(
            [
                [+1, -1, -1],
                [+1, +1, -1],
                [-1, +1, -1],
                [-1, -1, +1],
                [-0.5, -0.5, -0.5],  # should be selected as furthest from ligand
                [+1, -1, +1],
                [+1, +1, +1],
                [-1, +1, +1],
            ]
        )
        * openmm.unit.angstrom
    )

    expected_distance = 10.0 * openmm.unit.angstrom

    displacement = select_displacement(receptor, ligand_1, None, expected_distance)

    expected_displacement = (
        numpy.array([-1, -1, -1]) / numpy.sqrt(3) * expected_distance
    )
    assert all_close(displacement, expected_displacement)


def test_offset_ligand():
    ligand = build_mock_structure(["[Ar]"])

    system = openmm.System()
    system.addParticle(1.0)
    force = openmm.NonbondedForce()
    force.addParticle(0.0, 1.0, 0.0)
    system.addForce(force)

    coords_0 = ligand.xyz.value_in_unit(openmm.unit.angstrom)
    offset = numpy.array([5.0, 4.0, 3.0])

    ligand_offset = _offset_ligand(ligand, offset * openmm.unit.angstrom)

    assert numpy.allclose(ligand.xyz.value_in_unit(openmm.unit.angstrom), coords_0)
    assert numpy.allclose(ligand_offset.xyz, coords_0 + offset)


def test_setup_system_abfe(temoa_ligand_1, temoa_receptor, mock_setup_config, mocker):
    n_ligand_atoms = len(temoa_ligand_1.atoms)
    n_receptor_atoms = len(temoa_receptor.atoms)

    def mock_prepare_system(receptor, lig_1, lig_2, *_, **__):
        assert lig_2 is None

        lig_1.residues[0].name = "L1"

        bound = lig_1 + receptor
        bound.box = numpy.eye(3) * 100.0 * openmm.unit.angstrom

        mock_system = openmm.System()
        nb_force = openmm.NonbondedForce()

        for _ in range(bound.n_atoms):
            mock_system.addParticle(1.0 * openmm.unit.amu)
            nb_force.addParticle(0.0, 1.0, 0.0)

        mock_system.addForce(nb_force)

        return bound, mock_system

    mock_prepare = mocker.patch(
        "femto.md.prepare.prepare_system",
        autospec=True,
        side_effect=mock_prepare_system,
    )

    mock_apply_hmr = mocker.patch("femto.md.prepare.apply_hmr", autospec=True)
    mock_apply_rest = mocker.patch("femto.md.rest.apply_rest", autospec=True)

    mock_setup_config.apply_rest = True

    expected_h_mass = 2.0 * openmm.unit.amu
    mock_setup_config.hydrogen_mass = expected_h_mass

    topology, system = setup_system(
        mock_setup_config,
        temoa_receptor,
        temoa_ligand_1,
        None,
        [],
        numpy.ones(3) * 22.0 * openmm.unit.angstrom,
        receptor_ref_query="resi 1",
        ligand_1_ref_query=None,
        ligand_2_ref_query=None,
    )

    assert isinstance(topology, mdtop.Topology)
    assert isinstance(system, openmm.System)

    mock_prepare.assert_called_once()
    mock_apply_hmr.assert_called_once_with(mocker.ANY, mocker.ANY, expected_h_mass)

    mock_apply_rest.assert_called_once()
    assert mock_apply_rest.call_args.args[1] == set(range(n_ligand_atoms))

    assert len(topology.atoms) == system.getNumParticles()
    assert len(topology[f"resn {LIGAND_1_RESIDUE_NAME}"].residues) == 1

    forces = collections.defaultdict(list)

    for force in system.getForces():
        forces[force.getForceGroup()].append(force)

    assert len(forces[OpenMMForceGroup.NONBONDED]) == 1
    nonbonded_force = forces[OpenMMForceGroup.NONBONDED][0]
    assert nonbonded_force.getNumParticles() == len(topology.atoms)

    assert len(forces[OpenMMForceGroup.COM_RESTRAINT]) == 1
    com_force = forces[OpenMMForceGroup.COM_RESTRAINT][0]

    assert isinstance(com_force, openmm.CustomCentroidBondForce)
    assert com_force.getNumBonds() == 1  # between ligand 1 com and receptor com

    expected_ligand_com_idxs = tuple(range(n_ligand_atoms))
    ligand_com_idxs, _ = com_force.getGroupParameters(0)
    assert ligand_com_idxs == expected_ligand_com_idxs

    expected_receptor_com_idxs = tuple(
        range(n_ligand_atoms, n_ligand_atoms + n_receptor_atoms)
    )
    receptor_com_idxs, _ = com_force.getGroupParameters(1)
    assert receptor_com_idxs == expected_receptor_com_idxs

    # make sure the receptor position restraints are applied to the right atoms
    restraint_forces = forces[OpenMMForceGroup.POSITION_RESTRAINT]
    assert len(restraint_forces) == 1

    restraint_force: openmm.CustomExternalForce = restraint_forces[0]
    assert restraint_force.getNumParticles() == n_receptor_atoms

    atom_idx, restraint_params = restraint_force.getParticleParameters(0)
    assert atom_idx == n_ligand_atoms

    x0, y0, z0, k, radius = restraint_params

    assert numpy.isclose(x0, 0.39454)
    assert numpy.isclose(y0, -0.02540)
    assert numpy.isclose(z0, 0.24646)

    expected_k = mock_setup_config.restraints.receptor.k.value_in_unit_system(
        openmm.unit.md_unit_system
    )
    expected_radius = (
        mock_setup_config.restraints.receptor.radius
    ).value_in_unit_system(openmm.unit.md_unit_system)

    assert numpy.isclose(k, expected_k)
    assert numpy.isclose(radius, expected_radius)


def test_setup_system_rbfe(
    temoa_ligand_1, temoa_ligand_2, temoa_receptor, mock_setup_config, mocker
):
    n_ligand_1_atoms = len(temoa_ligand_1.atoms)
    n_ligand_2_atoms = len(temoa_ligand_2.atoms)
    n_receptor_atoms = len(temoa_receptor.atoms)

    def mock_prepare_system(receptor, lig_1, lig_2, *_, **__):
        lig_1.residues[0].name = "L1"
        lig_2.residues[0].name = "R1"

        bound = lig_1 + lig_2 + receptor
        bound.box = numpy.eye(3) * 100.0 * openmm.unit.angstrom

        mock_system = openmm.System()

        for _ in range(bound.n_atoms):
            mock_system.addParticle(1.0 * openmm.unit.amu)

        return bound, mock_system

    mock_prepare = mocker.patch(
        "femto.md.prepare.prepare_system",
        autospec=True,
        side_effect=mock_prepare_system,
    )

    mock_setup_config.cation = "K+"

    topology, system = setup_system(
        mock_setup_config,
        receptor=temoa_receptor,
        receptor_ref_query="resi 1",
        ligand_1=temoa_ligand_1,
        ligand_1_ref_query=("index 1", "index 2", "index 3"),
        ligand_2=temoa_ligand_2,
        ligand_2_ref_query=("index 4", "index 5", "index 6"),
        cofactors=[],
        displacement=numpy.ones(3) * 22.0 * openmm.unit.angstrom,
    )

    mock_prepare.assert_called_once()

    assert len(topology[f"resn {LIGAND_1_RESIDUE_NAME}"].atoms) == n_ligand_1_atoms
    assert len(topology[f"resn {LIGAND_2_RESIDUE_NAME}"].atoms) == n_ligand_2_atoms

    forces = collections.defaultdict(list)

    for force in system.getForces():
        forces[force.getForceGroup()].append(force)

    assert len(forces[OpenMMForceGroup.COM_RESTRAINT]) == 2
    com_force_1 = forces[OpenMMForceGroup.COM_RESTRAINT][0]
    com_force_2 = forces[OpenMMForceGroup.COM_RESTRAINT][1]

    for com_force in (com_force_1, com_force_2):
        assert isinstance(com_force, openmm.CustomCentroidBondForce)
        # between ligand 1 com and receptor com or ligand 2 com and receptor com
        assert com_force.getNumBonds() == 1

    expected_ligand_1_com_idxs = tuple(range(n_ligand_1_atoms))
    ligand_1_com_idxs, _ = com_force_1.getGroupParameters(0)
    assert ligand_1_com_idxs == expected_ligand_1_com_idxs

    expected_ligand_2_com_idxs = tuple(
        range(n_ligand_1_atoms, n_ligand_1_atoms + n_ligand_2_atoms)
    )
    ligand_2_com_idxs, _ = com_force_2.getGroupParameters(0)
    assert ligand_2_com_idxs == expected_ligand_2_com_idxs

    expected_receptor_com_idxs = tuple(
        range(
            n_ligand_1_atoms + n_ligand_2_atoms,
            n_ligand_1_atoms + n_ligand_2_atoms + n_receptor_atoms,
        )
    )
    receptor_com_idxs, _ = com_force_1.getGroupParameters(1)
    assert receptor_com_idxs == expected_receptor_com_idxs
    receptor_com_idxs, _ = com_force_2.getGroupParameters(1)
    assert receptor_com_idxs == expected_receptor_com_idxs

    assert len(forces[OpenMMForceGroup.ALIGNMENT_RESTRAINT]) == 3

    ligand_2_center_orig = temoa_ligand_2.xyz.value_in_unit(openmm.unit.angstrom).mean(
        axis=0
    )
    ligand_2_center = (
        topology[f":{LIGAND_2_RESIDUE_NAME}"]
        .xyz.value_in_unit(openmm.unit.angstrom)
        .mean(axis=0)
    )

    for i in range(3):
        assert numpy.isclose(
            ligand_2_center[i] - ligand_2_center_orig[i], 22.0, atol=0.5
        )
