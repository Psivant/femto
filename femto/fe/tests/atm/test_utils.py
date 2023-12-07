import numpy
import openmm
import openmm.unit
import pytest

import femto.fe.atm
import femto.fe.config
import femto.md.config
from femto.fe.atm._utils import create_alignment_restraints, create_atm_force
from femto.md.constants import (
    LIGAND_1_RESIDUE_NAME,
    LIGAND_2_RESIDUE_NAME,
    OpenMMForceGroup,
)
from femto.md.tests.mocking import build_mock_structure
from femto.md.utils.openmm import assign_force_groups


@pytest.fixture
def mock_system() -> openmm.System:
    system = openmm.System()

    for _ in range(4):
        system.addParticle(17.0)

    bond_force = openmm.HarmonicBondForce()
    bond_force.addBond(0, 1, 1.0, 100.0)
    system.addForce(bond_force)

    angle_force = openmm.HarmonicAngleForce()
    angle_force.addAngle(0, 1, 2, 1.25, 200.0)
    system.addForce(angle_force)

    angle_force = openmm.PeriodicTorsionForce()
    angle_force.addTorsion(0, 1, 2, 3, 1, 3.14, 300.0)
    system.addForce(angle_force)

    nonbonded_force = openmm.NonbondedForce()

    for i in range(system.getNumParticles()):
        nonbonded_force.addParticle(0.0, 1.0, i / 10.0)

    atm_force = openmm.ATMForce(0.0, 0.0, 0.0, 0.1, 0.0, 110.0, 0.1, 1.0 / 16.0, 1)
    atm_force.addForce(nonbonded_force)
    system.addForce(atm_force)

    assign_force_groups(system)

    return system


def test_create_alignment_restraints():
    forces = create_alignment_restraints(
        (0, 1, 2),
        (3, 4, 5),
        1.234 * openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom**2,
        4.321 * openmm.unit.kilocalorie_per_mole,
        5.7654 * openmm.unit.kilocalorie_per_mole,
        [-0.1, -0.2, -0.3] * openmm.unit.angstrom,
    )

    coords = numpy.array(
        [
            [-3.4, -1.1, 0.0],
            [-2.1, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
            [4.0, 0.23, 0.1],
            [4.20, 2.34, 0.0],
        ]
    )

    # computed with openmm-atmmetaforce-plugin ==0.3.5
    expected_energies = [
        71.74066162109375 * openmm.unit.kilojoules_per_mole,
        1.6197073459625244 * openmm.unit.kilojoules_per_mole,
        12.197089195251465 * openmm.unit.kilojoules_per_mole,
    ]

    for force, expected_energy in zip(forces, expected_energies, strict=True):
        system = openmm.System()

        for _ in range(6):
            system.addParticle(1.0)

        system.addForce(force)

        ctx = openmm.Context(system, openmm.VerletIntegrator(0.0001))
        ctx.setPositions(coords * openmm.unit.angstrom)

        actual_energy = ctx.getState(getEnergy=True).getPotentialEnergy()
        assert femto.md.utils.openmm.is_close(actual_energy, expected_energy)


def test_create_atm_force():
    receptor = build_mock_structure(["CC"])

    ligand_1 = build_mock_structure(["[Cl-]"])
    ligand_1.residues[0].name = LIGAND_1_RESIDUE_NAME

    ligand_2 = build_mock_structure(["[Na+]"])
    ligand_2.residues[0].name = LIGAND_2_RESIDUE_NAME

    topology = receptor + ligand_2 + ligand_1

    expected_u_max = 10.0 * openmm.unit.kilojoules_per_mole
    expected_u0 = 2.0 * openmm.unit.kilojoules_per_mole
    expected_a = 3.0

    config = femto.fe.atm.ATMSoftCore(
        u_max=expected_u_max, u0=expected_u0, a=expected_a
    )

    expected_offset = numpy.array([0.1, 0.2, 0.3]) * openmm.unit.nanometers

    force = create_atm_force(topology, config, expected_offset)

    assert isinstance(force, openmm.ATMForce)
    assert force.getForceGroup() == OpenMMForceGroup.ATM

    assert force.getNumParticles() == len(topology.atoms)

    global_parameters = {
        force.getGlobalParameterName(i): force.getGlobalParameterDefaultValue(i)
        for i in range(force.getNumGlobalParameters())
    }

    assert numpy.isclose(
        global_parameters[openmm.ATMForce.Umax()],
        expected_u_max.value_in_unit_system(openmm.unit.md_unit_system),
    )
    assert numpy.isclose(
        global_parameters[openmm.ATMForce.Ubcore()],
        expected_u0.value_in_unit_system(openmm.unit.md_unit_system),
    )
    assert numpy.isclose(
        global_parameters[openmm.ATMForce.Acore()],
        expected_a,
    )

    # TODO: fix this test when OpenMM #4313 is resolved
    # for i in range(len(receptor.atoms)):
    #     disp_a, disp_b = openmm.Vec3(0, 0, 0), openmm.Vec3(0, 0, 0)
    #
    #     idx, dx, dy, dz = force.getParticleParameters(i, disp_a, disp_b)
    #     assert idx == i
    #
    #     assert numpy.isclose(dx, 0.0)
    #     assert numpy.isclose(dy, 0.0)
    #     assert numpy.isclose(dz, 0.0)
    #
    # for i in range(len(receptor.atoms), len(receptor.atoms) + len(ligand_2.atoms)):
    #
    #     (dx, dy, dz), _ = force.getParticleParameters(i)
    #     assert idx == i
    #
    #     assert numpy.isclose(
    #         dx, -expected_offset[0].value_in_unit_system(openmm.unit.md_unit_system)
    #     )
    #     assert numpy.isclose(
    #         dy, -expected_offset[1].value_in_unit_system(openmm.unit.md_unit_system)
    #     )
    #     assert numpy.isclose(
    #         dz, -expected_offset[2].value_in_unit_system(openmm.unit.md_unit_system)
    #     )
    #
    # for i in range(
    #     len(receptor.atoms) + len(ligand_2.atoms),
    #     len(receptor.atoms) + len(ligand_2.atoms) + len(ligand_1.atoms),
    # ):
    #     (dx, dy, dz), _ = force.getParticleParameters(i)
    #     assert idx == i
    #
    #     assert numpy.isclose(
    #         dx, expected_offset[0].value_in_unit_system(openmm.unit.md_unit_system)
    #     )
    #     assert numpy.isclose(
    #         dy, expected_offset[1].value_in_unit_system(openmm.unit.md_unit_system)
    #     )
    #     assert numpy.isclose(
    #         dz, expected_offset[2].value_in_unit_system(openmm.unit.md_unit_system)
    #     )
