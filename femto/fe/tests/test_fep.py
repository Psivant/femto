import numpy
import openmm
import openmm.app
import openmm.unit
import pytest

import femto.fe.config
import femto.md.utils.openmm
from femto.fe.fep import (
    LAMBDA_CHARGES_LIGAND_1,
    LAMBDA_CHARGES_LIGAND_2,
    LAMBDA_VDW_LIGAND_1,
    LAMBDA_VDW_LIGAND_2,
    _convert_intramolecular_interactions,
    apply_fep,
)

KJ_PER_MOL = openmm.unit.kilojoule_per_mole


E_CHARGE = 1.602176634e-19 * openmm.unit.coulomb
EPSILON0 = (
    1e-6
    * 8.8541878128e-12
    / (openmm.unit.AVOGADRO_CONSTANT_NA * E_CHARGE**2)
    * openmm.unit.farad
    / openmm.unit.meter
)
ONE_4PI_EPS0 = 1 / (4 * numpy.pi * EPSILON0) * EPSILON0.unit * 10.0  # nm -> angstrom


@pytest.fixture
def fep_config() -> femto.fe.config.FEP:
    return femto.fe.config.FEP(ligands_can_interact=False)


def compute_interaction_energy(
    epsilon,
    sigma,
    charge,
    distance,
    lambda_vdw: float = 1.0,
    lambda_charges: float = 1.0,
):
    r_electrostatics = distance
    r_vdw = (0.5 * sigma**6 * (1.0 - lambda_vdw) + distance**6) ** (1.0 / 6.0)

    return (
        # vdw
        4.0 * lambda_vdw * epsilon * ((sigma / r_vdw) ** 12 - (sigma / r_vdw) ** 6)
        # electrostatics
        + ONE_4PI_EPS0 * lambda_charges * charge / r_electrostatics
    ) * KJ_PER_MOL


@pytest.fixture
def three_particle_system():
    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    force.setUseDispersionCorrection(False)

    charges = 0.1, 0.2, -0.3
    sigmas = 1.1, 1.2, 1.3
    epsilons = 210, 220, 230

    force.addParticle(charges[0], sigmas[0] * openmm.unit.angstrom, epsilons[0])
    force.addParticle(charges[1], sigmas[1] * openmm.unit.angstrom, epsilons[1])
    force.addParticle(charges[2], sigmas[2] * openmm.unit.angstrom, epsilons[2])

    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    system.addParticle(1.0)
    system.addForce(force)

    distances = [[0.0, 4.0, 3.0], [4.0, 0.0, 5.0], [3.0, 5.0, 0.0]]

    def interaction_energy_fn(
        idx_a, idx_b, lambda_vdw: float = 1.0, lambda_charges: float = 1.0
    ):
        epsilon = numpy.sqrt(epsilons[idx_a] * epsilons[idx_b])
        sigma = 0.5 * (sigmas[idx_a] + sigmas[idx_b])
        charge = charges[idx_a] * charges[idx_b]

        return compute_interaction_energy(
            epsilon, sigma, charge, distances[idx_a][idx_b], lambda_vdw, lambda_charges
        )

    coords = (
        numpy.array(
            [[0.0, 0.0, 0.0], [distances[0][1], 0.0, 0.0], [0.0, distances[0][2], 0.0]]
        )
        * openmm.unit.angstrom
    )

    return system, coords, interaction_energy_fn


class TestNonbondedInteractions:
    def test_convert_intramolecular_interactions(self):
        system = openmm.System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        system.addParticle(1.0)
        system.addParticle(1.0)
        system.addParticle(1.0)

        bond_force = openmm.HarmonicBondForce()
        bond_force.addBond(0, 1, 1.0 * openmm.unit.angstrom, 1.0)
        bond_force.addBond(1, 2, 1.0 * openmm.unit.angstrom, 1.0)
        bond_force.addBond(2, 3, 1.0 * openmm.unit.angstrom, 1.0)
        bond_force.addBond(3, 4, 1.0 * openmm.unit.angstrom, 1.0)
        system.addForce(bond_force)

        epsilons = (numpy.arange(5) * 10.0 + 200.0).tolist()
        sigmas = (numpy.arange(5) / 10.0 + 1.0).tolist()

        charges = (numpy.arange(5) / 10.0).tolist()

        force = openmm.NonbondedForce()

        for i in range(5):
            force.addParticle(charges[i], sigmas[i], epsilons[i])

        for i, j in [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
        ]:
            force.addException(
                i,
                j,
                charges[i] * charges[j],
                0.5 * (sigmas[i] + sigmas[j]),
                numpy.sqrt(epsilons[i] * epsilons[j]),
            )

        system.addForce(force)

        coords = (
            numpy.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 1.0, 0.0],
                    [4.0, 0.0, 0.0],
                ]
            )
            * openmm.unit.angstrom
        )

        existing_exclusions = [
            force.getExceptionParameters(i) for i in range(force.getNumExceptions())
        ]
        energy_0 = femto.md.utils.openmm.compute_energy(system, coords, None)

        # we expect a 1-5 exclusion to be added.
        _convert_intramolecular_interactions(force, {0, 1, 2, 3, 4}, set())

        modified_exclusions = [
            force.getExceptionParameters(i) for i in range(force.getNumExceptions())
        ]
        assert len(modified_exclusions) == len(existing_exclusions) + 1

        for i in range(len(existing_exclusions)):
            assert existing_exclusions[i][0] == modified_exclusions[i][0]
            assert existing_exclusions[i][1] == modified_exclusions[i][1]

            assert all(
                femto.md.utils.openmm.is_close(v_a, v_b)
                for v_a, v_b in zip(
                    existing_exclusions[i][2:], modified_exclusions[i][2:], strict=True
                )
            )

        assert modified_exclusions[-1][:2] == [0, 4]

        energy_1 = femto.md.utils.openmm.compute_energy(system, coords, None)

        assert femto.md.utils.openmm.is_close(energy_0, energy_1)

    def test_one_ligand(self, three_particle_system, fep_config):
        """Test scaling the nonbonded interactions of single particles."""

        system, coords, energy_fn = three_particle_system
        apply_fep(system, {0}, set(), fep_config)

        # expect lig_1 + solvent, lig_1 + lig_2 and lig_2 + solvent interaction when
        # lambda=0
        energy_0 = femto.md.utils.openmm.compute_energy(
            system,
            coords,
            None,
            {
                LAMBDA_VDW_LIGAND_1: 0.0,
                LAMBDA_CHARGES_LIGAND_1: 0.0,
            },
        )
        expected_energy_0 = energy_fn(0, 2) + energy_fn(0, 1) + energy_fn(1, 2)
        assert femto.md.utils.openmm.is_close(energy_0, expected_energy_0)

        # expect only lig_2 + solvent interaction when lambda=1
        energy_1 = femto.md.utils.openmm.compute_energy(
            system,
            coords,
            None,
            {
                LAMBDA_VDW_LIGAND_1: 1.0,
                LAMBDA_CHARGES_LIGAND_1: 1.0,
            },
        )
        expected_energy_1 = energy_fn(1, 2)
        assert femto.md.utils.openmm.is_close(energy_1, expected_energy_1)

        # expect all particles to interact but only lig - solvent interactions to be
        # scaled
        energy_05 = femto.md.utils.openmm.compute_energy(
            system,
            coords,
            None,
            {
                LAMBDA_VDW_LIGAND_1: 0.5,
                LAMBDA_CHARGES_LIGAND_1: 0.5,
            },
        )
        expected_energy_05 = (
            energy_fn(1, 2) + energy_fn(0, 2, 0.5, 0.5) + energy_fn(0, 1, 0.5, 0.5)
        )
        assert femto.md.utils.openmm.is_close(energy_05, expected_energy_05)

    def test_two_ligands(self, three_particle_system, fep_config):
        """Test scaling the nonbonded interactions of single particles."""

        system, coords, energy_fn = three_particle_system
        apply_fep(system, {0}, {1}, fep_config)

        # expect only lig_1 + solvent interaction when lambda=0
        energy_0 = femto.md.utils.openmm.compute_energy(
            system,
            coords,
            None,
            {
                LAMBDA_VDW_LIGAND_1: 0.0,
                LAMBDA_CHARGES_LIGAND_1: 0.0,
                LAMBDA_VDW_LIGAND_2: 1.0,
                LAMBDA_CHARGES_LIGAND_2: 1.0,
            },
        )
        expected_energy_0 = energy_fn(0, 2)
        assert femto.md.utils.openmm.is_close(energy_0, expected_energy_0)

        # expect only lig_2 + solvent interaction when lambda=1
        energy_1 = femto.md.utils.openmm.compute_energy(
            system,
            coords,
            None,
            {
                LAMBDA_VDW_LIGAND_1: 1.0,
                LAMBDA_CHARGES_LIGAND_1: 1.0,
                LAMBDA_VDW_LIGAND_2: 0.0,
                LAMBDA_CHARGES_LIGAND_2: 0.0,
            },
        )
        expected_energy_1 = energy_fn(1, 2)
        assert femto.md.utils.openmm.is_close(energy_1, expected_energy_1)

        # expect lig_1 + solvent and lig_2 + solvent interaction when lambda=0.5
        # but no lig_1 + lig_2 interaction by default
        energy_05 = femto.md.utils.openmm.compute_energy(
            system,
            coords,
            None,
            {
                LAMBDA_VDW_LIGAND_1: 0.5,
                LAMBDA_CHARGES_LIGAND_1: 0.5,
                LAMBDA_VDW_LIGAND_2: 0.5,
                LAMBDA_CHARGES_LIGAND_2: 0.5,
            },
        )
        expected_energy_05 = energy_fn(0, 2, 0.5, 0.5) + energy_fn(1, 2, 0.5, 0.5)
        assert femto.md.utils.openmm.is_close(energy_05, expected_energy_05)

    def test_exception(self, three_particle_system, fep_config):
        """Test that 1-n exceptions are properly created."""

        system, coords, energy_fn = three_particle_system

        apply_fep(system, {0, 1}, set(), fep_config)

        for lambda_value in [0.0, 0.5, 1.0]:
            energy = femto.md.utils.openmm.compute_energy(
                system,
                coords,
                None,
                {
                    LAMBDA_VDW_LIGAND_1: 1.0 - lambda_value,
                    LAMBDA_CHARGES_LIGAND_1: 1.0 - lambda_value,
                },
            )
            expected_energy = (
                energy_fn(0, 2, lambda_value, lambda_value)
                + energy_fn(1, 2, lambda_value, lambda_value)
                + energy_fn(0, 1, 1.0, 1.0)
            )
            assert femto.md.utils.openmm.is_close(energy, expected_energy)

    def test_existing_exception(self, three_particle_system, fep_config):
        """Test that existing exceptions aren't changed."""

        system, coords, energy_fn = three_particle_system

        charge, sigma, epsilon = 0.4, 1.4, 2.4

        force: openmm.NonbondedForce = next(iter(system.getForces()))
        force.addException(0, 1, charge, sigma * openmm.unit.angstrom, epsilon)

        apply_fep(system, {0, 1}, set(), fep_config)

        for lambda_value in [0.0, 0.5, 1.0]:
            energy = femto.md.utils.openmm.compute_energy(
                system,
                coords,
                None,
                {
                    LAMBDA_VDW_LIGAND_1: 1.0 - lambda_value,
                    LAMBDA_CHARGES_LIGAND_1: 1.0 - lambda_value,
                },
            )
            expected_energy = (
                energy_fn(0, 2, lambda_value, lambda_value)
                + energy_fn(1, 2, lambda_value, lambda_value)
                + compute_interaction_energy(epsilon, sigma, charge, 4.0, 1.0, 1.0)
            )
            assert femto.md.utils.openmm.is_close(energy, expected_energy)
