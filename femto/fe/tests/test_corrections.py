import openmm.unit

import femto.fe.config
import femto.md.config
import femto.md.utils.openmm
from femto.fe.corrections import analytical_correction_boresch


def test_analytical_correction_boresch():
    temperature = 298.15 * openmm.unit.kelvin

    dist_0 = 1.359 * openmm.unit.nanometers

    theta_a_0 = 45.36 * openmm.unit.degrees
    theta_b_0 = 120.95 * openmm.unit.degrees

    # taken from the SepTop SI
    restraint = femto.md.config.BoreschRestraint(
        k_distance=8368 * openmm.unit.kilojoule_per_mole / openmm.unit.nanometer**2,
        k_angle_a=1236.22 * openmm.unit.kilojoule_per_mole / openmm.unit.radian**2,
        k_angle_b=83.68 * openmm.unit.kilojoule_per_mole / openmm.unit.radian**2,
        k_dihedral_a=83.68 * openmm.unit.kilojoule_per_mole / openmm.unit.radian**2,
        k_dihedral_b=83.68 * openmm.unit.kilojoule_per_mole / openmm.unit.radian**2,
        k_dihedral_c=83.68 * openmm.unit.kilojoule_per_mole / openmm.unit.radian**2,
    )
    expected_correction = -7.968 * openmm.unit.kilocalorie_per_mole

    correction = analytical_correction_boresch(
        dist_0, theta_a_0, theta_b_0, restraint, temperature
    )
    correction = correction.in_units_of(openmm.unit.kilocalorie_per_mole)

    assert femto.md.utils.openmm.is_close(correction, expected_correction, atol=0.001)
