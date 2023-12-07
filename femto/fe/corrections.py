"""Compute analytical corrections to free energies."""
import numpy
import openmm.unit

import femto.md.config


def analytical_correction_boresch(
    dist_0: openmm.unit.Quantity,
    theta_a_0: openmm.unit.Quantity,
    theta_b_0: openmm.unit.Quantity,
    restraint: femto.md.config.BoreschRestraint,
    temperature: openmm.unit.Quantity,
) -> openmm.unit.Quantity:
    """Compute the analytical correction to the free energy due to disabling a
    Boresch-style restraint as described by Eq. 32 in [1].

    References:
        [1] Boresch, Stefan, et al. "Absolute binding free energies: a quantitative
        approach for their calculation." The Journal of Physical Chemistry B 107.35
        (2003): 9535-9551.

    Args:
        dist_0: The equilibrium distance between r3 and l1.
        theta_a_0: The equilibrium angle between r2, r3, and l1.
        theta_b_0: The equilibrium angle between r3, l1, and l2.
        restraint: The Boresch restraint parameters.
        temperature: The temperature of the system.

    Returns:
        The analytical correction.
    """

    sin_theta_a_0 = numpy.sin(theta_a_0.value_in_unit(openmm.unit.radian))
    sin_theta_b_0 = numpy.sin(theta_b_0.value_in_unit(openmm.unit.radian))

    k_dist = restraint.k_distance

    k_theta_a = restraint.k_angle_a
    k_theta_b = restraint.k_angle_b

    k_phi_a = restraint.k_dihedral_a
    k_phi_b = restraint.k_dihedral_b
    k_phi_c = restraint.k_dihedral_c

    volume_0 = 1660.0 * openmm.unit.angstrom**3  # taken from [1].

    kt = openmm.unit.MOLAR_GAS_CONSTANT_R * temperature

    correction = -kt * numpy.log(
        ((8.0 * numpy.pi**2 * volume_0) / (dist_0**2 * sin_theta_a_0 * sin_theta_b_0))
        * (
            numpy.sqrt(k_dist * k_theta_a * k_theta_b * k_phi_a * k_phi_b * k_phi_c)
            / (2.0 * numpy.pi * kt) ** 3
        )
    )

    return correction
