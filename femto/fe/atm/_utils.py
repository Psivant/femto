"""Utilities for preparing ATM simulations."""

import copy
import typing

import openmm
import openmm.app
import openmm.unit
import parmed

import femto.md.constants
import femto.md.rest

if typing.TYPE_CHECKING:
    import femto.fe.atm


def create_state_dicts(
    states: "femto.fe.atm.ATMSamplingStage",
) -> list[dict[str, float]]:
    """Map the lambda states specified in the configuration to a list of dictionaries.

    Args:
        states: The lambda states.

    Returns:
        The dictionaries of lambda states.
    """
    return [
        {
            openmm.ATMForce.Lambda1(): states.lambda_1[i],
            openmm.ATMForce.Lambda2(): states.lambda_2[i],
            openmm.ATMForce.Direction(): states.direction[i],
            openmm.ATMForce.Alpha(): (
                states.alpha[i] * states.alpha_unit
            ).value_in_unit(openmm.unit.kilojoules_per_mole**-1),
            openmm.ATMForce.Uh(): (states.u0[i] * states.u0_unit).value_in_unit(
                openmm.unit.kilojoules_per_mole
            ),
            openmm.ATMForce.W0(): (states.w0[i] * states.w0_unit).value_in_unit(
                openmm.unit.kilojoules_per_mole
            ),
            **(
                {femto.md.rest.REST_CTX_PARAM: states.bm_b0[i]}
                if states.bm_b0 is not None
                else {}
            ),
        }
        for i in range(len(states.lambda_1))
    ]


def create_atm_force(
    topology: parmed.Structure,
    soft_core: "femto.fe.atm.ATMSoftCore",
    offset: openmm.unit.Quantity,
) -> openmm.ATMForce:
    """Creates an ATM force initialized to lambda=0.0

    Args:
        topology: The topology of the system being simulation.
        soft_core: The soft core parameters to use.
        offset: The ligand offset.

    Returns:
        The created force.
    """

    ligand_1_idxs = [
        i
        for i, atom in enumerate(topology.atoms)
        if atom.residue.name == femto.md.constants.LIGAND_1_RESIDUE_NAME
    ]
    ligand_2_idxs = [
        i
        for i, atom in enumerate(topology.atoms)
        if atom.residue.name == femto.md.constants.LIGAND_2_RESIDUE_NAME
    ]
    assert len(ligand_1_idxs) > 0, "no ligands were found to add to the ATM force"

    atm_force = openmm.ATMForce(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        soft_core.u_max.value_in_unit(openmm.unit.kilojoules_per_mole),
        soft_core.u0.value_in_unit(openmm.unit.kilojoules_per_mole),
        soft_core.a,
        1,
    )
    atm_force.setForceGroup(femto.md.constants.OpenMMForceGroup.ATM)

    for _ in range(len(topology.atoms)):
        atm_force.addParticle(openmm.Vec3(0.0, 0.0, 0.0))

    offset = offset.in_units_of(openmm.unit.nanometers)

    for i in ligand_1_idxs:
        atm_force.setParticleParameters(i, openmm.Vec3(*offset))

    for i in [] if ligand_2_idxs is None else ligand_2_idxs:
        atm_force.setParticleParameters(i, -openmm.Vec3(*offset))

    return atm_force


def add_atm_force(
    system: openmm.System,
    topology: parmed.Structure,
    soft_core: "femto.fe.atm.ATMSoftCore",
    offset: openmm.unit.Quantity,
):
    """Adds an ATM force to the system *in-place*, making any non-bonded forces
    children.

    Args:
        system: The system to add the force to.
        topology: The topology of the system being simulated.
        soft_core: The soft core parameters to use.
        offset: The ligand offset.
    """
    import femto.fe.atm._utils

    nonbonded_idxs = [
        i
        for i, force in enumerate(system.getForces())
        if isinstance(force, openmm.NonbondedForce)
    ]
    assert len(nonbonded_idxs) == 1, "expected exactly one nonbonded force"
    nonbonded_idx = nonbonded_idxs[0]

    nonbonded_force = copy.deepcopy(system.getForce(nonbonded_idx))
    system.removeForce(nonbonded_idx)

    atm_force = femto.fe.atm._utils.create_atm_force(topology, soft_core, offset)
    atm_force.addForce(nonbonded_force)
    system.addForce(atm_force)


def create_com_restraint(
    idxs_a: list[int],
    idxs_b: list[int],
    k: openmm.unit.Quantity,
    radius: openmm.unit.Quantity,
    offset: openmm.unit.Quantity,
) -> openmm.CustomCentroidBondForce:
    """Create a flat-bottom center of mass restraint between two groups of atoms.

    Args:
        idxs_a: The indexes of the first group of atoms.
        idxs_b: The indexes of the second group of atoms.
        k: The force constant [kcal / mol].
        radius: The radius of the flat-bottom restraint [angstrom].
        offset: The distances to separate the two atom groups by [angstrom].

    Returns:
        The created restraint force.
    """

    assert len(idxs_a) > 0 and len(idxs_b) > 0, "no atoms were specified for restraint"

    energy_fn = (
        "0.5 * k * step(dist - radius) * (dist - radius)^2;"
        "dist = sqrt((x1 - x2 - dx)^2 + (y1 - y2 - dy)^2 + (z1 - z2 - dz)^2);"
    )
    force = openmm.CustomCentroidBondForce(2, energy_fn)
    force.addPerBondParameter("k")
    force.addPerBondParameter("radius")
    force.addPerBondParameter("dx")
    force.addPerBondParameter("dy")
    force.addPerBondParameter("dz")

    force.addGroup(idxs_a)
    force.addGroup(idxs_b)

    parameters = (
        k.value_in_unit(openmm.unit.kilojoules_per_mole / openmm.unit.nanometer**2),
        radius.value_in_unit(openmm.unit.nanometer),
        offset[0].value_in_unit(openmm.unit.nanometer),
        offset[1].value_in_unit(openmm.unit.nanometer),
        offset[2].value_in_unit(openmm.unit.nanometer),
    )
    force.addBond((0, 1), parameters)

    return force


def create_alignment_restraints(
    idxs_a: tuple[int, int, int],
    idxs_b: tuple[int, int, int],
    k_distance: openmm.unit.Quantity,
    k_angle: openmm.unit.Quantity,
    k_dihedral: openmm.unit.Quantity,
    offset: openmm.unit.Quantity,
) -> tuple[
    openmm.CustomCompoundBondForce,
    openmm.CustomCompoundBondForce,
    openmm.CustomCompoundBondForce,
]:
    """Create a ligand alignment restraint force following
    https://pubs.acs.org/doi/10.1021/acs.jcim.1c01129

    Args:
        idxs_a: The reference indexes of the first ligand.
        idxs_b: The reference indexes of the second ligand.
        k_distance: The force constant for the distance restraint [kcal / mol / A^2].
        k_angle: The force constant for the angle restraint [kcal / mol].
        k_dihedral: The force constant for the dihedral restraint [kcal / mol].
        offset: The offset to apply to the distance restraint [angstrom].

    Returns:
        The created bond, angle and dihedral restraint forces.
    """

    distance_energy_fn = (
        "0.5 * k * ((x1 - x2 - dx)^2 + (y1 - y2 - dy)^2 + (z1 - z2 - dz)^2);"
    )
    distance_force = openmm.CustomCompoundBondForce(2, distance_energy_fn)
    distance_force.addPerBondParameter("k")
    distance_force.addPerBondParameter("dx")
    distance_force.addPerBondParameter("dy")
    distance_force.addPerBondParameter("dz")

    distance_parameters = [
        k_distance.value_in_unit(
            openmm.unit.kilojoules_per_mole / openmm.unit.nanometer**2
        ),
        offset[0].value_in_unit(openmm.unit.nanometer),
        offset[1].value_in_unit(openmm.unit.nanometer),
        offset[2].value_in_unit(openmm.unit.nanometer),
    ]
    distance_force.addBond((idxs_b[0], idxs_a[0]), distance_parameters)

    angle_energy_fn = (
        "0.5 * k * (1 - cos_theta);"
        ""
        "cos_theta = (dx_1 * dx_2 + dy_1 * dy_2 + dz_1 * dz_2) / (norm_1 * norm_2);"
        ""
        "norm_1 = sqrt(dx_1^2 + dy_1^2 + dz_1^2);"
        "dx_1 = x2 - x1; dy_1 = y2 - y1; dz_1 = z2 - z1;"
        ""
        "norm_2 = sqrt(dx_2^2 + dy_2^2 + dz_2^2);"
        "dx_2 = x4 - x3; dy_2 = y4 - y3; dz_2 = z4 - z3;"
    )
    angle_force = openmm.CustomCompoundBondForce(4, angle_energy_fn)
    angle_force.addPerBondParameter("k")
    angle_force.addBond(
        (idxs_b[0], idxs_b[1], idxs_a[0], idxs_a[1]),
        [k_angle.value_in_unit(openmm.unit.kilojoules_per_mole)],
    )

    dihedral_energy_fn = (
        "0.5 * k * (1 - cos_phi);"
        ""
        "cos_phi = (v_x * w_x + v_y * w_y + v_z * w_z) / (norm_v * norm_w);"
        ""
        "norm_v = sqrt(v_x^2 + v_y^2 + v_z^2);"
        "v_x = dx_31 - dot_31 * dx_21 / norm_21;"
        "v_y = dy_31 - dot_31 * dy_21 / norm_21;"
        "v_z = dz_31 - dot_31 * dz_21 / norm_21;"
        ""
        "dot_31 = (dx_31 * dx_21 + dy_31 * dy_21 + dz_31 * dz_21) / norm_21;"
        "dx_31 = x3 - x1; dy_31 = y3 - y1; dz_31 = z3 - z1;"
        ""
        "norm_w = sqrt(w_x^2 + w_y^2 + w_z^2);"
        "w_x = dx_54 - dot_54 * dx_21 / norm_21;"
        "w_y = dy_54 - dot_54 * dy_21 / norm_21;"
        "w_z = dz_54 - dot_54 * dz_21 / norm_21;"
        ""
        "dot_54 =(dx_54 * dx_21 + dy_54 * dy_21 + dz_54 * dz_21) / norm_21;"
        "dx_54 = x5 - x4; dy_54 = y5 - y4; dz_54 = z5 - z4;"
        ""
        "norm_21 = sqrt(dx_21^2 + dy_21^2 + dz_21^2);"
        "dx_21 = x2 - x1; dy_21 = y2 - y1; dz_21 = z2 - z1;"
    )
    dihedral_force = openmm.CustomCompoundBondForce(5, dihedral_energy_fn)
    dihedral_force.addPerBondParameter("k")
    dihedral_force.addBond(
        (idxs_b[0], idxs_b[1], idxs_b[2], idxs_a[0], idxs_a[2]),
        [0.5 * k_dihedral.value_in_unit(openmm.unit.kilojoules_per_mole)],
    )
    dihedral_force.addBond(
        (idxs_a[0], idxs_a[1], idxs_a[2], idxs_b[0], idxs_b[2]),
        [0.5 * k_dihedral.value_in_unit(openmm.unit.kilojoules_per_mole)],
    )

    return distance_force, angle_force, dihedral_force
