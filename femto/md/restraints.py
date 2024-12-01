"""Create OpenMM restraint forces."""

import typing

import numpy.linalg
import openmm.unit

import femto.md.config
import femto.md.constants
import femto.md.utils.geometry
import femto.top

_FLAT_BOTTOM_ENERGY_FN = (
    "0.5 * k * select(step(dist - radius), (dist - radius) ^ 2, 0);"
    "dist = periodicdistance(x,y,z,x0,y0,z0)"
).replace(" ", "")


_BORESCH_ENERGY_FN = (
    "0.5 * E;"
    "E = k_dist_a  * (distance(p3,p4) - dist_0)    ^ 2"
    "  + k_theta_a * (angle(p2,p3,p4) - theta_a_0) ^ 2"
    "  + k_theta_b * (angle(p3,p4,p5) - theta_b_0) ^ 2"
    "  + k_phi_a   * (d_phi_a_wrap)                ^ 2"
    "  + k_phi_b   * (d_phi_b_wrap)                ^ 2"
    "  + k_phi_c   * (d_phi_c_wrap)                ^ 2;"
    # compute the periodic dihedral delta (e.g. distance between -180 and 180 is 0)
    "d_phi_a_wrap = d_phi_a - floor(d_phi_a / (2.0 * pi) + 0.5) * (2.0 * pi);"
    "d_phi_a = dihedral(p1,p2,p3,p4) - phi_a_0;"
    "d_phi_b_wrap = d_phi_b - floor(d_phi_b / (2.0 * pi) + 0.5) * (2.0 * pi);"
    "d_phi_b = dihedral(p2,p3,p4,p5) - phi_b_0;"
    "d_phi_c_wrap = d_phi_c - floor(d_phi_c / (2.0 * pi) + 0.5) * (2.0 * pi);"
    "d_phi_c = dihedral(p3,p4,p5,p6) - phi_c_0;"
    f"pi = {numpy.pi}"
).replace(" ", "")


_ANGSTROM = openmm.unit.angstrom
_RADIANS = openmm.unit.radian


class _BoreschGeometry(typing.NamedTuple):
    dist_0: openmm.unit.Quantity

    theta_a_0: openmm.unit.Quantity
    theta_b_0: openmm.unit.Quantity

    phi_a_0: openmm.unit.Quantity
    phi_b_0: openmm.unit.Quantity
    phi_c_0: openmm.unit.Quantity


def create_flat_bottom_restraint(
    config: femto.md.config.FlatBottomRestraint,
    coords: dict[int, openmm.unit.Quantity],
) -> openmm.CustomExternalForce:
    """Creates a flat bottom position restraint.

    Args:
        config: The restraint configuration.
        coords: A dictionary of indices of atoms to restrain and the corresponding
            coordinates to restrain them to.

    Returns:
        The restraint force.
    """

    force = openmm.CustomExternalForce(_FLAT_BOTTOM_ENERGY_FN)

    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    force.addPerParticleParameter("k")
    force.addPerParticleParameter("radius")

    k = config.k.value_in_unit_system(openmm.unit.md_unit_system)
    radius = config.radius.value_in_unit_system(openmm.unit.md_unit_system)

    for idx, coord in coords.items():
        force.addParticle(
            idx,
            [
                coord[0].value_in_unit_system(openmm.unit.md_unit_system),
                coord[1].value_in_unit_system(openmm.unit.md_unit_system),
                coord[2].value_in_unit_system(openmm.unit.md_unit_system),
                k,
                radius,
            ],
        )

    force.setForceGroup(femto.md.constants.OpenMMForceGroup.POSITION_RESTRAINT)
    force.setName(femto.md.constants.OpenMMForceName.POSITION_RESTRAINT)

    return force


def create_position_restraints(
    topology: femto.top.Topology,
    mask: str,
    config: femto.md.config.FlatBottomRestraint,
) -> openmm.CustomExternalForce:
    """Creates position restraints for all ligand atoms.

    Args:
        topology: The topology of the system being simulation.
        mask: The mask that defines which atoms to restrain.
        config: The restraint configuration.

    Returns:
        The created restraint force.
    """

    selection_idxs = topology.select(mask)
    assert len(selection_idxs) > 0, "no atoms were found to restrain"

    assert topology.xyz is not None, "topology must have coordinates to restrain"
    xyz = topology.xyz.value_in_unit(openmm.unit.angstrom)

    coords = {i: openmm.Vec3(*xyz[i]) * openmm.unit.angstrom for i in selection_idxs}

    if not isinstance(config, femto.md.config.FlatBottomRestraint):
        raise NotImplementedError("only flat bottom restraints are currently supported")

    return femto.md.restraints.create_flat_bottom_restraint(config, coords)


def _compute_boresch_geometry(
    receptor_atoms: tuple[int, int, int],
    ligand_atoms: tuple[int, int, int],
    coords: openmm.unit.Quantity,
) -> _BoreschGeometry:
    """Computes the equilibrium distances, angles, and dihedrals used by a Boresch
    restraint."""

    r1, r2, r3 = receptor_atoms
    l1, l2, l3 = ligand_atoms

    coords = coords.value_in_unit(openmm.unit.angstrom)

    dist_0 = (
        femto.md.utils.geometry.compute_distances(coords, numpy.array([[r3, l1]]))
        * _ANGSTROM
    )

    theta_a_0 = (
        femto.md.utils.geometry.compute_angles(coords, numpy.array([[r3, l1, l2]]))
        * _RADIANS
    )
    theta_b_0 = (
        femto.md.utils.geometry.compute_angles(coords, numpy.array([[r2, r3, l1]]))
        * _RADIANS
    )

    phi_a_0 = (
        femto.md.utils.geometry.compute_dihedrals(
            coords, numpy.array([[r3, l1, l2, l3]])
        )
        * _RADIANS
    )
    phi_b_0 = (
        femto.md.utils.geometry.compute_dihedrals(
            coords, numpy.array([[r2, r3, l1, l2]])
        )
        * _RADIANS
    )
    phi_c_0 = (
        femto.md.utils.geometry.compute_dihedrals(
            coords, numpy.array([[r1, r2, r3, l1]])
        )
        * _RADIANS
    )

    return _BoreschGeometry(dist_0, theta_a_0, theta_b_0, phi_a_0, phi_b_0, phi_c_0)


def create_boresch_restraint(
    config: femto.md.config.BoreschRestraint,
    receptor_atoms: tuple[int, int, int],
    ligand_atoms: tuple[int, int, int],
    coords: openmm.unit.Quantity,
    ctx_parameter: str | None = None,
) -> openmm.CustomCompoundBondForce:
    """Creates a 'Boresch' style restraint force, useful for aligning a receptor and
    ligand.

    Namely, the following will be restrained:
        * ``receptor[2]`` -- ``ligand[0]`` distance.
        * ``receptor[2]`` -- ``ligand[0]``   -- ``ligand[1]`` angle.
        * ``receptor[1]`` -- ``receptor[2]`` -- ``ligand[0]`` angle.
        * ``receptor[2]`` -- ``ligand[0]``   -- ``ligand[1]``   -- ``ligand[2]`` dih.
        * ``receptor[1]`` -- ``receptor[2]`` -- ``ligand[0]``   -- ``ligand[1]`` dih.
        * ``receptor[0]`` -- ``receptor[1]`` -- ``receptor[2]`` -- ``ligand[0]`` dih.

    Args:
        config: The restraint configuration.
        receptor_atoms: The indices of the three receptor atoms to restrain.
        ligand_atoms: The indices of the three ligand atoms to restrain.
        coords: The coordinates of the *full* system.
        ctx_parameter: An optional context parameter to use to scale the strength of
            the restraint.

    Returns:
        The restraint force.
    """
    n_particles = 6  # 3 receptor + 3 ligand

    energy_fn = _BORESCH_ENERGY_FN

    if ctx_parameter is not None:
        energy_fn = f"{ctx_parameter} * {energy_fn}"

    force = openmm.CustomCompoundBondForce(n_particles, energy_fn)

    if ctx_parameter is not None:
        force.addGlobalParameter(ctx_parameter, 1.0)

    geometry = _compute_boresch_geometry(receptor_atoms, ligand_atoms, coords)

    parameters = []

    for key, value in [
        ("k_dist_a", config.k_distance),
        ("k_theta_a", config.k_angle_a),
        ("k_theta_b", config.k_angle_b),
        ("k_phi_a", config.k_dihedral_a),
        ("k_phi_b", config.k_dihedral_b),
        ("k_phi_c", config.k_dihedral_c),
        ("dist_0", geometry.dist_0),
        ("theta_a_0", geometry.theta_a_0),
        ("theta_b_0", geometry.theta_b_0),
        ("phi_a_0", geometry.phi_a_0),
        ("phi_b_0", geometry.phi_b_0),
        ("phi_c_0", geometry.phi_c_0),
    ]:
        force.addPerBondParameter(key)
        parameters.append(value.value_in_unit_system(openmm.unit.md_unit_system))

    force.addBond(receptor_atoms + ligand_atoms, parameters)
    force.setUsesPeriodicBoundaryConditions(False)
    force.setName(femto.md.constants.OpenMMForceName.ALIGNMENT_RESTRAINT)
    force.setForceGroup(femto.md.constants.OpenMMForceGroup.ALIGNMENT_RESTRAINT)

    return force
