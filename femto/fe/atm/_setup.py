"""Set up the system for ATM calculations."""

import copy
import logging
import typing

import numpy
import openmm
import openmm.app
import openmm.unit
import scipy.spatial.distance

import femto.fe.reference
import femto.md.prepare
import femto.md.rest
import femto.md.restraints
import femto.md.utils.openmm
import femto.top
from femto.md.constants import OpenMMForceGroup, OpenMMForceName

if typing.TYPE_CHECKING:
    import femto.fe.atm

_LOGGER = logging.getLogger(__name__)


def select_displacement(
    receptor: femto.top.Topology,
    ligand_1: femto.top.Topology,
    ligand_2: femto.top.Topology | None,
    distance: openmm.unit.Quantity,
) -> openmm.unit.Quantity:
    """Attempts to automatically select a displacement vector for the ligands.

    Args:
        receptor: The receptor.
        ligand_1: The first ligand positioned in the binding site.
        ligand_2: The second ligand positioned in the binding site.
        distance: The distance to translate ligands along the displacement vector by.

    Returns:
        The displacement vector.
    """

    ligand_coords = numpy.vstack(
        [ligand_1.xyz.value_in_unit(openmm.unit.angstrom)]
        + (
            []
            if ligand_2 is None
            else [ligand_2.xyz.value_in_unit(openmm.unit.angstrom)]
        )
    )
    receptor_coords = receptor.xyz.value_in_unit(openmm.unit.angstrom)

    directions = numpy.array(
        [
            [-1.0, -1.0, -1.0],
            [+1.0, -1.0, -1.0],
            [+1.0, +1.0, -1.0],
            [-1.0, +1.0, -1.0],
            [-1.0, -1.0, +1.0],
            [+1.0, -1.0, +1.0],
            [+1.0, +1.0, +1.0],
            [-1.0, +1.0, +1.0],
        ]
    )
    directions /= numpy.linalg.norm(directions, axis=1, keepdims=True)

    closest_distances = []

    for direction in directions:
        displacement = direction * distance.value_in_unit(openmm.unit.angstrom)

        offset_coords = ligand_coords + displacement

        distances = scipy.spatial.distance.cdist(offset_coords, receptor_coords)
        closest_distances.append(distances.min())

    direction = directions[numpy.argmax(closest_distances)]
    return direction.flatten() * distance


def _offset_ligand(
    ligand: femto.top.Topology, offset: openmm.unit.Quantity
) -> femto.top.Topology:
    """Offsets the coordinates of the specified ligand by a specified amount.

    Args:
        ligand: The ligand to offset.
        offset: The amount to offset the ligand by.

    Returns:
        The offset ligand.
    """

    ligand = copy.deepcopy(ligand)
    ligand.xyz += offset

    return ligand


def _apply_atm_restraints(
    system: openmm.System,
    config: "femto.fe.atm.ATMRestraints",
    ligand_1_com_idxs: list[int],
    ligand_1_ref_idxs: tuple[int, int, int] | None,
    ligand_2_com_idxs: list[int] | None,
    ligand_2_ref_idxs: tuple[int, int, int] | None,
    receptor_ref_idxs: list[int],
    offset: openmm.unit.Quantity,
):
    """Adds center of mass (COM) and optionally alignment restraints (if running RBFE)
    to a system.

    Args:
        system: The system to add the constraints to in-place.
        config: The restraint configuration.
        ligand_1_com_idxs: The indices to use when computing the COM of the first
            ligand.
        ligand_1_ref_idxs: The indices of the first ligand to align on.
        ligand_2_com_idxs: The indices to use when computing the COM of the second
            ligand.
        ligand_2_ref_idxs: The indices of the second ligand to align on.
        receptor_ref_idxs: The indices of the receptor atoms that form the binding site.
        offset: The vector that the ligand will be offset by during the ATM calculation.
    """
    import femto.fe.atm._utils

    com_restraint = femto.fe.atm._utils.create_com_restraint(
        ligand_1_com_idxs,
        receptor_ref_idxs,
        config.com.k,
        config.com.radius,
        [0.0, 0.0, 0.0] * openmm.unit.angstrom,
    )
    com_restraint.setForceGroup(OpenMMForceGroup.COM_RESTRAINT)
    com_restraint.setName(OpenMMForceName.COM_RESTRAINT)
    system.addForce(com_restraint)

    if ligand_2_com_idxs is None:
        return

    com_restraint = femto.fe.atm._utils.create_com_restraint(
        ligand_2_com_idxs,
        receptor_ref_idxs,
        config.com.k,
        config.com.radius,
        [*offset.value_in_unit(openmm.unit.angstrom)] * openmm.unit.angstrom,
    )
    com_restraint.setForceGroup(OpenMMForceGroup.COM_RESTRAINT)
    com_restraint.setName(OpenMMForceName.COM_RESTRAINT)
    system.addForce(com_restraint)

    alignment_forces = femto.fe.atm._utils.create_alignment_restraints(
        ligand_1_ref_idxs,
        ligand_2_ref_idxs,
        config.alignment.k_distance,
        config.alignment.k_angle,
        config.alignment.k_dihedral,
        offset=[*offset.value_in_unit(openmm.unit.angstrom)] * openmm.unit.angstrom,
    )

    for alignment_force in alignment_forces:
        alignment_force.setForceGroup(OpenMMForceGroup.ALIGNMENT_RESTRAINT)
        alignment_force.setName(OpenMMForceName.ALIGNMENT_RESTRAINT)
        system.addForce(alignment_force)

    return system


def _apply_receptor_restraints(
    system: openmm.System,
    config: "femto.fe.atm.ATMRestraints",
    restrained_coords: dict[int, openmm.unit.Quantity],
):
    """Apply position restraints to the receptor atoms.

    Args:
        system: The system to add the constraints to in-place.
        config: The restraint configuration.
        restrained_coords: The coordinates of the receptor atoms to restrain.
    """

    if len(restrained_coords) == 0:
        raise RuntimeError("No receptor atoms to restrain were found.")

    _LOGGER.info(f"restrained receptor idxs={[*restrained_coords]}")

    force = femto.md.restraints.create_flat_bottom_restraint(
        config.receptor, restrained_coords
    )
    system.addForce(force)


def setup_system(
    config: "femto.fe.atm.ATMSetupStage",
    receptor: femto.top.Topology,
    ligand_1: femto.top.Topology,
    ligand_2: femto.top.Topology | None,
    displacement: openmm.unit.Quantity,
    receptor_ref_query: str | None,
    ligand_1_ref_query: tuple[str, str, str] | None = None,
    ligand_2_ref_query: tuple[str, str, str] | None = None,
) -> tuple[femto.top.Topology, openmm.System]:
    """Prepares a system ready for running the ATM method.

    Returns:
        The prepared topology and OpenMM system object.
    """

    _LOGGER.info(f"setting up an {'ABFE' if ligand_2 is None else 'RBFE'} calculation")

    if receptor_ref_query is None:
        # we need to select the receptor cavity atoms before offsetting any ligands
        # as the query is distance based

        _LOGGER.info("selecting receptor reference atoms")
        receptor_ref_query = femto.fe.reference.select_protein_cavity_atoms(
            receptor,
            [ligand_1] + ([] if ligand_2 is None else [ligand_2]),
            config.reference.receptor_cutoff,
        )

    ligand_1_ref_idxs, ligand_2_ref_idxs = None, None

    # we carve out a 'cavity' where the first ligand will be displaced into during the
    # ATM calculations. this should make equilibration at all states easier.
    cavity_formers = [_offset_ligand(ligand_1, displacement)]

    if ligand_2 is not None:
        # we make sure that when placing solvent molecules we don't accidentally place
        # any on top of the ligands in the cavity itself
        cavity_formers.append(ligand_2)

        (
            ligand_1_ref_idxs,
            ligand_2_ref_idxs,
        ) = femto.fe.reference.select_ligand_idxs(
            ligand_1,
            ligand_2,
            config.reference.ligand_method,
            ligand_1_ref_query,
            ligand_2_ref_query,
        )
        assert ligand_2_ref_idxs is not None, "ligand 2 ref atoms were not selected"
        ligand_2_ref_idxs = tuple(i + len(ligand_1.atoms) for i in ligand_2_ref_idxs)

        ligand_2 = _offset_ligand(ligand_2, displacement)

    _LOGGER.info("preparing system")
    topology, system = femto.md.prepare.prepare_system(
        receptor,
        ligand_1,
        ligand_2,
        config.solvent,
        [],
        displacement,
        cavity_formers=cavity_formers,
    )

    if config.apply_hmr:
        _LOGGER.info("applying HMR.")

        hydrogen_mass = config.hydrogen_mass
        femto.md.prepare.apply_hmr(system, topology, hydrogen_mass)

    ligand_1_idxs = list(range(len(ligand_1.atoms)))
    ligand_2_idxs = None

    if ligand_2 is not None:
        ligand_2_idxs = [i + len(ligand_1_idxs) for i in range(len(ligand_2.atoms))]

    if config.apply_rest:
        _LOGGER.info("applying REST2.")

        solute_idxs = ligand_1_idxs + ([] if ligand_2_idxs is None else ligand_2_idxs)
        femto.md.rest.apply_rest(system, set(solute_idxs), config.rest_config)

    _LOGGER.info("applying restraints.")
    ligands = [ligand_1] + ([] if ligand_2 is None else [ligand_2])
    idx_offset = sum(len(ligand.atoms) for ligand in ligands)

    receptor_ref_idxs = receptor.select(receptor_ref_query) + idx_offset
    _LOGGER.info(f"receptor ref idxs={receptor_ref_idxs}")

    _apply_atm_restraints(
        system,
        config.restraints,
        ligand_1_com_idxs=ligand_1_idxs,
        ligand_1_ref_idxs=ligand_1_ref_idxs,
        ligand_2_com_idxs=ligand_2_idxs,
        ligand_2_ref_idxs=ligand_2_ref_idxs,
        receptor_ref_idxs=receptor_ref_idxs,
        offset=displacement,
    )

    restraint_idxs = receptor.select(config.restraints.receptor_query)

    _apply_receptor_restraints(
        system, config.restraints, {i: topology.xyz[i] for i in restraint_idxs}
    )
    femto.md.utils.openmm.assign_force_groups(system)

    return topology, system
