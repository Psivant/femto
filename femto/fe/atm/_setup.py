"""Set up the system for ATM calculations."""

import copy
import logging
import pathlib
import typing

import mdtop
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
from femto.md.constants import OpenMMForceGroup, OpenMMForceName

if typing.TYPE_CHECKING:
    import femto.fe.atm

_LOGGER = logging.getLogger(__name__)


def select_displacement(
    receptor: mdtop.Topology,
    ligand_1: mdtop.Topology,
    ligand_2: mdtop.Topology | None,
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
    ligand: mdtop.Topology, offset: openmm.unit.Quantity
) -> mdtop.Topology:
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
    ligand_1_com_idxs: numpy.ndarray,
    ligand_1_ref_idxs: tuple[int, int, int] | None,
    ligand_2_com_idxs: numpy.ndarray | None,
    ligand_2_ref_idxs: tuple[int, int, int] | None,
    receptor_ref_idxs: numpy.ndarray,
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
    receptor: mdtop.Topology,
    ligand_1: mdtop.Topology,
    ligand_2: mdtop.Topology | None,
    cofactors: list[mdtop.Topology] | None,
    displacement: openmm.unit.Quantity,
    receptor_ref_query: str | None,
    ligand_1_ref_query: tuple[str, str, str] | None = None,
    ligand_2_ref_query: tuple[str, str, str] | None = None,
    extra_params: list[pathlib.Path] | None = None,
) -> tuple[mdtop.Topology, openmm.System]:
    """Prepares a system ready for running the ATM method.

    Args:
        config: The configuration for setting up the system.
        receptor: The receptor topology.
        ligand_1: The first ligand.
        ligand_2: The second ligand if one is present.
        cofactors: Any cofactors.
        displacement: The displacement vector to use for the ligands.
        receptor_ref_query: The query to select the receptor reference atoms.
        ligand_1_ref_query: The query to select the first ligand reference atoms.
        ligand_2_ref_query: The query to select the second ligand reference atoms.
        extra_params: The paths to any extra parameter files (.xml, .parm) to use
            when parameterizing the system.

    Returns:
        The prepared topology and OpenMM system object.
    """

    _LOGGER.info(f"setting up an {'ABFE' if ligand_2 is None else 'RBFE'} calculation")

    if receptor_ref_query is None:
        # we need to select the receptor cavity atoms before offsetting any ligands
        # as the query is distance based
        receptor_cutoff = config.reference.receptor_cutoff.value_in_unit(
            openmm.unit.angstrom
        )
        receptor_ref_query = (
            f"name CA near_to {receptor_cutoff} of "
            f"(resn {femto.md.constants.LIGAND_1_RESIDUE_NAME} or "
            f" resn {femto.md.constants.LIGAND_2_RESIDUE_NAME})"
        )

    # prefix zero to denote that 0 maps to atom 0 of the receptor, not the topology.
    receptor_ref_idxs_0 = receptor.select(receptor_ref_query)

    # we carve out a 'cavity' where the first ligand will be displaced into during the
    # ATM calculations. this should make equilibration at all states easier.
    cavity_formers = [_offset_ligand(ligand_1, displacement)]

    if ligand_2 is not None:
        # we make sure that when placing solvent molecules we don't accidentally place
        # any on top of the ligands in the cavity itself
        cavity_formers.append(ligand_2)
        ligand_2 = _offset_ligand(ligand_2, displacement)

    _LOGGER.info("preparing system")
    topology, system = femto.md.prepare.prepare_system(
        receptor,
        ligand_1,
        ligand_2,
        cofactors,
        config,
        displacement,
        cavity_formers=cavity_formers,
        extra_params=extra_params,
    )

    if config.apply_hmr:
        _LOGGER.info("applying HMR.")
        femto.md.prepare.apply_hmr(system, topology, config.hydrogen_mass)

    ligand_1_idxs = topology.select(f"resn {femto.md.constants.LIGAND_1_RESIDUE_NAME}")
    ligand_2_idxs = topology.select(f"resn {femto.md.constants.LIGAND_2_RESIDUE_NAME}")

    ligand_1 = topology.subset(ligand_1_idxs)
    ligand_2 = topology.subset(ligand_2_idxs) if ligand_2 is not None else None

    if config.apply_rest:
        _LOGGER.info("applying REST2.")

        solute_idxs = {*ligand_1_idxs, *({} if ligand_2 is None else ligand_2_idxs)}
        femto.md.rest.apply_rest(system, solute_idxs, config.rest_config)

    _LOGGER.info("applying restraints.")
    ligand_1_ref_idxs, ligand_2_ref_idxs = None, None

    if ligand_2 is not None:
        (
            ligand_1_ref_idxs_0,
            ligand_2_ref_idxs_0,
        ) = femto.fe.reference.select_ligand_idxs(
            ligand_1,
            ligand_2,
            config.reference.ligand_method,
            ligand_1_ref_query,
            ligand_2_ref_query,
        )
        assert ligand_2_ref_idxs_0 is not None, "ligand 2 ref atoms were not selected"

        ligand_1_ref_idxs = [ligand_1_idxs[i] for i in ligand_1_ref_idxs_0]
        ligand_2_ref_idxs = [ligand_2_idxs[i] for i in ligand_2_ref_idxs_0]

        _LOGGER.info(f"ligand 1 ref idxs={ligand_1_idxs}")
        _LOGGER.info(f"ligand 2 ref idxs={ligand_2_idxs}")

    receptor_start_idx = ligand_1.n_atoms + (
        0 if ligand_2 is None else ligand_2.n_atoms
    )
    receptor_ref_idxs = receptor_ref_idxs_0 + receptor_start_idx
    _LOGGER.info(f"receptor cavity idxs={receptor_ref_idxs}")

    _apply_atm_restraints(
        system,
        config.restraints,
        ligand_1_com_idxs=ligand_1_idxs,
        ligand_1_ref_idxs=ligand_1_ref_idxs,
        ligand_2_com_idxs=ligand_2_idxs if ligand_2 is not None else None,
        ligand_2_ref_idxs=ligand_2_ref_idxs,
        receptor_ref_idxs=receptor_ref_idxs,
        offset=displacement,
    )

    restraint_idxs = (
        receptor.select(config.restraints.receptor_query) + receptor_start_idx
    )
    _LOGGER.info(f"receptor restrained idxs={receptor_ref_idxs}")

    _apply_receptor_restraints(
        system, config.restraints, {i: topology.xyz[i] for i in restraint_idxs}
    )

    femto.md.utils.openmm.assign_force_groups(system)

    return topology, system
