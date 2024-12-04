"""Set up the system for SepTop calculations."""

import copy
import logging
import pathlib

import mdtop
import numpy.linalg.linalg
import openmm
import openmm.app
import openmm.unit

import femto.fe.fep
import femto.fe.reference
import femto.md.constants
import femto.md.prepare
import femto.md.rest
import femto.md.restraints
import femto.md.utils.openmm

_LOGGER = logging.getLogger(__name__)


_ANGSTROM = openmm.unit.angstrom


LAMBDA_BORESCH_LIGAND_1 = "lambda_boresch_lig_1"
"""The name of the context variable used to control the Boresch-style restraints on the
first ligand."""
LAMBDA_BORESCH_LIGAND_2 = "lambda_boresch_lig_2"
"""The name of the context variable used to control the Boresch-style restraints on the
second ligand."""


def _compute_ligand_offset(
    ligand_1: mdtop.Topology, ligand_2: mdtop.Topology
) -> openmm.unit.Quantity:
    """Computes the amount to offset the second ligand by in the solution phase during
    RBFE calculations.

    Args:
        ligand_1: The first ligand.
        ligand_2: The second ligand.

    Returns:
        The amount to offset the second ligand by.
    """

    xyz_1 = ligand_1.xyz.value_in_unit(openmm.unit.angstrom)
    xyz_2 = ligand_2.xyz.value_in_unit(openmm.unit.angstrom)

    ligand_1_radius = numpy.linalg.norm(xyz_1 - xyz_1.mean(axis=0), axis=1).max()
    ligand_2_radius = numpy.linalg.norm(xyz_2 - xyz_2.mean(axis=0), axis=1).max()
    ligand_distance = (ligand_1_radius + ligand_2_radius) * 1.5
    ligand_offset = xyz_1.mean(0) - xyz_2.mean(0)
    ligand_offset[0] += ligand_distance
    return ligand_offset * _ANGSTROM


def _apply_complex_restraints(
    topology: mdtop.Topology,
    receptor_ref_idxs: tuple[int, int, int],
    ligand_ref_idxs: tuple[int, int, int],
    config: "femto.fe.septop.SepTopComplexRestraints",
    system: openmm.System,
    ctx_variable: str,
):
    """Apply Boresch-style restraints to align a ligand with the complex.

    Args:
        topology: The full topology of the complex phase.
        receptor_ref_idxs: The reference indices of the receptor, i.e. r1, r2, r3.
        ligand_ref_idxs: The reference indices of the ligand, i.e. l1, l2, l3.
        system: The OpenMM system to add the restraints to.
        ctx_variable: The name of the context variable to use to control the restraint
            strength.
    """

    coords = topology.xyz.value_in_unit(openmm.unit.angstrom)

    distance_0 = 5.0  # based on original SepTop implementation.

    distance_r1_l1 = numpy.linalg.norm(
        coords[receptor_ref_idxs[0]] - coords[ligand_ref_idxs[0]]
    )
    scale = (distance_r1_l1 / distance_0) ** 2 if config.scale_k_angle_a else 1.0

    config_scaled = copy.deepcopy(config)
    config_scaled.k_angle_a *= scale

    force = femto.md.restraints.create_boresch_restraint(
        config_scaled,
        receptor_ref_idxs[::-1],  # expects [r3, r2, r1], not [r1, r2, r3]
        ligand_ref_idxs,
        coords * _ANGSTROM,
        ctx_variable,
    )
    system.addForce(force)

    femto.md.utils.openmm.assign_force_groups(system)


def _apply_solution_restraints(
    topology: mdtop.Topology,
    ligand_1_ref_idx: int,
    ligand_2_ref_idx: int,
    config: "femto.fe.septop.SepTopSolutionRestraints",
    system: openmm.System,
):
    """Apply a distance restraints between the ligands.

    Args:
        topology: The full topology of the complex phase.
        ligand_1_ref_idx: The reference index of the first ligand.
        ligand_2_ref_idx: The reference index of the second ligand.
        system: The OpenMM system to add the restraints to.
    """

    coords = topology.xyz.value_in_unit(openmm.unit.angstrom)

    distance = numpy.linalg.norm(coords[ligand_1_ref_idx] - coords[ligand_2_ref_idx])

    force = openmm.HarmonicBondForce()
    force.addBond(
        ligand_1_ref_idx,
        ligand_2_ref_idx,
        distance * openmm.unit.angstrom,
        config.k_distance,
    )
    force.setName(femto.md.constants.OpenMMForceName.ALIGNMENT_RESTRAINT)
    force.setForceGroup(femto.md.constants.OpenMMForceGroup.ALIGNMENT_RESTRAINT)

    system.addForce(force)


def _setup_system(
    config: "femto.fe.septop.SepTopSetupStage",
    ligand_1: mdtop.Topology,
    ligand_2: mdtop.Topology | None,
    receptor: mdtop.Topology | None,
    cofactors: list[mdtop.Topology] | None,
    ligand_1_ref_query: tuple[str, str, str] | None,
    ligand_2_ref_query: tuple[str, str, str] | None,
    ligand_2_offset: openmm.unit.Quantity | None,
    extra_params: list[pathlib.Path] | None,
) -> tuple[
    openmm.System, mdtop.Topology, tuple[int, int, int], tuple[int, int, int] | None
]:
    _LOGGER.info("preparing system.")
    topology, system = femto.md.prepare.prepare_system(
        receptor,
        ligand_1,
        ligand_2,
        cofactors,
        config,
        ligand_1_offset=None,
        ligand_2_offset=ligand_2_offset,
        extra_params=extra_params,
    )

    if config.apply_hmr:
        _LOGGER.info("applying HMR.")
        femto.md.prepare.apply_hmr(system, topology, config.hydrogen_mass)

    _LOGGER.info("applying FEP.")
    ligand_1_idxs = topology.select(f"resn {femto.md.constants.LIGAND_1_RESIDUE_NAME}")
    ligand_2_idxs = topology.select(f"resn {femto.md.constants.LIGAND_2_RESIDUE_NAME}")

    ligand_1 = topology.subset(ligand_1_idxs)
    ligand_2 = topology.subset(ligand_2_idxs) if ligand_2 is not None else None

    femto.fe.fep.apply_fep(
        system, {*ligand_1_idxs}, {*ligand_2_idxs}, config.fep_config
    )

    if config.apply_rest:
        _LOGGER.info("applying REST2.")

        solute_idxs = {*ligand_1_idxs, *({} if ligand_2 is None else ligand_2_idxs)}
        femto.md.rest.apply_rest(system, solute_idxs, config.rest_config)

    ligand_1_ref_idxs, ligand_2_ref_idxs = femto.fe.reference.select_ligand_idxs(
        ligand_1, ligand_2, "baumann", ligand_1_ref_query, ligand_2_ref_query
    )

    ligand_1_ref_idxs = tuple(int(ligand_1_idxs[i]) for i in ligand_1_ref_idxs)
    _LOGGER.info(f"ligand 1 ref idxs={ligand_1_idxs}")

    if ligand_2 is not None and ligand_2_ref_idxs is not None:
        ligand_2_ref_idxs = tuple(int(ligand_2_idxs[i]) for i in ligand_2_ref_idxs)
        _LOGGER.info(f"ligand 2 ref idxs={ligand_2_idxs}")

    return system, topology, ligand_1_ref_idxs, ligand_2_ref_idxs


def setup_complex(
    config: "femto.fe.septop.SepTopSetupStage",
    receptor: mdtop.Topology,
    ligand_1: mdtop.Topology,
    ligand_2: mdtop.Topology | None,
    cofactors: list[mdtop.Topology] | None,
    receptor_ref_query: tuple[str, str, str] | None = None,
    ligand_1_ref_query: tuple[str, str, str] | None = None,
    ligand_2_ref_query: tuple[str, str, str] | None = None,
    extra_params: list[pathlib.Path] | None = None,
) -> tuple[mdtop.Topology, openmm.System]:
    """Prepares a system ready for running the SepTop method.

    Args:
        config: The configuration for setting up the system.
        receptor: The receptor.
        ligand_1: The first ligand.
        ligand_2: The second ligand if one is present.
        cofactors: Any cofactors.
        receptor_ref_query: The query to select the reference atoms of the receptor.
        ligand_1_ref_query: The query to select the reference atoms of the first ligand.
        ligand_2_ref_query: The query to select the reference atoms of the second ligand
        extra_params: The paths to any extra parameter files (.xml, .parm) to use
            when parameterizing the system.

    Returns:
        The prepared topology and OpenMM system object.
    """
    import femto.fe.septop

    if not isinstance(config.restraints, femto.fe.septop.SepTopComplexRestraints):
        raise ValueError("invalid restraint config")

    system, topology, ligand_1_ref_idxs, ligand_2_ref_idxs = _setup_system(
        config,
        ligand_1,
        ligand_2,
        receptor,
        cofactors,
        ligand_1_ref_query,
        ligand_2_ref_query,
        None,
        extra_params,
    )

    _LOGGER.info("applying restraints.")

    if receptor_ref_query is None:
        _LOGGER.info("selecting receptor reference atoms")

        receptor_ref_idxs_1 = femto.fe.reference.select_receptor_idxs(
            topology, ligand_1_ref_idxs
        )
    else:
        receptor_ref_idxs_1 = femto.fe.reference.queries_to_idxs(
            receptor, receptor_ref_query
        )

        ligand_idxs = topology.select(
            f"resn {femto.md.constants.LIGAND_1_RESIDUE_NAME} | "
            f"resn {femto.md.constants.LIGAND_2_RESIDUE_NAME}"
        )
        receptor_start_idx = 0 if len(ligand_idxs) == 0 else (max(ligand_idxs) + 1)
        receptor_ref_idxs_1 = tuple(v + receptor_start_idx for v in receptor_ref_idxs_1)

    _LOGGER.info(f"receptor ref idxs for ligand 1={receptor_ref_idxs_1}")

    _apply_complex_restraints(
        topology,
        receptor_ref_idxs_1,
        ligand_1_ref_idxs,
        config.restraints,
        system,
        LAMBDA_BORESCH_LIGAND_1,
    )

    if ligand_2 is None:
        return topology, system

    receptor_ref_idxs_2 = receptor_ref_idxs_1

    if receptor_ref_query is None and femto.fe.reference.check_receptor_idxs(
        topology, receptor_ref_idxs_2, ligand_2_ref_idxs
    ):
        _LOGGER.info("selecting alternate receptor reference atoms for ligand 2")

        receptor_ref_idxs_2 = femto.fe.reference.select_receptor_idxs(
            topology, ligand_2_ref_idxs
        )

    _LOGGER.info(f"receptor ref idxs for ligand 2={receptor_ref_idxs_2}")

    _apply_complex_restraints(
        topology,
        receptor_ref_idxs_2,
        ligand_2_ref_idxs,
        config.restraints,
        system,
        LAMBDA_BORESCH_LIGAND_2,
    )
    return topology, system


def setup_solution(
    config: "femto.fe.septop.SepTopSetupStage",
    ligand_1: mdtop.Topology,
    ligand_2: mdtop.Topology | None,
    ligand_1_ref_query: tuple[str, str, str] | None = None,
    ligand_2_ref_query: tuple[str, str, str] | None = None,
    extra_params: list[pathlib.Path] | None = None,
) -> tuple[mdtop.Topology, openmm.System]:
    """Prepares a system ready for running the SepTop method.

    Args:
        config: The configuration for setting up the system.
        ligand_1: The first ligand.
        ligand_2: The second ligand if one is present.
        ligand_1_ref_query: The query to select the reference atoms of the first ligand.
        ligand_2_ref_query: The query to select the reference atoms of the second ligand
        extra_params: The paths to any extra parameter files (.xml, .parm) to use
            when parameterizing the system.

    Returns:
        The prepared topology and OpenMM system object.
    """
    import femto.fe.septop

    config = copy.deepcopy(config)

    if config.box_padding is None:
        raise NotImplementedError("box padding must be set for solution phase")

    restraint_config = config.restraints

    if not isinstance(restraint_config, femto.fe.septop.SepTopSolutionRestraints):
        raise ValueError("invalid restraint config")

    if ligand_2 is not None:
        ligand_2.xyz += _compute_ligand_offset(ligand_1, ligand_2)

    system, topology, ligand_1_ref_idxs, ligand_2_ref_idxs = _setup_system(
        config,
        ligand_1,
        ligand_2,
        None,
        None,
        ligand_1_ref_query,
        ligand_2_ref_query,
        None,
        extra_params,
    )

    if ligand_2 is not None:
        _apply_solution_restraints(
            topology,
            ligand_1_ref_idxs[1],
            ligand_2_ref_idxs[1],
            config.restraints,
            system,
        )
    femto.md.utils.openmm.assign_force_groups(system)

    return topology, system
