"""Set up the system for SepTop calculations."""

import copy
import logging

import numpy.linalg.linalg
import openmm
import openmm.app
import openmm.unit
import parmed

import femto.fe.fep
import femto.fe.reference
import femto.md.constants
import femto.md.rest
import femto.md.restraints
import femto.md.solvate
import femto.md.system
import femto.md.utils.openmm

_LOGGER = logging.getLogger(__name__)


_ANGSTROM = openmm.unit.angstrom


LAMBDA_BORESCH_LIGAND_1 = "lambda_boresch_lig_1"
"""The name of the context variable used to control the Boresch-style restraints on the
first ligand."""
LAMBDA_BORESCH_LIGAND_2 = "lambda_boresch_lig_2"
"""The name of the context variable used to control the Boresch-style restraints on the
second ligand."""


def _offset_ligand(ligand: parmed.Structure, offset: openmm.unit.Quantity):
    """Offsets the coordinates of the specified ligand by an offset.

    Args:
        ligand: The ligand to offset.
        offset: The amount to offset the ligand by.
    """

    for atom in ligand.atoms:
        atom.xx += offset[0].value_in_unit(openmm.unit.angstrom)
        atom.xy += offset[1].value_in_unit(openmm.unit.angstrom)
        atom.xz += offset[2].value_in_unit(openmm.unit.angstrom)


def _compute_ligand_offset(
    ligand_1: parmed.Structure, ligand_2: parmed.Structure
) -> openmm.unit.Quantity:
    """Computes the amount to offset the second ligand by in the solution phase during
    RBFE calculations.

    Args:
        ligand_1: The first ligand.
        ligand_2: The second ligand.

    Returns:
        The amount to offset the second ligand by.
    """

    ligand_1_radius = numpy.linalg.norm(
        ligand_1.coordinates - ligand_1.coordinates.mean(axis=0), axis=1
    ).max()
    ligand_2_radius = numpy.linalg.norm(
        ligand_2.coordinates - ligand_2.coordinates.mean(axis=0), axis=1
    ).max()
    ligand_distance = (ligand_1_radius + ligand_2_radius) * 1.5
    ligand_offset = ligand_1.coordinates.mean(0) - ligand_2.coordinates.mean(0)
    ligand_offset[0] += ligand_distance
    return ligand_offset * _ANGSTROM


def _apply_complex_restraints(
    topology: parmed.Structure,
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

    coords = topology.coordinates

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


def _apply_solution_restraints(
    topology: parmed.Structure,
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

    coords = topology.coordinates * openmm.unit.angstrom

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
    ligand_1: parmed.amber.AmberParm,
    ligand_2: parmed.amber.AmberParm | None,
    receptor: parmed.amber.AmberParm | None,
    ligand_1_ref_query: tuple[str, str, str] | None,
    ligand_2_ref_query: tuple[str, str, str] | None,
    ligand_2_offset: openmm.unit.Quantity | None = None,
) -> tuple[
    openmm.System, parmed.Structure, tuple[int, int, int], tuple[int, int, int] | None
]:
    _LOGGER.info("solvating system")
    topology = femto.md.solvate.solvate_system(
        receptor, ligand_1, ligand_2, config.solvent, ligand_2_offset=ligand_2_offset
    )

    _LOGGER.info("creating OpenMM system")
    system = topology.createSystem(
        nonbondedMethod=openmm.app.PME,
        nonbondedCutoff=0.9 * openmm.unit.nanometer,
        constraints=openmm.app.HBonds,
        rigidWater=True,
    )

    if config.apply_hmr:
        _LOGGER.info("applying HMR.")
        femto.md.system.apply_hmr(system, topology, config.hydrogen_mass)

    _LOGGER.info("applying FEP.")
    ligand_1_idxs = set(range(len(ligand_1.atoms)))
    ligand_2_idxs = None

    if ligand_2 is not None:
        ligand_2_idxs = {i + len(ligand_1_idxs) for i in range(len(ligand_2.atoms))}

    femto.fe.fep.apply_fep(system, ligand_1_idxs, ligand_2_idxs, config.fep_config)

    if config.apply_rest:
        _LOGGER.info("applying REST2.")

        solute_idxs = {*ligand_1_idxs, *({} if ligand_2 is None else ligand_2_idxs)}
        femto.md.rest.apply_rest(system, solute_idxs, config.rest_config)

    ligand_ref_idxs = femto.fe.reference.select_ligand_idxs(
        ligand_1, ligand_2, "baumann", ligand_1_ref_query, ligand_2_ref_query
    )

    ligand_1_ref_idxs, ligand_2_ref_idxs = ligand_ref_idxs

    if ligand_2_ref_idxs is not None:
        ligand_2_ref_idxs = tuple(i + len(ligand_1.atoms) for i in ligand_2_ref_idxs)

    return system, topology, ligand_1_ref_idxs, ligand_2_ref_idxs


def setup_complex(
    config: "femto.fe.septop.SepTopSetupStage",
    receptor: parmed.amber.AmberParm,
    ligand_1: parmed.amber.AmberParm,
    ligand_2: parmed.amber.AmberParm | None,
    receptor_ref_query: tuple[str, str, str] | None = None,
    ligand_1_ref_query: tuple[str, str, str] | None = None,
    ligand_2_ref_query: tuple[str, str, str] | None = None,
) -> tuple[parmed.Structure, openmm.System]:
    """Prepares a system ready for running the SepTop method.

    Returns:
        The prepared topology and OpenMM system object.
    """
    import femto.fe.septop

    config = copy.deepcopy(config)

    restraint_config = config.restraints

    if not isinstance(restraint_config, femto.fe.septop.SepTopComplexRestraints):
        raise ValueError("invalid restraint config")

    system, topology, ligand_1_ref_idxs, ligand_2_ref_idxs = _setup_system(
        config, ligand_1, ligand_2, receptor, ligand_1_ref_query, ligand_2_ref_query
    )

    ligands = [ligand_1] + ([] if ligand_2 is None else [ligand_2])
    idx_offset = sum(len(ligand.atoms) for ligand in ligands)

    _LOGGER.info("applying restraints.")

    if receptor_ref_query is None:
        _LOGGER.info("selecting receptor reference atoms")

        receptor_ref_idxs_1 = femto.fe.reference.select_receptor_idxs(
            receptor, ligand_1, ligand_1_ref_idxs
        )
        receptor_ref_idxs_2 = receptor_ref_idxs_1

        if ligand_2 is not None and not femto.fe.reference.check_receptor_idxs(
            receptor, receptor_ref_idxs_1, ligand_2, ligand_1_ref_idxs
        ):
            _LOGGER.info("selecting alternate receptor reference atoms for ligand 2")
            receptor_ref_idxs_2 = femto.fe.reference.select_receptor_idxs(
                receptor, ligand_2, ligand_2_ref_idxs
            )

    else:
        receptor_ref_idxs_1 = femto.fe.reference.queries_to_idxs(
            receptor, receptor_ref_query
        )
        receptor_ref_idxs_2 = receptor_ref_idxs_1

    _LOGGER.info(f"receptor ref idxs for ligand 1={receptor_ref_idxs_1}")
    receptor_ref_idxs_1 = tuple(i + idx_offset for i in receptor_ref_idxs_1)

    _apply_complex_restraints(
        topology,
        receptor_ref_idxs_1,
        ligand_1_ref_idxs,
        restraint_config,
        system,
        LAMBDA_BORESCH_LIGAND_1,
    )

    if ligand_2 is not None:
        _LOGGER.info(f"receptor ref idxs for ligand 2={receptor_ref_idxs_2}")
        receptor_ref_idxs_2 = tuple(i + idx_offset for i in receptor_ref_idxs_2)

        _apply_complex_restraints(
            topology,
            receptor_ref_idxs_2,
            ligand_2_ref_idxs,
            restraint_config,
            system,
            LAMBDA_BORESCH_LIGAND_2,
        )

    femto.md.utils.openmm.assign_force_groups(system)

    return topology, system


def setup_solution(
    config: "femto.fe.septop.SepTopSetupStage",
    ligand_1: parmed.amber.AmberParm,
    ligand_2: parmed.amber.AmberParm | None,
    ligand_1_ref_query: tuple[str, str, str] | None = None,
    ligand_2_ref_query: tuple[str, str, str] | None = None,
) -> tuple[parmed.Structure, openmm.System]:
    """Prepares a system ready for running the SepTop method.

    Returns:
        The prepared topology and OpenMM system object.
    """
    import femto.fe.septop

    config = copy.deepcopy(config)

    if config.solvent.box_padding is None:
        raise NotImplementedError

    restraint_config = config.restraints

    if not isinstance(restraint_config, femto.fe.septop.SepTopSolutionRestraints):
        raise ValueError("invalid restraint config")

    ligand_offset = None

    if ligand_2 is not None:
        ligand_offset = _compute_ligand_offset(ligand_1, ligand_2)
        _offset_ligand(ligand_2, ligand_offset)

    system, topology, ligand_1_ref_idxs, ligand_2_ref_idxs = _setup_system(
        config,
        ligand_1,
        ligand_2,
        None,
        ligand_1_ref_query,
        ligand_2_ref_query,
        None if ligand_2 is None else -ligand_offset,
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
