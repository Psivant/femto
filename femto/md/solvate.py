"""Solvate protein ligand systems"""

import logging

import numpy
import openmm.app
import openmm.unit
import parmed
import parmed.openmm

import femto.md.config
from femto.md.utils import amber as amber_utils

_LOGGER = logging.getLogger(__name__)


class _MockForceField:
    """Create a mocked version of the OpenMM force field class so that we can
    provide a parmed structure directly to Modeller
    """

    def __init__(self, structure: parmed.structure):
        self._system = openmm.System()
        force = openmm.NonbondedForce()

        for atom in structure.atoms:
            self._system.addParticle(atom.mass * openmm.unit.dalton)
            force.addParticle(
                atom.charge * openmm.unit.elementary_charge,
                atom.sigma * openmm.unit.angstrom,
                abs(atom.epsilon) * openmm.unit.kilocalories_per_mole,
            )

        self._system.addForce(force)

    def createSystem(self, *_, **__) -> openmm.System:
        return self._system


def _compute_box_size(
    receptor: parmed.Structure | None,
    ligand_1: parmed.Structure,
    ligand_2: parmed.Structure | None,
    padding: openmm.unit.Quantity,
    ligand_1_offset: openmm.unit.Quantity | None = None,
    ligand_2_offset: openmm.unit.Quantity | None = None,
    cavity_formers: list[parmed.Structure] | None = None,
) -> openmm.unit.Quantity:
    """Computes the size of the simulation box based on the coordinates of complex with
    the ligand offset outside the binding site and a specified additional padding.

    Args:
        receptor: The (optional) receptor.
        ligand_1: The first ligand.
        ligand_2: The (optional) second ligand.
        padding: The amount of extra padding to add.
        ligand_1_offset: The amount to offset the first ligand by before computing the
            box size.
        ligand_2_offset: The amount to offset the second ligand by before computing the
            box size.

    Returns:
        The length of each box axis.
    """
    cavity_formers = [] if cavity_formers is None else cavity_formers

    ligand_1_offset = (
        ligand_1_offset.value_in_unit(openmm.unit.angstrom)
        if ligand_1_offset is not None
        else numpy.zeros((1, 3))
    )
    ligand_2_offset = (
        ligand_2_offset.value_in_unit(openmm.unit.angstrom)
        if ligand_2_offset is not None
        else numpy.zeros((1, 3))
    )

    vdw_radii = numpy.array(
        [
            *([atom.rmin for atom in receptor.atoms] if receptor is not None else []),
            *[atom.rmin for atom in ligand_1.atoms],
            *([atom.rmin for atom in ligand_2.atoms] if ligand_2 is not None else []),
            *([atom.rmin for former in cavity_formers for atom in former.atoms]),
        ]
    )
    vdw_radii = vdw_radii.reshape(-1, 1) * 2.0

    complex_coords = numpy.vstack(
        [
            numpy.array(receptor.coordinates)
            if receptor is not None
            else numpy.zeros((0, 3)),
            numpy.array(ligand_1.coordinates) + ligand_1_offset,
            numpy.array(ligand_2.coordinates) + ligand_2_offset
            if ligand_2 is not None
            else numpy.zeros((0, 3)),
        ]
        + [former.coordinates for former in cavity_formers]
    )

    padding = padding.value_in_unit(openmm.unit.angstrom)

    min_coords = (complex_coords - vdw_radii).min(axis=0) - padding
    max_coords = (complex_coords + vdw_radii).max(axis=0) + padding

    box_lengths = numpy.maximum(
        max_coords - min_coords,
        2.0 * numpy.abs(numpy.maximum(ligand_1_offset, ligand_2_offset)),
    )
    return box_lengths.flatten() * openmm.unit.angstrom


def solvate_system(
    receptor: parmed.Structure | None,
    ligand_1: parmed.Structure,
    ligand_2: parmed.Structure | None,
    solvent: femto.md.config.Solvent,
    ligand_1_offset: openmm.unit.Quantity | None = None,
    ligand_2_offset: openmm.unit.Quantity | None = None,
    cavity_formers: list[parmed.Structure] | None = None,
) -> parmed.Structure:
    """Solvates a system.

    Args:
        receptor: The (optional) receptor.
        ligand_1: The first ligand.
        ligand_2: The (optional) second ligand.
        solvent: The solvent configuration.
        ligand_1_offset: The amount to offset the first ligand by before computing the
            box size if using a padded box.
        ligand_2_offset: The amount to offset the second ligand by before computing the
            box size if using a padded box.
        cavity_formers: The (optional) list of structures that should be considered
            'present' when placing the solvent molecules such that they leave cavities,
            but are not added to the final topology themselves.

            Note that cavity formers will be considered when determining the box size.

    Returns:
        The solvated system containing the ligands, the receptor, ions and the solvent.
    """

    bound = ligand_1

    if ligand_2 is not None:
        bound = bound + ligand_2
    if receptor is not None:
        bound = bound + receptor

    cavity = bound

    if cavity_formers is not None:
        for former in cavity_formers:
            cavity = cavity + former

    modeller = openmm.app.Modeller(cavity.topology, cavity.positions)

    box_size = None

    if solvent.box_padding is not None:
        box_size = _compute_box_size(
            receptor,
            ligand_1,
            ligand_2,
            solvent.box_padding,
            ligand_1_offset,
            ligand_2_offset,
            cavity_formers,
        )

        _LOGGER.info(f"using a box size of {box_size}")

    modeller.addSolvent(
        _MockForceField(cavity),
        model=solvent.water_model.lower(),
        boxSize=box_size,
        numAdded=None if box_size is not None else solvent.n_waters,
        boxShape="cube",
        positiveIon=solvent.cation,
        negativeIon=solvent.anion,
        neutralize=solvent.neutralize,
        ionicStrength=solvent.ionic_strength,
    )

    solution: parmed.Structure = parmed.openmm.load_topology(
        modeller.topology, None, modeller.positions
    )
    solution_box = solution.box

    _LOGGER.info("parameterizing solvent")

    solvent_mask = amber_utils.extract_water_and_ions_mask(bound)
    bound.strip(solvent_mask)

    complex_mask = [1 - i for i in amber_utils.extract_water_and_ions_mask(solution)]
    solution.strip(complex_mask)

    solution = amber_utils.parameterize_structure(solution, solvent.tleap_sources)

    system = bound + solution
    system.box = solution_box

    center_offset = numpy.array(bound.coordinates).mean(axis=0)

    coordinates = numpy.array(system.coordinates)
    coordinates -= center_offset
    coordinates += system.box[:3] * 0.5

    system.coordinates = coordinates.tolist()

    return system
