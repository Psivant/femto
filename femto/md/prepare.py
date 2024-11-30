"""Preparing systems ready for simulation."""

import copy
import logging
import pathlib

import numpy
import openmm
import openmm.app
import openmm.unit
from rdkit import Chem

import femto.md.config
import femto.md.constants
import femto.top

_LOGGER = logging.getLogger(__name__)


def load_ligand(path: pathlib.Path, residue_name: str = "LIG") -> femto.top.Topology:
    """Load a ligand from its coordinates.

    Args:
        path: The path to the ligand file (.sdf)
        residue_name: The residue name to assign to the ligand.

    Returns:
        The loaded ligand
    """
    residue_name = "LIG" if residue_name is None else residue_name

    if path.suffix.lower() == ".sdf":
        mol = Chem.AddHs(Chem.MolFromMolFile(str(path), removeHs=False))
    elif path.suffix.lower() == ".mol2":
        mol = Chem.AddHs(Chem.MolFromMol2File(str(path), removeHs=False))
    else:
        raise NotImplementedError(f"unsupported file format: {path.suffix}")

    top = femto.top.Topology.from_rdkit(mol, residue_name)

    return top


def load_ligands(
    ligand_1_path: pathlib.Path,
    ligand_2_path: pathlib.Path | None,
) -> tuple[femto.top.Topology, femto.top.Topology | None]:
    """Load the first, and optionally second, ligand from their coordinates.

    Args:
        ligand_1_path: The path to the first ligand.
        ligand_2_path: The (optional) path of the second ligand.

    Returns:
        The loaded ligands.
    """

    ligand_1 = load_ligand(ligand_1_path, femto.md.constants.LIGAND_1_RESIDUE_NAME)

    if ligand_2_path is None:
        return ligand_1, None

    ligand_2 = load_ligand(ligand_2_path, femto.md.constants.LIGAND_2_RESIDUE_NAME)

    return ligand_1, ligand_2


def load_receptor(path: pathlib.Path) -> femto.top.Topology:
    """Loads a receptor from its coordinates.

    Args:
        path: The path to the receptor (.pdb).

    Returns:
        The loaded receptor.
    """
    if path.suffix.lower() == ".pdb":
        return femto.top.Topology.from_file(path)

    raise NotImplementedError(f"unsupported file format: {path.suffix}")


def apply_hmr(
    system: openmm.System,
    topology: femto.top.Topology,
    hydrogen_mass: openmm.unit.Quantity = 1.5 * openmm.unit.amu,
):
    """Apply hydrogen mass repartitioning to a system.

    Args:
        system: The system to modify in-place.
        topology: The topology of the system.
        hydrogen_mass: The mass to use ofr hydrogen atoms.
    """

    atoms = topology.atoms

    for bond in topology.bonds:
        atom_1: femto.top.Atom = atoms[bond.idx_1]
        atom_2: femto.top.Atom = atoms[bond.idx_2]

        if atom_1.atomic_num == 1:
            (atom_1, atom_2) = (atom_2, atom_1)

        if atom_2.atomic_num != 1:
            continue
        if atom_1.atomic_num == 1:
            continue

        elements = sorted(a.atomic_num for a in atom_2.residue.atoms)

        if elements == [1, 1, 8]:
            continue

        mass_delta = hydrogen_mass - system.getParticleMass(atom_2.index)

        system.setParticleMass(atom_2.index, hydrogen_mass)
        system.setParticleMass(
            atom_1.index, system.getParticleMass(atom_1.index) - mass_delta
        )


def _compute_box_size(
    receptor: femto.top.Topology | None,
    ligand_1: femto.top.Topology,
    ligand_2: femto.top.Topology | None,
    cofactors: list[femto.top.Topology],
    padding: openmm.unit.Quantity,
    ligand_1_offset: openmm.unit.Quantity | None = None,
    ligand_2_offset: openmm.unit.Quantity | None = None,
    cavity_formers: list[femto.top.Topology] | None = None,
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
    cavity_formers = [] if cavity_formers is None else []

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

    def _rmin(atom: femto.top.Atom) -> float:
        return Chem.PeriodicTable.GetRvdw(Chem.GetPeriodicTable(), atom.atomic_num)

    vdw_radii = numpy.array(
        [
            *([_rmin(atom) for atom in receptor.atoms] if receptor is not None else []),
            *[_rmin(atom) for atom in ligand_1.atoms],
            *([_rmin(atom) for atom in ligand_2.atoms] if ligand_2 is not None else []),
            *([_rmin(atom) for cofactor in cofactors for atom in cofactor.atoms]),
            *([_rmin(atom) for former in cavity_formers for atom in former.atoms]),
        ]
    )
    vdw_radii = vdw_radii.reshape(-1, 1)

    complex_coords = numpy.vstack(
        [
            receptor.xyz.value_in_unit(openmm.unit.angstrom)
            if receptor is not None
            else numpy.zeros((0, 3)),
            ligand_1.xyz.value_in_unit(openmm.unit.angstrom) + ligand_1_offset,
            ligand_2.xyz.value_in_unit(openmm.unit.angstrom) + ligand_2_offset
            if ligand_2 is not None
            else numpy.zeros((0, 3)),
        ]
        + [cofactor.xyz.value_in_unit(openmm.unit.angstrom) for cofactor in cofactors]
        + [former.xyz.value_in_unit(openmm.unit.angstrom) for former in cavity_formers]
    )

    padding = padding.value_in_unit(openmm.unit.angstrom)

    min_coords = (complex_coords - vdw_radii).min(axis=0) - padding
    max_coords = (complex_coords + vdw_radii).max(axis=0) + padding

    box_lengths = numpy.maximum(
        max_coords - min_coords,
        2.0 * numpy.abs(numpy.maximum(ligand_1_offset, ligand_2_offset)),
    )
    return box_lengths.flatten() * openmm.unit.angstrom


def _register_openff_generator(
    ligand_1: femto.top.Topology | None,
    ligand_2: femto.top.Topology | None,
    cofactors: list[femto.top.Topology],
    force_field: openmm.app.ForceField,
):
    """Registers an OpenFF template generator with the force field, to use to
    parameterize ligands and cofactors."""
    from openff.toolkit.topology import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    def _top_to_openff(top: femto.top.Topology) -> Molecule:
        return Molecule.from_rdkit(top.to_rdkit(), allow_undefined_stereo=True)

    molecules = [
        *([_top_to_openff(ligand_1)] if ligand_1 is not None else []),
        *([_top_to_openff(ligand_2)] if ligand_2 is not None else []),
        *([_top_to_openff(cofactor) for cofactor in cofactors]),
    ]

    smirnoff = SMIRNOFFTemplateGenerator(molecules=molecules).generator
    force_field.registerTemplateGenerator(smirnoff)


def prepare_system(
    receptor: femto.top.Topology | None,
    ligand_1: femto.top.Topology | None,
    ligand_2: femto.top.Topology | None,
    solvent: femto.md.config.Solvent,
    cofactors: list[femto.top.Topology] | None = None,
    ligand_1_offset: openmm.unit.Quantity | None = None,
    ligand_2_offset: openmm.unit.Quantity | None = None,
    cavity_formers: list[femto.top.Topology] | None = None,
    extra_params: list[pathlib.Path] | None = None,
) -> tuple[femto.top.Topology, openmm.System]:
    """Solvates and parameterizes a system.

    Args:
        receptor: A receptor to include in the system.
        ligand_1: A primary ligand to include in the system.
        ligand_2: A secondary ligand to include in the system if doing RBFE.
        solvent: The solvent configuration.
        cofactors: Any cofactors to include in the system.
        ligand_1_offset: The amount to offset the first ligand by before computing the
            box size if using a padded box.
        ligand_2_offset: The amount to offset the second ligand by before computing the
            box size if using a padded box.
        cavity_formers: A (optional) list of topologies that should be considered
            'present' when placing the solvent molecules such that they leave cavities,
            but are not added to the final topology themselves.

            Note that cavity formers will be considered when determining the box size.
        extra_params: The paths to any extra parameter files (.xml, .parm) to use
            when parameterizing the system.

    Returns:
        The solvated and parameterized topology and system, containing the ligands, the
        receptor, cofactors, ions and solvent.
    """
    cofactors = [] if cofactors is None else cofactors
    cavity_formers = [] if cavity_formers is None else copy.deepcopy(cavity_formers)

    force_field = openmm.app.ForceField(
        *solvent.default_protein_ff, *([] if extra_params is None else extra_params)
    )

    if solvent.default_ligand_ff is not None:
        _register_openff_generator(ligand_1, ligand_2, cofactors, force_field)

    topology = femto.top.Topology.merge(
        *([] if ligand_1 is None else [ligand_1]),
        *([] if ligand_2 is None else [ligand_2]),
        *([] if receptor is None else [receptor]),
        *cofactors,
    )

    box_size = None

    if solvent.box_padding is not None:
        box_size = _compute_box_size(
            receptor,
            ligand_1,
            ligand_2,
            cofactors,
            solvent.box_padding,
            ligand_1_offset,
            ligand_2_offset,
            cavity_formers,
        )
        _LOGGER.info(f"using a box size of {box_size}")

    for former in cavity_formers:
        for residue in former.residues:
            residue.name = "CAV"

    cavity = femto.top.Topology.merge(topology, *cavity_formers)

    _LOGGER.info("adding solvent and ions")
    modeller = openmm.app.Modeller(cavity.to_openmm(), cavity.xyz)
    modeller.addSolvent(
        force_field,
        model=solvent.water_model.lower(),
        boxSize=box_size,
        numAdded=None if box_size is not None else solvent.n_waters,
        boxShape="cube",
        positiveIon=solvent.cation,
        negativeIon=solvent.anion,
        neutralize=solvent.neutralize,
        ionicStrength=solvent.ionic_strength,
    )
    modeller.addExtraParticles(force_field)

    topology = femto.top.Topology.from_openmm(modeller.topology)
    topology.xyz = (
        numpy.array(modeller.positions.value_in_unit(openmm.unit.angstrom))
        * openmm.unit.angstrom
    )
    topology = topology["not r. CAV"]  # strip cavity formers

    _LOGGER.info("parameterizing the system")

    system = force_field.createSystem(
        topology.to_openmm(),
        nonbondedMethod=openmm.app.PME,
        nonbondedCutoff=0.9 * openmm.unit.nanometer,
        constraints=openmm.app.HBonds,
        rigidWater=True,
    )

    # TODO: is this still needed??
    bound = femto.top.Topology.merge(
        *([] if ligand_1 is None else [ligand_1]),
        *([] if ligand_2 is None else [ligand_2]),
        *([] if receptor is None else [receptor]),
        *cofactors,
    )

    center_offset = (
        bound.xyz.value_in_unit(openmm.unit.angstrom).mean(axis=0)
        * openmm.unit.angstrom
    )
    box_center = (
        numpy.diag(topology.box.value_in_unit(openmm.unit.angstrom)) * 0.5
    ) * openmm.unit.angstrom

    topology.xyz = topology.xyz - center_offset + box_center

    return topology, system