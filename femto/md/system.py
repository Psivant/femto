"""Utilities for preparing and modifying OpenMM systems."""
import logging
import pathlib

import openmm
import parmed

import femto.md.config
import femto.md.utils.amber
from femto.md.constants import LIGAND_1_RESIDUE_NAME, LIGAND_2_RESIDUE_NAME

_LOGGER = logging.getLogger(__name__)


def load_ligand(
    coord_path: pathlib.Path, param_path: pathlib.Path, residue_name: str | None = None
) -> parmed.amber.AmberParm:
    """Load a ligand from its coordinate and parameter definition.

    Args:
        coord_path: The path to the ligand coordinate file (.rst7 / .mol2)
        param_path: The path to the ligand parameter file (.parm7)
        residue_name: The (optional) residue name to assign to the ligand.

    Returns:
        The loaded ligand
    """

    ligand = parmed.amber.AmberParm(str(param_path), str(coord_path))

    for residue in ligand.residues:
        residue.name = residue_name

    ligand.parm_data["RESIDUE_LABEL"] = [residue_name]

    return ligand


def load_ligands(
    ligand_1_coords: pathlib.Path,
    ligand_1_params: pathlib.Path,
    ligand_2_coords: pathlib.Path | None,
    ligand_2_params: pathlib.Path | None,
) -> tuple[parmed.amber.AmberParm, parmed.amber.AmberParm | None]:
    """Load the first, and optionally second, ligand from their coordinates and
    parameters.

    Args:
        ligand_1_coords: The coordinates of the first ligand.
        ligand_1_params: The parameters of the first ligand.
        ligand_2_coords: The (optional) coordinates of the second ligand.
        ligand_2_params: The (optional) parameters of the second ligand.

    Returns:
        The loaded ligands.
    """

    assert (ligand_2_params is None and ligand_2_coords is None) or (
        ligand_2_params is not None and ligand_2_coords is not None
    ), "both or neither of ligand_2_coords and ligand_2_params must be provided"

    ligand_1 = load_ligand(ligand_1_coords, ligand_1_params, LIGAND_1_RESIDUE_NAME)

    if ligand_2_coords is None:
        return ligand_1, None

    ligand_2 = load_ligand(ligand_2_coords, ligand_2_params, LIGAND_2_RESIDUE_NAME)

    return ligand_1, ligand_2


def load_receptor(
    coord_path: pathlib.Path,
    param_path: pathlib.Path | None,
    tleap_sources: list[str] | None = None,
) -> parmed.amber.AmberParm:
    """Loads a receptor from its coordinates and optionally parameters.

    If no parameters are provided, the receptor will be parameterized using tLeap.

    Args:
        coord_path: The coordinates of the receptor.
        param_path: The parameters of the receptor.
        tleap_sources: The tLeap sources to use to parameterize the receptor.
            See ``femto.md.config.DEFAULT_TLEAP_SOURCES`` for the defaults.

    Returns:
        The loaded receptor.
    """
    tleap_sources = (
        femto.md.config.DEFAULT_TLEAP_SOURCES
        if tleap_sources is None
        else tleap_sources
    )

    if param_path is not None:
        return parmed.amber.AmberParm(str(param_path), str(coord_path))

    receptor = parmed.load_file(str(coord_path), structure=True)

    _LOGGER.info(
        f"no receptor parameters provided, the receptor will parameterize using "
        f"tLeap: {tleap_sources}"
    )
    return femto.md.utils.amber.parameterize_structure(receptor, tleap_sources)


def apply_hmr(
    system: openmm.System,
    topology: parmed.Structure,
    hydrogen_mass: openmm.unit.Quantity = 1.5 * openmm.unit.amu,
):
    """Apply hydrogen mass repartitioning to a system.

    Args:
        system: The system to modify in-place.
        topology: The topology of the system.
        hydrogen_mass: The mass to use ofr hydrogen atoms.
    """

    for bond in topology.bonds:
        atom_1, atom_2 = bond.atom1, bond.atom2

        if atom_1.atomic_number == 1:
            (atom_1, atom_2) = (atom_2, atom_1)

        if atom_2.atomic_number != 1:
            continue
        if atom_1.atomic_number == 1:
            continue

        elements = sorted(a.atomic_number for a in atom_2.residue.atoms)

        if elements == [1, 1, 8]:
            continue

        mass_delta = hydrogen_mass - system.getParticleMass(atom_2.idx)

        system.setParticleMass(atom_2.idx, hydrogen_mass)
        system.setParticleMass(
            atom_1.idx, system.getParticleMass(atom_1.idx) - mass_delta
        )
