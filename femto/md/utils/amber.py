"""Utilities for manipulating AMBER data and tools"""

import copy
import pathlib
import subprocess
import tempfile

import parmed


def extract_water_and_ions_mask(structure: parmed.Structure) -> list[bool]:
    """Returns a per atom mask that is true if the atom belongs to a water molecule
    or an ion, and false otherwise.

    Args:
        structure: The structure to extract the mask from.

    Returns:
        The selection mask.
    """

    solvent_residues = {
        i
        for i, residue in enumerate(structure.residues)
        if sorted(a.element for a in residue.atoms) == [1, 1, 8]
        # a bit of a hack to check for ions.
        or len(residue.atoms) == 1
    }
    solvent_mask = [
        i in solvent_residues
        for i, residue in enumerate(structure.residues)
        for _ in residue.atoms
    ]

    return solvent_mask


def parameterize_structure(
    structure: parmed.Structure, tleap_sources: list[str]
) -> parmed.amber.AmberParm:
    """Parameterizes a given structure using tLeap.

    Args:
        structure: The structure to parameterize.
        tleap_sources: The tLeap parameters to source.

    Returns:
        The parameterized structure
    """

    if len(structure.atoms) == 0:
        return copy.deepcopy(structure)

    control_file = "\n".join(
        [
            *[f"source {source}" for source in tleap_sources],
            'addPdbAtomMap { { "Na" "NA" } { "Na+" "NA" } { "Cl" "CL" } { "Cl-" "CL" } }',  # noqa: E501
            "structure = loadpdb structure.pdb",
            "saveAmberParm structure structure.parm7 structure.rst7",
            "quit",
        ]
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = pathlib.Path(tmp_dir)

        structure.save(str(tmp_dir / "structure.pdb"))
        (tmp_dir / "tleap.in").write_text(control_file)

        result = subprocess.run(
            ["tleap", "-f", "tleap.in"], cwd=tmp_dir, capture_output=True, text=True
        )

        param_path, coord_path = tmp_dir / "structure.parm7", tmp_dir / "structure.rst7"

        if result.returncode != 0 or not param_path.exists() or not coord_path.exists():
            raise RuntimeError(f"TLeap failed - {result.stdout}\n{result.stderr}")

        return parmed.amber.AmberParm(str(param_path), str(coord_path))
