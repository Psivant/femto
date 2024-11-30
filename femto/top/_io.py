import pathlib
import warnings

import openmm.app
from rdkit import Chem

import femto.top


def _find_section(lines: list[str], header: str) -> int | None:
    """Finds the index of a section in a MOL2 file."""
    idxs = [
        i
        for i, line in enumerate(lines)
        if line.strip().startswith(f"@<TRIPOS>{header}")
    ]

    if len(idxs) > 1:
        raise ValueError(f"found multiple {header} sections")

    return idxs[0] if len(idxs) > 0 else None


def _guess_atomic_num(name: str) -> int:
    """Hackish attempt to guess the atomic number of an atom based on its name."""

    name = "".join(c for c in name if c.isalpha()).strip()

    if len(name) == 0:
        return 0

    symbols = [
        f"{name[0].upper()}{name[1].lower()}" if len(name) > 1 else name[0].upper(),
        name[0].upper(),
    ]

    for symbol in symbols:
        try:
            return openmm.app.Element.getBySymbol(symbol).atomic_number
        except KeyError:
            continue

    return 0


def _parse_molecule_section(lines: list[str]) -> tuple[int, int]:
    """Parses the MOLECULE section of a MOL2 file."""

    start_idx = _find_section(lines, "MOLECULE")

    if start_idx is None:
        raise ValueError("no MOLECULE section found")

    info_split = [int(x) for x in lines[start_idx + 2].split()]

    n_atoms = info_split[0]
    n_bonds = info_split[1]

    return n_atoms, n_bonds


def _parse_atom_section(lines: list[str], n_atoms: int, n_bonds: int) -> Chem.RWMol:
    """Parses the ATOM and BOND section of a MOL2 file."""

    start_idx = _find_section(lines, "ATOM")

    if start_idx is None:
        raise ValueError("no ATOM section found")

    mol = Chem.RWMol()

    xyz = []

    for i in range(start_idx + 1, start_idx + n_atoms + 1):
        atom_line = lines[i].split()

        _, name, x, y, z, type_ = atom_line[:6]

        atomic_num = _guess_atomic_num(name)

        if atomic_num == 0:
            atomic_num = _guess_atomic_num(type_)

        atom_idx = mol.AddAtom(Chem.Atom(atomic_num))
        mol.GetAtomWithIdx(atom_idx).SetProp("_Name", name)

        xyz.append([float(x), float(y), float(z)])

    conf = Chem.Conformer(n_atoms)

    for i, coord in enumerate(xyz):
        conf.SetAtomPosition(i, coord)

    mol.AddConformer(conf, assignId=True)

    start_idx = _find_section(lines, "BOND")

    if start_idx is None:
        raise ValueError("no BOND section found")

    rdkit_bond_order_map = {
        "1": Chem.BondType.SINGLE,
        "2": Chem.BondType.DOUBLE,
        "3": Chem.BondType.TRIPLE,
        "ar": Chem.BondType.AROMATIC,
    }

    for i in range(start_idx + 1, start_idx + n_bonds + 1):
        bond_line = lines[i].split()

        _, idx_1, idx_2 = bond_line[:3]

        order_str = 1.0 if len(bond_line) < 4 else bond_line[3]
        order = rdkit_bond_order_map[order_str]

        mol.AddBond(int(idx_1) - 1, int(idx_2) - 1, order)

    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)

    for _, atom in enumerate(mol.GetAtoms()):
        valence = (
            sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()])
            + atom.GetNumExplicitHs()
        )
        expected_valence = Chem.GetPeriodicTable().GetDefaultValence(
            atom.GetAtomicNum()
        )
        formal_charge = int(valence - expected_valence)
        atom.SetFormalCharge(formal_charge)

    mol = mol.GetMol()
    Chem.SanitizeMol(mol)

    return mol


def parse_mol2(
    path: pathlib.Path, residue_name: str = "UNK", chain: str = ""
) -> femto.top.Topology:
    """Parses a MOL2 file.

    Args:
        path: The path to the MOL2 file.
        residue_name: The residue name to use for the ligand.
        chain: The chain ID to use for the ligand.

    Returns:
        The parsed topology.
    """

    warnings.warn(
        "Parsing MOL2 is dangerous and deprecated, and will be removed in a future "
        "release. We recommend using SDF files instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    lines = path.read_text().splitlines()
    lines = [line for line in lines if not line.startswith("#")]

    n_atoms, n_bonds = _parse_molecule_section(lines)

    mol = _parse_atom_section(lines, n_atoms, n_bonds)
    return femto.top.Topology.from_rdkit(mol, residue_name, chain)
