"""Utilities for automatically selecting 'reference' atoms for alignment."""
import itertools
import logging
import typing

import mdtraj
import networkx
import numpy
import openmm.unit
import parmed
import scipy.spatial
import scipy.spatial.distance

import femto.fe.config
import femto.md.utils.geometry

_LOGGER = logging.getLogger(__name__)


_COLLINEAR_THRESHOLD = 0.9  # roughly 25 degrees

# values taken from the SepTop reference implementation at commit 7af0b4d
_ANGLE_CHECK_FORCE_CONSTANT = 20.0 * openmm.unit.kilocalorie_per_mole
_ANGLE_CHECK_T = 298.15 * openmm.unit.kelvin
_ANGLE_CHECK_RT = openmm.unit.MOLAR_GAS_CONSTANT_R * _ANGLE_CHECK_T

_ANGLE_CHECK_FACTOR = 0.5 * _ANGLE_CHECK_FORCE_CONSTANT / _ANGLE_CHECK_RT
_ANGLE_CHECK_CUTOFF = 10.0  # units of kT

_DIHEDRAL_CHECK_CUTOFF = numpy.deg2rad(150.0)


def _is_angle_linear(coords: numpy.ndarray, idxs: tuple[int, int, int]) -> bool:
    """Check if angle is within 10 kT from 0 or 180 following the SepTop reference
    implementation.

    Args:
        coords: The full set of coordinates.
        idxs: The indices of the three atoms that form the angle.

    Returns:
        True if the angle is linear, False otherwise.
    """

    angle = femto.md.utils.geometry.compute_angles(coords, numpy.array([idxs]))

    check_1 = _ANGLE_CHECK_FACTOR * angle**2
    check_2 = _ANGLE_CHECK_FACTOR * (angle - numpy.pi) ** 2

    return check_1 < _ANGLE_CHECK_CUTOFF or check_2 < _ANGLE_CHECK_CUTOFF


def _is_dihedral_trans(coords: numpy.ndarray, idxs: tuple[int, int, int, int]) -> bool:
    """Check if a dihedral angle is within -150 and 150 degrees.

    Args:
        coords: The full set of coordinates.
        idxs: The indices of the four atoms that form the dihedral.

    Returns:
        True if the dihedral is planar.
    """

    dihedral = femto.md.utils.geometry.compute_dihedrals(coords, numpy.array([idxs]))
    return numpy.abs(dihedral) > _DIHEDRAL_CHECK_CUTOFF


def _are_collinear(
    coords: numpy.ndarray, idxs: typing.Sequence[int] | None = None
) -> bool:
    """Checks whether a sequence of coordinates are collinear.

    Args:
        coords: The full set of coordinates.
        idxs: The sequence of indices of those coordinates to check for collinearity.

    Returns:
        True if any sequential pair of vectors is collinear.
    """

    idxs = idxs if idxs is not None else list(range(len(coords)))

    for i in range(len(idxs) - 2):
        v_1 = coords[idxs[i + 1], :] - coords[idxs[i], :]
        v_1 /= numpy.linalg.norm(v_1)
        v_2 = coords[idxs[i + 2], :] - coords[idxs[i + 1], :]
        v_2 /= numpy.linalg.norm(v_2)

        if numpy.dot(v_1, v_2) > _COLLINEAR_THRESHOLD:
            return True

    return False


def queries_to_idxs(
    structure: parmed.Structure, queries: typing.Iterable[str]
) -> tuple[int, ...]:
    """Find the indices of those atoms matched by a set of AMBER style reference atom
    queries.

    Args:
        structure: The ligand to query.
        queries: The amber style selection queries.

    Returns:
        The indices of the matched atoms.
    """
    ref_idxs = []

    for query in queries:
        mask = parmed.amber.AmberMask(structure, query).Selection()
        mask_idxs = tuple(i for i, matches in enumerate(mask) if matches)

        if len(mask_idxs) != 1:
            raise ValueError(
                f"{query} matched {len(mask_idxs)} atoms while exactly 1 atom was "
                f"expected."
            )

        ref_idxs.extend(mask_idxs)

    return tuple(ref_idxs)


def _create_ligand_queries_baumann(
    ligand: parmed.Structure, snapshots: list[openmm.unit.Quantity] | None
) -> tuple[str, str, str]:
    """Creates AMBER style masks for selecting three atoms from a ligand for use in
    Boresch-likes restraints using the method described by Baumann et al.

    References:
        [1] Baumann, Hannah M., et al. "Broadening the scope of binding free energy
            calculations using a Separated Topologies approach." (2023).
    """

    ligand_graph = networkx.from_edgelist(
        (bond.atom1.idx, bond.atom2.idx)
        for bond in ligand.bonds
        if bond.atom1.atomic_number != 1 and bond.atom2.atomic_number != 1
    )

    all_paths = [
        path
        for node_paths in networkx.shortest_path(ligand_graph).values()
        for path in node_paths.values()
    ]
    path_lengths = {(path[0], path[-1]): len(path) for path in all_paths}

    longest_path = max(all_paths, key=len)
    center_idx = longest_path[len(longest_path) // 2]

    cycles = networkx.cycle_basis(ligand_graph)

    if len(cycles) >= 1 and snapshots is not None:
        ligand_trajectory = mdtraj.Trajectory(
            [snapshot.value_in_unit(openmm.unit.nanometers) for snapshot in snapshots],
            ligand.topology,
        )
        ligand_trajectory.superpose(ligand_trajectory)

        rmsf = mdtraj.rmsf(ligand_trajectory, ligand_trajectory, 0)
        cycles = [cycle for cycle in cycles if rmsf[cycle].max() < 0.1]

    if len(cycles) >= 1:
        open_list = [atom_idx for cycle in cycles for atom_idx in cycle]
    else:
        open_list = [atom.idx for atom in ligand.atoms if atom.atomic_number != 1]

    distances = [path_lengths[(center_idx, atom_idx)] for atom_idx in open_list]
    closest_idx = open_list[numpy.argmin(distances)]

    if len(cycles) >= 1:
        # restrict the list of reference atoms to select from to those that are in the
        # same cycle as the closest atom.
        cycle_idx = next(
            iter(i for i, cycle in enumerate(cycles) if closest_idx in cycle)
        )
        open_list = cycles[cycle_idx]

        distances = [path_lengths[(closest_idx, atom_idx)] for atom_idx in open_list]

    open_list = [
        idx
        for _, idx in sorted(zip(distances, open_list, strict=True))
        if idx != closest_idx
    ]
    ref_masks = (f"@{open_list[0] + 1}", f"@{closest_idx + 1}", f"@{open_list[1] + 1}")

    # TODO: check if the reference atoms are co-linear
    # TODO: handle the unhappy paths of not enough atoms are found.

    return ref_masks


def _create_ligand_queries_chen(
    ligand_1: parmed.Structure, ligand_2: parmed.Structure
) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """Creates AMBER style masks for selecting three atoms from a ligand for use in
    Boresch-likes restraints using the approach defined in ``siflow`` by Erik Chen."""

    coords_1 = numpy.array(ligand_1.coordinates)
    coords_2 = numpy.array(ligand_2.coordinates)

    distances = scipy.spatial.distance_matrix(coords_1, coords_2)

    counter = 0

    ref_atoms_1, ref_atoms_2 = [], []

    while counter < distances.size and len(ref_atoms_1) != 3:
        idx_1, idx_2 = numpy.unravel_index(distances.argmin(), distances.shape)
        distances[idx_1, idx_2] = numpy.inf

        counter += 1

        if (
            ligand_1.atoms[idx_1].atomic_number == 1
            or ligand_2.atoms[idx_2].atomic_number == 1
        ):
            continue

        if len(ref_atoms_1) == 2 and (
            _are_collinear(coords_1[ref_atoms_1 + [idx_1], :])
            or _are_collinear(coords_2[ref_atoms_2 + [idx_2], :])
        ):
            continue

        distances[idx_1, :] = numpy.inf
        distances[:, idx_2] = numpy.inf

        ref_atoms_1.append(idx_1)
        ref_atoms_2.append(idx_2)

    if len({*ref_atoms_1}) != 3 or len({*ref_atoms_2}) != 3:
        raise RuntimeError("Could not find three non-co-linear reference atoms.")

    return (
        tuple(f"@{ref_atoms_1[i] + 1}" for i in range(3)),
        tuple(f"@{ref_atoms_2[i] + 1}" for i in range(3)),
    )


def _create_ligand_queries(
    ligand_1: parmed.Structure,
    ligand_2: parmed.Structure | None,
    method: femto.fe.config.LigandReferenceMethod,
) -> tuple[tuple[str, str, str], tuple[str, str, str] | None]:
    """Creates AMBER style masks for selecting three atoms from a ligand for use in
    Boresch-likes alignment restraints.

    Args:
        ligand_1: The first ligand.
        ligand_2: The second ligand.
        method: The method to use to select the reference atoms.

    Returns:
        The AMBER style queries that will select the reference atoms of the first and
        second ligands respectively.
    """

    method = method.lower()

    if method in {"chen"} and ligand_2 is None:
        raise ValueError(f"two ligands must be provided to use the {method} method.")

    if method.lower() == "chen":
        return _create_ligand_queries_chen(ligand_1, ligand_2)
    elif method.lower() == "baumann":
        query_1 = _create_ligand_queries_baumann(ligand_1, None)
        query_2 = (
            None if ligand_2 is None else _create_ligand_queries_baumann(ligand_2, None)
        )

        return query_1, query_2

    raise NotImplementedError(f"Unknown method: {method}")


def select_ligand_idxs(
    ligand_1: parmed.amber.AmberParm,
    ligand_2: parmed.amber.AmberParm | None,
    method: femto.fe.config.LigandReferenceMethod,
    ligand_1_queries: tuple[str, str, str] | None = None,
    ligand_2_queries: tuple[str, str, str] | None = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int] | None]:
    """Returns the indices of the reference atoms that may be used to align ligands.

    Notes:
        * Some methods, e.g. ``chen``, select reference atoms based on the two ligands
          meaning they can only be used in RBFE calculations, whereas others, e.g.
          ``baumann``, select reference atoms based on a single ligand.

    Args:
        ligand_1: The first ligand.
        ligand_2: The second ligand.
        method: The method to use to select the reference atoms if none are specified.
        ligand_1_queries: Three (optional) AMBER style queries to use to manually
            select atoms from the first ligand.
        ligand_2_queries: Three (optional) AMBER style queries to use to manually
            select atoms from the second ligand

    Returns:
        The indices of the first and second ligand respectively. No offset is applied
        to the second ligand indices so a query of ``"@1"`` would yield ``0`` rather
        than ``n_ligand_1_atoms``.
    """
    if ligand_1_queries is None or (ligand_2 is not None and ligand_2_queries is None):
        _LOGGER.info("selecting ligand reference atoms")

        ligand_queries = _create_ligand_queries(ligand_1, ligand_2, method)

        if ligand_1_queries is None:
            ligand_1_queries = ligand_queries[0]
        if ligand_2_queries is None and ligand_2 is not None:
            ligand_2_queries = ligand_queries[1]

    _LOGGER.info(f"ligand 1 ref queries={ligand_1_queries}")
    _LOGGER.info(f"ligand 2 ref queries={ligand_2_queries}")

    ligand_1_idxs = queries_to_idxs(ligand_1, ligand_1_queries)
    _LOGGER.info(f"ligand 1 ref idxs={ligand_1_idxs}")

    if ligand_2 is not None:
        ligand_2_idxs = queries_to_idxs(ligand_2, ligand_2_queries)
        _LOGGER.info(f"ligand 2 ref idxs={ligand_2_idxs}")
    else:
        ligand_2_idxs = None

    return ligand_1_idxs, ligand_2_idxs


def _filter_receptor_atoms(
    receptor: parmed.Structure,
    ligand: parmed.Structure,
    ligand_ref_idx: int,
    min_helix_size: int = 8,
    min_sheet_size: int = 8,
    skip_residues_start: int = 20,
    skip_residues_end: int = 10,
    minimum_distance: openmm.unit.Quantity = 1.0 * openmm.unit.nanometers,
    maximum_distance: openmm.unit.Quantity = 3.0 * openmm.unit.nanometers,
) -> list[int]:
    """Select possible protein atoms for Boresch-style restraints based on the criteria
    outlined by Baumann et al [1].

    References:
        [1] Baumann...

    Args:
        receptor: The receptor structure.
        ligand: The ligand structure.
        ligand_ref_idx: The index of the first reference ligand atom.
        min_helix_size: The minimum number of residues that have to be in an alpha-helix
            for it to be considered stable.
        min_sheet_size: The minimum number of residues that have to be in a beta-sheet
            for it to be considered stable.
        skip_residues_start: The number of residues to skip at the start of the protein
            as these tend to be more flexible.
        skip_residues_end: The number of residues to skip at the end of the protein
            as these tend to be more flexible
        minimum_distance: Discard any protein atoms that are closer than this distance
            to the ligand.
        maximum_distance: Discard any protein atoms that are further than this distance
            from the ligand.

    Returns:
        list of indices of possible protein atoms
    """

    assert min_helix_size >= 7, "helices must be at least 7 residues long"
    assert min_sheet_size >= 7, "sheets must be at least 7 residues long"

    receptor_topology = mdtraj.Topology.from_openmm(receptor.topology)
    receptor_trajectory = mdtraj.Trajectory(
        (receptor.coordinates * openmm.unit.angstrom).value_in_unit(
            openmm.unit.nanometer
        ),
        receptor_topology,
    )

    structure = mdtraj.compute_dssp(receptor_trajectory, simplified=True).tolist()[0]

    # following the SepTop reference implementation we prefer to select from alpha
    # helices if they are dominant in the protein, but otherwise select from sheets
    # as well.
    n_helix_residues = structure.count("H")
    n_sheet_residues = structure.count("E")

    allowed_motifs = ["H"] if n_helix_residues >= n_sheet_residues else ["H", "E"]
    min_motif_size = {"H": min_helix_size, "E": min_sheet_size}

    residues_to_keep = []

    for motif, idxs in itertools.groupby(enumerate(structure), lambda x: x[1]):
        idxs = list(idxs)

        if motif not in allowed_motifs or len(idxs) < min_motif_size[motif]:
            continue

        # discard the first and last 3 residues of the helix / sheet
        start_idx, end_idx = idxs[0][0] + 3, idxs[-1][0] - 3

        residues_to_keep.extend(
            f"resid {idx}"
            for idx in range(start_idx, end_idx + 1)
            if skip_residues_start <= idx < len(structure) - skip_residues_end
        )

    residue_mask = " ".join(residues_to_keep)

    atom_mask = f"protein and (backbone or name CB) and ({residue_mask})"
    atom_idxs = receptor_topology.select(atom_mask)

    # TODO: discard atoms with RMSF > 0.1

    if len(atom_idxs) == 0:
        raise ValueError("no suitable receptor atoms could be found")

    distances = (
        scipy.spatial.distance.cdist(
            receptor.coordinates[atom_idxs, :], ligand.coordinates[[ligand_ref_idx], :]
        )
        * openmm.unit.angstrom
    )

    distance_mask = (distances > minimum_distance).all(axis=1)
    distance_mask &= (distances <= maximum_distance).any(axis=1)

    return atom_idxs[distance_mask].tolist()


def _is_valid_r1(
    receptor: parmed.Structure,
    receptor_idx: int,
    ligand: parmed.Structure,
    ligand_ref_idxs: tuple[int, int, int],
) -> bool:
    """Check whether a given receptor atom would be a valid 'R1' atom given the
    following criteria:

    * L2,L1,R1 angle not 'close' to 0 or 180 degrees
    * L3,L2,L1,R1 dihedral between -150 and 150 degrees

    Args:
        receptor: The receptor structure.
        receptor_idx: The index of the receptor atom to check.
        ligand: The ligand structure.
        ligand_ref_idxs: The three reference ligand atoms.
    """

    coords = numpy.vstack([ligand.coordinates, receptor.coordinates])

    l1, l2, l3 = ligand_ref_idxs
    r1 = receptor_idx + len(ligand.atoms)

    # TODO: angle and dihedral variance checks

    if _are_collinear(coords, (r1, l1, l2, l3)):
        return False

    if _is_angle_linear(coords, (r1, l1, l2)):
        return False

    if _is_dihedral_trans(coords, (r1, l1, l2, l3)):
        return False

    return True


def _is_valid_r2(
    receptor: parmed.Structure,
    receptor_idx: int,
    receptor_ref_idx_1: int,
    ligand: parmed.Structure,
    ligand_ref_idxs: tuple[int, int, int],
) -> bool:
    """Check whether a given receptor atom would be a valid 'R2' atom given the
    following criteria:

    * R1,R2 are further apart than 5 Angstroms
    * R2,R1,L1,L2 are not collinear
    * R2,R1,L1 angle not 'close' to 0 or 180 degrees
    * R2,R1,L1,L2 dihedral between -150 and 150 degrees

    Args:
        receptor: The receptor structure.
        receptor_idx: The index of the receptor atom to check.
        receptor_ref_idx_1: The index of the first receptor reference atom.
        ligand: The ligand structure.
        ligand_ref_idxs: The three reference ligand atoms.
    """

    coords = numpy.vstack([ligand.coordinates, receptor.coordinates])

    l1, l2, l3 = ligand_ref_idxs
    r1, r2 = receptor_ref_idx_1 + len(ligand.atoms), receptor_idx + len(ligand.atoms)

    # TODO: angle and dihedral variance checks

    if r1 == r2:
        return False

    if numpy.linalg.norm(coords[r1, :] - coords[r2, :]) < 5.0:
        return False

    if _are_collinear(coords, (r2, r1, l1, l2)):
        return False

    if _is_angle_linear(coords, (r2, r1, l1)):
        return False

    if _is_dihedral_trans(coords, (r2, r1, l1, l2)):
        return False

    return True


def _is_valid_r3(
    receptor: parmed.Structure,
    receptor_idx: int,
    receptor_ref_idx_1: int,
    receptor_ref_idx_2: int,
    ligand: parmed.Structure,
    ligand_ref_idxs: tuple[int, int, int],
) -> bool:
    """Check whether a given receptor atom would be a valid 'R3' atom given the
    following criteria:

    * R1,R2,R3,L1 are not collinear
    * R3,R2,R1,L1 dihedral between -150 and 150 degrees

    Args:
        receptor: The receptor structure.
        receptor_idx: The index of the receptor atom to check.
        receptor_ref_idx_1: The index of the first receptor reference atom.
        receptor_ref_idx_2: The index of the second receptor reference atom.
        ligand: The ligand structure.
        ligand_ref_idxs: The three reference ligand atoms.
    """

    coords = numpy.vstack([ligand.coordinates, receptor.coordinates])

    l1, l2, l3 = ligand_ref_idxs
    r1, r2, r3 = (
        receptor_ref_idx_1 + len(ligand.atoms),
        receptor_ref_idx_2 + len(ligand.atoms),
        receptor_idx + len(ligand.atoms),
    )

    if len({r1, r2, r3}) != 3:
        return False

    # TODO: angle and dihedral variance checks

    if _are_collinear(coords, (r3, r2, r1, l1)):
        return False

    if _is_dihedral_trans(coords, (r3, r2, r1, l1)):
        return False

    return True


def select_receptor_idxs(
    receptor: parmed.Structure,
    ligand: parmed.Structure,
    ligand_ref_idxs: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Select possible protein atoms for Boresch-style restraints.

    Args:
        receptor: The receptor structure.
        ligand: The ligand structure.
        ligand_ref_idxs: The indices of the three ligands atoms that will be restrained.

    Returns:
        The indices of the three atoms to use for the restraint
    """

    receptor_idxs = _filter_receptor_atoms(receptor, ligand, ligand_ref_idxs[0])

    valid_r1_idxs = (
        idx
        for idx in receptor_idxs
        if _is_valid_r1(receptor, idx, ligand, ligand_ref_idxs)
    )

    found_r1, found_r2 = next(
        (
            (r1, r2)
            for r1 in valid_r1_idxs
            for r2 in receptor_idxs
            if _is_valid_r2(receptor, r2, r1, ligand, ligand_ref_idxs)
        ),
        None,
    )

    if found_r1 is None or found_r2 is None:
        raise ValueError("could not find valid R1 / R2 atoms")

    valid_r3_idxs = [
        idx
        for idx in receptor_idxs
        if _is_valid_r3(receptor, idx, found_r1, found_r2, ligand, ligand_ref_idxs)
    ]

    if len(valid_r3_idxs) == 0:
        raise ValueError("could not find a valid R3 atom")

    r3_distances = scipy.spatial.distance.cdist(
        receptor.coordinates[valid_r3_idxs, :],
        receptor.coordinates[[found_r1, found_r2], :],
    )

    found_r3 = valid_r3_idxs[r3_distances.min(axis=1).argmax()]

    return found_r1, found_r2, found_r3


def select_protein_cavity_atoms(
    protein: parmed.Structure,
    ligands: list[parmed.Structure],
    cutoff: openmm.unit.Quantity,
) -> str:
    """Select the alpha carbon atoms that define the binding cavity of the protein based
    on their distance to ligand atoms.

    Args:
        protein: The protein.
        ligands: The ligands to consider.
        cutoff: Residues further than this distance from a ligand will be excluded.

    Returns:
        The AMBER style query that will select the reference atoms of the protein.
    """

    ref_atoms = []

    for residue in protein.residues:
        if (
            sorted(a.element for a in residue.atoms) == [1, 1, 8]
            # a bit of a hack to check for ions.
            or len(residue.atoms) == 1
        ):
            continue

        c_alpha_idxs = [a.idx for a in residue.atoms if a.name == "CA"]

        if len(c_alpha_idxs) < 1:
            continue

        ref_atoms.extend(c_alpha_idxs)

    protein_coords = numpy.array(protein.coordinates)[ref_atoms, :]

    cutoff = cutoff.value_in_unit(openmm.unit.angstrom)

    is_reference = numpy.array([False] * len(ref_atoms))

    for ligand in ligands:
        ligand_coords = numpy.array(ligand.coordinates)

        distances = scipy.spatial.distance_matrix(protein_coords, ligand_coords)
        is_reference = is_reference | (distances < cutoff).any(axis=1)

    n_atoms = is_reference.sum()

    if n_atoms < 1:
        raise RuntimeError("Could not find the protein binding site reference atoms.")

    ref_mask = "@" + ",".join(
        str(i + 1) for i, keep in zip(ref_atoms, is_reference, strict=True) if keep
    )
    return ref_mask
