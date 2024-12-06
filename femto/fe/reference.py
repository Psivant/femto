"""Utilities for automatically selecting 'reference' atoms for alignment."""

import copy
import itertools
import logging
import typing

import mdtop
import mdtraj
import networkx
import numpy
import openmm.unit
import scipy

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

_ANGLE_CHECK_MAX_VAR = 100.0  # units of degrees^2

_DIHEDRAL_CHECK_CUTOFF = 150.0  # units of degrees
_DIHEDRAL_CHECK_MAX_VAR = 300.0  # units of degrees^2

_RMSF_CUTOFF = 0.1  # nm


def _is_angle_linear(coords: numpy.ndarray, idxs: tuple[int, int, int]) -> bool:
    """Check if angle is within 10 kT from 0 or 180 following the SepTop reference
    implementation.

    Args:
        coords: The full set of coordinates.
        idxs: The indices of the three atoms that form the angle.

    Returns:
        True if the angle is linear, False otherwise.
    """

    angles = numpy.rad2deg(
        femto.md.utils.geometry.compute_angles(coords, numpy.array([idxs]))
    )

    angle_avg_rad = numpy.deg2rad(scipy.stats.circmean(angles, low=-180.0, high=180.0))
    angle_var_deg = scipy.stats.circvar(angles, low=-180.0, high=180.0)

    check_1 = _ANGLE_CHECK_FACTOR * angle_avg_rad**2
    check_2 = _ANGLE_CHECK_FACTOR * (angle_avg_rad - numpy.pi) ** 2

    return (
        check_1 < _ANGLE_CHECK_CUTOFF
        or check_2 < _ANGLE_CHECK_CUTOFF
        or angle_var_deg > _ANGLE_CHECK_MAX_VAR
    )


def _is_dihedral_trans(coords: numpy.ndarray, idxs: tuple[int, int, int, int]) -> bool:
    """Check if a dihedral angle is within -150 and 150 degrees.

    Args:
        coords: The full set of coordinates.
        idxs: The indices of the four atoms that form the dihedral.

    Returns:
        True if the dihedral is planar.
    """

    dihedrals = numpy.rad2deg(
        femto.md.utils.geometry.compute_dihedrals(coords, numpy.array([idxs]))
    )

    dihedral_avg = scipy.stats.circmean(dihedrals, low=-180.0, high=180.0)
    dihedral_var = scipy.stats.circvar(dihedrals, low=-180.0, high=180.0)

    return (
        numpy.abs(dihedral_avg) > _DIHEDRAL_CHECK_CUTOFF
        or dihedral_var > _DIHEDRAL_CHECK_MAX_VAR
    )


def _are_collinear(
    coords: numpy.ndarray, idxs: typing.Sequence[int] | None = None
) -> bool:
    """Checks whether a sequence of coordinates are collinear.

    Args:
        coords: The full set of coordinates, either with ``shape=(n_coords, 3)`` or
            ``shape=(n_frames, n_coords, 3)``.
        idxs: The sequence of indices of those coordinates to check for collinearity.

    Returns:
        True if any sequential pair of vectors is collinear.
    """

    if coords.ndim == 2:
        coords = coords.reshape(1, *coords.shape)

    idxs = idxs if idxs is not None else list(range(coords.shape[1]))

    for i in range(len(idxs) - 2):
        v_1 = coords[:, idxs[i + 1], :] - coords[:, idxs[i], :]
        v_1 /= numpy.linalg.norm(v_1, axis=-1, keepdims=True)
        v_2 = coords[:, idxs[i + 2], :] - coords[:, idxs[i + 1], :]
        v_2 /= numpy.linalg.norm(v_2, axis=-1, keepdims=True)

        if (numpy.abs((v_1 * v_2).sum(axis=-1)) > _COLLINEAR_THRESHOLD).any():
            return True

    return False


def queries_to_idxs(
    topology: mdtop.Topology, queries: typing.Iterable[str]
) -> tuple[int, ...]:
    """Find the indices of those atoms matched by a set of atom queries.

    Args:
        topology: The ligand to query.
        queries: The atom selection queries.

    Returns:
        The indices of the matched atoms.
    """
    ref_idxs = []

    for query in queries:
        mask_idxs = topology.select(query)

        if len(mask_idxs) != 1:
            raise ValueError(
                f"{query} matched {len(mask_idxs)} atoms. exactly 1 atom was expected."
            )

        ref_idxs.extend(mask_idxs)

    return tuple(ref_idxs)


def _create_ligand_queries_baumann(
    ligand: mdtop.Topology, snapshots: list[openmm.unit.Quantity] | None
) -> tuple[str, str, str]:
    """Creates AMBER style masks for selecting three atoms from a ligand for use in
    Boresch-likes restraints using the method described by Baumann et al.

    References:
        [1] Baumann, Hannah M., et al. "Broadening the scope of binding free energy
            calculations using a Separated Topologies approach." (2023).
    """

    atoms = ligand.atoms

    ligand_graph = networkx.from_edgelist(
        (bond.idx_1, bond.idx_2)
        for bond in ligand.bonds
        if atoms[bond.idx_1].atomic_num != 1 and atoms[bond.idx_2].atomic_num != 1
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
            ligand.to_openmm(),
        )
        ligand_trajectory.superpose(ligand_trajectory)

        rmsf = mdtraj.rmsf(ligand_trajectory, ligand_trajectory, 0)
        cycles = [cycle for cycle in cycles if rmsf[cycle].max() < _RMSF_CUTOFF]

    if len(cycles) >= 1:
        open_list = [atom_idx for cycle in cycles for atom_idx in cycle]
    else:
        open_list = [atom.index for atom in ligand.atoms if atom.atomic_num != 1]

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
    ref_masks = (
        f"index {open_list[0] + 1}",
        f"index {closest_idx + 1}",
        f"index {open_list[1] + 1}",
    )

    # TODO: check if the reference atoms are co-linear
    # TODO: handle the unhappy paths of not enough atoms are found.

    return ref_masks


def _create_ligand_queries_chen(
    ligand_1: mdtop.Topology, ligand_2: mdtop.Topology
) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    """Creates selection masks for selecting three atoms from a ligand for use in
    Boresch-likes restraints using the approach defined in ``siflow`` by Erik Chen."""

    coords_1 = ligand_1.xyz.value_in_unit(openmm.unit.angstrom)
    coords_2 = ligand_2.xyz.value_in_unit(openmm.unit.angstrom)

    distances = scipy.spatial.distance_matrix(coords_1, coords_2)

    counter = 0

    ref_atoms_1, ref_atoms_2 = [], []

    while counter < distances.size and len(ref_atoms_1) != 3:
        idx_1, idx_2 = numpy.unravel_index(distances.argmin(), distances.shape)
        distances[idx_1, idx_2] = numpy.inf

        counter += 1

        if (
            ligand_1.atoms[idx_1].atomic_num == 1
            or ligand_2.atoms[idx_2].atomic_num == 1
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
        tuple(f"index {ref_atoms_1[i] + 1}" for i in range(3)),
        tuple(f"index {ref_atoms_2[i] + 1}" for i in range(3)),
    )


def _create_ligand_queries(
    ligand_1: mdtop.Topology,
    ligand_2: mdtop.Topology | None,
    method: femto.fe.config.LigandReferenceMethod,
) -> tuple[tuple[str, str, str], tuple[str, str, str] | None]:
    """Creates selection masks for selecting three atoms from a ligand for use in
    Boresch-likes alignment restraints.

    Args:
        ligand_1: The first ligand.
        ligand_2: The second ligand.
        method: The method to use to select the reference atoms.

    Returns:
        The atom queries that will select the reference atoms of the first and
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
    ligand_1: mdtop.Topology,
    ligand_2: mdtop.Topology | None,
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
        ligand_1_queries: Three (optional) selection queries to use to manually
            select atoms from the first ligand.
        ligand_2_queries: Three (optional) selection queries to use to manually
            select atoms from the second ligand

    Returns:
        The indices of the first and second ligand respectively. No offset is applied
        to the second ligand indices so a query of ``"idx. 1"`` would yield ``0``
        rather than ``n_ligand_1_atoms``.
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

    if ligand_2 is not None:
        ligand_2_idxs = queries_to_idxs(ligand_2, ligand_2_queries)
    else:
        ligand_2_idxs = None

    return ligand_1_idxs, ligand_2_idxs


def _filter_receptor_atoms(
    topology: mdtraj.Trajectory,
    ligand_ref_idx: int,
    min_helix_size: int = 8,
    min_sheet_size: int = 8,
    skip_residues_start: int = 20,
    skip_residues_end: int = 10,
    minimum_distance: openmm.unit.Quantity = 1.0 * openmm.unit.nanometers,
    maximum_distance: openmm.unit.Quantity = 3.0 * openmm.unit.nanometers,
) -> list[int]:
    """Select possible protein atoms for Boresch-style restraints based on the criteria
    outlined by Baumann et al.

    Args:
        topology: The system topology.
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
        The indices of protein atoms that should be considered for use in Boresch-style
        restraints.
    """

    assert min_helix_size >= 7, "helices must be at least 7 residues long"
    assert min_sheet_size >= 7, "sheets must be at least 7 residues long"

    backbone_idxs = topology.top.select("protein and (backbone or name CB)")
    backbone: mdtraj.Trajectory = topology.atom_slice(backbone_idxs)

    structure = mdtraj.compute_dssp(backbone, simplified=True).tolist()[0]

    # following the SepTop reference implementation we prefer to select from alpha
    # helices if they are dominant in the protein, but otherwise select from sheets
    # as well.
    n_helix_residues = structure.count("H")
    n_sheet_residues = structure.count("E")

    allowed_motifs = ["H"] if n_helix_residues >= n_sheet_residues else ["H", "E"]
    min_motif_size = {"H": min_helix_size, "E": min_sheet_size}

    residues_to_keep = []

    structure = structure[skip_residues_start : -(skip_residues_end + 1)]

    for motif, idxs in itertools.groupby(enumerate(structure), lambda x: x[1]):
        idxs = [(idx + skip_residues_start, motif) for idx, motif in idxs]

        if motif not in allowed_motifs or len(idxs) < min_motif_size[motif]:
            continue

        # discard the first and last 3 residues of the helix / sheet
        start_idx, end_idx = idxs[0][0] + 3, idxs[-1][0] - 3

        residues_to_keep.extend(f"resid {idx}" for idx in range(start_idx, end_idx + 1))

    rigid_backbone_idxs = backbone.top.select(" ".join(residues_to_keep))

    if len(rigid_backbone_idxs) == 0:
        raise ValueError("no suitable receptor atoms could be found")

    if backbone.n_frames > 1:
        superposed = copy.deepcopy(backbone)
        superposed.superpose(superposed)

        rmsf = mdtraj.rmsf(superposed, superposed, 0)  # nm

        rigid_backbone_idxs = rigid_backbone_idxs[
            rmsf[rigid_backbone_idxs] < _RMSF_CUTOFF
        ]

    distances = scipy.spatial.distance.cdist(
        backbone.xyz[0, rigid_backbone_idxs, :], topology.xyz[0, [ligand_ref_idx], :]
    )

    minimum_distance = minimum_distance.value_in_unit(openmm.unit.nanometer)
    maximum_distance = maximum_distance.value_in_unit(openmm.unit.nanometer)

    distance_mask = (distances > minimum_distance).all(axis=1)
    distance_mask &= (distances <= maximum_distance).any(axis=1)

    return backbone_idxs[rigid_backbone_idxs[distance_mask]].tolist()


def _is_valid_r1(
    topology: mdtraj.Trajectory, r1: int, l1: int, l2: int, l3: int
) -> bool:
    """Check whether a given receptor atom would be a valid 'R1' atom given the
    following criteria:

    * L2,L1,R1 angle not 'close' to 0 or 180 degrees
    * L3,L2,L1,R1 dihedral between -150 and 150 degrees
    """

    coords = topology.xyz

    if _are_collinear(coords, (r1, l1, l2, l3)):
        return False

    if _is_angle_linear(coords, (r1, l1, l2)):
        return False

    if _is_dihedral_trans(coords, (r1, l1, l2, l3)):
        return False

    return True


def _is_valid_r2(
    topology: mdtraj.Trajectory, r1: int, r2: int, l1: int, l2: int
) -> bool:
    """Check whether a given receptor atom would be a valid 'R2' atom given the
    following criteria:

    * R1,R2 are further apart than 5 Angstroms
    * R2,R1,L1,L2 are not collinear
    * R2,R1,L1 angle not 'close' to 0 or 180 degrees
    * R2,R1,L1,L2 dihedral between -150 and 150 degrees
    """

    coords = topology.xyz

    if r1 == r2:
        return False

    if numpy.linalg.norm(coords[:, r1, :] - coords[:, r2, :], axis=-1).mean() < 0.5:
        return False

    if _are_collinear(coords, (r2, r1, l1, l2)):
        return False

    if _is_angle_linear(coords, (r2, r1, l1)):
        return False

    if _is_dihedral_trans(coords, (r2, r1, l1, l2)):
        return False

    return True


def _is_valid_r3(
    topology: mdtraj.Trajectory, r1: int, r2: int, r3: int, l1: int
) -> bool:
    """Check whether a given receptor atom would be a valid 'R3' atom given the
    following criteria:

    * R1,R2,R3,L1 are not collinear
    * R3,R2,R1,L1 dihedral between -150 and 150 degrees
    """

    coords = topology.xyz

    if len({r1, r2, r3}) != 3:
        return False

    if _are_collinear(coords[[0]], (r3, r2, r1, l1)):
        return False

    if _is_dihedral_trans(coords, (r3, r2, r1, l1)):
        return False

    return True


def _topology_to_mdtraj(topology: mdtop.Topology) -> mdtraj.Trajectory:
    coords = topology.xyz.value_in_unit(openmm.unit.nanometer)

    # if the structure has no box vectors defined, or the box vectors are smaller than
    # the structure, we will use the structure's coordinates to define the box so we
    # have at least a reasonable guess.
    coords_min = coords.min(axis=0)
    coords_max = coords.max(axis=0)

    box_delta = (coords_max - coords_min) * 1.1  # add some slight padding
    box_from_coords = numpy.diag(box_delta)

    box = (
        numpy.array(topology.box.value_in_unit(openmm.unit.nanometer))
        if topology.box is not None
        else None
    )

    if box is None or (box < box_from_coords).any():
        box = box_from_coords

    trajectory = mdtraj.Trajectory(
        coords, mdtraj.Topology.from_openmm(topology.to_openmm())
    )
    trajectory.unitcell_vectors = box.reshape(1, 3, 3)

    return trajectory


def select_receptor_idxs(
    topology: mdtop.Topology | mdtraj.Trajectory,
    ligand_ref_idxs: tuple[int, int, int],
) -> tuple[int, int, int]:
    """Select possible protein atoms for Boresch-style restraints using the method
    outlined by Baumann et al [1].

    References:
        [1] Baumann, Hannah M., et al. "Broadening the scope of binding free energy
            calculations using a Separated Topologies approach." (2023).

    Args:
        topology: The topology containing the receptor and ligands.
        ligand_ref_idxs: The indices of the three ligands atoms that will be restrained.

    Returns:
        The indices of the three atoms to use for the restraint
    """

    if isinstance(topology, mdtop.Topology):
        topology = _topology_to_mdtraj(topology)

    receptor_idxs = _filter_receptor_atoms(topology, ligand_ref_idxs[0])

    l1, l2, l3 = ligand_ref_idxs

    valid_r1_idxs = [
        r1 for r1 in receptor_idxs if _is_valid_r1(topology, r1, l1, l2, l3)
    ]

    found_r1, found_r2 = next(
        (
            (r1, r2)
            for r1 in valid_r1_idxs
            for r2 in receptor_idxs
            if _is_valid_r2(topology, r1, r2, l1, l2)
        ),
        None,
    )

    if found_r1 is None or found_r2 is None:
        raise ValueError("could not find valid R1 / R2 atoms")

    valid_r3_idxs = [
        r3 for r3 in receptor_idxs if _is_valid_r3(topology, found_r1, found_r2, r3, l1)
    ]

    if len(valid_r3_idxs) == 0:
        raise ValueError("could not find a valid R3 atom")

    r3_distances_per_frame = []

    for frame in topology.xyz:
        r3_r_distances = scipy.spatial.distance.cdist(
            frame[valid_r3_idxs, :], frame[[found_r1, found_r2], :]
        )
        r3_l_distances = scipy.spatial.distance.cdist(
            frame[valid_r3_idxs, :], frame[[ligand_ref_idxs[0]], :]
        )

        r3_distances_per_frame.append(numpy.hstack([r3_r_distances, r3_l_distances]))

    # chosen to match the SepTop reference implementation at commit 3705ba5
    max_distance = 0.8 * (topology.unitcell_lengths.mean(axis=0).min(axis=-1) / 2)

    r3_distances_avg = numpy.stack(r3_distances_per_frame).mean(axis=0)

    max_distance_mask = r3_distances_avg.max(axis=-1) < max_distance
    r3_distances_avg = r3_distances_avg[max_distance_mask]

    valid_r3_idxs = numpy.array(valid_r3_idxs)[max_distance_mask].tolist()

    r3_distances_prod = r3_distances_avg[:, 0] * r3_distances_avg[:, 1]
    found_r3 = valid_r3_idxs[r3_distances_prod.argmax()]

    return found_r1, found_r2, found_r3


def check_receptor_idxs(
    topology: mdtop.Topology | mdtraj.Trajectory,
    receptor_idxs: tuple[int, int, int],
    ligand_ref_idxs: tuple[int, int, int],
) -> bool:
    """Check if the specified receptor atoms meet the criteria for use in Boresch-style
    restraints as defined by Baumann et al [1].

    References:
        [1] Baumann, Hannah M., et al. "Broadening the scope of binding free energy
            calculations using a Separated Topologies approach." (2023).

    Args:
        topology: The system topology.
        receptor_idxs: The indices of the three receptor atoms that will be restrained.
        ligand_ref_idxs: The indices of the three ligand atoms that will be restrained.

    Returns:
        True if the atoms meet the criteria, False otherwise.
    """

    if isinstance(topology, mdtop.Topology):
        topology = _topology_to_mdtraj(topology)

    r1, r2, r3 = receptor_idxs
    l1, l2, l3 = ligand_ref_idxs

    is_valid_r1 = _is_valid_r1(topology, r1, l1, l2, l3)
    is_valid_r2 = _is_valid_r2(topology, r1, r2, l1, l2)
    is_valid_r3 = _is_valid_r3(topology, r1, r2, r3, l1)

    r3_distances_per_frame = [
        scipy.spatial.distance.cdist(frame[[r3], :], frame[[r1, r2], :])
        for frame in topology.xyz
    ]
    r3_distance_avg = numpy.stack(r3_distances_per_frame).mean(axis=0)

    max_distance = 0.8 * (topology.unitcell_lengths[-1][0] / 2)
    is_valid_distance = r3_distance_avg.max(axis=-1) < max_distance

    return is_valid_r1 and is_valid_r2 and is_valid_r3 and is_valid_distance


def select_protein_cavity_atoms(
    protein: mdtop.Topology,
    ligands: list[mdtop.Topology],
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
            sorted(a.atomic_num for a in residue.atoms) == [1, 1, 8]
            # a bit of a hack to check for ions.
            or len(residue.atoms) == 1
        ):
            continue

        c_alpha_idxs = [a.index for a in residue.atoms if a.name == "CA"]

        if len(c_alpha_idxs) < 1:
            continue

        ref_atoms.extend(c_alpha_idxs)

    protein_coords = protein.xyz.value_in_unit(openmm.unit.angstrom)[ref_atoms, :]

    cutoff = cutoff.value_in_unit(openmm.unit.angstrom)

    is_reference = numpy.array([False] * len(ref_atoms))

    for ligand in ligands:
        ligand_coords = ligand.xyz.value_in_unit(openmm.unit.angstrom)

        distances = scipy.spatial.distance_matrix(protein_coords, ligand_coords)
        is_reference = is_reference | (distances < cutoff).any(axis=1)

    n_atoms = is_reference.sum()

    if n_atoms < 1:
        raise RuntimeError("Could not find the protein binding site reference atoms.")

    ref_mask = "index " + "+".join(
        str(i + 1) for i, keep in zip(ref_atoms, is_reference, strict=True) if keep
    )
    return ref_mask
