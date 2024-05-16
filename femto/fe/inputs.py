"""Helpers for loading inputs from a standard directory structure."""

import pathlib
import typing

import pydantic
import yaml

import femto.fe.config


@pydantic.dataclasses.dataclass
class Structure:
    """A basic representation of a single structure (i.e. a ligand or a receptor)."""

    name: str
    """The name of the structure."""
    coords: pathlib.Path
    """The path to the associated coordinate file (.rst7, .mol2, .pdb)."""
    params: pathlib.Path | None
    """The path to the associated parameter file (.parm7)."""

    metadata: dict[str, typing.Any]
    """Any additional metadata associated with the structure."""


@pydantic.dataclasses.dataclass
class Edge:
    """A basic representation of an edge in a free energy network."""

    ligand_1: Structure
    """The first ligand."""
    ligand_2: Structure | None
    """The second ligand if computing a RBFE."""


@pydantic.dataclasses.dataclass
class Network:
    """A basic definition of a free energy network of edges."""

    receptor: Structure
    """The receptor."""
    edges: list[Edge]
    """The edges in the network."""

    def find_edge(self, ligand_1: str, ligand_2: str | None = None) -> Edge:
        """Find an edge in the network.

        Args:
            ligand_1: The name of the first ligand.
            ligand_2: The name of the second ligand.

        Returns:
            The edge.
        """

        edges = [edge for edge in self.edges if edge.ligand_1.name == ligand_1]

        filter_fn = (
            (lambda e: e.ligand_2 is not None and e.ligand_2.name == ligand_2)
            if ligand_2 is not None
            else (lambda e: e.ligand_2 is None)
        )
        edges = [edge for edge in edges if filter_fn(edge)]

        if len(edges) == 0:
            raise RuntimeError(f"Could not find {ligand_1}~{ligand_2}")

        assert len(edges) == 1, f"found multiple edges for {ligand_1}~{ligand_2}"
        return edges[0]


def _find_config(
    root_dir: pathlib.Path,
    config_cls: type[femto.fe.config.Network] = femto.fe.config.Network,
) -> femto.fe.config.Network:
    """Attempt to load the free energy network definition from the input file.

    First an ``'edges.yaml'`` file is searched for. If this is not found, then a
    legacy ``'Morph.in'`` file is searched for. An error is raised if neither is found.

    Args:
        root_dir: The root of the BFE directory structure.
        config_cls: The class to use to parse the ``'edges.yaml'`` file if present.

    Returns:
        The free energy network definition.
    """

    config_file = root_dir / "edges.yaml"
    morph_file = root_dir / "Morph.in"

    if config_file.exists():
        return config_cls(**yaml.safe_load(config_file.read_text()))

    elif not morph_file.exists():
        raise RuntimeError(f"Could not find either {config_file} or {morph_file}.")

    edges = [
        tuple(line.strip().split("~"))
        for line in morph_file.read_text().splitlines()
        if len(line.strip()) > 0
    ]
    return config_cls(
        edges=[
            {
                "ligand_1": ligand_1,
                "ligand_2": ligand_2 if ligand_1 != ligand_2 else None,
            }
            for ligand_1, ligand_2 in edges
        ]
    )


def _find_receptor(
    root_dir: pathlib.Path,
    name: str | None = None,
    metadata: dict[str, typing.Any] | None = None,
) -> Structure:
    """Attempts to find the receptor coordinates from a standard BFE directory
    structure."""

    receptor_paths = [
        path
        for suffix in ("pdb", "rst7", "mol2")
        for path in (root_dir / "proteins").glob(f"*/protein.{suffix}")
    ]

    if name is not None:
        receptor_paths = [path for path in receptor_paths if path.parent.name == name]

    if len(receptor_paths) != 1:
        raise RuntimeError("Expected to find exactly one receptor file.")

    receptor_coords = receptor_paths[0]
    receptor_params = receptor_coords.with_suffix(".parm7")

    if not receptor_params.exists():
        receptor_params = None

    receptor_name = receptor_coords.parent.name

    return Structure(
        receptor_name,
        receptor_coords,
        receptor_params,
        metadata=metadata if metadata is not None else {},
    )


def _find_edge(root_dir: pathlib.Path, edge: femto.fe.config.Edge) -> Edge:
    """Attempts to find the files corresponding to a given edge in a standard BFE
    directory structure."""

    ligand_1_coords = root_dir / "forcefield" / edge.ligand_1 / "vacuum.mol2"

    if not ligand_1_coords.exists():
        ligand_1_coords = ligand_1_coords.with_suffix(".rst7")

    ligand_1_params = root_dir / "forcefield" / edge.ligand_1 / "vacuum.parm7"

    if not ligand_1_coords.exists() or not ligand_1_params.exists():
        raise RuntimeError(f"Could not find files for {edge.ligand_1}")

    ligand_1 = Structure(
        edge.ligand_1, ligand_1_coords, ligand_1_params, edge.ligand_1_metadata
    )
    ligand_2 = None

    if edge.ligand_2 is not None:
        ligand_2_coords = root_dir / "forcefield" / edge.ligand_2 / "vacuum.mol2"
        ligand_2_params = root_dir / "forcefield" / edge.ligand_2 / "vacuum.parm7"

        if not ligand_2_coords.exists():
            ligand_2_coords = ligand_2_coords.with_suffix(".rst7")

        if not ligand_2_coords.exists() or not ligand_2_params.exists():
            raise RuntimeError(f"Could not find files for {edge.ligand_2}")

        ligand_2 = Structure(
            edge.ligand_2, ligand_2_coords, ligand_2_params, edge.ligand_2_metadata
        )

    return Edge(ligand_1, ligand_2)


def find_edges(
    root_dir: pathlib.Path,
    config_cls: type[femto.fe.config.Network] = femto.fe.config.Network,
    config_path: pathlib.Path | None = None,
) -> Network:
    """Attempts to find the input files for a network free energy 'edges' in the
    standard directory structure.

    Args:
        root_dir: The root of the directory structure.
        config_cls: The class to use to parse the ``'edges.yaml'`` file if present.
        config_path: The path to the file defining the edges to run.
    """

    config = (
        _find_config(root_dir, config_cls)
        if config_path is None
        else config_cls(**yaml.safe_load(config_path.read_text()))
    )

    receptor = _find_receptor(root_dir, config.receptor, config.receptor_metadata)
    edges = [_find_edge(root_dir, edge) for edge in config.edges]

    return Network(receptor, edges)
