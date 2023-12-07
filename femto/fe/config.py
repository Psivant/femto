"""Common configuration models."""
import typing

import openmm.unit
import pydantic

import femto.md.utils.models

_T = typing.TypeVar("_T")


DEFAULT_INITIAL_TEMPERATURE = 50.0 * openmm.unit.kelvin
"""The default temperature to begin annealing from during equilibration"""


LigandReferenceMethod = typing.Literal["chen", "baumann"]
"""The method to use when automatically selecting ligand atoms to use in alignment
restraints."""


class FEP(femto.md.utils.models.BaseModel):
    """Configure modifying a system to be scalable by FEP lambdas."""

    scale_vdw: bool = pydantic.Field(
        True, description="Whether to scale the vdW non-bonded interactions."
    )
    scale_charges: bool = pydantic.Field(
        True, description="Whether to scale the electrostatic non-bonded interactions."
    )

    ligands_can_interact: bool = pydantic.Field(
        False, description="Whether ligands are allowed to interact with each other."
    )


class Edge(femto.md.utils.models.BaseModel):
    """Defines a basic edge in a free energy network."""

    ligand_1: str = pydantic.Field(..., description="The name of the first ligand.")
    ligand_2: str | None = pydantic.Field(
        ...,
        description="The name of the second ligand. This should be ``None`` if running "
        "an ABFE calculation.",
    )

    @property
    def ligand_1_metadata(self) -> dict[str, typing.Any]:
        """Any additional metadata about ligand 1."""
        return {}

    @property
    def ligand_2_metadata(self) -> dict[str, typing.Any]:
        """Any additional metadata about ligand 2."""
        return {}


class Network(femto.md.utils.models.BaseModel):
    """Defines a basic free energy network."""

    receptor: str | None = pydantic.Field(
        None,
        description="The name of the receptor. If ``None``, the receptor will be "
        "identified from the input directory structure",
    )

    edges: list[Edge] = pydantic.Field(
        ..., description="The edges in the free energy network."
    )

    @property
    def receptor_metadata(self) -> dict[str, typing.Any]:
        """Any additional metadata about the receptor."""
        return {}
