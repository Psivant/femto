"""Configure ATM calculations."""

import pathlib
import typing

import numpy
import openmm.unit
import pydantic
import pydantic_units
import yaml
from pydantic_units import OpenMMQuantity

import femto.fe.config
import femto.md.config
from femto.md.utils.models import BaseModel

_ANGSTROM = openmm.unit.angstrom

_KCAL_PER_MOL = openmm.unit.kilocalorie_per_mole
_KCAL_PER_ANG_SQR = openmm.unit.kilocalorie_per_mole / _ANGSTROM**2


# fmt: off
DEFAULT_LAMBDA_1  = (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00)  # noqa: E201,E221,E241,E501
"""The default lambda 1 schedule."""
DEFAULT_LAMBDA_2  = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00)  # noqa: E201,E221,E241,E501
"""The default lambda 2 schedule."""
DEFAULT_DIRECTION = (   1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1)  # noqa: E201,E221,E241,E501
"""The default direction schedule."""
DEFAULT_ALPHA     = (0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10)  # noqa: E201,E221,E241,E501
"""The default alpha schedule."""
DEFAULT_U0        = (110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110., 110.)  # noqa: E201,E221,E241,E501
"""The default u0 schedule."""
DEFAULT_W0        = (   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0)  # noqa: E201,E221,E241,E501
"""The default w0 schedule."""

DEFAULT_MAX_REST_TEMPERATURE = 900.0
"""The default maximum temperature to use during the REST2 calculations."""

_DEFAULT_REST_TEMPERATURES = numpy.linspace(femto.md.config.DEFAULT_TEMPERATURE.value_in_unit(openmm.unit.kelvin), DEFAULT_MAX_REST_TEMPERATURE, len(DEFAULT_LAMBDA_1) // 2)  # noqa: E201,E221,E241,E501
_DEFAULT_REST_TEMPERATURES = numpy.concatenate([_DEFAULT_REST_TEMPERATURES, _DEFAULT_REST_TEMPERATURES[::-1]]) * openmm.unit.kelvin  # noqa: E201,E221,E241,E501

DEFAULT_BM_B0 = tuple((femto.md.config.DEFAULT_TEMPERATURE / _DEFAULT_REST_TEMPERATURES).tolist())  # noqa: E201,E221,E241,E501
"""The default beta scaling factors to use if running with REST2."""
# fmt: on

DEFAULT_RESTRAINT_MASK = "!(:WAT,CL,NA,K) & !@/H"
"""The default Amber style selection mask to apply position restraints to."""

DEFAULT_EQUILIBRATE_INTEGRATOR = femto.md.config.LangevinIntegrator(
    timestep=2.0 * openmm.unit.femtosecond,
    friction=1.0 / openmm.unit.picosecond,
)
"""The default integrator to use during equilibration."""
DEFAULT_EQUILIBRATE_RESTRAINTS = {
    DEFAULT_RESTRAINT_MASK: femto.md.config.FlatBottomRestraint(
        k=25.0 * _KCAL_PER_ANG_SQR, radius=1.5 * _ANGSTROM
    )
}
"""The default position restraints to apply during equilibration."""


class ATMSoftCore(BaseModel):
    """Configuration for the ATM soft-core potential."""

    u_max: OpenMMQuantity[_KCAL_PER_MOL] = pydantic.Field(
        ..., description="The 'u max' [kcal/mol] parameter."
    )
    u0: OpenMMQuantity[_KCAL_PER_MOL] = pydantic.Field(
        ..., description="The 'u0' [kcal/mol] parameter."
    )

    a: float = pydantic.Field(..., description="The 'a' parameter.")


class ATMAlignmentRestraint(BaseModel):
    """Configuration for an ATM alignment restraint."""

    type: typing.Literal["atm"] = "atm"

    k_distance: OpenMMQuantity[_KCAL_PER_ANG_SQR] = pydantic.Field(
        ...,
        description="Force constant [kcal/mol/Å^2] of the flat-bottom potential "
        "restraining the distance between the ligands.",
    )
    k_angle: OpenMMQuantity[_KCAL_PER_MOL] = pydantic.Field(
        ...,
        description="Force constant [kcal/mol] of the 1-cos(theta) potential "
        "restraining the angle between the ligands.",
    )
    k_dihedral: OpenMMQuantity[_KCAL_PER_MOL] = pydantic.Field(
        ...,
        description="Force constant [kcal/mol] of the 1-cos(phi) potential restraining "
        "the dihedral angle between the ligands.",
    )


class ATMRestraints(BaseModel):
    """Configure the restraints that will be applied during the ATM calculations."""

    com: femto.md.config.FlatBottomRestraint = pydantic.Field(
        femto.md.config.FlatBottomRestraint(
            k=25.0 * _KCAL_PER_ANG_SQR, radius=5.0 * _ANGSTROM
        ),
        description="The potential that restrains the ligands to the binding site.",
    )
    alignment: ATMAlignmentRestraint | None = pydantic.Field(
        ATMAlignmentRestraint(
            k_distance=2.5 * _KCAL_PER_ANG_SQR,
            k_angle=25.0 * _KCAL_PER_MOL,
            k_dihedral=25.0 * _KCAL_PER_MOL,
        ),
        description="The potential that restrains the orientation of the two ligands "
        "during an RBFE calculation.",
    )

    receptor: femto.md.config.FlatBottomRestraint = pydantic.Field(
        femto.md.config.FlatBottomRestraint(
            k=25.0 * _KCAL_PER_ANG_SQR, radius=1.5 * _ANGSTROM
        ),
        description="The potential that restrains specified receptor atoms to their "
        "initial coordinates.",
    )
    receptor_query: str = pydantic.Field(
        "@CA",
        description="An Amber query used to identify which receptor atoms to restrain.",
    )


class ATMReferenceSelection(BaseModel):
    """Configure how receptor binding sites and ligand alignment reference atoms are
    selected if the user does not explicitly provide them."""

    receptor_cutoff: OpenMMQuantity[_ANGSTROM] = pydantic.Field(
        5.0 * _ANGSTROM,
        description="The minimum distance between a residues' alpha carbon and a "
        "ligand atom to be considered part of the binding site.",
    )

    ligand_method: femto.fe.config.LigandReferenceMethod = pydantic.Field(
        "chen",
        description="The default method to use to select ligand reference atoms during "
        "RBFE calculations if the user does not explicitly provide them.",
    )


class ATMSetupStage(BaseModel):
    """Configure how the complex will be solvated and restrained prior to
    equilibration
    """

    displacement: (
        OpenMMQuantity[_ANGSTROM]
        | tuple[
            OpenMMQuantity[_ANGSTROM],
            OpenMMQuantity[_ANGSTROM],
            OpenMMQuantity[_ANGSTROM],
        ]
    ) = pydantic.Field(
        38.0 * _ANGSTROM,
        description="The distance to displace ligands from the binding site along "
        "an automatically selected displacement vector, or the vector to displace "
        "the ligands by.",
    )

    solvent: femto.md.config.Solvent = pydantic.Field(
        femto.md.config.Solvent(),
        description="Control how the system should be solvated.",
    )

    reference: ATMReferenceSelection = pydantic.Field(
        ATMReferenceSelection(),
        description="Selection of receptor and ligand reference atoms.",
    )
    restraints: ATMRestraints = pydantic.Field(
        ATMRestraints(), description="Control how the system should be restrained."
    )

    apply_hmr: bool = pydantic.Field(
        True, description="Whether to aply hydrogen mass repartitioning to the system."
    )
    hydrogen_mass: OpenMMQuantity[openmm.unit.amu] = pydantic.Field(
        1.5 * openmm.unit.amu,
        description="The mass to assign to hydrogen atoms when applying HMR.",
    )

    apply_rest: bool = pydantic.Field(
        False, description="Whether to prepare the system for REST sampling."
    )
    rest_config: femto.md.config.REST | None = pydantic.Field(
        femto.md.config.REST(
            scale_nonbonded=True,
            scale_torsions=True,
            scale_angles=False,
            scale_bonds=False,
        ),
        description="The REST configuration to use if ``apply_rest`` is True.",
    )

    @pydantic.field_serializer("displacement")
    def _serialize_displacement(v) -> str | list[str]:
        if isinstance(v, openmm.unit.Quantity):
            return pydantic_units.quantity_serializer(v)

        elif isinstance(v, tuple):
            return [pydantic_units.quantity_serializer(x) for x in v]

        raise NotImplementedError

    @pydantic.field_validator("rest_config")
    def _validate_rest_config(cls, v, info: pydantic.ValidationInfo):
        assert (
            info.data["apply_rest"] is False or v is not None
        ), "REST configuration must be provided if ``apply_rest`` is True."
        return v


class ATMStates(BaseModel):
    """Configure the lambda schedules."""

    lambda_1: list[float] = pydantic.Field(DEFAULT_LAMBDA_1)
    lambda_2: list[float] = pydantic.Field(DEFAULT_LAMBDA_2)

    direction: list[typing.Literal[-1, 1]] = pydantic.Field(DEFAULT_DIRECTION)

    alpha: list[float] = pydantic.Field(DEFAULT_ALPHA)
    alpha_unit: typing.ClassVar = openmm.unit.kilocalorie_per_mole**-1
    u0: list[float] = pydantic.Field(DEFAULT_U0)
    u0_unit: typing.ClassVar = openmm.unit.kilocalorie_per_mole
    w0: list[float] = pydantic.Field(DEFAULT_W0)
    w0_unit: typing.ClassVar = openmm.unit.kilocalorie_per_mole

    bm_b0: list[float] | None = pydantic.Field(
        None,  # DEFAULT_BM_B0,
        description="The REST2 beta scaling factors (beta_m / beta_0) to use. "
        "``apply_rest`` must be set to true in the setup config if using this.",
    )

    @pydantic.model_validator(mode="after")
    def _validate_lambda_lengths(self):
        fields = ["lambda_1", "lambda_2", "direction", "alpha", "u0", "w0"]

        if self.bm_b0 is not None:
            fields.append("bm_b0")

        lengths = {len(getattr(self, field)) for field in fields}
        assert len(lengths) == 1, f"the {fields} fields must have the same length."

        return self


class ATMEquilibrateStage(BaseModel):
    """Configure how the system will be equilibrated prior to replica exchange."""

    stages: list[femto.md.config.SimulationStage] = pydantic.Field(
        [
            femto.md.config.Minimization(restraints=DEFAULT_EQUILIBRATE_RESTRAINTS),
            femto.md.config.Anneal(
                integrator=DEFAULT_EQUILIBRATE_INTEGRATOR,
                restraints=DEFAULT_EQUILIBRATE_RESTRAINTS,
                temperature_initial=femto.fe.config.DEFAULT_INITIAL_TEMPERATURE,
                temperature_final=femto.md.config.DEFAULT_TEMPERATURE,
                n_steps=50000,
                frequency=5000,
            ),
            femto.md.config.Simulation(
                integrator=DEFAULT_EQUILIBRATE_INTEGRATOR,
                restraints=DEFAULT_EQUILIBRATE_RESTRAINTS,
                temperature=femto.md.config.DEFAULT_TEMPERATURE,
                pressure=femto.md.config.DEFAULT_PRESSURE,
                n_steps=150000,
            ),
        ]
    )
    report_interval: int = pydantic.Field(
        5000,
        description="The number of steps to report energy, volume, etc after.",
    )

    soft_core: ATMSoftCore = pydantic.Field(
        ATMSoftCore(u_max=1000 * _KCAL_PER_MOL, u0=500 * _KCAL_PER_MOL, a=1.0 / 16.0),
        description="The ATM soft-core potential parameters to use during "
        "equilibration.",
    )


class ATMSamplingStage(femto.md.config.HREMD):
    """Configure how the system will be sampled using Hamiltonian replica exchange."""

    integrator: femto.md.config.LangevinIntegrator = pydantic.Field(
        femto.md.config.LangevinIntegrator(
            timestep=4.0 * openmm.unit.femtosecond,
            friction=1.0 / openmm.unit.picosecond,
        ),
        description="The MD integrator to use.",
    )

    soft_core: ATMSoftCore = pydantic.Field(
        ATMSoftCore(u_max=200 * _KCAL_PER_MOL, u0=100 * _KCAL_PER_MOL, a=1.0 / 16.0),
        description="The ATM soft-core potential parameters to use.",
    )

    analysis_interval: int | None = pydantic.Field(
        None,
        description="The interval (in number of cycles) between estimating and "
        "reporting the free energy. If ``None``, no analysis will be performed.",
    )


class ATMConfig(BaseModel):
    """Configuration the stages of the ATM calculation."""

    type: typing.Literal["atm"] = "atm"

    setup: ATMSetupStage = pydantic.Field(
        ATMSetupStage(), description="Prepare the system for equilibration."
    )

    states: ATMStates = pydantic.Field(
        ATMStates(), description="Configure the lambda schedules."
    )

    equilibrate: ATMEquilibrateStage = pydantic.Field(
        ATMEquilibrateStage(), description="Equilibrate the system."
    )
    sample: ATMSamplingStage = pydantic.Field(
        ATMSamplingStage(),
        description="Sample across lambda windows using replica exchange.",
    )


class ATMEdge(femto.fe.config.Edge):
    """Defines an ATM specific edge in a free energy network."""

    ligand_1_ref_atoms: tuple[str, str, str] | None = pydantic.Field(
        None,
        description="Three (optional) AMBER style queries that select the atoms of the "
        "first ligand to align during an RBFE calculation.",
    )
    ligand_2_ref_atoms: tuple[str, str, str] | None = pydantic.Field(
        None,
        description="Three (optional) AMBER style queries that select the atoms of the "
        "second ligand to align during an RBFE calculation.",
    )

    @property
    def ligand_1_metadata(self) -> dict[str, typing.Any]:
        """Any additional metadata about ligand 1."""
        return {"ref_atoms": self.ligand_1_ref_atoms}

    @property
    def ligand_2_metadata(self) -> dict[str, typing.Any]:
        """Any additional metadata about ligand 2."""
        return {"ref_atoms": self.ligand_2_ref_atoms}


class ATMNetwork(femto.fe.config.Network):
    """Defines an ATM specific free energy network."""

    receptor_ref_query: str | None = pydantic.Field(
        None,
        description="An (optional) AMBER style query to manually select the receptor "
        "atoms that define the binding site. If unspecified, they will be determined "
        "automatically based on the config.",
    )

    edges: list[ATMEdge] = pydantic.Field(
        ..., description="The edges in the free energy network."
    )

    @property
    def receptor_metadata(self) -> dict[str, typing.Any]:
        """Any additional metadata about the receptor."""
        return {"ref_atoms": self.receptor_ref_query}


def load_config(path: pathlib.Path) -> ATMConfig:
    """Load a configuration from a YAML file.

    Args:
        path: The path to the YAML configuration.

    Returns:
        The loaded configuration.
    """
    return ATMConfig(**yaml.safe_load(path.read_text()))
