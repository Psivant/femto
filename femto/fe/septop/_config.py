"""Configure seperated topology calculations."""

import pathlib
import typing

import numpy
import openmm.unit
import pydantic
import yaml

import femto.fe.config
import femto.md.config
from femto.md.utils.models import BaseModel, OpenMMQuantity

_ANGSTROM = openmm.unit.angstrom

_KCAL_PER_MOL = openmm.unit.kilocalorie_per_mole

_KCAL_PER_ANG_SQR = openmm.unit.kilocalorie_per_mole / _ANGSTROM**2
_KCAL_PER_RAD_SQR = openmm.unit.kilocalorie_per_mole / openmm.unit.radians**2

# fmt: off
DEFAULT_LAMBDA_VDW_1_COMPLEX      = [0.0] * 8                                  + [0.00, 0.0, 0.00] + numpy.linspace(0.0, 1.0, 8).tolist()  # noqa: E201,E221,E241,E501
"""The default vdW lambda schedule of the first ligand in the complex phase."""
DEFAULT_LAMBDA_CHARGES_1_COMPLEX  = [0.0] * 8                                  + [0.25, 0.5, 0.75] + [1.0] * 8                             # noqa: E201,E221,E241,E501
"""The default charge lambda schedule of the first ligand in the complex phase."""
DEFAULT_LAMBDA_VDW_2_COMPLEX      = numpy.linspace(1.0, 0.0, 8).tolist()       + [0.00, 0.0, 0.00] + [0.0] * 8                             # noqa: E201,E221,E241,E501
"""The default vdW lambda schedule of the second ligand in the complex phase."""
DEFAULT_LAMBDA_CHARGES_2_COMPLEX  = [1.0] * 8                                  + [0.75, 0.5, 0.25] + [0.0] * 8                             # noqa: E201,E221,E241,E501
"""The default charge lambda schedule of the second ligand in the complex phase."""

DEFAULT_LAMBDA_BORESCH_LIGAND_1   = [0.0, 0.05, 0.1, 0.3, 0.5, 0.75, 1.0, 1.0] + [1.0] * 3         + [1.0] * 8                                   # noqa: E201,E221,E241,E501
"""The default lambda schedule of the Boresch restraint on the first ligand in the
complex phase."""
DEFAULT_LAMBDA_BORESCH_LIGAND_2   = [1.0] * 8                                  + [1.0] * 3         + [1.0, 0.95, 0.9, 0.7, 0.5, 0.25, 0.0, 0.0]  # noqa: E201,E221,E241,E501
"""The default lambda schedule of the Boresch restraint on the second ligand in the
complex phase."""

DEFAULT_LAMBDA_VDW_1_SOLUTION     = [0.0, 0.000, 0.00, 0.000, 0.0, 0.000, 0.00, 0.000, 0.00, 0.12, 0.240, 0.36, 0.480, 0.6, 0.700, 0.77, 0.850, 1.0]  # noqa: E201,E221,E241,E501
"""The default vdW lambda schedule of the first ligand in the solution phase."""
DEFAULT_LAMBDA_CHARGES_1_SOLUTION = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.00, 1.00, 1.000, 1.00, 1.000, 1.0, 1.000, 1.00, 1.000, 1.0]  # noqa: E201,E221,E241,E501
"""The default charge lambda schedule of the first ligand in the solution phase."""
DEFAULT_LAMBDA_VDW_2_SOLUTION     = [1.0, 0.850, 0.77, 0.700, 0.6, 0.480, 0.36, 0.240, 0.12, 0.00, 0.000, 0.00, 0.000, 0.0, 0.000, 0.00, 0.000, 0.0]  # noqa: E201,E221,E241,E501
"""The default vdW lambda schedule of the second ligand in the solution phase."""
DEFAULT_LAMBDA_CHARGES_2_SOLUTION = [1.0, 1.000, 1.00, 1.000, 1.0, 1.000, 1.00, 1.000, 1.00, 1.00, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]  # noqa: E201,E221,E241,E501
"""The default charge lambda schedule of the second ligand in the solution phase."""
# fmt: on

DEFAULT_BORESCH_K_DISTANCE = 20.0 * _KCAL_PER_ANG_SQR
"""The default force constant of the Boresch distance restraint."""
DEFAULT_BORESCH_K_THETA = 20.0 * _KCAL_PER_RAD_SQR
"""The default force constant of the Boresch angle restraint.""" ""

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


class SepTopComplexRestraints(femto.md.config.BoreschRestraint):
    """Configure the restraints to apply in the complex phase."""

    scale_k_angle_a: typing.Literal[True] = pydantic.Field(
        True,
        description="Whether to scale the force constant for the r2, r3, and l1 angle "
        "based upon the *initial* distance between r3 and l1.",
    )


class SepTopSolutionRestraints(BaseModel):
    """Configure the restraints to apply in the solution phase."""

    type: typing.Literal["harmonic"] = "harmonic"

    k_distance: OpenMMQuantity[_KCAL_PER_ANG_SQR] = pydantic.Field(
        2.4 * _KCAL_PER_ANG_SQR,
        description="Force constant [kcal/mol/Å^2] of the distance restraint that will "
        "separate the ligands during an RBFE calculation.",
    )


DEFAULT_COMPLEX_RESTRAINTS = SepTopComplexRestraints(
    k_distance=DEFAULT_BORESCH_K_DISTANCE,
    k_angle_a=DEFAULT_BORESCH_K_THETA * 2.0,
    k_angle_b=DEFAULT_BORESCH_K_THETA,
    k_dihedral_a=DEFAULT_BORESCH_K_THETA,
    k_dihedral_b=DEFAULT_BORESCH_K_THETA,
    k_dihedral_c=DEFAULT_BORESCH_K_THETA,
)
DEFAULT_SOLUTION_RESTRAINTS = SepTopSolutionRestraints()


class SepTopSetupStage(BaseModel):
    """Configure how the complex will be solvated and restrained prior to
    equilibration
    """

    solvent: femto.md.config.Solvent = pydantic.Field(
        femto.md.config.Solvent(),
        description="Control how the system should be solvated.",
    )

    restraints: SepTopComplexRestraints | SepTopSolutionRestraints = pydantic.Field(
        ...,
        description="Control how the system should be restrained.",
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

    fep_config: femto.fe.config.FEP = pydantic.Field(
        femto.fe.config.FEP(ligands_can_interact=False),
        description="Configure how to alchemically couple the ligands.",
    )

    @pydantic.field_validator("rest_config")
    def _validate_rest_config(cls, v, info: pydantic.ValidationInfo):
        assert (
            info.data["apply_rest"] is False or v is not None
        ), "REST configuration must be provided if ``apply_rest`` is True."
        return v


class SepTopStates(BaseModel):
    """Configure the lambda schedules."""

    lambda_vdw_ligand_1: list[float] = pydantic.Field(
        ..., description="The vdW lambda schedule of the first ligand."
    )
    lambda_vdw_ligand_2: list[float] | None = pydantic.Field(
        ..., description="The vdW lambda schedule of the second ligand."
    )

    lambda_charges_ligand_1: list[float] = pydantic.Field(
        ..., description="The charge lambda schedule of the first ligand."
    )
    lambda_charges_ligand_2: list[float] | None = pydantic.Field(
        ..., description="The charge lambda schedule of the second ligand."
    )

    lambda_boresch_ligand_1: list[float] | None = pydantic.Field(
        ...,
        description="The lambda schedule of the boresch restraint on the first ligand.",
    )
    lambda_boresch_ligand_2: list[float] | None = pydantic.Field(
        ...,
        description="The lambda schedule of the boresch restraint on the second "
        "ligand.",
    )

    bm_b0: list[float] | None = pydantic.Field(
        None,
        description="The REST2 beta scaling factors (beta_m / beta_0) to use. Set this "
        "to ``None`` to disable REST2 scaling.",
    )

    @pydantic.model_validator(mode="after")
    def _validate_lambda_lengths(self):
        fields = ["lambda_vdw_ligand_1", "lambda_charges_ligand_1"]

        assert (
            self.lambda_vdw_ligand_2 is None and self.lambda_charges_ligand_2 is None
        ) or (
            self.lambda_vdw_ligand_2 is not None
            and self.lambda_charges_ligand_2 is not None
        ), "if ligand 2 is present both a vdW and charge schedule should be set"

        if self.lambda_vdw_ligand_2 is not None:
            fields.extend(["lambda_vdw_ligand_2", "lambda_charges_ligand_2"])

        if self.bm_b0 is not None:
            fields.append("bm_b0")

        lengths = {len(getattr(self, field)) for field in fields}
        assert len(lengths) == 1, f"the {fields} fields must have the same length."

        return self


class SepTopEquilibrateStage(BaseModel):
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


class SepTopSamplingStage(femto.md.config.HREMD):
    """Configure how the system will be sampled using Hamiltonian replica exchange."""

    integrator: femto.md.config.LangevinIntegrator = pydantic.Field(
        femto.md.config.LangevinIntegrator(
            timestep=4.0 * openmm.unit.femtosecond,
            friction=1.0 / openmm.unit.picosecond,
        ),
        description="The MD integrator to use.",
    )

    pressure: OpenMMQuantity[openmm.unit.atmosphere] | None = pydantic.Field(
        femto.md.config.DEFAULT_PRESSURE,
        description="The pressure to simulate at, or ``None`` to run in NVT.",
    )
    barostat_frequency: int = pydantic.Field(
        25,
        description="The frequency at which to apply the barostat. This is ignored if "
        "``pressure`` is ``None``.",
    )

    analysis_interval: int | None = pydantic.Field(
        None,
        description="The interval (in number of cycles) between estimating and "
        "reporting the free energy. If ``None``, no analysis will be performed.",
    )


class SepTopPhaseConfig(BaseModel):
    """Configure one phase (i.e. complex or solution) of a separated topology
    FE calculation."""

    setup: SepTopSetupStage = pydantic.Field(
        ..., description="Prepare the system for equilibration."
    )

    states: SepTopStates = pydantic.Field(
        ..., description="Configure the lambda schedules."
    )

    equilibrate: SepTopEquilibrateStage = pydantic.Field(
        SepTopEquilibrateStage(), description="Equilibrate the system."
    )
    sample: SepTopSamplingStage = pydantic.Field(
        SepTopSamplingStage(),
        description="Sample the system across lambda windows using HREMD.",
    )


class SepTopConfig(BaseModel):
    """Configuration a separated topology FE calculation."""

    type: typing.Literal["septop"] = "septop"

    complex: SepTopPhaseConfig = pydantic.Field(
        SepTopPhaseConfig(
            setup=SepTopSetupStage(restraints=DEFAULT_COMPLEX_RESTRAINTS),
            states=SepTopStates(
                lambda_vdw_ligand_1=DEFAULT_LAMBDA_VDW_1_COMPLEX,
                lambda_charges_ligand_1=DEFAULT_LAMBDA_CHARGES_1_COMPLEX,
                lambda_boresch_ligand_1=DEFAULT_LAMBDA_BORESCH_LIGAND_1,
                lambda_vdw_ligand_2=DEFAULT_LAMBDA_VDW_2_COMPLEX,
                lambda_charges_ligand_2=DEFAULT_LAMBDA_CHARGES_2_COMPLEX,
                lambda_boresch_ligand_2=DEFAULT_LAMBDA_BORESCH_LIGAND_2,
            ),
        ),
        description="Configure the complex phase calculations.",
    )
    solution: SepTopPhaseConfig = pydantic.Field(
        SepTopPhaseConfig(
            setup=SepTopSetupStage(restraints=DEFAULT_SOLUTION_RESTRAINTS),
            states=SepTopStates(
                lambda_vdw_ligand_1=DEFAULT_LAMBDA_VDW_1_SOLUTION,
                lambda_charges_ligand_1=DEFAULT_LAMBDA_CHARGES_1_SOLUTION,
                lambda_vdw_ligand_2=DEFAULT_LAMBDA_VDW_2_SOLUTION,
                lambda_charges_ligand_2=DEFAULT_LAMBDA_CHARGES_2_SOLUTION,
                lambda_boresch_ligand_1=None,
                lambda_boresch_ligand_2=None,
            ),
        ),
        description="Configure the solution phase calculations.",
    )


def load_config(path: pathlib.Path) -> SepTopConfig:
    """Load a configuration from a YAML file.

    Args:
        path: The path to the YAML configuration.

    Returns:
        The loaded configuration.
    """
    return SepTopConfig(**yaml.safe_load(path.read_text()))
