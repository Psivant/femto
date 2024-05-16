"""Common configuration models."""

import abc
import enum
import typing

import omegaconf
import openmm.unit
import pydantic

from femto.md.utils.models import BaseModel, OpenMMQuantity

_T = typing.TypeVar("_T")

_ANGSTROM = openmm.unit.angstrom

_KCAL_PER_MOL = openmm.unit.kilocalorie_per_mole
_KCAL_PER_ANG_SQR = openmm.unit.kilocalorie_per_mole / _ANGSTROM**2
_KCAL_PER_RAD_SQR = openmm.unit.kilocalorie_per_mole / openmm.unit.radians**2

DEFAULT_TEMPERATURE = 298.15 * openmm.unit.kelvin
"""The default temperature to simulate at"""
DEFAULT_PRESSURE = 1.0 * openmm.unit.bar
"""The default pressure to simulate at"""


DEFAULT_TLEAP_SOURCES = ["leaprc.water.tip3p", "leaprc.protein.ff14SB"]
"""The default Leap parameter files to load when parameterizing the solvent /
receptor"""


class FlatBottomRestraint(BaseModel):
    """Configuration for a flat bottom restraint."""

    type: typing.Literal["flat-bottom"] = "flat-bottom"

    k: OpenMMQuantity[_KCAL_PER_ANG_SQR] = pydantic.Field(
        ..., description="Force constant [kcal/mol/Å^2] of the restraint."
    )
    radius: OpenMMQuantity[_ANGSTROM] = pydantic.Field(
        ..., description="The radius [Å] of the restraint."
    )


class BoreschRestraint(BaseModel):
    """Configuration for a Boresch style restraint between three receptor atoms
    (r1, r2, r3) and three ligand atoms (l1, l2, l3).
    """

    type: typing.Literal["boresch"] = "boresch"

    k_distance: OpenMMQuantity[_KCAL_PER_ANG_SQR] = pydantic.Field(
        ...,
        description="Force constant [kcal/mol/Å^2] of the harmonic distance restraint "
        "between r3 and l1.",
    )

    k_angle_a: OpenMMQuantity[_KCAL_PER_RAD_SQR] = pydantic.Field(
        ...,
        description="Force constant [kcal/mol/rad^2] of the harmonic angle restraint "
        "on the angle formed by r2, r3, and l1.",
    )
    k_angle_b: OpenMMQuantity[_KCAL_PER_RAD_SQR] = pydantic.Field(
        ...,
        description="Force constant [kcal/mol/rad^2] of the harmonic angle restraint "
        "on the angle formed by r3, l1, and l2.",
    )

    k_dihedral_a: OpenMMQuantity[_KCAL_PER_RAD_SQR] = pydantic.Field(
        ...,
        description="Force constant [kcal/mol/rad^2] of the harmonic dihedral "
        "restraint on the dihedral angle formed by r1, r2, r3, and l1.",
    )
    k_dihedral_b: OpenMMQuantity[_KCAL_PER_RAD_SQR] = pydantic.Field(
        ...,
        description="Force constant [kcal/mol/rad^2] of the harmonic dihedral "
        "restraint on the dihedral angle formed by r2, r3, l1, and l2.",
    )
    k_dihedral_c: OpenMMQuantity[_KCAL_PER_RAD_SQR] = pydantic.Field(
        ...,
        description="Force constant [kcal/mol/rad^2] of the harmonic dihedral "
        "restraint on the dihedral angle formed by r3, l1, l2, and l3.",
    )


class Solvent(BaseModel):
    """Configuration for solvating a system."""

    ionic_strength: OpenMMQuantity[openmm.unit.molar] = pydantic.Field(
        0.0 * openmm.unit.molar,
        description="The total concentration of ions pairs (``anion`` and ``cation``) "
        "to add to approximate an ionic strength. This does not include ions that are "
        "added to neutralize the system.",
    )

    neutralize: bool = pydantic.Field(
        True, description="Whether to add counter ions to neutralize the system."
    )

    cation: typing.Literal["Na+", "K+"] = pydantic.Field(
        "K+", description="The cation to use when neutralizing the system."
    )
    anion: typing.Literal["Cl-"] = pydantic.Field(
        "Cl-", description="The anion to use when neutralizing the system."
    )

    water_model: typing.Literal["tip3p"] = pydantic.Field(
        "tip3p", description="The water model to use."
    )
    tleap_sources: list[str] = pydantic.Field(
        [*DEFAULT_TLEAP_SOURCES],
        description="The tLeap parameters to source when parameterizing the system "
        "minus any ligands (and possibly receptors) which should be handled separately",
    )

    box_padding: OpenMMQuantity[_ANGSTROM] | None = pydantic.Field(
        10.0 * _ANGSTROM,
        description="The minimum distance between any complex atom (including any "
        "offset ligands) and the box wall. This option is mutually exclusive with "
        "``n_waters``.",
    )

    n_waters: int | None = pydantic.Field(
        None,
        description="The number of extra waters to solvate the complex using. This "
        "option is mutually exclusive with ``box_padding``.",
    )

    @pydantic.model_validator(mode="after")
    def _validate_n_waters(self) -> "Solvent":
        assert (
            self.box_padding is None or self.n_waters is None
        ), "`box_padding` and `n_waters` are mutually exclusive"

        return self


class LangevinIntegrator(BaseModel):
    """Configuration for a Langevin integrator."""

    type: typing.Literal["langevin"] = "langevin"

    timestep: OpenMMQuantity[openmm.unit.picosecond] = pydantic.Field(
        0.002 * openmm.unit.picosecond
    )
    friction: OpenMMQuantity[openmm.unit.picosecond**-1] = pydantic.Field(
        0.5 / openmm.unit.picosecond
    )

    constraint_tolerance: float = pydantic.Field(
        1.0e-6, description="The tolerance with which constraints are maintained."
    )


class REST(BaseModel):
    """Configure REST2 setup."""

    scale_bonds: bool = pydantic.Field(
        False,
        description="Whether to scale bond force constants by ``beta_m / beta_0``.",
    )
    scale_angles: bool = pydantic.Field(
        False,
        description="Whether to scale angle force constants by ``beta_m / beta_0``.",
    )
    scale_torsions: bool = pydantic.Field(
        True,
        description="Whether to scale torsion barrier heights  by ``beta_m / beta_0``.",
    )
    scale_nonbonded: bool = pydantic.Field(
        True,
        description="Whether to scale non-bonded epsilons by ``beta_m / beta_0`` and "
        "charges by ``sqrt(beta_m / beta_0)``.",
    )


class _SimulationStep(BaseModel, abc.ABC):
    """Common configuration for a simulation 'step'."""

    type: str

    integrator: LangevinIntegrator = pydantic.Field(
        ..., description="The integrator to use during the simulation."
    )

    restraints: dict[str, FlatBottomRestraint] = pydantic.Field(
        {},
        description="The position restraints to apply during the minimization. The "
        "keys should be AMBER style selection masks.",
    )


class Minimization(BaseModel):
    """Configuration for a minimization."""

    type: typing.Literal["minimization"] = "minimization"

    restraints: dict[str, FlatBottomRestraint] = pydantic.Field(
        {},
        description="The position restraints to apply during the minimization. The "
        "keys should be AMBER style selection masks.",
    )

    tolerance: OpenMMQuantity[_KCAL_PER_MOL / _ANGSTROM] = pydantic.Field(
        10.0 * _KCAL_PER_MOL / _ANGSTROM,
        description="Minimization will be halted once the root-mean-square value of "
        "all force components reaches this tolerance.",
    )
    max_iterations: int = pydantic.Field(
        0,
        description="The maximum number of iterations to perform. If 0, minimization "
        "will continue until the tolerance is met.",
    )


class Simulation(_SimulationStep):
    """Configuration for an MD simulation."""

    type: typing.Literal["simulation"] = "simulation"

    temperature: OpenMMQuantity[openmm.unit.kelvin] = pydantic.Field(
        ...,
        description="The temperature to simulate at.",
    )

    pressure: OpenMMQuantity[openmm.unit.atmospheres] | None = pydantic.Field(
        ...,
        description="The pressure to simulate at, or none to run in NVT.",
    )
    barostat_frequency: int = pydantic.Field(
        25,
        description="The frequency at which to apply the barostat. This is ignored if "
        "``pressure`` is ``None``.",
    )

    n_steps: int = pydantic.Field(
        ..., description="The number of steps to simulate for."
    )


class Anneal(_SimulationStep):
    """Configuration for a temperature annealing NVT MD simulation."""

    type: typing.Literal["anneal"] = "anneal"

    temperature_initial: OpenMMQuantity[openmm.unit.kelvin] = pydantic.Field(
        ...,
        description="The starting temperature to simulate at.",
    )
    temperature_final: OpenMMQuantity[openmm.unit.kelvin] = pydantic.Field(
        ...,
        description="The final temperature to simulate at.",
    )

    n_steps: int = pydantic.Field(
        ..., description="The number of steps to simulate for."
    )
    frequency: int = pydantic.Field(
        ..., description="The frequency at which to increment the temperature."
    )


SimulationStage = Minimization | Simulation | Anneal


class HREMDSwapMode(str, enum.Enum):
    """The mode in which to propose state swaps between replicas.

    * ``NEIGHBOURS``: Only allow swapping the current state with neighbouring states.
    * ``ALL``: Allow swapping the current state with any other state.
    """

    NEIGHBOURS = "neighbours"
    ALL = "all"


HREMDSwapModeLiteral = typing.Literal["neighbours", "all"]


class HREMD(BaseModel):
    """Configuration for Hamiltonian replica exchange sampling"""

    temperature: OpenMMQuantity[openmm.unit.kelvin] = pydantic.Field(
        DEFAULT_TEMPERATURE, description="The temperature to sample at."
    )

    n_warmup_steps: int = pydantic.Field(
        150000,
        description="The number of steps to run each replica for before starting hremd "
        "trials. All energies gathered during this period will be discarded.",
    )

    n_steps_per_cycle: int = pydantic.Field(
        1000,
        description="The number of steps to propagate the system by before attempting "
        "an exchange.",
    )
    n_cycles: int = pydantic.Field(
        2500,
        description="The number of cycles of "
        "'propagate the system' -> 'exchange replicas' to run.",
    )

    max_step_retries: int = pydantic.Field(
        5,
        description="The maximum number of times to attempt to step if a NaN is "
        "encountered before raising an exception",
    )

    swap_mode: HREMDSwapModeLiteral | None = pydantic.Field(
        HREMDSwapMode.ALL.value,
        description="The mode in which to propose state swaps between replicas. This "
        "can either be: 'neighbours', only try and swap adjacent states or ii. 'all', "
        "try and swap all states stochastically. If ``None``, no replica exchanges "
        "will be attempted.",
    )
    max_swaps: int | None = pydantic.Field(
        None,
        description="The maximum number of swap proposals to make if running in 'all' "
        "mode. This variable does nothing when running in 'neighbours' mode.",
    )

    trajectory_interval: int | None = pydantic.Field(
        None,
        description="The number of cycles to run before saving the current replica "
        "states to DCD trajectory files. If ``None``, no trajectories will be saved.",
    )
    trajectory_enforce_pbc: bool = pydantic.Field(
        False,
        description="Whether to apply periodic boundary conditions when retrieving "
        "coordinates for writing to trajectory files.",
    )

    checkpoint_interval: int | None = pydantic.Field(
        None,
        description="The number of cycles to run before saving the current replica "
        "states to checkpoint files. If ``None``, no checkpoints will be saved.",
    )


def merge_configs(*configs: _T) -> dict[str, typing.Any]:
    """Merge multiple configurations together.

    Args:
        configs: The configurations to merge. These should either be dataclasses or
            plain dictionaries. Values in subsequent configs will overwrite values in
            previous ones.

    Returns:
        The merged configuration.
    """

    if len(configs) == 0:
        raise ValueError("At least one config must be provided")

    configs = [
        config.model_dump() if isinstance(config, pydantic.BaseModel) else config
        for config in configs
    ]

    return omegaconf.OmegaConf.to_object(omegaconf.OmegaConf.merge(*configs))
