"""Utilities for manipulating OpenMM objects."""

import numpy
import openmm
import openmm.app
import openmm.unit
import parmed

import femto.md.config
from femto.md.constants import OpenMMForceGroup, OpenMMForceName, OpenMMPlatform


def is_close(
    v1: openmm.unit.Quantity,
    v2: openmm.unit.Quantity,
    rtol=1.0e-5,
    atol=1.0e-8,
    equal_nan=False,
) -> bool | numpy.ndarray:
    """Compares if two unit wrapped values are close using ``numpy.is_close``"""

    if not v1.unit.is_compatible(v2.unit):
        return False

    return numpy.isclose(
        v1.value_in_unit(v1.unit),
        v2.value_in_unit(v1.unit),
        atol=atol,
        rtol=rtol,
        equal_nan=equal_nan,
    )


def all_close(
    v1: openmm.unit.Quantity,
    v2: openmm.unit.Quantity,
    rtol=1.0e-5,
    atol=1.0e-8,
    equal_nan=False,
) -> bool:
    """Compares if all values in two unit wrapped array are close using
    ``numpy.allclose``
    """

    if not v1.unit.is_compatible(v2.unit):
        return False

    if v1.shape != v2.shape:
        return False

    return numpy.allclose(
        v1.value_in_unit(v1.unit),
        v2.value_in_unit(v1.unit),
        atol=atol,
        rtol=rtol,
        equal_nan=equal_nan,
    )


def assign_force_groups(system: openmm.System):
    """Assign standard force groups to forces in a system.

    Notes:
        * COM, alignment, and position restraints are detected by their name. If their
          name is not set to a ``OpenMMForceName``, they will be assigned a force group
          of ``OTHER``.

    Args:
        system: The system to modify in-place.
    """

    force: openmm.Force

    for force in system.getForces():
        if force.getName() == OpenMMForceName.COM_RESTRAINT:
            force.setForceGroup(OpenMMForceGroup.COM_RESTRAINT)
        elif force.getName() == OpenMMForceName.ALIGNMENT_RESTRAINT:
            force.setForceGroup(OpenMMForceGroup.ALIGNMENT_RESTRAINT)
        elif force.getName().startswith(OpenMMForceName.POSITION_RESTRAINT):
            force.setForceGroup(OpenMMForceGroup.POSITION_RESTRAINT)

        elif isinstance(force, openmm.HarmonicBondForce):
            force.setForceGroup(OpenMMForceGroup.BOND)
        elif isinstance(force, openmm.HarmonicAngleForce):
            force.setForceGroup(OpenMMForceGroup.ANGLE)
        elif isinstance(
            force, (openmm.PeriodicTorsionForce, openmm.CustomTorsionForce)
        ):
            force.setForceGroup(OpenMMForceGroup.DIHEDRAL)
        elif isinstance(force, (openmm.NonbondedForce, openmm.CustomNonbondedForce)):
            force.setForceGroup(OpenMMForceGroup.NONBONDED)
        elif isinstance(force, openmm.ATMForce):
            force.setForceGroup(OpenMMForceGroup.ATM)
        elif isinstance(force, openmm.MonteCarloBarostat):
            force.setForceGroup(OpenMMForceGroup.BAROSTAT)
        else:
            force.setForceGroup(OpenMMForceGroup.OTHER)


def check_for_nans(coords: openmm.State):
    """Checks whether a state has NaN coordinates.

    Raises:
        openmm.OpenMMException: If any of the coordinates are NaN.
    """

    if numpy.isnan(
        coords.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom)
    ).any():
        raise openmm.OpenMMException("Positions were NaN")


def compute_energy(
    system: openmm.System,
    positions: openmm.unit.Quantity,
    box_vectors: openmm.unit.Quantity | None,
    context_params: dict[str, float] | None = None,
    platform: OpenMMPlatform = OpenMMPlatform.REFERENCE,
    groups: int | set[int] = -1,
) -> openmm.unit.Quantity:
    """Computes the potential energy of a system at a given set of positions.

    Args:
        system: The system to compute the energy of.
        positions: The positions to compute the energy at.
        box_vectors: The box vectors to use if any.
        context_params: Any global context parameters to set.
        platform: The platform to use.
        groups: The force groups to include in the energy calculation.

    Returns:
        The computed energy.
    """
    context_params = context_params if context_params is not None else {}

    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.0001 * openmm.unit.femtoseconds),
        openmm.Platform.getPlatformByName(str(platform)),
    )

    for key, value in context_params.items():
        context.setParameter(key, value)

    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)

    return context.getState(getEnergy=True, groups=groups).getPotentialEnergy()


def evaluate_ctx_parameters(
    state: dict[str, float], system: openmm.System
) -> dict[str, float]:
    """Inspects an OpenMM system for any context parameters that contain special
    keywords that indicate a computed value (e.g. ``sqrt<bm_b0>``) and injects their
    values into a state dictionary.

    Args:
        state: The core set of context parameter values that may appear in any
            expressions.
        system: The system to inspect for context parameters.

    Returns:
        The updated state dictionary.
    """
    import sympy

    # create a context to easily see which global parameters are available to be set
    ctx = openmm.Context(
        system,
        openmm.VerletIntegrator(0.0001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    found_ctx_parameters = {*ctx.getParameters()}

    updated_state = {**state}

    for ctx_parameter in found_ctx_parameters:
        expression = sympy.sympify(ctx_parameter.replace("<", "(").replace(">", ")"))

        if expression.is_symbol:
            continue

        value = expression.subs(state)

        try:
            value = float(value)
        except TypeError as e:
            raise ValueError(
                f"could not evaluate context parameter {ctx_parameter}"
            ) from e

        updated_state[ctx_parameter] = value

    missing = {*state} - {*updated_state}
    assert len(missing) == 0, f"missing ctx parameters: {missing}"

    return updated_state


def get_simulation_summary(
    simulation: openmm.app.Simulation, groups: set[int] | int = -1
) -> str:
    """Returns a string summarising the current energy and volume of a system being
    simulated.

    Args:
        simulation: The current simulation.
        groups: The force groups to include in the energy calculation.

    Returns:
        The formatted string.
    """

    state = simulation.context.getState(getEnergy=True, groups=groups)

    box_vectors = state.getPeriodicBoxVectors()
    volume = box_vectors[0][0] * box_vectors[1][1] * box_vectors[2][2]

    return f"energy={state.getPotentialEnergy()} volume={volume}"


def create_integrator(
    config: femto.md.config.LangevinIntegrator, temperature: openmm.unit.Quantity
) -> openmm.Integrator:
    """Creates an OpenMM integrator.

    Args:
        config: The configuration of the integrator.
        temperature: The temperature to simulate at.

    Returns:
        The created integrator.
    """

    if isinstance(config, femto.md.config.LangevinIntegrator):
        integrator = openmm.LangevinMiddleIntegrator(
            temperature, config.friction, config.timestep
        )
        integrator.setConstraintTolerance(config.constraint_tolerance)
    else:
        raise NotImplementedError(f"integrator type {type(config)} is not supported")

    return integrator


def create_simulation(
    system: openmm.System,
    topology: parmed.Structure,
    coords: openmm.State | None,
    integrator: openmm.Integrator,
    state: dict[str, float] | None,
    platform: femto.md.constants.OpenMMPlatform,
) -> openmm.app.Simulation:
    """Creates an OpenMM simulation object.

    Args:
        system: The system to simulate
        topology: The topology being simulated.
        coords: The initial coordinates and box vectors. If ``None``, the coordinates
            and box vectors from the topology will be used.
        integrator: The integrator to evolve the system with.
        state: The state of the system to simulate.
        platform: The accelerator to run using.

    Returns:
        The created simulation.
    """

    platform_properties = (
        {"Precision": "mixed"} if platform.upper() in ["CUDA", "OPENCL"] else {}
    )
    platform = openmm.Platform.getPlatformByName(platform)

    if coords is not None:
        system.setDefaultPeriodicBoxVectors(*coords.getPeriodicBoxVectors())
    else:
        system.setDefaultPeriodicBoxVectors(*topology.box_vectors)

    simulation = openmm.app.Simulation(
        topology.topology, system, integrator, platform, platform_properties
    )

    if coords is None:
        simulation.context.setPeriodicBoxVectors(*topology.box_vectors)
        simulation.context.setPositions(topology.positions)
    else:
        simulation.context.setState(coords)

    state = femto.md.utils.openmm.evaluate_ctx_parameters(state, simulation.system)

    for k, v in state.items():
        simulation.context.setParameter(k, v)

    return simulation


def get_pressure(system: openmm.System) -> openmm.unit.Quantity | None:
    """Extracts the pressure from a system if it has a barostat.

    Notes:
        * If the system has no barostat, this function will return ``None``.
        * Only the first barostat found will be used.
    """

    barostats = [
        force
        for force in system.getForces()
        if isinstance(
            force,
            (
                openmm.MonteCarloBarostat,
                openmm.MonteCarloAnisotropicBarostat,
                openmm.MonteCarloFlexibleBarostat,
                openmm.MonteCarloMembraneBarostat,
            ),
        )
    ]
    assert len(barostats) == 0 or len(barostats) == 1

    return (
        None
        if len(barostats) == 0 or barostats[0].getFrequency() <= 0
        else barostats[0].getDefaultPressure()
    )
