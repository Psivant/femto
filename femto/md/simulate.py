"""Run OpenMM simulations."""

import collections
import copy
import logging
import time

import openmm.app
import openmm.unit
import parmed

import femto.md.anneal
import femto.md.config
import femto.md.constants
import femto.md.reporting
import femto.md.reporting.openmm
import femto.md.restraints
import femto.md.utils
import femto.md.utils.mpi
import femto.md.utils.openmm

_LOGGER = logging.getLogger(__name__)

_KJ_PER_NM = openmm.unit.kilojoules_per_mole / openmm.unit.nanometer


def _validate_system(system: openmm.System):
    """Ensure that a system does not already contain a barostat or integrator."""

    n_barostats = sum(
        1
        for force in system.getForces()
        if isinstance(force, openmm.MonteCarloBarostat)
    )
    if n_barostats > 0:
        raise RuntimeError("the system should not contain a barostat already")


def _prepare_simulation(
    system: openmm.System,
    topology: parmed.Structure,
    state: dict[str, float],
    coords: openmm.State | None,
    config: femto.md.config.SimulationStage,
    platform: femto.md.constants.OpenMMPlatform,
) -> openmm.app.Simulation:
    """Prepare an OpenMM simulation object ready for a given stage."""
    system = copy.deepcopy(system)

    for mask, restraint in config.restraints.items():
        system.addForce(
            femto.md.restraints.create_position_restraints(topology, mask, restraint)
        )
    if isinstance(config, femto.md.config.Simulation) and config.pressure is not None:
        barostat = openmm.MonteCarloBarostat(
            config.pressure, config.temperature, config.barostat_frequency
        )
        system.addForce(barostat)

    if isinstance(config, femto.md.config.Anneal):
        integrator = femto.md.utils.openmm.create_integrator(
            config.integrator, config.temperature_initial
        )
    elif isinstance(config, femto.md.config.Simulation):
        integrator = femto.md.utils.openmm.create_integrator(
            config.integrator, config.temperature
        )
    else:
        integrator = openmm.VerletIntegrator(0.0001)

    femto.md.utils.openmm.assign_force_groups(system)

    simulation = femto.md.utils.openmm.create_simulation(
        system, topology, coords, integrator, state, platform
    )
    return simulation


def simulate_state(
    system: openmm.System,
    topology: parmed.Structure,
    state: dict[str, float],
    stages: list[femto.md.config.SimulationStage],
    platform: femto.md.constants.OpenMMPlatform,
    reporter: femto.md.reporting.openmm.OpenMMStateReporter | None = None,
    enforce_pbc: bool = False,
) -> openmm.State:
    """Simulate a system following the specified ``stages``, at a given 'state' (i.e.
    a set of context parameters, such as free energy lambda values)

    Args:
        system: The system to simulate.
        topology: The topology to simulate.
        state: The state to simulate at.
        stages: The stages to run.
        platform: The accelerator to use.
        reporter: The reporter to use to record system statistics such as volume and
            energy.
        enforce_pbc: Whether to enforce periodic boundary conditions when retrieving
            the final coordinates.

    Returns:
        The final coordinates and box vectors.
    """

    reporter = (
        reporter
        if reporter is not None
        else femto.md.reporting.openmm.OpenMMStateReporter(
            femto.md.reporting.NullReporter(), "", 999999999
        )
    )

    stage_counter = collections.defaultdict(int)

    coords = None

    for stage_idx, stage in enumerate(stages):
        stage_name = f"{stage.type}-{stage_counter[stage.type]}"
        stage_counter[stage.type] += 1

        reporter.tag = f"equilibration/{stage_name}"

        with femto.md.utils.timer.timeit(f"stage={stage_idx}-{stage.type}"):
            simulation = _prepare_simulation(
                system, topology, state, coords, stage, platform
            )
            simulation.reporters.append(reporter)

            if isinstance(stage, femto.md.config.Minimization):
                reporter.report(simulation, simulation.context.getState(getEnergy=True))
                simulation.minimizeEnergy(
                    stage.tolerance.value_in_unit(_KJ_PER_NM), stage.max_iterations
                )
                reporter.report(simulation, simulation.context.getState(getEnergy=True))
            elif isinstance(stage, femto.md.config.Anneal):
                start_time = time.perf_counter()
                femto.md.anneal.anneal_temperature(
                    simulation,
                    stage.temperature_initial,
                    stage.temperature_final,
                    stage.n_steps,
                    stage.frequency,
                )
                end_time = time.perf_counter()

                ns = stage.n_steps * stage.integrator.timestep.value_in_unit(
                    openmm.unit.nanosecond
                )

                elapsed_time_s = end_time - start_time
                elapsed_time_day = elapsed_time_s / (24 * 60 * 60)

                ns_per_day = ns / elapsed_time_day
                _LOGGER.info(f"annealing had performance of {ns_per_day:.4f} ns/day.")
            elif isinstance(stage, femto.md.config.Simulation):
                start_time = time.perf_counter()
                simulation.step(stage.n_steps)
                end_time = time.perf_counter()

                ns = stage.n_steps * stage.integrator.timestep.value_in_unit(
                    openmm.unit.nanosecond
                )

                elapsed_time_s = end_time - start_time
                elapsed_time_day = elapsed_time_s / (24 * 60 * 60)

                ns_per_day = ns / elapsed_time_day
                _LOGGER.info(f"simulation had performance of {ns_per_day:.4f} ns/day.")
            else:
                raise NotImplementedError(f"unknown stage type {type(stage)}")

            _LOGGER.info(
                f"after {stage_name} "
                f"{femto.md.utils.openmm.get_simulation_summary(simulation)}"
            )
            coords = simulation.context.getState(
                getPositions=True,
                getVelocities=True,
                getForces=True,
                getEnergy=True,
                enforcePeriodicBox=enforce_pbc,
            )

    return coords


def simulate_states(
    system: openmm.System,
    topology: parmed.Structure,
    states: list[dict[str, float]],
    stages: list[femto.md.config.SimulationStage],
    platform: femto.md.constants.OpenMMPlatform,
    reporter: femto.md.reporting.openmm.OpenMMStateReporter | None = None,
    enforce_pbc: bool = False,
) -> list[openmm.State]:
    """Simulate the system at each 'state' using ``simulate_state``.

    If running using MPI, each process will be responsible for simulating at a subset
    of the states.

    Args:
        system: The system to simulate.
        topology: The topology of the system to simulate.
        states: The states of the system to simulate.
        stages: The stages to run.
        platform: The accelerator to use.
        reporter: The reporter to use to record system statistics such as volume and
            energy.
        enforce_pbc: Whether to enforce periodic boundary conditions when retrieving
            the final coordinates.

    Returns:
        The final coordinates at each state.
    """

    with femto.md.utils.mpi.get_mpi_comm() as mpi_comm:
        # figure out how many states will be run in this MPI process
        n_local_states, state_offset = femto.md.utils.mpi.divide_tasks(
            mpi_comm, len(states)
        )

        final_coords: dict[int, openmm.State] = {}

        for i in range(n_local_states):
            state_idx = i + state_offset

            final_coords[state_idx] = simulate_state(
                system,
                topology,
                states[i + state_offset],
                stages,
                platform,
                reporter if state_idx == 0 else None,
                enforce_pbc=enforce_pbc,
            )

        final_coords = femto.md.utils.mpi.reduce_dict(final_coords, mpi_comm)

    return [final_coords[i] for i in range(len(states))]
