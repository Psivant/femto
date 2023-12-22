"""Perform Hamiltonian replica exchange sampling."""
import contextlib
import itertools
import logging
import pathlib
import typing

import numpy
import openmm.app
import openmm.unit
import pyarrow
import tqdm

import femto.md.config
import femto.md.utils.mpi
import femto.md.utils.openmm

if typing.TYPE_CHECKING:
    from mpi4py import MPI


_LOGGER = logging.getLogger(__name__)


_T = typing.TypeVar("_T")


_KJ_PER_MOL = openmm.unit.kilojoules_per_mole


_HREMDStorage = typing.NamedTuple(
    "HREMDStorage",
    [
        ("file", pyarrow.OSFile),
        ("writer", pyarrow.RecordBatchStreamWriter),
        ("schema", pyarrow.Schema),
    ],
)


@contextlib.contextmanager
def _create_storage(
    mpi_comm: "MPI.Intracomm", output_path: pathlib.Path, n_states: int
) -> _HREMDStorage | None:
    """Open a storage ready for writing.

    Args:
        mpi_comm: The main MPI communicator.
        output_path: The path to write the output to.

    Returns:
        The report object if running on rank 0, or none otherwise.
    """
    if mpi_comm.rank != 0:
        yield None
        return

    schema = pyarrow.schema(
        [
            ("step", pyarrow.int64()),
            (
                "u_kn",
                pyarrow.list_(pyarrow.list_(pyarrow.float64(), n_states), n_states),
            ),
            ("replica_to_state_idx", pyarrow.list_(pyarrow.int16(), n_states)),
            (
                "n_proposed_swaps",
                pyarrow.list_(pyarrow.list_(pyarrow.int64(), n_states), n_states),
            ),
            (
                "n_accepted_swaps",
                pyarrow.list_(pyarrow.list_(pyarrow.int64(), n_states), n_states),
            ),
        ]
    )

    output_path.unlink(missing_ok=True)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with pyarrow.OSFile(str(output_path), "wb") as file:
        with pyarrow.RecordBatchStreamWriter(file, schema) as writer:
            yield _HREMDStorage(file, writer, schema)


def _create_trajectory_storage(
    simulation: openmm.app.Simulation,
    n_replicas: int,
    replica_idx_offset: int,
    n_steps_per_cycle: int,
    trajectory_interval: int | None,
    output_dir: pathlib.Path,
    exit_stack: contextlib.ExitStack,
) -> list[openmm.app.DCDFile] | None:
    """Open a DCD trajectory reporter per replica.

    Args:
        simulation: The simulation to report.
        n_replicas: The number of replicas being sampled on this process.
        replica_idx_offset: The index of the first replica being sampled on this process
        n_steps_per_cycle: The number of steps per cycle.
        trajectory_interval: The interval with which to write the trajectory.
        output_dir: The root output directory. The trajectories will be written to
            `output_dir/trajectories/r{replica_idx}.dcd`.
        exit_stack: The exit stack to use for opening the files.

    Returns:
        The trajectory reporters if the trajectory interval is greater than zero.
    """
    if trajectory_interval is None or trajectory_interval <= 0:
        return

    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(exist_ok=True, parents=True)

    return [
        openmm.app.DCDFile(
            exit_stack.enter_context(
                (trajectory_dir / f"r{replica_idx_offset + i}.dcd").open("wb")
            ),
            simulation.topology,
            simulation.integrator.getStepSize(),
            n_steps_per_cycle * trajectory_interval,
        )
        for i in range(n_replicas)
    ]


def _compute_reduced_potentials(
    context: openmm.Context,
    states: list[dict[str, float]],
    temperature: openmm.unit.Quantity,
    pressure: openmm.unit.Quantity | None,
    force_groups: set[int] | int,
) -> numpy.ndarray:
    """Compute the reduced potential of the given coordinates for each state being
    sampled.

    Args:
        context: The current simulation config.
        states: The states being sampled.
        temperature: The temperature being sampled at.
        force_groups: The force groups to consider. Use -1 if all groups should be
            considered.

    Returns:
        The reduced potentials with ``shape=(n_states,)``.
    """
    beta = 1.0 / (openmm.unit.BOLTZMANN_CONSTANT_kB * temperature)

    reduced_potentials = numpy.zeros(len(states))

    for state_idx, state in enumerate(states):
        for key, value in state.items():
            context.setParameter(key, value)

        reduced_potential = (
            context.getState(getEnergy=True, groups=force_groups).getPotentialEnergy()
            / openmm.unit.AVOGADRO_CONSTANT_NA
        )

        if pressure is not None:
            reduced_potential += pressure * context.getState().getPeriodicBoxVolume()

        reduced_potentials[state_idx] = beta * reduced_potential

    return reduced_potentials


def _propagate_replicas(
    simulation: openmm.app.Simulation,
    temperature: openmm.unit.Quantity,
    pressure: openmm.unit.Quantity | None,
    states: list[dict[str, float]],
    coords: list[openmm.State],
    n_steps: int,
    replica_to_state_idx: numpy.ndarray,
    replica_idx_offset: int,
    force_groups: set[int] | int,
    max_retries: int,
    enforcePB: bool = True
):
    """Propagate all replica states forward in time.

    Args:
        simulation: The main simulation object.
        states: The states being sampled.
        temperature: The temperature being sampled at.
        pressure: The pressure being sampled at.
        coords: The starting coordinates of each replica.
        n_steps: The number of steps to propagate by.
        replica_to_state_idx: A map between each replica and the state it is sampling.
        replica_idx_offset: The index of the first state sampled by this worker.
        force_groups: The force groups to consider. Use -1 if all groups should be
            considered.
        max_retries: The maximum number of times to attempt to step a replica if a NaN
            is encountered
        enforcePB: Maintain wrapped coordinates in output.
    """

    if n_steps <= 0:
        return

    n_states = len(states)
    reduced_potentials = numpy.zeros((n_states, n_states))

    for local_replica_idx in range(len(coords)):
        replica_idx = local_replica_idx + replica_idx_offset
        state_idx = replica_to_state_idx[replica_idx]

        local_replica_coords = coords[local_replica_idx]

        for attempt in range(max_retries):
            try:
                simulation.context.setState(coords[local_replica_idx])

                for key, value in states[state_idx].items():
                    simulation.context.setParameter(key, value)

                simulation.step(n_steps)

                local_replica_coords = simulation.context.getState(
                    getPositions=True, getVelocities=True, enforcePeriodicBox=enforcePB
                )
                femto.md.utils.openmm.check_for_nans(local_replica_coords)
            except openmm.OpenMMException:
                # randomize the velocities and try again
                simulation.context.setVelocitiesToTemperature(temperature)
                message = f"NaN detected for replica={replica_idx} state={state_idx}"

                if attempt == max_retries - 1:
                    _LOGGER.warning(f"{message} that could not be resolved by retries.")
                    raise

                _LOGGER.warning(f"{message}, retrying {attempt + 1}/{max_retries}")
                continue

            break

        coords[local_replica_idx] = local_replica_coords

        reduced_potentials[:, replica_idx] = _compute_reduced_potentials(
            simulation.context, states, temperature, pressure, force_groups
        )

    return reduced_potentials


def _propose_swap(
    state_idx_i: int,
    state_idx_j: int,
    reduced_potentials: numpy.ndarray,
    n_proposed_swaps: numpy.ndarray,
    n_accepted_swaps: numpy.ndarray,
    replica_to_state_idx: numpy.ndarray,
):
    """Attempt to swap a pair of states between replicas.

    Notes:
        replica_to_state_idx
    """

    state_to_replica_idx = {
        state_idx: replica_idx
        for replica_idx, state_idx in enumerate(replica_to_state_idx)
    }

    replica_idx_i = state_to_replica_idx[state_idx_i]
    replica_idx_j = state_to_replica_idx[state_idx_j]

    potential_ii = reduced_potentials[state_idx_i, replica_idx_i]
    potential_ij = reduced_potentials[state_idx_i, replica_idx_j]
    potential_ji = reduced_potentials[state_idx_j, replica_idx_i]
    potential_jj = reduced_potentials[state_idx_j, replica_idx_j]

    criteria = -((potential_ij - potential_ii) - (potential_jj - potential_ji))

    if any(
        numpy.isnan(x) for x in [potential_ii, potential_ij, potential_ji, potential_jj]
    ):
        return

    n_proposed_swaps[state_idx_i, state_idx_j] += 1
    n_proposed_swaps[state_idx_j, state_idx_i] += 1

    if criteria >= 0.0 or numpy.random.rand() < numpy.exp(criteria):
        replica_to_state_idx[replica_idx_i] = state_idx_j
        replica_to_state_idx[replica_idx_j] = state_idx_i

        n_accepted_swaps[state_idx_i, state_idx_j] += 1
        n_accepted_swaps[state_idx_j, state_idx_i] += 1


def _propose_swaps(
    replica_to_state_idx: numpy.ndarray,
    reduced_potentials: numpy.ndarray,
    n_proposed_swaps: numpy.ndarray,
    n_accepted_swaps: numpy.ndarray,
    mask: set[tuple[int, int]],
    mode: femto.md.config.HREMDSwapMode | None,
    max_swaps: int | None,
):
    """Attempt to swap states between replicas.

    Args:
        replica_to_state_idx: The replica to state index map to modify in-place.
        reduced_potentials: The matrix of reduced potentials with
            ``reduced_potentials[state_idx][replica_idx] = value``
        n_proposed_swaps: An array tracking the number of proposed swaps to modify
            in-place.
        n_accepted_swaps: An array tracking the number of accepted swaps to modify
            in-place.
        mask: Pairs of state indices that cannot be swapped.
        mode: The swap mode. This can either be:
            * Neighbours: only try and swap adjacent states
            * All: try and swap all states stochastically
        max_swaps: The maximum number of proposals to make if running in 'all' mode.
            This variable does nothing when running in 'neighbours' mode.
    """

    if mode is None:
        return

    n_states = len(replica_to_state_idx)

    if mode == femto.md.config.HREMDSwapMode.NEIGHBOURS:
        # select whether to swap [0, 1], [2, 3], ... OR [1, 2], [3, 4], ...
        state_idx_offset = numpy.random.randint(2)

        pairs = [
            (state_idx_i, state_idx_i + 1)
            for state_idx_i in range(state_idx_offset, n_states - 1, 2)
            if (state_idx_i, state_idx_i + 1) not in mask
            and (state_idx_i + 1, state_idx_i) not in mask
        ]

    elif mode == femto.md.config.HREMDSwapMode.ALL:
        pairs = [
            (state_idx_i, state_idx_j)
            for state_idx_i, state_idx_j in itertools.combinations(range(n_states), r=2)
            if (state_idx_i, state_idx_j) not in mask
            and (state_idx_j, state_idx_i) not in mask
        ]
        max_swaps = len(pairs) if max_swaps is None else max_swaps

        pairs = (
            pairs
            if len(pairs) <= max_swaps
            else [
                pairs[i]
                for i in numpy.random.random_integers(0, len(pairs) - 1, max_swaps)
            ]
        )

    else:
        raise NotImplementedError

    for state_idx_i, state_idx_j in pairs:
        _propose_swap(
            state_idx_i,
            state_idx_j,
            reduced_potentials,
            n_proposed_swaps,
            n_accepted_swaps,
            replica_to_state_idx,
        )


def _store_potentials(
    replica_to_state_idx: numpy.ndarray,
    reduced_potentials: numpy.ndarray,
    n_proposed_swaps: numpy.ndarray,
    n_accepted_swaps: numpy.ndarray,
    storage: _HREMDStorage | None,
    step: int,
):
    """Report the current state of the replica exchange simulation to an output file."""

    if storage is None:
        return

    record = pyarrow.record_batch(
        [
            (step,),
            (reduced_potentials.tolist(),),
            (replica_to_state_idx,),
            (n_proposed_swaps.tolist(),),
            (n_accepted_swaps.tolist(),),
        ],
        schema=storage.schema,
    )
    storage.writer.write_batch(record)


def _store_trajectory(
    coords: list[openmm.State], storage: list[openmm.app.DCDFile] | None
):
    """Store the current replica states to DCD files."""

    if storage is None:
        return

    for state, output_file in zip(coords, storage, strict=True):
        output_file.writeModel(
            positions=state.getPositions(),
            periodicBoxVectors=state.getPeriodicBoxVectors(),
        )


def run_hremd(
    simulation: openmm.app.Simulation,
    states: list[dict[str, float]],
    config: femto.md.config.HREMD,
    output_dir: pathlib.Path,
    swap_mask: set[tuple[int, int]] | None = None,
    force_groups: set[int] | int = -1,
    initial_coords: list[openmm.State] | None = None,
    analysis_fn: typing.Callable[[int, numpy.ndarray, numpy.ndarray], None]
    | None = None,
    analysis_interval: int | None = None,
    enforcePB: bool = True
):
    """Run a Hamiltonian replica exchange simulation.

    Args:
        simulation: The main simulation object to sample using.
        states: The states to sample at. This should be a dictionary with keys
            corresponding to global context parameters.
        config: The sampling configuration.
        output_dir: The directory to store the sampled energies and statistics to, and
            any trajectory files if requested in the config.
        swap_mask: Pairs of states that should not be swapped.
        force_groups: The force groups to consider when computing the reduced potentials
        initial_coords: The initial coordinates of each state. If not provided, the
            coordinates will be taken from the simulation object.
        analysis_fn: A function to call after every ``analysis_interval`` cycles. It
            should take as arguments the current cycle number, the reduced potentials
            with ``shape=(n_states, n_samples)`` and the number of samples of each
            state with ``shape=(n_states,)``.
        analysis_interval: The interval with which to call the analysis function.
            If ``None``, no analysis will be performed.
        enforcePB: Maintain wrapped coordinates in output.
    """
    from mpi4py import MPI

    n_states = len(states)

    states = [
        femto.md.utils.openmm.evaluate_ctx_parameters(state, simulation.system)
        for state in states
    ]

    swap_mask = set() if swap_mask is None else swap_mask

    n_proposed_swaps = numpy.zeros((n_states, n_states))
    n_accepted_swaps = numpy.zeros((n_states, n_states))

    replica_to_state_idx = numpy.arange(n_states)

    u_kn, n_k = (
        numpy.empty((n_states, n_states * config.n_cycles)),
        numpy.zeros(n_states, dtype=int),
    )
    has_sampled = numpy.zeros(n_states * config.n_cycles, bool)

    barostats = [
        force
        for force in simulation.system.getForces()
        if isinstance(force, openmm.MonteCarloBarostat)
    ]
    assert len(barostats) == 0 or len(barostats) == 1

    pressure = (
        None
        if len(barostats) == 0 or barostats[0].getFrequency() <= 0
        else barostats[0].getDefaultPressure()
    )

    samples_path = output_dir / "samples.arrow"

    with (
        femto.md.utils.mpi.get_mpi_comm() as mpi_comm,
        _create_storage(mpi_comm, samples_path, n_states) as storage,
        contextlib.ExitStack() as exit_stack,
    ):
        # each MPI process may be responsible for propagating multiple states,
        # e.g. if we have 20 states to simulate windows but only 4 GPUs to run on.
        n_replicas, replica_idx_offset = femto.md.utils.mpi.divide_tasks(
            mpi_comm, n_states
        )

        if initial_coords is None:
            # enforcePeriodicBox
            coords = [simulation.context.getState(getPositions=True,
                                                  enforcePeriodicBox=enforcePB)
                                                  ] * n_replicas
        else:
            coords = [initial_coords[i + replica_idx_offset] for i in range(n_replicas)]

        if mpi_comm.rank == 0:
            _LOGGER.info(f"running {config.n_warmup_steps} warm-up steps")

        _propagate_replicas(
            simulation,
            config.temperature,
            pressure,
            states,
            coords,
            config.n_warmup_steps,
            replica_to_state_idx,
            replica_idx_offset,
            force_groups,
            config.max_step_retries,
            enforcePB=enforcePB
        )

        if mpi_comm.rank == 0:
            _LOGGER.info(f"running {config.n_cycles} replica exchange cycles")

        trajectory_storage = _create_trajectory_storage(
            simulation,
            n_replicas,
            replica_idx_offset,
            config.n_steps_per_cycle,
            config.trajectory_interval,
            output_dir,
            exit_stack,
        )

        for cycle in tqdm.tqdm(
            range(config.n_cycles), total=config.n_cycles, disable=mpi_comm.rank != 0
        ):
            reduced_potentials = _propagate_replicas(
                simulation,
                config.temperature,
                pressure,
                states,
                coords,
                config.n_steps_per_cycle,
                replica_to_state_idx,
                replica_idx_offset,
                force_groups,
                config.max_step_retries,
                enforcePB=enforcePB
            )
            reduced_potentials = mpi_comm.reduce(reduced_potentials, MPI.SUM, 0)

            has_sampled[replica_to_state_idx * config.n_cycles + cycle] = True
            u_kn[:, replica_to_state_idx * config.n_cycles + cycle] = reduced_potentials

            n_k += 1

            should_save_trajectory = (
                config.trajectory_interval is not None
                and cycle % config.trajectory_interval == 0
            )

            if should_save_trajectory:
                _store_trajectory(coords, trajectory_storage)

            should_analyze = (
                analysis_fn is not None
                and analysis_interval is not None
                and cycle % analysis_interval == 0
            )

            if should_analyze:
                analysis_fn(cycle, u_kn[:, has_sampled], n_k)

            if mpi_comm.rank == 0:
                _store_potentials(
                    replica_to_state_idx,
                    reduced_potentials,
                    n_proposed_swaps,
                    n_accepted_swaps,
                    storage,
                    cycle * config.n_steps_per_cycle,
                )
                _propose_swaps(
                    replica_to_state_idx,
                    reduced_potentials,
                    n_proposed_swaps,
                    n_accepted_swaps,
                    swap_mask,
                    config.swap_mode,
                    config.max_swaps,
                )

            replica_to_state_idx = mpi_comm.bcast(replica_to_state_idx, 0)

        mpi_comm.barrier()

    return u_kn, n_k
