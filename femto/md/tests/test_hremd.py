import mdtraj
import numpy
import openmm
import openmm.app
import openmm.unit
import pyarrow
import pymbar
import pytest

import femto.fe.ddg
import femto.md.config
import femto.md.hremd
import femto.md.reporting
from femto.md.hremd import (
    _compute_reduced_potentials,
    _propagate_replicas,
    _propose_swaps,
    run_hremd,
)


def _create_harmonic_simulation(
    temperature: openmm.unit.Quantity, mass: openmm.unit.Quantity
) -> openmm.app.Simulation:
    system = openmm.System()
    system.addParticle(mass)

    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(999 * openmm.unit.angstroms, 0, 0),
        openmm.Vec3(0, 999 * openmm.unit.angstroms, 0),
        openmm.Vec3(0, 0, 999 * openmm.unit.angstroms),
    )

    default_k = 1.0 * openmm.unit.kilojoules_per_mole / openmm.unit.angstrom**2
    default_k = default_k.value_in_unit_system(openmm.unit.md_unit_system)

    force = openmm.CustomExternalForce("0.5 * k * (x^2 + y^2 + z^2);")
    force.addGlobalParameter("k", default_k)
    force.addParticle(0, [])
    system.addForce(force)

    positions = numpy.zeros((1, 3)) * openmm.unit.angstroms

    topology = openmm.app.Topology()
    topology.addAtom(
        "Ar",
        openmm.app.Element.getBySymbol("Ar"),
        topology.addResidue("MOL", topology.addChain("A")),
    )

    simulation = openmm.app.Simulation(
        topology,
        system,
        openmm.LangevinMiddleIntegrator(
            temperature, 1.0 / openmm.unit.picosecond, 2.0 * openmm.unit.femtosecond
        ),
        openmm.Platform.getPlatformByName("Reference"),
    )
    simulation.context.setPositions(positions)

    return simulation


def _create_harmonic_states(
    temperature: openmm.unit.Quantity, n_states: int
) -> tuple[list[dict[str, float]], numpy.ndarray]:
    boltzmann_temperature = openmm.unit.MOLAR_GAS_CONSTANT_R * temperature

    f_i = numpy.zeros([n_states])
    states = []

    for i in range(n_states):
        sigma = (1.0 + 0.2 * i) * openmm.unit.angstroms
        k = boltzmann_temperature / sigma**2

        states.append({"k": k.value_in_unit_system(openmm.unit.md_unit_system)})

        f_i[i] = -numpy.log(2 * numpy.pi * (sigma / openmm.unit.angstroms) ** 2) * (
            3.0 / 2.0
        )

    delta_f_ij = f_i - f_i[:, numpy.newaxis]

    return states, delta_f_ij


@pytest.fixture
def harmonic_test_case():
    temperature = 300.0 * openmm.unit.kelvin

    simulation = _create_harmonic_simulation(temperature, 12.0 * openmm.unit.daltons)
    states, expected_delta_f_ij = _create_harmonic_states(temperature, n_states=5)

    return simulation, temperature, states, expected_delta_f_ij


def test_compute_reduced_potentials():
    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    force = openmm.CustomBondForce("lambda")
    force.addGlobalParameter("lambda", 1.0)
    force.addBond(0, 1)
    system.addForce(force)

    temperature = 300.0 * openmm.unit.kelvin
    beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * temperature)

    pressure = 1.0 * openmm.unit.atmospheres

    volume = 9.0 * openmm.unit.nanometers**3
    coords = numpy.zeros((2, 3)) * openmm.unit.angstrom

    lambda_values = numpy.arange(3)
    states = [{"lambda": v} for v in lambda_values]

    context = openmm.Context(
        system,
        openmm.VerletIntegrator(1),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(coords)
    context.setPeriodicBoxVectors(*(numpy.eye(3) * volume ** (1 / 3)))

    # nvt
    expected_u = beta * (lambda_values * openmm.unit.kilojoules_per_mole)
    actual_u = _compute_reduced_potentials(context, states, temperature, None, -1)
    assert numpy.allclose(actual_u, expected_u)

    # npt
    expected_u = beta * (
        lambda_values * openmm.unit.kilojoules_per_mole
        + (pressure * volume * openmm.unit.AVOGADRO_CONSTANT_NA)
    )
    actual_u = _compute_reduced_potentials(context, states, temperature, pressure, -1)
    assert numpy.allclose(actual_u, expected_u)


@pytest.mark.parametrize(
    "n_states, state_idx_offset, mask, expected_pairs",
    [
        (4, 0, set(), [(0, 1), (2, 3)]),
        (4, 1, set(), [(1, 2)]),
        (4, 0, {(1, 0)}, [(2, 3)]),
        (4, 0, {(0, 1)}, [(2, 3)]),
        (3, 0, set(), [(0, 1)]),
        (3, 1, set(), [(1, 2)]),
    ],
)
def test_propose_swaps_neighbours(
    n_states, state_idx_offset, mask, expected_pairs, mocker
):
    mock_propose_swap = mocker.patch("femto.md.hremd._propose_swap", autospec=True)
    mocker.patch("numpy.random.randint", autospec=True, side_effect=[state_idx_offset])

    _propose_swaps(
        numpy.arange(n_states),
        numpy.zeros((n_states, n_states)),
        numpy.zeros((n_states, n_states)),
        numpy.zeros((n_states, n_states)),
        mask,
        "neighbours",
        None,
    )

    proposed_pairs = [call.args[:2] for call in mock_propose_swap.call_args_list]
    assert proposed_pairs == expected_pairs


@pytest.mark.parametrize(
    "n_states, n_proposals, mask, random_idxs, expected_pairs",
    [
        (4, 6, set(), [], [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
        (4, 2, set(), [0, 2], [(0, 1), (0, 3)]),
        (4, 2, {(0, 2)}, [0, 2], [(0, 1), (1, 2)]),
    ],
)
def test_propose_swaps_all(
    n_states, n_proposals, mask, random_idxs, expected_pairs, mocker
):
    mock_propose_swap = mocker.patch("femto.md.hremd._propose_swap", autospec=True)
    mocker.patch(
        "numpy.random.random_integers", autospec=True, return_value=random_idxs
    )

    _propose_swaps(
        numpy.arange(n_states),
        numpy.zeros((n_states, n_states)),
        numpy.zeros((n_states, n_states)),
        numpy.zeros((n_states, n_states)),
        mask,
        "all",
        n_proposals,
    )

    proposed_pairs = [call.args[:2] for call in mock_propose_swap.call_args_list]
    assert proposed_pairs == expected_pairs


def test_propagate_replica_retries_fixed(mocker):
    mock_simulation = mocker.MagicMock()
    mocker.patch(
        "femto.md.hremd._compute_reduced_potentials",
        autospec=True,
        return_value=numpy.zeros((1,)),
    )

    n_steps = 0
    n_expected_steps = 5

    def step_side_effect(*_):
        nonlocal n_steps
        n_steps += 1

        if n_steps < n_expected_steps:
            raise openmm.OpenMMException("Particle coordinate is NaN")

    mock_simulation.step.side_effect = step_side_effect

    _propagate_replicas(
        mock_simulation,
        300 * openmm.unit.kelvin,
        None,
        [{}],
        [openmm.State()],
        1,
        numpy.zeros(1, dtype=int),
        0,
        -1,
        max_retries=n_expected_steps,
    )

    assert n_steps == n_expected_steps


def test_propagate_replica_retries_fails(mocker):
    mock_simulation = mocker.MagicMock()

    error_msg = "Particle coordinate is NaN"

    def step_side_effect(*_):
        raise openmm.OpenMMException(error_msg)

    mock_simulation.step.side_effect = step_side_effect

    with pytest.raises(openmm.OpenMMException, match=error_msg):
        _propagate_replicas(
            mock_simulation,
            300 * openmm.unit.kelvin,
            None,
            [{}],
            [openmm.State()],
            1,
            numpy.zeros(1, dtype=int),
            0,
            -1,
            max_retries=1,
        )


@pytest.mark.parametrize(
    "swap_mode",
    [
        femto.md.config.HREMDSwapMode.ALL.value,
        femto.md.config.HREMDSwapMode.NEIGHBOURS.value,
    ],
)
def test_hremd_sampling(harmonic_test_case, tmp_cwd, swap_mode):
    simulation, temperature, states, expected_delta_f_ij = harmonic_test_case
    n_cycles, n_steps_per_cycle = 200, 100

    trajectory_interval = 5
    analysis_interval = 10

    output_dir = tmp_cwd

    store_trajectories = swap_mode == femto.md.config.HREMDSwapMode.ALL

    def analysis_fn(cycle, u_kn, n_k):
        assert u_kn.shape[0] == len(states)
        assert u_kn.shape[1] == len(states) + cycle * len(states)

        assert n_k.shape == (len(states),)
        assert numpy.sum(n_k) == len(states) + cycle * len(states)

    run_hremd(
        simulation=simulation,
        states=states,
        config=femto.md.config.HREMD(
            temperature=temperature,
            n_warmup_steps=0,
            n_cycles=n_cycles,
            n_steps_per_cycle=n_steps_per_cycle,
            swap_mode=swap_mode,
            trajectory_interval=trajectory_interval if store_trajectories else None,
        ),
        output_dir=output_dir,
        analysis_fn=analysis_fn,
        analysis_interval=analysis_interval,
    )

    samples_path = output_dir / "samples.arrow"
    u_kn, n_k = femto.fe.ddg.load_u_kn(samples_path)

    mbar = pymbar.MBAR(u_kn, n_k)
    mbar_result = mbar.compute_free_energy_differences()

    delta_f_ij, delta_f_ij_std = mbar_result["Delta_f"], mbar_result["dDelta_f"]

    error = numpy.abs(delta_f_ij - expected_delta_f_ij)

    indices = numpy.where(delta_f_ij_std > 0.0)

    n_sigma = numpy.zeros_like(delta_f_ij)
    n_sigma[indices] = error[indices] / delta_f_ij_std[indices]

    max_sigma = 6.0
    assert numpy.all(n_sigma <= max_sigma)

    with pyarrow.OSFile(str(samples_path), "rb") as file:
        with pyarrow.RecordBatchStreamReader(file) as reader:
            output_table = reader.read_all()

    # make sure some replica exchanges did actually take place
    n_proposed_swaps = numpy.sum(
        numpy.array(x) for x in output_table["n_proposed_swaps"].to_pylist()
    )
    n_accepted_swaps = numpy.sum(
        numpy.array(x) for x in output_table["n_accepted_swaps"].to_pylist()
    )

    assert (n_accepted_swaps > 0).any()

    acceptance_rate = n_accepted_swaps / numpy.maximum(1.0, n_proposed_swaps)

    assert numpy.allclose(acceptance_rate - acceptance_rate.T, 0.0)
    assert all(
        acceptance_rate[i, i + 1] > 0.65 for i in range(len(acceptance_rate) - 1)
    )

    exepected_trajectories = [
        output_dir / f"trajectories/r{i}.dcd" for i in range(len(states))
    ]

    for trajectory_path in exepected_trajectories:
        if not store_trajectories:
            assert not trajectory_path.exists()
            continue

        assert trajectory_path.exists()
        trajectory = mdtraj.load(
            str(trajectory_path), top=mdtraj.Topology.from_openmm(simulation.topology)
        )
        assert len(trajectory) == n_cycles // trajectory_interval


def test_hremd_sampling_checkpoint(harmonic_test_case, tmp_cwd, mocker):
    spied_load_checkpoint = mocker.spy(femto.md.hremd, "_load_checkpoint")

    simulation, temperature, states, expected_delta_f_ij = harmonic_test_case
    top = mdtraj.Topology.from_openmm(simulation.topology)

    n_steps_1 = 1
    n_steps_2 = 3

    config_1 = femto.md.config.HREMD(
        n_warmup_steps=0,
        n_cycles=n_steps_1,
        n_steps_per_cycle=1,
        checkpoint_interval=2,
        trajectory_interval=1,
    )
    checkpoint_path = tmp_cwd / "checkpoint.pkl"

    u_kn_1, n_k_1, coords_1 = run_hremd(
        simulation=simulation, states=states, config=config_1, output_dir=tmp_cwd
    )
    assert checkpoint_path.exists()

    traj_1 = mdtraj.load_dcd(str(tmp_cwd / "trajectories/r1.dcd"), top=top)
    traj_1.save(str(tmp_cwd / "top.pdb"))
    assert len(traj_1) == n_steps_1

    assert n_k_1.sum() == len(states)
    assert u_kn_1.shape[1] == len(states)

    config_2 = femto.md.config.HREMD(
        n_warmup_steps=0,
        n_cycles=n_steps_2,
        n_steps_per_cycle=1,
        checkpoint_interval=2,
        trajectory_interval=1,
    )
    u_kn_2, n_k_2, coords_2 = run_hremd(
        simulation=simulation, states=states, config=config_2, output_dir=tmp_cwd
    )

    expected_init_coords = [coord.getPositions(asNumpy=True) for coord in coords_1]
    actual_init_coords = [
        coord.getPositions(asNumpy=True)
        for coord in spied_load_checkpoint.spy_return[1]
    ]
    assert all(
        numpy.allclose(
            actual_coord.value_in_unit(openmm.unit.angstrom),
            expected_coord.value_in_unit(openmm.unit.angstrom),
        )
        for actual_coord, expected_coord in zip(
            actual_init_coords, expected_init_coords, strict=True
        )
    )

    traj_2 = mdtraj.load_dcd(str(tmp_cwd / "trajectories/r1.dcd"), top=top)
    assert len(traj_2) == n_steps_2

    assert n_k_2.sum() == len(states) * n_steps_2
    assert u_kn_2.shape[1] == len(states) * n_steps_2

    for col in range(len(states)):
        assert numpy.allclose(u_kn_2[:, col * n_steps_2], u_kn_1[:, col])

        for i in range(2):
            assert not numpy.allclose(u_kn_2[:, col * n_steps_2 + i], 0.0)

    loaded_u_kn_2, loaded_n_k_2 = femto.fe.ddg.load_u_kn(tmp_cwd / "samples.arrow")
    assert numpy.allclose(loaded_u_kn_2, u_kn_2)
    assert numpy.allclose(loaded_n_k_2, n_k_2)
