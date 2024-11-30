import numpy
import openmm
import pytest

import femto.fe.fep
import femto.md.config
import femto.md.constants
import femto.md.utils.openmm
import femto.top
from femto.md.simulate import _validate_system, simulate_states
from femto.md.tests.mocking import build_mock_structure


@pytest.fixture
def mock_topology() -> femto.top.Topology:
    topology = build_mock_structure(["[Ar]"])
    topology.residues[0].name = femto.md.constants.LIGAND_1_RESIDUE_NAME
    topology.box = numpy.eye(3) * 50.0 * openmm.unit.angstrom

    return topology


@pytest.fixture
def mock_system(mock_topology) -> openmm.System:
    system = openmm.System()
    system.addParticle(1.0)
    system.setDefaultPeriodicBoxVectors(*mock_topology.box)

    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    force.addGlobalParameter(femto.fe.fep.LAMBDA_VDW_LIGAND_1, 1.0)
    force.addParticle(1.0, 1.0, 0.0)
    force.setForceGroup(femto.md.constants.OpenMMForceGroup.NONBONDED)
    system.addForce(force)

    return system


def mock_openmm_state(system: openmm.System, coords: numpy.ndarray):
    context = openmm.Context(system, openmm.VerletIntegrator(0.001))
    context.setPositions(coords)
    return context.getState(getPositions=True)


def test_validate_system_with_barostat():
    system = openmm.System()
    system.addForce(openmm.MonteCarloBarostat(1.0, 298.15))

    with pytest.raises(RuntimeError, match="the system should not contain a barostat"):
        _validate_system(system)


# def test_simulate_state(mock_system, mock_topology, mocker):
#     spied_add_restraints = mocker.spy(
#         femto.fe.septop.equilibrate, "_create_position_restraints"
#     )
#     spied_minimize = mocker.spy(openmm.app.Simulation, "minimizeEnergy")
#     spied_heat = mocker.spy(femto.md.anneal, "anneal_temperature")
#
#     mock_config = femto.fe.septop.config.SepTopEquilibrateStage(
#         n_thermal_annealing_steps=1,
#         thermal_annealing_frequency=1,
#         n_thermal_npt_steps=5,
#         barostat_frequency=1,
#     )
#
#     final_state = femto.fe.septop.equilibrate.equilibrate_state(
#         mock_system,
#         mock_topology,
#         {femto.fe.fep.LAMBDA_VDW_LIGAND_1: 1.0},
#         mock_config,
#         platform="Reference",
#     )
#     assert isinstance(final_state, openmm.State)
#
#     spied_add_restraints.assert_called_once_with(
#         mock_topology, mock_config.restraint_mask, mock_config.restraint
#     )
#     spied_minimize.assert_called_once()
#     spied_heat.assert_called_once()
#
#     initial_vectors = (
#         numpy.array(mock_topology.box_vectors.value_in_unit(openmm.unit.angstrom))
#         * openmm.unit.angstrom
#     )
#     final_vectors = final_state.getPeriodicBoxVectors(asNumpy=True)
#
#     assert not femto.md.utils.openmm.all_close(initial_vectors, final_vectors)


def test_simulate_states(mock_system, mock_topology, mocker):
    n_expected_states = 2

    expected_coords = [
        numpy.array([[i, 0.0, 0.0]]) * openmm.unit.angstrom
        for i in range(n_expected_states)
    ]

    mock_equilibrate = mocker.patch(
        "femto.md.simulate.simulate_state",
        side_effect=[
            mock_openmm_state(mock_system, coords) for coords in expected_coords
        ],
    )
    mock_config = [femto.md.config.Minimization()]
    mock_states = [{"lambda_vdw_ligand_1": 0.0}] * n_expected_states

    outputs = simulate_states(
        mock_system,
        mock_topology,
        mock_states,
        mock_config,
        platform="Reference",
    )

    assert mock_equilibrate.call_count == n_expected_states
    assert len(outputs) == n_expected_states

    actual_coords = [output.getPositions(asNumpy=True) for output in outputs]

    assert [
        femto.md.utils.openmm.all_close(actual, expected)
        for actual, expected in zip(actual_coords, expected_coords, strict=True)
    ]
