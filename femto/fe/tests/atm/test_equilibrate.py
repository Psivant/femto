import numpy
import openmm
import openmm.app
import openmm.unit
import pytest

import femto.fe.atm
import femto.fe.config
import femto.md.config
import femto.md.constants
import femto.md.reporting
import femto.md.reporting.openmm
import femto.md.utils.openmm
import femto.top
from femto.fe.atm._equilibrate import equilibrate_states
from femto.md.tests.mocking import build_mock_structure


@pytest.fixture
def mock_system() -> openmm.System:
    system = openmm.System()
    system.addParticle(1.0)

    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(50.0, 0.0, 0.0) * openmm.unit.angstrom,
        openmm.Vec3(0.0, 50.0, 0.0) * openmm.unit.angstrom,
        openmm.Vec3(0.0, 0.0, 50.0) * openmm.unit.angstrom,
    )

    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    force.addParticle(1.0, 1.0, 0.0)
    force.setForceGroup(femto.md.constants.OpenMMForceGroup.NONBONDED)
    system.addForce(force)

    return system


@pytest.fixture
def mock_topology(mock_system) -> femto.top.Topology:
    topology = build_mock_structure(["[Ar]"])
    topology.xyz = numpy.array([[0.0, 0.0, 0.0]]) * openmm.unit.angstrom
    topology.residues[0].name = femto.md.constants.LIGAND_1_RESIDUE_NAME
    topology.box = (
        numpy.array(
            mock_system.getDefaultPeriodicBoxVectors().value_in_unit(
                openmm.unit.angstrom
            )
        )
        * openmm.unit.angstrom
    )

    return topology


@pytest.fixture
def mock_coords(mock_system) -> list[openmm.State]:
    coords = [numpy.array([[1.234, 0.0, 0.0]]), numpy.array([[4.321, 0.0, 0.0]])]
    box_vectors = [
        numpy.array([[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 3.0]]),
        numpy.array([[6.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.1]]),
    ]

    states = []

    for coord, box in zip(coords, box_vectors, strict=True):
        context = openmm.Context(mock_system, openmm.VerletIntegrator(0.0001))
        context.setPeriodicBoxVectors(*(box * openmm.unit.angstrom))
        context.setPositions(coord * openmm.unit.angstrom)

        states.append(context.getState(getPositions=True))

    return states


def test_equilibrate_system(mock_system, mock_topology, mock_coords, mocker):
    n_expected_states = 2

    mock_simulate = mocker.patch(
        "femto.md.simulate.simulate_states",
        return_value=mock_coords,
    )
    mock_stages = [femto.md.config.Minimization()]

    mock_config = femto.fe.atm.ATMEquilibrateStage(stages=mock_stages)
    mock_states = femto.fe.atm.ATMStates(
        lambda_1=[0.0] * n_expected_states,
        lambda_2=[0.0] * n_expected_states,
        direction=[1, -1],
        alpha=[0.0] * n_expected_states,
        u0=[0.0] * n_expected_states,
        w0=[0.0] * n_expected_states,
    )

    outputs = equilibrate_states(
        mock_system,
        mock_topology,
        mock_states,
        mock_config,
        offset=[1.0, 1.0, 1.0] * openmm.unit.angstrom,
        platform="Reference",
    )

    mock_simulate.assert_called_once()
    (
        called_system,
        called_topology,
        called_states,
        called_stages,
        called_platform,
        _,
    ) = mock_simulate.call_args.args

    assert femto.md.constants.OpenMMForceGroup.ATM in {
        force.getForceGroup() for force in called_system.getForces()
    }

    assert called_topology == mock_topology
    assert called_platform == "Reference"
    assert called_stages == mock_stages

    assert outputs == mock_coords

    expected_coords = [
        numpy.array([[1.234, 0.0, 0.0]]),
        numpy.array([[4.321, 0.0, 0.0]]),
    ]
    expected_box = numpy.array([[6.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 3.0]])

    for output, expected_coord in zip(outputs, expected_coords, strict=True):
        assert isinstance(output, openmm.State)

        assert numpy.allclose(
            output.getPositions(asNumpy=True).value_in_unit(openmm.unit.angstrom),
            expected_coord,
        )
        assert numpy.allclose(
            output.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                openmm.unit.angstrom
            ),
            expected_box,
        )
