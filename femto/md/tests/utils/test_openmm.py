import copy

import numpy
import openmm.app
import openmm.unit
import pytest

import femto.fe.fep
from femto.md.constants import OpenMMForceGroup
from femto.md.tests.mocking import build_mock_structure
from femto.md.utils.openmm import (
    all_close,
    assign_force_groups,
    check_for_nans,
    create_simulation,
    evaluate_ctx_parameters,
    get_simulation_summary,
    is_close,
)


@pytest.fixture
def mock_system() -> openmm.System:
    system = openmm.System()

    for _ in range(4):
        system.addParticle(17.0)

    bond_force = openmm.HarmonicBondForce()
    bond_force.addBond(0, 1, 1.0, 100.0)
    system.addForce(bond_force)

    angle_force = openmm.HarmonicAngleForce()
    angle_force.addAngle(0, 1, 2, 1.25, 200.0)
    system.addForce(angle_force)

    angle_force = openmm.PeriodicTorsionForce()
    angle_force.addTorsion(0, 1, 2, 3, 1, 3.14, 300.0)
    system.addForce(angle_force)

    nonbonded_force = openmm.NonbondedForce()

    for i in range(system.getNumParticles()):
        nonbonded_force.addParticle(0.0, 1.0, i / 10.0)

    system.addForce(nonbonded_force)

    return system


@pytest.fixture
def mock_atm_system(mock_system) -> openmm.System:
    [(nonbonded_idx, nonbonded_force)] = (
        (i, force)
        for i, force in enumerate(mock_system.getForces())
        if isinstance(force, openmm.NonbondedForce)
    )
    nonbonded_force.setForceGroup(OpenMMForceGroup.NONBONDED)

    atm_force = openmm.ATMForce(0.0, 0.0, 0.0, 0.1, 0.0, 110.0, 0.1, 1.0 / 16.0, 1)
    atm_force.addForce(copy.deepcopy(nonbonded_force))
    mock_system.addForce(atm_force)
    mock_system.removeForce(nonbonded_idx)

    return mock_system


@pytest.mark.parametrize(
    "v1, v2, expected_close",
    [
        (
            1.0 * openmm.unit.kilocalorie_per_mole,
            4.184 * openmm.unit.kilojoules_per_mole,
            True,
        ),
        (
            1.0 * openmm.unit.kilocalorie_per_mole,
            1.0 * openmm.unit.kilojoules_per_mole,
            False,
        ),
        (1.0 * openmm.unit.kilocalorie_per_mole, 1.0 * openmm.unit.kelvin, False),
    ],
)
def test_is_close(v1, v2, expected_close):
    assert is_close(v1, v2) == expected_close


@pytest.mark.parametrize(
    "v1, v2, expected_close",
    [
        (
            numpy.ones((1, 3)) * openmm.unit.kilocalorie_per_mole,
            numpy.ones((1, 3)) * openmm.unit.kilocalorie_per_mole,
            True,
        ),
        (
            numpy.ones((1, 3)) * openmm.unit.kilocalorie_per_mole,
            numpy.ones((1, 2)) * openmm.unit.kilocalorie_per_mole,
            False,
        ),
        (
            numpy.ones((1, 3)) * openmm.unit.kilocalorie_per_mole,
            numpy.ones((1, 3)) * openmm.unit.kelvin,
            False,
        ),
        (
            numpy.ones((1, 3)) * openmm.unit.kilocalorie_per_mole,
            (numpy.ones((1, 3)) * 2) * openmm.unit.kilocalorie_per_mole,
            False,
        ),
    ],
)
def test_all_close(v1, v2, expected_close):
    assert all_close(v1, v2) == expected_close


def test_check_for_nans():
    system = openmm.System()
    system.addParticle(1.0 * openmm.unit.amu)
    system.addParticle(1.0 * openmm.unit.amu)

    context = openmm.Context(system, openmm.VerletIntegrator(0.0001))
    context.setPositions(
        numpy.array([[0.0, 0.0, 0.0], [0.0, numpy.nan, 0.0]]) * openmm.unit.angstrom
    )

    with pytest.raises(openmm.OpenMMException, match="Positions were NaN"):
        check_for_nans(context.getState(getPositions=True))


def test_assign_force_groups(mock_atm_system):
    mock_atm_system.addForce(openmm.MonteCarloBarostat(1.0, 298.15))
    mock_atm_system.addForce(openmm.CMMotionRemover())

    assign_force_groups(mock_atm_system)

    forces = {}

    for force in mock_atm_system.getForces():
        forces[force.getForceGroup()] = type(force)

    assert forces == {
        0: openmm.HarmonicBondForce,
        1: openmm.HarmonicAngleForce,
        2: openmm.PeriodicTorsionForce,
        7: openmm.MonteCarloBarostat,
        8: openmm.ATMForce,
        16: openmm.CMMotionRemover,
    }


def test_evaluate_ctx_parameters():
    expression = "x*sqrt<y>"

    system = openmm.System()
    system.addParticle(1.0 * openmm.unit.amu)
    force_1 = openmm.NonbondedForce()
    force_1.addParticle(0.0, 1.0, 0.0)
    force_1.addGlobalParameter("x", 1.0)
    system.addForce(force_1)
    force_2 = openmm.CustomTorsionForce("y")
    force_2.addGlobalParameter("y", 1.0)
    system.addForce(force_2)
    force_3 = openmm.CustomExternalForce("0.0")
    force_3.addGlobalParameter(expression, 1.0)
    system.addForce(force_3)
    # add a force without global parameters to make sure it doesn't crash the method
    system.addForce(openmm.PeriodicTorsionForce())

    x, y = 4.0, 9.0

    output = evaluate_ctx_parameters({"x": x, "y": y}, system)

    expected_output = {"x": x, "y": y, expression: x * numpy.sqrt(y)}
    assert output == expected_output


def test_evaluate_ctx_parameters_cant_evaluate():
    system = openmm.System()
    system.addParticle(1.0 * openmm.unit.amu)
    force = openmm.NonbondedForce()
    force.addParticle(0.0, 1.0, 0.0)
    force.addGlobalParameter("x", 1.0)
    force.addGlobalParameter("x*y", 1.0)
    system.addForce(force)

    with pytest.raises(ValueError, match="could not evaluate context parameter"):
        evaluate_ctx_parameters({"x": 1.0}, system)


def test_get_simulation_summary(mocker):
    mock_simulation = mocker.MagicMock()

    mock_state = mock_simulation.context.getState.return_value

    expected_potential = 1.0 * openmm.unit.kilojoules_per_mole
    mock_state.getPotentialEnergy.return_value = expected_potential

    expected_length = 3.0 * openmm.unit.angstrom
    expected_volume = expected_length**3

    mock_state.getPeriodicBoxVectors.return_value = numpy.eye(3) * expected_length

    summary = get_simulation_summary(mock_simulation)

    expected_summary = f"energy={expected_potential} volume={expected_volume}"
    assert summary == expected_summary


def test_create_simulation():
    topology = build_mock_structure(["[Ar]"])
    topology.residues[0].name = femto.md.constants.LIGAND_1_RESIDUE_NAME
    topology.box = (numpy.eye(3) * 50.0) * openmm.unit.angstrom

    system = openmm.System()
    system.addParticle(1.0)
    system.setDefaultPeriodicBoxVectors(*topology.box)
    force = openmm.NonbondedForce()
    force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    force.addGlobalParameter(femto.fe.fep.LAMBDA_VDW_LIGAND_1, 1.0)
    force.addParticle(1.0, 1.0, 0.0)
    force.setForceGroup(femto.md.constants.OpenMMForceGroup.NONBONDED)
    system.addForce(force)

    integrator = openmm.VerletIntegrator(0.001)

    expected_lambda = 0.5

    simulation = create_simulation(
        system,
        topology,
        None,
        integrator,
        {femto.fe.fep.LAMBDA_VDW_LIGAND_1: expected_lambda},
        femto.md.constants.OpenMMPlatform.CPU,
    )
    assert isinstance(simulation, openmm.app.Simulation)

    positions = simulation.context.getState(getPositions=True).getPositions(
        asNumpy=True
    )
    assert femto.md.utils.openmm.all_close(positions, topology.xyz)

    expected_box_vectors = topology.box
    box_vectors = simulation.context.getState().getPeriodicBoxVectors(asNumpy=True)
    assert femto.md.utils.openmm.all_close(box_vectors, expected_box_vectors)

    assert simulation.context.getParameter(
        femto.fe.fep.LAMBDA_VDW_LIGAND_1
    ) == pytest.approx(expected_lambda)


@pytest.mark.parametrize(
    "pressure, freq, expected_pressure",
    [
        (1.0 * openmm.unit.atmosphere, 25, 1.0 * openmm.unit.atmosphere),
        (1.0 * openmm.unit.atmosphere, 0, None),
    ],
)
def test_get_pressure(pressure, freq, expected_pressure):
    system = openmm.System()
    system.addForce(openmm.MonteCarloBarostat(pressure, 298.15, freq))

    pressure = femto.md.utils.openmm.get_pressure(system)
    assert pressure == expected_pressure


def test_get_pressure_no_barostat():
    pressure = femto.md.utils.openmm.get_pressure(openmm.System())
    assert pressure is None
