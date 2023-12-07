import numpy
import openmm
import openmm.app
import openmm.unit
import pytest

from femto.md.anneal import anneal_state, anneal_temperature


@pytest.fixture
def mock_system() -> openmm.System:
    force = openmm.NonbondedForce()
    force.addGlobalParameter("lambda_x", 1.0)
    force.addParticle(1.0, 1.0, 0.0)
    system = openmm.System()
    system.addParticle(1.0)
    system.addForce(force)

    return system


def test_anneal_temperature(mock_system, mocker):
    mock_step = mocker.patch("openmm.app.Simulation.step", autospec=True)
    mock_set_temperature = mocker.patch(
        "openmm.LangevinIntegrator.setTemperature", autospec=True
    )

    simulation = openmm.app.Simulation(
        None, mock_system, openmm.openmm.LangevinIntegrator(298, 1, 0.1)
    )

    expected_n_steps = 10

    anneal_temperature(
        simulation,
        temperature_initial=100.0 * openmm.unit.kelvin,
        temperature_final=110.5 * openmm.unit.kelvin,
        n_steps=expected_n_steps,
        frequency=1,
    )

    # + 1 for start temperature
    expected_temperatures = numpy.linspace(100.0, 110.5, expected_n_steps + 1).tolist()
    # we make sure to set final temperature exactly
    expected_temperatures.append(110.5)

    assert mock_step.call_count == expected_n_steps

    temperatures = [call.args[-1] for call in mock_set_temperature.call_args_list]
    assert temperatures == pytest.approx(expected_temperatures)


def test_anneal_lambda(mock_system, mocker):
    mock_step = mocker.patch("openmm.app.Simulation.step", autospec=True)

    simulation = openmm.app.Simulation(None, mock_system, openmm.VerletIntegrator(0.01))
    set_parameter_func = mocker.spy(simulation.context, "setParameter")

    expected_n_steps = 10

    anneal_state(simulation, {"lambda_x": 0.0}, {"lambda_x": 0.5}, expected_n_steps, 1)

    expected_lambdas = numpy.linspace(0.0, 0.5, expected_n_steps + 1).tolist()
    # final lambda is set exactly even if increment would reach it
    expected_lambdas.append(0.5)

    assert mock_step.call_count == expected_n_steps

    lambdas = [dict([call.args]) for call in set_parameter_func.call_args_list]
    lambdas = [v["lambda_x"] for v in lambdas]

    assert lambdas == pytest.approx(expected_lambdas)
