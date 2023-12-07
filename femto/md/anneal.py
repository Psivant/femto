"""Perform annealing of the temperature and/or global context parameters."""
import openmm.app
import openmm.unit

import femto.md.utils.openmm


def _set_temperature(integrator: openmm.Integrator, temperature: openmm.unit.Quantity):
    """Sets the temperature of an OpenMM integrator.

    Args:
        integrator: The integrator to set the temperature of.
        temperature: The temperature to set the integrator to.
    """

    if hasattr(integrator, "setTemperature"):
        integrator.setTemperature(temperature.value_in_unit(openmm.unit.kelvin))
    else:
        integrator.setGlobalVariableByName(
            "kT", openmm.unit.MOLAR_GAS_CONSTANT_R * temperature
        )


def anneal_temperature(
    simulation: openmm.app.Simulation,
    temperature_initial: openmm.unit.Quantity,
    temperature_final: openmm.unit.Quantity,
    n_steps: int,
    frequency: int,
):
    """Gradually ramp the system temperature from a starting value to the final value.

    Args:
        simulation: The current simulation.
        temperature_initial: The initial temperature.
        temperature_final: The final temperature.
        n_steps: The number of steps to anneal over.
        frequency: The frequency at which to increment the temperature.
    """

    n_increments = n_steps // frequency
    increment = (temperature_final - temperature_initial) / n_increments

    temperature = temperature_initial
    _set_temperature(simulation.integrator, temperature)

    for _ in range(n_increments):
        simulation.step(frequency)

        temperature += increment
        _set_temperature(simulation.integrator, temperature)

    _set_temperature(simulation.integrator, temperature_final)


def anneal_state(
    simulation: openmm.app.Simulation,
    state_initial: dict[str, float],
    state_final: dict[str, float],
    n_steps: int,
    frequency: int,
):
    """Gradually anneal from an initial state (i.e. set of global context parameters)
    to a final one.

    Args:
        simulation: The current simulation.
        state_initial: The initial state.
        state_final: The final state.
        n_steps: The number of steps to anneal over.
        frequency: The frequency at which to update the state.
    """

    n_increments = n_steps // frequency

    state_initial = femto.md.utils.openmm.evaluate_ctx_parameters(
        state_initial, simulation.system
    )
    state_final = femto.md.utils.openmm.evaluate_ctx_parameters(
        state_final, simulation.system
    )

    state = {**state_initial}

    increments = {
        key: (state_final[key] - state_initial[key]) / n_increments
        for key in state_initial
    }

    for key in state:
        simulation.context.setParameter(key, state[key])

    for _ in range(n_increments):
        simulation.step(frequency)

        for key in state:
            state[key] += increments[key]
            simulation.context.setParameter(key, state[key])

    for key in state:
        simulation.context.setParameter(key, state_final[key])
