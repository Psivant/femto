"""Equilibration stage of the ATM free energy calculation."""
import copy
import logging
import typing

import numpy
import openmm.unit
import parmed

import femto.md.constants
import femto.md.reporting
import femto.md.reporting.openmm
import femto.md.simulate

if typing.TYPE_CHECKING:
    import femto.fe.atm


_LOGGER = logging.getLogger(__name__)


def equilibrate_states(
    system: openmm.System,
    topology: parmed.Structure,
    states: "femto.fe.atm.ATMStates",
    config: "femto.fe.atm.ATMEquilibrateStage",
    offset: openmm.unit.Quantity,
    platform: femto.md.constants.OpenMMPlatform,
    reporter: femto.md.reporting.Reporter | None = None,
) -> list[openmm.State]:
    """Equilibrate the system at each lambda window.

    Args:
        system: The system to simulate.
        topology: The topology of the system to simulate.
        states: The states of the system to simulate.
        config: Configuration settings.
        offset: The vector to offset the ligand by using the ATM force.
        platform: The accelerator to use.
        reporter: The (optional) reporter to use to record system statistics such as
            volume and energy.

    Returns:
        The final equilibrated state.
    """
    import femto.fe.atm._utils

    reporter = femto.md.reporting.NullReporter() if reporter is None else reporter

    openmm_reporter = femto.md.reporting.openmm.OpenMMStateReporter(
        reporter, "equilibration", config.report_interval
    )

    state_dicts = femto.fe.atm._utils.create_state_dicts(states)

    system = copy.deepcopy(system)
    femto.fe.atm._utils.add_atm_force(system, topology, config.soft_core, offset)

    equilibrated_coords = femto.md.simulate.simulate_states(
        system, topology, state_dicts, config.stages, platform, openmm_reporter
    )

    box_vectors = [
        coords.getPeriodicBoxVectors(asNumpy=True).value_in_unit(openmm.unit.angstrom)
        for coords in equilibrated_coords
    ]
    box_vectors_max = numpy.max(numpy.stack(box_vectors), axis=0) * openmm.unit.angstrom

    for i, coords in enumerate(equilibrated_coords):
        context = openmm.Context(system, openmm.VerletIntegrator(0.00001))
        context.setState(coords)
        context.setPeriodicBoxVectors(*box_vectors_max)

        equilibrated_coords[i] = context.getState(getPositions=True)

    return equilibrated_coords
