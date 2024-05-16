"""Equilibration stage of the SepTop free energy calculation."""

import logging
import typing

import openmm.unit
import parmed

import femto.md.constants
import femto.md.reporting
import femto.md.reporting.openmm
import femto.md.simulate

if typing.TYPE_CHECKING:
    import femto.fe.septop

_LOGGER = logging.getLogger(__name__)


def equilibrate_states(
    system: openmm.System,
    topology: parmed.Structure,
    states: "femto.fe.septop.SepTopStates",
    config: "femto.fe.septop.SepTopEquilibrateStage",
    platform: femto.md.constants.OpenMMPlatform,
    reporter: femto.md.reporting.Reporter | None = None,
) -> list[openmm.State]:
    """Equilibrate the system at each lambda window.

    Args:
        system: The system to simulate.
        topology: The topology of the system to simulate.
        states: The states of the system to simulate.
        config: Configuration settings.
        platform: The accelerator to use.
        reporter: The (optional) reporter to use to record system statistics such as
            volume and energy.

    Returns:
        The final equilibrated state.
    """
    import femto.fe.septop

    reporter = femto.md.reporting.NullReporter() if reporter is None else reporter

    openmm_reporter = femto.md.reporting.openmm.OpenMMStateReporter(
        reporter, "equilibration", config.report_interval
    )

    state_dicts = femto.fe.septop.create_state_dicts(states, system)

    equilibrated_coords = femto.md.simulate.simulate_states(
        system, topology, state_dicts, config.stages, platform, openmm_reporter
    )
    return equilibrated_coords
