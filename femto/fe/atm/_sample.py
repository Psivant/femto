"""Sample along the two ATM legs using HREMD."""

import copy
import functools
import logging
import pathlib
import typing

import mdtop
import numpy
import openmm

import femto.md.constants
import femto.md.hremd
import femto.md.reporting
import femto.md.utils.openmm

if typing.TYPE_CHECKING:
    import femto.fe.atm

_LOGGER = logging.getLogger(__name__)


def _analyze(
    cycle: int,
    u_kn: numpy.ndarray,
    n_k: numpy.ndarray,
    config: "femto.fe.atm.ATMSamplingStage",
    states: "femto.fe.atm.ATMStates",
    reporter: femto.md.reporting.Reporter,
):
    """Estimate the current free energy and its error."""

    try:
        ddg_data = femto.fe.atm.compute_ddg(config, states, u_kn, n_k)

        for column in ddg_data.columns:
            reporter.report_scalar(column, cycle, ddg_data[column].values[0])

        reporter.flush()
    except:  # noqa: E722
        _LOGGER.warning(
            f"could not compute online estimates of the free energies at cycle={cycle}"
        )


def run_hremd(
    system: openmm.System,
    topology: mdtop.Topology,
    coords: list[openmm.State],
    states: "femto.fe.atm.ATMStates",
    config: "femto.fe.atm.ATMSamplingStage",
    offset: openmm.unit.Quantity,
    platform: femto.md.constants.OpenMMPlatform,
    output_dir: pathlib.Path,
    reporter: femto.md.reporting.Reporter | None = None,
):
    """Perform replica exchange sampling for a system prepared for ATM calculations.

    Args:
        system: The system to simulate. It should *not* already contain an ATM force.
        topology: The topology associated with the system.
        coords: The starting coordinates for each state.
        states: The lambda states to sample.
        config: Configuration settings.
        offset: The vector to offset the ligand by using the ATM force.
        platform: The platform to run on.
        output_dir: The directory to store the sampled energies and statistics to, and
            any trajectory files if requested.
        reporter: The reporter to log statistics such as online estimates of the
            free energy to.
    """
    import femto.fe.atm._utils

    state_dicts = femto.fe.atm._utils.create_state_dicts(states)

    system = copy.deepcopy(system)
    femto.fe.atm._utils.add_atm_force(system, topology, config.soft_core, offset)

    integrator = femto.md.utils.openmm.create_integrator(
        config.integrator, config.temperature
    )
    simulation = femto.md.utils.openmm.create_simulation(
        system, topology, coords[0], integrator, state_dicts[0], platform
    )

    n_states = len(state_dicts)

    swap_mask = {
        (i, j)
        for i in range(n_states)
        for j in range(n_states)
        if states.direction[i] != states.direction[j]
    }

    analysis_fn = None

    if reporter is not None and config.analysis_interval is not None:
        analysis_fn = functools.partial(
            _analyze, config=config, states=states, reporter=reporter
        )

    return femto.md.hremd.run_hremd(
        simulation,
        state_dicts,
        config,
        output_dir,
        swap_mask,
        initial_coords=coords,
        analysis_fn=analysis_fn,
        analysis_interval=config.analysis_interval,
    )
