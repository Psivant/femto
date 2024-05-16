"""Sample along the lambda schedule using HREMD."""

import copy
import functools
import logging
import pathlib
import typing

import numpy
import openmm
import parmed

import femto.md.constants
import femto.md.hremd
import femto.md.reporting
import femto.md.utils.openmm
import femto.fe.ddg

if typing.TYPE_CHECKING:
    import femto.fe.septop

_LOGGER = logging.getLogger(__name__)

_KJ_PER_MOL = openmm.unit.kilojoules_per_mole


def _analyze(
    cycle: int,
    u_kn: numpy.ndarray,
    n_k: numpy.ndarray,
    config: "femto.fe.septop.SepTopSamplingStage",
    reporter: femto.md.reporting.Reporter,
):
    """Estimate the current free energy and its error."""

    try:
        estimated, _ = femto.fe.ddg.estimate_ddg(u_kn, n_k, config.temperature)

        ddg = estimated["ddG_kcal_mol"]
        ddg_error = estimated["ddG_error_kcal_mol"]

        reporter.report_scalar("ddG_kcal_mol", cycle, ddg)
        reporter.report_scalar("ddG_error_kcal_mol", cycle, ddg_error)
    except:  # noqa: E722
        _LOGGER.warning(
            f"could not compute online estimates of the free energies at cycle={cycle}"
        )


def run_hremd(
    system: openmm.System,
    topology: parmed.Structure,
    coords: list[openmm.State],
    states: "femto.fe.septop.SepTopStates",
    config: "femto.fe.septop.SepTopSamplingStage",
    platform: femto.md.constants.OpenMMPlatform,
    output_dir: pathlib.Path,
    reporter: femto.md.reporting.Reporter | None = None,
):
    """Perform replica exchange sampling for a system prepared for SepTop calculations.

    Args:
        system: The system.
        topology: The topology associated with the system.
        coords: The starting coordinates for each state.
        states: The lambda states to sample.
        config: Configuration settings.
        platform: The platform to run on.
        output_dir: The directory to store the sampled energies and statistics to, and
            any trajectory files if requested.
        reporter: The reporter to log statistics such as online estimates of the
            free energy to.
    """
    import femto.fe.septop

    system = copy.deepcopy(system)

    n_barostats = sum(
        1
        for force in system.getForces()
        if isinstance(force, openmm.MonteCarloBarostat)
    )
    if n_barostats > 0:
        raise RuntimeError("the system should not contain a barostat already")

    if config.pressure is not None:
        barostat = openmm.MonteCarloBarostat(
            config.pressure, config.temperature, config.barostat_frequency
        )
        system.addForce(barostat)

    femto.md.utils.openmm.assign_force_groups(system)

    state_dicts = femto.fe.septop.create_state_dicts(states, system)

    integrator = femto.md.utils.openmm.create_integrator(
        config.integrator, config.temperature
    )
    simulation = femto.md.utils.openmm.create_simulation(
        system, topology, coords[0], integrator, state_dicts[0], platform
    )

    analysis_fn = None

    if reporter is not None and config.analysis_interval is not None:
        analysis_fn = functools.partial(_analyze, config=config, reporter=reporter)

    return femto.md.hremd.run_hremd(
        simulation,
        state_dicts,
        config,
        output_dir,
        initial_coords=coords,
        analysis_fn=analysis_fn,
        analysis_interval=config.analysis_interval,
    )
