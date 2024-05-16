"""Estimate ddG values from sampled data."""

import logging
import pathlib

import numpy
import openmm.unit
import pyarrow

import femto.fe.uwham

_LOGGER = logging.getLogger(__name__)


def _uncorrelated_frames(length: int, g: float) -> list[int]:
    """Return the indices of frames that are un-correlated.

    Args:
        length: The total number of correlated frames.
        g: The statistical inefficiency of the data.

    Returns:
        The indices of un-correlated frames.
    """
    indices = []
    n = 0

    while int(round(n * g)) < length:
        t = int(round(n * g))
        if n == 0 or t != indices[n - 1]:
            indices.append(t)
        n += 1

    return indices


def load_u_kn(results_path: pathlib.Path) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Loads the reduced potentials from a replica exchange sampler output file and
    re-orders / de-correlates them into a form acceptable by MBAR.

    Note:
        Samples will be de-correlated using the ``pymbar.timeseries`` module.

    Args:
        results_path: The path to the arrow replica exchange output file.

    Returns:
        The loaded ``u_kn`` and ``n_k`` arrays.
    """
    import pymbar.timeseries

    with pyarrow.OSFile(str(results_path), "rb") as file:
        with pyarrow.RecordBatchStreamReader(file) as reader:
            output_table = reader.read_all()

    replica_to_state_idx = numpy.hstack(
        [numpy.array(x) for x in output_table["replica_to_state_idx"].to_pylist()]
    )

    # group the data along axis 1 so that data sampled in the same state is grouped.
    # this will let us more easily de-correlate the data.
    u_kn = numpy.hstack([numpy.array(x) for x in output_table["u_kn"].to_pylist()])
    u_kn_per_k = [u_kn[:, replica_to_state_idx == i] for i in range(len(u_kn))]

    n_uncorrelated = u_kn.shape[1] // u_kn.shape[0]

    g = pymbar.timeseries.statistical_inefficiency_multiple(
        [
            u_kn_per_k[i][i, i * n_uncorrelated : (i + 1) * n_uncorrelated]
            for i in range(len(u_kn))
        ]
    )
    uncorrelated_frames = _uncorrelated_frames(n_uncorrelated, g)

    for state_idx, state_u_kn in enumerate(u_kn_per_k):
        u_kn_per_k[state_idx] = state_u_kn[:, uncorrelated_frames]

    u_kn = numpy.hstack(u_kn_per_k)
    n_k = numpy.array([len(uncorrelated_frames)] * u_kn.shape[0])

    return u_kn, n_k


def _estimate_f_i(
    u_kn: numpy.ndarray, n_k: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Attempts to estimate free energies for each state using MBAR, and falling back
    to UWHAM if that fails.
    """
    import pymbar.timeseries

    try:
        mbar = pymbar.MBAR(u_kn, n_k)
        ddg_dict = mbar.compute_free_energy_differences()

        return ddg_dict["Delta_f"][0, :], ddg_dict["dDelta_f"][0, :] ** 2, mbar.W_nk
    except:  # noqa: E722
        _LOGGER.warning(
            "pymbar could not be used to estimate free energies, falling back to "
            "internal UWHAM implementation"
        )
        return femto.fe.uwham.estimate_f_i(u_kn, n_k)


def estimate_ddg(
    u_kn: numpy.ndarray,
    n_k: numpy.ndarray,
    temperature: openmm.unit.Quantity,
    state_groups: list[tuple[int, float]] | None = None,
) -> tuple[dict[str, float], dict[str, numpy.ndarray]]:
    """Estimate the free energy change for each group of states as well as the total
    free energy change.

    Args:
        u_kn: The reduced potentials.
        n_k: The number of samples per state.
        temperature: The temperature at which the samples were collected.
        state_groups: If certain states should be grouped together (e.g. leg 1 and
            leg 2 in ATM), this should be a list of tuples containing the number of
            states belonging to each group as well as the multiplier to use when
            summing the total free energy.

    Returns:
        The free energy estimates and the overlap matrices.
    """
    beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * temperature)

    state_idx = 0
    sample_idx = 0

    total_ddg = 0.0
    total_ddg_error = 0.0

    estimates = {}
    overlaps = {}

    state_groups = state_groups if state_groups is not None else [(len(n_k), 1.0)]

    for group_idx, (n_states, factor) in enumerate(state_groups):
        group_n_k = n_k[state_idx : state_idx + n_states]
        group_u_kn = u_kn[
            state_idx : state_idx + n_states, sample_idx : sample_idx + group_n_k.sum()
        ]

        f_i, f_i_variance, weights = _estimate_f_i(group_u_kn, group_n_k)
        overlaps[f"overlap_{group_idx}"] = group_n_k * (weights.T @ weights)

        ddg = (f_i[-1] - f_i[0]) / beta
        ddg = float(ddg.value_in_unit(openmm.unit.kilocalorie_per_mole))

        ddg_error = numpy.sqrt(f_i_variance[-1] + f_i_variance[0]) / beta
        ddg_error = float(ddg_error.value_in_unit(openmm.unit.kilocalorie_per_mole))

        total_ddg += factor * ddg
        total_ddg_error += ddg_error**2

        state_idx += n_states
        sample_idx += group_n_k.sum()

        estimates[f"ddG_{group_idx}_kcal_mol"] = ddg
        estimates[f"ddG_{group_idx}_error_kcal_mol"] = ddg_error

    total_ddg_error = numpy.sqrt(total_ddg_error)

    estimates["ddG_kcal_mol"] = total_ddg
    estimates["ddG_error_kcal_mol"] = total_ddg_error

    return estimates, overlaps
