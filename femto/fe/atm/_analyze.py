"""Analyze the output of ATM calculations."""
import typing

import numpy

import femto.fe.ddg

if typing.TYPE_CHECKING:
    import pandas
    import femto.fe.atm


def compute_ddg(
    config: "femto.fe.atm.ATMSamplingStage",
    states: "femto.fe.atm.ATMStates",
    u_kn: numpy.ndarray,
    n_k: numpy.ndarray,
) -> "pandas.DataFrame":
    """Computes the total binding free energy.

    Args:
        config: The sampling configuration.
        states: The sampled states.
        u_kn: The samples.
        n_k: The sample counts.

    Returns:
        A pandas DataFrame containing the total binding free energy and its components.
    """
    import pandas

    n_states = len(states.lambda_1)
    n_states_leg_1 = states.direction.index(-1)

    state_groups = [(n_states_leg_1, 1.0), (n_states - n_states_leg_1, 1.0)]

    estimated, _ = femto.fe.ddg.estimate_ddg(
        u_kn, n_k, config.temperature, state_groups
    )
    return pandas.DataFrame([estimated])
