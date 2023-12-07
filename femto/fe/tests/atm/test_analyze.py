import numpy
from pymbar.testsystems import harmonic_oscillators

import femto.fe.atm
from femto.fe.atm._analyze import compute_ddg


def test_compute_ddg():
    n_states = 22

    testcase = harmonic_oscillators.HarmonicOscillatorsTestCase(
        [0.0] * n_states, [1.0] * n_states
    )
    _, u_kn, n_k, _ = testcase.sample([10] * n_states, mode="u_kn")

    config = femto.fe.atm.ATMConfig()
    ddg = compute_ddg(config.sample, config.states, u_kn, n_k)

    assert len(ddg) == 1
    assert ddg.columns.tolist() == [
        "ddG_0_kcal_mol",
        "ddG_0_error_kcal_mol",
        "ddG_1_kcal_mol",
        "ddG_1_error_kcal_mol",
        "ddG_kcal_mol",
        "ddG_error_kcal_mol",
    ]

    assert numpy.isclose(ddg["ddG_0_kcal_mol"].values, 0.0, atol=0.1)
