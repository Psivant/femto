import numpy
import openmm.unit
import pyarrow
import pymbar
import pytest
from pymbar.testsystems import harmonic_oscillators

import femto.fe.uwham
from femto.fe.ddg import _uncorrelated_frames, estimate_ddg, load_u_kn


@pytest.fixture
def mock_samples_path(tmp_path):
    n_states = 2

    u_kn = numpy.array(
        [
            [[0.0, 1.0], [8.0, 9.0]],
            [[2.0, 3.0], [10.0, 11.0]],
            [[4.0, 5.0], [12.0, 13.0]],
            [[6.0, 7.0], [14.0, 15.0]],
        ]
    )
    replica_to_state_idx = numpy.array([[0, 1], [1, 0], [1, 0], [0, 1]])

    arrow_list = pyarrow.list_

    schema = pyarrow.schema(
        [
            ("u_kn", arrow_list(arrow_list(pyarrow.float64(), n_states), n_states)),
            ("replica_to_state_idx", arrow_list(pyarrow.int16(), n_states)),
        ]
    )

    path = tmp_path / "samples.arrow"

    with pyarrow.OSFile(str(path), "wb") as file:
        with pyarrow.RecordBatchStreamWriter(file, schema) as writer:
            table = pyarrow.table(
                [u_kn.tolist(), replica_to_state_idx.tolist()], schema=schema
            )
            writer.write_table(table)

            yield path


@pytest.mark.parametrize(
    "g, expected_idxs", [(1.0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), (2.0, [0, 2, 4, 6, 8])]
)
def test_uncorrelated_frames(g, expected_idxs):
    assert _uncorrelated_frames(10, g) == expected_idxs


def test_load_u_kn(mock_samples_path, mocker):
    mocker.patch(
        "pymbar.timeseries.statistical_inefficiency_multiple",
        autospec=True,
        # every other frame should be skipped
        return_value=2.0,
    )

    u_kn, n_k = load_u_kn(mock_samples_path)

    expected_u_kn = numpy.array([[0.0, 5.0, 1.0, 4.0], [8.0, 13.0, 9.0, 12.0]])
    expected_n_k = numpy.array([2, 2])

    assert u_kn.shape == expected_u_kn.shape
    assert numpy.allclose(u_kn, expected_u_kn)

    assert n_k.shape == expected_n_k.shape
    assert numpy.allclose(n_k, expected_n_k)


@pytest.mark.parametrize("force_mbar_failure", [False, True])
def test_estimate_ddg(force_mbar_failure, mocker):
    n_samples = 20
    n_states = 2

    spied_uwham_fn = mocker.spy(femto.fe.uwham, "estimate_f_i")

    testcase = harmonic_oscillators.HarmonicOscillatorsTestCase(
        O_k=list(range(n_states)), K_k=list(range(1, n_states + 1))
    )

    temperature = (
        1.0 / openmm.unit.MOLAR_GAS_CONSTANT_R * openmm.unit.kilojoules_per_mole
    )
    beta = 1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * temperature)

    _, u_kn_1, n_k_1, _ = testcase.sample(N_k=[n_samples] * n_states, mode="u_kn")
    f_i_1 = pymbar.MBAR(u_kn_1, n_k_1).f_k

    _, u_kn_2, n_k_2, _ = testcase.sample(N_k=[n_samples] * n_states, mode="u_kn")
    f_i_2 = pymbar.MBAR(u_kn_2, n_k_2).f_k

    filler = numpy.ones((n_states, n_samples * n_states)) * numpy.nan

    u_kn = numpy.asarray(numpy.bmat([[u_kn_1, filler], [filler, u_kn_2]]))

    if force_mbar_failure:

        def mock_fail(*_, **__):
            raise pymbar.utils.ConvergenceError("test error")

        mocker.patch("pymbar.MBAR", autospec=True, side_effect=mock_fail)

    n_state_groups = 2

    results, overlaps = estimate_ddg(
        u_kn,
        numpy.array([n_samples] * n_states * n_state_groups),
        temperature,
        [(2, -1.0), (2, 1.0)],
    )

    expected_uwham_calls = n_state_groups if force_mbar_failure else 0
    assert spied_uwham_fn.call_count == expected_uwham_calls

    expected_ddg_1 = ((f_i_1[-1] - f_i_1[0]) / beta).value_in_unit(
        openmm.unit.kilocalories_per_mole
    )
    expected_ddg_2 = ((f_i_2[-1] - f_i_2[0]) / beta).value_in_unit(
        openmm.unit.kilocalories_per_mole
    )

    expected_ddg = -expected_ddg_1 + expected_ddg_2

    assert numpy.isclose(results["ddG_0_kcal_mol"], expected_ddg_1, atol=0.005)
    assert numpy.isclose(results["ddG_1_kcal_mol"], expected_ddg_2, atol=0.005)

    assert numpy.isclose(results["ddG_kcal_mol"], expected_ddg, atol=0.005)

    assert overlaps["overlap_0"].shape == (2, 2)
    assert (overlaps["overlap_0"] > 0.1).all()

    assert overlaps["overlap_1"].shape == (2, 2)
    assert (overlaps["overlap_1"] > 0.1).all()
