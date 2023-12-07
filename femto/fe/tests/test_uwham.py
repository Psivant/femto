import numpy
import pymbar
import pytest
from pymbar.testsystems import harmonic_oscillators

from femto.fe.uwham import estimate_f_i


def test_estimate_f_i():
    testcase = harmonic_oscillators.HarmonicOscillatorsTestCase()
    _, u_kn, n_k, _ = testcase.sample(N_k=[500, 505, 510, 515, 520], mode="u_kn")

    mbar = pymbar.MBAR(u_kn, n_k)
    mbar_f_i = mbar.f_k
    mbar_weights = mbar.W_nk

    f_i, df_i, weights = estimate_f_i(u_kn, n_k)

    assert f_i.shape == mbar_f_i.shape
    assert numpy.allclose(mbar_f_i, f_i, atol=1.0e-3)
    assert numpy.allclose(mbar_weights, weights, atol=1.0e-3)


@pytest.mark.parametrize(
    "u_kn, n_k, expected_raises",
    [
        (
            numpy.zeros((3, 2)),
            numpy.zeros((2,)),
            pytest.raises(RuntimeError, match="The number of states do not match"),
        ),
        (
            numpy.zeros((2, 2)),
            numpy.zeros((2,)),
            pytest.raises(RuntimeError, match="The number of samples do not match"),
        ),
    ],
)
def test_estimate_f_i_shape_missmatch(u_kn, n_k, expected_raises):
    with expected_raises:
        estimate_f_i(u_kn, n_k)


def test_estimate_f_i_minimize_fails(mocker):
    mock_minimize = mocker.patch("scipy.optimize.minimize", autospec=True)
    mock_minimize.return_value.success = False

    with pytest.raises(
        RuntimeError, match="The UWHAM minimization failed to converge."
    ):
        estimate_f_i(numpy.zeros((1, 1)), numpy.ones(1))


def test_estimate_f_i_weight_sum(mocker):
    mock_minimize = mocker.patch("scipy.optimize.minimize", autospec=True)
    mock_minimize.return_value.success = True

    mocker.patch(
        "femto.fe.uwham._compute_weights",
        autospec=True,
        return_value=numpy.zeros((1, 1)),
    )

    with pytest.raises(RuntimeError, match="The UWHAM weights do not sum to 1.0"):
        estimate_f_i(numpy.zeros((1, 1)), numpy.ones(1))
