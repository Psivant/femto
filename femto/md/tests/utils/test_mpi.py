import os
import signal

import pytest

from femto.md.utils.mpi import (
    divide_gpus,
    divide_tasks,
    get_mpi_comm,
    is_rank_zero,
    reduce_dict,
    run_on_rank_zero,
)


def test_is_rank_zero_not_mpi(mocker):
    mocker.patch.dict(os.environ, {}, clear=True)
    assert is_rank_zero()


@pytest.mark.parametrize("rank, expected", [(0, True), (1, False)])
def test_is_rank_zero(rank, expected, mocker):
    import mpi4py.MPI

    mocker.patch.dict(os.environ, {"OMPI_COMM_WORLD_SIZE": "1"}, clear=True)
    mocker.patch.object(mpi4py.MPI, "COMM_WORLD", mocker.MagicMock(rank=rank))

    assert is_rank_zero() == expected


def test_get_mpi_comm_abort_on_error(mocker):
    mock_comm = mocker.patch("mpi4py.MPI.COMM_WORLD")
    mock_comm.size = 2

    sigint_handler = signal.getsignal(signal.SIGINT)

    with pytest.raises(RuntimeError, match="dummy-error"):
        with get_mpi_comm():
            assert signal.getsignal(signal.SIGINT) != sigint_handler
            raise RuntimeError("dummy-error")

    assert signal.getsignal(signal.SIGINT) == sigint_handler

    mock_comm.Abort.assert_called_once()


def test_get_mpi_comm_abort_on_signal(mocker):
    mock_comm = mocker.patch("mpi4py.MPI.COMM_WORLD")
    mock_comm.size = 2

    original_sigint_handler = signal.getsignal(signal.SIGINT)

    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        with get_mpi_comm():
            assert signal.getsignal(signal.SIGINT) != signal.SIG_IGN
            signal.raise_signal(signal.SIGINT)

        mock_comm.Abort.assert_called_once()
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)


def test_reduce_dict():
    value = {"a": 1.0, "b": 2.0, "c": 3.0}

    with get_mpi_comm() as mpi_comm:
        return_value = reduce_dict(value, mpi_comm)

    assert value == return_value


def test_divide_tasks(mocker):
    world_size = 5
    n_total_tasks = 9  # two tasks per worker except one worker with one

    return_values = []

    for rank in range(world_size):
        mock_comm = mocker.MagicMock()
        mock_comm.size = world_size
        mock_comm.rank = rank

        return_values.append(divide_tasks(mock_comm, n_total_tasks))

    # workers should receive two tasks each except the last worker,
    # i.e. (0, 1), (2, 3), (4, 5), (6, 7), (8,)
    assert return_values == [(2, 0), (2, 2), (2, 4), (2, 6), (1, 8)]


@pytest.mark.parametrize(
    "rank, expected_gpu_idx", [(0, 0), (1, 1), (2, 2), (3, 0), (4, 1)]
)
def test_divide_gpus(rank, expected_gpu_idx, mocker):
    world_size = 5

    mock_comm_ctx = mocker.MagicMock()
    mock_comm = mock_comm_ctx.__enter__.return_value
    mock_comm.size = world_size
    mock_comm.rank = rank

    mocker.patch(
        "femto.md.utils.mpi.get_mpi_comm",
        autospec=True,
        return_value=mock_comm_ctx,
    )
    mocker.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2"})

    divide_gpus()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == str(expected_gpu_idx)


def test_run_on_rank_zero():
    @run_on_rank_zero
    def dummy_func(arg_a):
        return arg_a * 2

    return_value = dummy_func(2)
    assert return_value == 4
