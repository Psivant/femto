import os
import signal

import pytest

import femto.md.utils.mpi


def test_is_rank_zero_not_mpi(mocker):
    mocker.patch.dict(os.environ, {}, clear=True)
    assert femto.md.utils.mpi.is_rank_zero()


@pytest.mark.parametrize("rank, expected", [(0, True), (1, False)])
def test_is_rank_zero(rank, expected, mocker):
    import mpi4py.MPI

    mocker.patch.dict(os.environ, {"OMPI_COMM_WORLD_SIZE": "1"}, clear=True)
    mocker.patch.object(mpi4py.MPI, "COMM_WORLD", mocker.MagicMock(rank=rank))

    assert femto.md.utils.mpi.is_rank_zero() == expected


def test_get_mpi_comm_nested(mocker):
    """Only the top level ctx manager should set the signals"""
    spied_signal = mocker.spy(signal, "getsignal")

    with femto.md.utils.mpi.get_mpi_comm():
        assert spied_signal.call_count == 3  # int term abrt
        assert femto.md.utils.mpi._INSIDE_MPI_COMM is True

        with femto.md.utils.mpi.get_mpi_comm():
            assert spied_signal.call_count == 3

        assert femto.md.utils.mpi._INSIDE_MPI_COMM is True

    assert femto.md.utils.mpi._INSIDE_MPI_COMM is False


def test_get_mpi_comm_abort_on_error(mocker):
    mock_comm = mocker.patch("mpi4py.MPI.COMM_WORLD")
    mock_comm.size = 2

    sigint_handler = signal.getsignal(signal.SIGINT)

    with pytest.raises(RuntimeError, match="dummy-error"):
        with femto.md.utils.mpi.get_mpi_comm():
            assert femto.md.utils.mpi._INSIDE_MPI_COMM is True
            assert signal.getsignal(signal.SIGINT) != sigint_handler
            raise RuntimeError("dummy-error")

    assert signal.getsignal(signal.SIGINT) == sigint_handler
    assert femto.md.utils.mpi._INSIDE_MPI_COMM is False

    mock_comm.Abort.assert_called_once()


def test_get_mpi_comm_abort_on_signal(mocker):
    mock_comm = mocker.patch("mpi4py.MPI.COMM_WORLD")
    mock_comm.size = 2

    original_sigint_handler = signal.getsignal(signal.SIGINT)

    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        with femto.md.utils.mpi.get_mpi_comm():
            assert signal.getsignal(signal.SIGINT) != signal.SIG_IGN
            signal.raise_signal(signal.SIGINT)

        mock_comm.Abort.assert_called_once()
        assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)


@pytest.mark.parametrize("rank", [None, 0])
def test_reduce_dict(rank):
    value = {"a": 1.0, "b": 2.0, "c": 3.0}

    with femto.md.utils.mpi.get_mpi_comm() as mpi_comm:
        return_value = femto.md.utils.mpi.reduce_dict(value, mpi_comm, rank)

    assert value == return_value


def test_divide_tasks(mocker):
    world_size = 5
    n_total_tasks = 9  # two tasks per worker except one worker with one

    return_values = []

    for rank in range(world_size):
        mock_comm = mocker.MagicMock()
        mock_comm.size = world_size
        mock_comm.rank = rank

        return_values.append(femto.md.utils.mpi.divide_tasks(mock_comm, n_total_tasks))

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

    femto.md.utils.mpi.divide_gpus()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == str(expected_gpu_idx)


def test_run_on_rank_zero():
    @femto.md.utils.mpi.run_on_rank_zero
    def dummy_func(arg_a):
        return arg_a * 2

    return_value = dummy_func(2)
    assert return_value == 4
