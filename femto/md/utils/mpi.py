"""Utilities for interacting with MPI."""

import contextlib
import functools
import logging
import os
import signal
import socket
import typing

import GPUtil
import numpy

_LOGGER = logging.getLogger(__name__)


if typing.TYPE_CHECKING:
    from mpi4py import MPI


_K = typing.TypeVar("_K", bound=str | int)
_T = typing.TypeVar("_T")


_REDUCE_DICT_OP = None
_INSIDE_MPI_COMM = False


def is_rank_zero() -> bool:
    """Returns true if the current MPI rank is zero, or if the application is not
    running using MPI."""

    mpi_env_vars = {"PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_SIZE"}

    if all(env_var not in os.environ for env_var in mpi_env_vars):
        return True

    from mpi4py import MPI

    return MPI.COMM_WORLD.rank == 0


@contextlib.contextmanager
def get_mpi_comm() -> typing.ContextManager["MPI.Intracomm"]:
    """A context manager that returns the main MPI communicator and installs signal
    handlers to abort MPI on exceptions.

    The signal handlers are restored to their defaults when the context manager exits.

    Returns:
        The global MPI communicator.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    global _INSIDE_MPI_COMM

    if _INSIDE_MPI_COMM:
        yield comm
        return

    _INSIDE_MPI_COMM = True

    original_signal_handlers = {
        signal_code: signal.getsignal(signal_code)
        for signal_code in [signal.SIGINT, signal.SIGTERM, signal.SIGABRT]
    }

    def abort_comm():
        if comm.size > 1:
            _LOGGER.warning("Aborting MPI")
            comm.Abort(1)

    def abort_comm_handler(signal_code, _):
        abort_comm()

        signal.signal(signal_code, original_signal_handlers[signal_code])
        signal.raise_signal(signal_code)

    try:
        for signal_code in original_signal_handlers:
            signal.signal(signal_code, abort_comm_handler)

        yield comm
    except BaseException as e:
        _LOGGER.exception(e)
        abort_comm()
        raise e
    finally:
        for signal_code in original_signal_handlers:
            signal.signal(signal_code, original_signal_handlers[signal_code])

        _INSIDE_MPI_COMM = False


def _reduce_dict_fn(dict_1: dict[str, float], dict_2: dict[str, float], _):
    """Sum the values of two dictionaries with the same keys."""
    for k, v in dict_2.items():
        if k not in dict_1:
            dict_1[k] = v
        else:
            dict_1[k] += v
    return dict_1


def reduce_dict(
    value: dict[_K, _T], mpi_comm: "MPI.Intracomm", root: int | None = None
) -> dict[_K, _T]:
    """Reduce a dictionary of values across MPI ranks.

    Args:
        value: The dictionary of values to reduce.
        mpi_comm: The MPI communicator to use for the reduction.
        root: The rank to which the reduced dictionary should be sent. If None, the
            reduced dictionary will be broadcast to all ranks.

    Returns:
        The reduced dictionary of values.
    """
    import mpi4py.MPI

    global _REDUCE_DICT_OP

    if _REDUCE_DICT_OP is None:
        _REDUCE_DICT_OP = mpi4py.MPI.Op.Create(_reduce_dict_fn, commute=True)

    if root is not None:
        return mpi_comm.reduce({**value}, op=_REDUCE_DICT_OP, root=root)
    else:
        return mpi_comm.allreduce({**value}, op=_REDUCE_DICT_OP)


def divide_tasks(mpi_comm: "MPI.Intracomm", n_tasks: int) -> tuple[int, int]:
    """Determine how many tasks the current MPI process should run given the total
    number that need to be distributed across all ranks.

    Args:
        mpi_comm: The main MPI communicator.
        n_tasks: The total number of tasks to run.

    Returns:
        The number of tasks to run on the current MPI process, and the index of the
        first task to be run by this worker.
    """
    n_workers = mpi_comm.size
    worker_idx = mpi_comm.rank

    n_each, n_extra = divmod(n_tasks, n_workers)

    replica_idx_offsets = numpy.array(
        [0] + n_extra * [n_each + 1] + (n_workers - n_extra) * [n_each]
    )
    replica_idx_offset = replica_idx_offsets.cumsum()[worker_idx]

    n_replicas = n_each + 1 if worker_idx < n_extra else n_each

    hostname = socket.gethostname()
    _LOGGER.debug(
        f"hostname={hostname} rank={mpi_comm.rank} will run {n_replicas} replicas"
    )

    return n_replicas, replica_idx_offset


def divide_gpus():
    """Attempts to divide the available GPUs across MPI ranks. If there are more ranks
    than GPUs, then each GPU will be assigned to multiple ranks.
    """
    import mpi4py.MPI

    hostname = socket.gethostname()

    with get_mpi_comm() as mpi_comm:
        n_cuda_devices = len(GPUtil.getGPUs())

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            n_cuda_devices = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

        if n_cuda_devices > 0:
            device_idx = mpi_comm.rank % n_cuda_devices
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{device_idx}"

            _LOGGER.debug(
                f"hostname={hostname} "
                f"rank={mpi4py.MPI.COMM_WORLD.rank} will use GPU={device_idx}"
            )

        else:
            _LOGGER.debug(f"hostname={hostname} has no GPUs")


def run_on_rank_zero(func):
    """A convenient decorator that ensures the function is only run on rank zero and
    that the outputs are broadcast to the other ranks.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        outputs = None
        rank = 0

        with get_mpi_comm() as mpi_comm:
            if mpi_comm.rank == rank:
                outputs = func(*args, **kwargs)

            outputs = mpi_comm.bcast(outputs, root=rank)

        return outputs

    return wrapper
