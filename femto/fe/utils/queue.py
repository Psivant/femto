"""Utilities for interacting with HPC queues"""
import functools
import logging
import pathlib
import shlex
import signal
import subprocess
import sys
import time
import typing

import pydantic

import femto.md.utils.models

_LOGGER = logging.getLogger(__name__)


class SLURMOptions(femto.md.utils.models.BaseModel):
    """Options for a Slurm job submission."""

    n_nodes: int = pydantic.Field(..., description="The number of nodes to request.")
    n_tasks: int = pydantic.Field(..., description="The number of tasks to request.")
    n_cpus_per_task: int = pydantic.Field(1, description="The cpus per task.")
    n_gpus_per_task: int = pydantic.Field(
        1, description="The number of gpus to request per task."
    )

    walltime: str = pydantic.Field("5-0", description="The maximum walltime.")

    partition: str = pydantic.Field(..., description="The partition to run on.")
    account: str | None = pydantic.Field(None, description="The account to use.")

    job_name: str = pydantic.Field("femto", description="The job name to use.")

    reservation: str | None = pydantic.Field(
        None, description="The reservation to use."
    )

    def to_flags(self) -> list[str]:
        """Convert this options to a list of sbatch / srun flags."""

        return [
            f"--nodes={self.n_nodes}",
            f"--ntasks={self.n_tasks}",
            f"--gpus-per-task={self.n_gpus_per_task}",
            f"--cpus-per-task={self.n_cpus_per_task}",
            f"--partition={self.partition}",
            f"--time={self.walltime}",
            f"--job-name={self.job_name}",
            *([f"--account={self.account}"] if self.account is not None else []),
            *(
                [f"--reservation={self.reservation}"]
                if self.reservation is not None
                else []
            ),
        ]


def submit_slurm_job(
    command: list[str],
    options: SLURMOptions,
    log_file: pathlib.Path,
    dependencies: list[str] | None = None,
) -> str:
    """Submit a set of SLURM jobs to the queue

    Args:
        command: The command to run.
        options: The SLURM options to use.
        log_file: The file to write the SLURM output to.
        dependencies: A list of SLURM job IDs to wait for before running this job.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)

    slurm_args = [f"--output={log_file}", *options.to_flags(), "--parsable"]

    if dependencies is not None and len(dependencies) > 0:
        slurm_args.append("--dependency=afterok:" + ",".join(dependencies))
        slurm_args.append("--kill-on-invalid-dep=yes")

    command_str = shlex.join(
        str(arg) if isinstance(arg, pathlib.Path) else arg for arg in command
    )
    run_args = ["sbatch", *slurm_args, "--wrap", command_str]

    result = subprocess.run(run_args, capture_output=True, text=True, check=True)
    result.check_returncode()

    job_id = result.stdout.strip().strip("\n")

    return job_id


def cancel_slurm_jobs(job_ids: typing.Iterable[str]):
    """Cancel a set of SLURM jobs

    Args:
        job_ids: The IDs of the jobs to cancel.
    """

    subprocess.run(["scancel", *job_ids])


def _cancel_slurm_job_and_exit(signal_code, _, job_ids: typing.Iterable[str]):
    """Attempts to cancel running SLURM jobs by calling ``scancel`` and then exits the
    program.

    Args:
        signal_code: The signal code that caused this handler to be called.
        job_ids: The IDs of the SLURM jobs to cancel.
    """
    cancel_slurm_jobs(job_ids)
    sys.exit(signal_code)


def wait_for_slurm_jobs(job_ids: typing.Iterable[str]):
    """Wait for a set of SLURM jobs to finish, or attempt to cancel them if the
    program fails before they do."""

    original_signal_handlers = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }

    cleanup_func = functools.partial(_cancel_slurm_job_and_exit, job_ids=job_ids)

    for signal_code in original_signal_handlers:
        signal.signal(signal_code, cleanup_func)

    try:
        remaining_job_ids = {*job_ids}

        while True:
            result = subprocess.run(
                ["squeue", "--job", ",".join(remaining_job_ids)],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                _LOGGER.warning(result.stdout)
                _LOGGER.warning(result.stderr)
                signal.raise_signal(signal.SIGTERM)

            finished_jobs = {
                job_id for job_id in remaining_job_ids if job_id not in result.stdout
            }
            remaining_job_ids -= finished_jobs

            if len(remaining_job_ids) == 0:
                break

            time.sleep(5.0)

    finally:
        for signal_code in original_signal_handlers:
            signal.signal(signal_code, original_signal_handlers[signal_code])
