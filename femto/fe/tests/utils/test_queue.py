import pytest

from femto.fe.utils.queue import (
    SLURMOptions,
    _cancel_slurm_job_and_exit,
    cancel_slurm_jobs,
    submit_slurm_job,
    wait_for_slurm_jobs,
)


def test_slurm_options_to_flags():
    options = SLURMOptions(
        n_nodes=1,
        n_tasks=2,
        n_cpus_per_task=3,
        n_gpus_per_task=4,
        walltime="6-0",
        partition="partition1",
        account="account2",
        job_name="job_name3",
    )

    flags = options.to_flags()

    expected_flags = [
        "--nodes=1",
        "--ntasks=2",
        "--gpus-per-task=4",
        "--cpus-per-task=3",
        "--partition=partition1",
        "--time=6-0",
        "--job-name=job_name3",
        "--account=account2",
    ]

    assert flags == expected_flags


@pytest.fixture
def mock_slurm_options() -> SLURMOptions:
    return SLURMOptions(
        n_nodes=1,
        n_tasks=2,
        n_cpus_per_task=3,
        n_gpus_per_task=4,
        walltime="6-0",
        partition="partition1",
        account="account2",
        job_name="job_name3",
    )


def test_submit_slurm_jobs(tmp_cwd, mock_slurm_options, mocker):
    expected_job_id = "id123"
    expected_command = ["my-command", "-a"]
    expected_log_file = tmp_cwd / "logs"
    expected_dependencies = ["id1", "id2"]

    mock_run_result = mocker.MagicMock(stdout=f"{expected_job_id}\n")
    mock_run = mocker.patch(
        "subprocess.run", autospec=True, return_value=mock_run_result
    )

    job_id = submit_slurm_job(
        expected_command, mock_slurm_options, expected_log_file, expected_dependencies
    )
    assert job_id == expected_job_id

    mock_run.assert_called_once_with(
        [
            "sbatch",
            f"--output={expected_log_file}",
            "--nodes=1",
            "--ntasks=2",
            "--gpus-per-task=4",
            "--cpus-per-task=3",
            "--partition=partition1",
            "--time=6-0",
            "--job-name=job_name3",
            "--account=account2",
            "--parsable",
            "--dependency=afterok:id1,id2",
            "--kill-on-invalid-dep=yes",
            "--wrap",
            " ".join(expected_command),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    mock_run.return_value.check_returncode.assert_called_once()
    assert expected_log_file.parent.exists()


def test_cancel_slurm_jobs(mocker):
    mock_run = mocker.patch("subprocess.run")

    expected_job_ids = {"a", "b", "c"}
    cancel_slurm_jobs(expected_job_ids)

    mock_run.assert_called_once_with(["scancel", *expected_job_ids])


def test_cancel_slurm_job_and_exit(mocker):
    mock_run = mocker.patch("subprocess.run")
    mock_exit = mocker.patch("sys.exit")

    expected_job_ids = {"a", "b", "c"}
    _cancel_slurm_job_and_exit(123, None, expected_job_ids)

    mock_run.assert_called_once_with(["scancel", *expected_job_ids])
    mock_exit.assert_called_once_with(123)


def test_wait_for_slurm_jobs(mocker):
    mock_exit = mocker.patch("sys.exit")
    mocker.patch("time.sleep")  # don't need the test to actually sleep

    mock_run = mocker.patch(
        "subprocess.run",
        side_effect=[
            mocker.Mock(returncode=0, stdout="a\nc"),
            mocker.Mock(returncode=0, stdout=""),
        ],
    )

    wait_for_slurm_jobs(["a", "b", "c"])

    mock_exit.assert_not_called()
    assert mock_run.call_count == 2
