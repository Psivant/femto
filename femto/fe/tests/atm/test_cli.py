import logging
import os
import typing

import click
import click.testing
import pytest
import yaml

from femto.fe.atm._cli import main as main_cli
from femto.fe.tests.systems import TEMOA_SYSTEM


@pytest.fixture
def click_runner() -> click.testing.CliRunner:
    runner = click.testing.CliRunner()
    yield runner


@pytest.fixture
def mock_cuda_devices(mocker):
    mocker.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})


def test_merge_config(click_runner, tmp_cwd):
    config = {"setup": {"hydrogen_mass": "1234.0 amu"}}

    config_path = tmp_cwd / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result: click.testing.Result = click_runner.invoke(
        typing.cast(click.Command, main_cli),
        ["--config", config_path, "run-workflow", "--help"],
    )
    assert result.exit_code == 0


def test_print_config(click_runner):
    result: click.testing.Result = click_runner.invoke(
        typing.cast(click.Command, main_cli), ["config"]
    )

    if result.exit_code != 0:
        raise result.exception

    assert "setup:" in result.output


def test_run_workflow_with_paths(click_runner, mock_cuda_devices, tmp_cwd, mocker):
    mock_run = mocker.patch("femto.fe.atm._runner.run_workflow", autospec=True)

    expected_output_dir = tmp_cwd / "outputs"
    expected_report_dir = tmp_cwd / "reports"

    result: click.testing.Result = click_runner.invoke(
        typing.cast(click.Command, main_cli),
        [
            "run-workflow",
            "--receptor-coords",
            TEMOA_SYSTEM.receptor_coords,
            "--receptor-params",
            TEMOA_SYSTEM.receptor_params,
            "--receptor-ref-atoms",
            TEMOA_SYSTEM.receptor_cavity_mask,
            "--ligand-1-coords",
            TEMOA_SYSTEM.ligand_1_coords,
            "--ligand-1-params",
            TEMOA_SYSTEM.ligand_1_params,
            "--ligand-1-ref-atoms",
            *TEMOA_SYSTEM.ligand_1_ref_atoms,
            "--ligand-2-coords",
            TEMOA_SYSTEM.ligand_2_coords,
            "--ligand-2-params",
            TEMOA_SYSTEM.ligand_2_params,
            "--ligand-2-ref-atoms",
            *TEMOA_SYSTEM.ligand_2_ref_atoms,
            "--output-dir",
            expected_output_dir,
            "--report-dir",
            expected_report_dir,
        ],
    )

    if result.exit_code != 0:
        raise result.exception

    mock_run.assert_called_once_with(
        mocker.ANY,
        TEMOA_SYSTEM.ligand_1_coords,
        TEMOA_SYSTEM.ligand_1_params,
        TEMOA_SYSTEM.ligand_2_coords,
        TEMOA_SYSTEM.ligand_2_params,
        TEMOA_SYSTEM.receptor_coords,
        TEMOA_SYSTEM.receptor_params,
        expected_output_dir,
        expected_report_dir,
        None,
        TEMOA_SYSTEM.ligand_1_ref_atoms,
        TEMOA_SYSTEM.ligand_2_ref_atoms,
        TEMOA_SYSTEM.receptor_cavity_mask,
    )


def test_run_workflow_missing_path(click_runner, mock_cuda_devices, tmp_cwd):
    mock_args = [
        "run-workflow",
        "--ligand-1-coords",
        TEMOA_SYSTEM.ligand_1_coords,
        "--ligand-1-params",
        TEMOA_SYSTEM.ligand_1_params,
        "--ligand-1-ref-atoms",
        *TEMOA_SYSTEM.ligand_1_ref_atoms,
        "--ligand-2-coords",
        TEMOA_SYSTEM.ligand_2_coords,
        "--ligand-2-params",
        TEMOA_SYSTEM.ligand_2_params,
        "--ligand-2-ref-atoms",
        *TEMOA_SYSTEM.ligand_2_ref_atoms,
        "--output-dir",
        ".",
    ]

    result: click.testing.Result = click_runner.invoke(
        typing.cast(click.Command, main_cli), mock_args
    )

    assert result.exit_code == 2
    assert "The receptor coordinates must be provided" in result.output


def test_run_workflow_with_directory(
    click_runner, mock_cuda_devices, tmp_cwd, mock_bfe_directory, mocker
):
    mock_run = mocker.patch("femto.fe.atm._runner.run_workflow", autospec=True)

    expected_output_dir = tmp_cwd / "outputs"

    result: click.testing.Result = click_runner.invoke(
        typing.cast(click.Command, main_cli),
        [
            "run-workflow",
            "--root-dir",
            mock_bfe_directory,
            "--ligand-1",
            "1h1q",
            "--ligand-2",
            "1oiu",
            "--output-dir",
            expected_output_dir,
        ],
    )

    if result.exit_code != 0:
        raise result.exception

    expected_ligand_1_coords = mock_bfe_directory / "forcefield/1h1q/vacuum.mol2"
    expected_ligand_1_params = mock_bfe_directory / "forcefield/1h1q/vacuum.parm7"

    expected_ligand_2_coords = mock_bfe_directory / "forcefield/1oiu/vacuum.mol2"
    expected_ligand_2_params = mock_bfe_directory / "forcefield/1oiu/vacuum.parm7"

    expected_receptor_coords = mock_bfe_directory / "proteins/cdk2/protein.pdb"
    expected_receptor_params = None

    expected_ligand_1_ref = None
    expected_ligand_2_ref = None
    expected_receptor_ref = None

    mock_run.assert_called_once_with(
        mocker.ANY,
        expected_ligand_1_coords,
        expected_ligand_1_params,
        expected_ligand_2_coords,
        expected_ligand_2_params,
        expected_receptor_coords,
        expected_receptor_params,
        expected_output_dir,
        None,
        None,
        expected_ligand_1_ref,
        expected_ligand_2_ref,
        expected_receptor_ref,
    )


def test_submit_workflows_cli(
    click_runner, tmp_cwd, mock_bfe_directory, mocker, caplog
):
    mock_submit = mocker.patch(
        "femto.fe.atm._runner.submit_network",
        autospec=True,
        return_value=["id-a", "id-b"],
    )
    mock_wait = mocker.patch("femto.fe.utils.queue.wait_for_slurm_jobs", autospec=True)

    with caplog.at_level(logging.INFO):
        result: click.testing.Result = click_runner.invoke(
            typing.cast(click.Command, main_cli),
            [
                "submit-workflows",
                "--root-dir",
                mock_bfe_directory,
                "--output-dir",
                tmp_cwd,
                "--slurm-nodes=1",
                "--slurm-tasks=2",
                "--slurm-cpus-per-task=3",
                "--slurm-gpus-per-task=4",
                "--slurm-walltime=1-00",
                "--slurm-partition=test",
                "--wait",
            ],
        )

    if result.exit_code != 0:
        raise result.exception

    assert "submitted 1h1q as job=id-a" in caplog.messages

    mock_submit.assert_called_once()
    mock_wait.assert_called_once()


def test_submit_replicas_cli(click_runner, tmp_cwd, mock_bfe_directory, mocker, caplog):
    mock_submit = mocker.patch(
        "femto.fe.atm._runner.submit_network",
        autospec=True,
        return_value=["id-a", "id-b"],
    )

    mock_wait = mocker.patch("femto.fe.utils.queue.wait_for_slurm_jobs", autospec=True)

    with caplog.at_level(logging.INFO):
        result: click.testing.Result = click_runner.invoke(
            typing.cast(click.Command, main_cli),
            [
                "submit-replicas",
                "--n-replicas=1",
                "--root-dir",
                mock_bfe_directory,
                "--output-dir",
                tmp_cwd,
                "--slurm-nodes=1",
                "--slurm-tasks=2",
                "--slurm-cpus-per-task=3",
                "--slurm-gpus-per-task=4",
                "--slurm-walltime=1-00",
                "--slurm-partition=test",
                "--wait",
            ],
        )

    if result.exit_code != 0:
        raise result.exception

    assert "submitted 1h1q replica=0 as job=id-a" in caplog.messages

    mock_submit.assert_called_once()
    mock_wait.assert_called_once()
