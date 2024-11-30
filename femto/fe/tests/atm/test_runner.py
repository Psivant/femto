import functools

import numpy
import openmm
import openmm.unit
import pandas
import pytest

import femto.fe.inputs
import femto.fe.utils.queue
import femto.md.constants
import femto.md.reporting
import femto.top
from femto.fe.atm._runner import _prepare_system, run_workflow, submit_network
from femto.fe.tests.systems import TEMOA_SYSTEM
from femto.md.tests.mocking import build_mock_structure


def test_prepare_system(tmp_cwd, mocker):
    expected_output_dir = tmp_cwd / "outputs"

    mock_setup_system = mocker.patch(
        "femto.fe.atm._setup.setup_system",
        autospec=True,
        return_value=(build_mock_structure(["[Ar]"]), openmm.System()),
    )

    mock_config = femto.fe.atm.ATMSetupStage()

    prepare_system_fn = functools.partial(
        _prepare_system,
        mock_config,
        TEMOA_SYSTEM.ligand_1_coords,
        TEMOA_SYSTEM.ligand_1_params,
        TEMOA_SYSTEM.ligand_2_coords,
        TEMOA_SYSTEM.ligand_2_params,
        TEMOA_SYSTEM.receptor_coords,
        TEMOA_SYSTEM.receptor_params,
        numpy.array([22, 22, 22]) * openmm.unit.angstrom,
        TEMOA_SYSTEM.ligand_1_ref_atoms,
        TEMOA_SYSTEM.ligand_2_ref_atoms,
        TEMOA_SYSTEM.receptor_cavity_mask,
        expected_output_dir,
    )

    topology, system, _ = prepare_system_fn()

    assert isinstance(topology, femto.top.Topology)
    assert isinstance(system, openmm.System)

    # caching should be hit now
    topology, system, _ = prepare_system_fn()

    assert isinstance(topology, femto.top.Topology)
    assert isinstance(system, openmm.System)

    assert mock_setup_system.call_count == 1


def test_run_workflow(tmp_cwd, mocker):
    mock_config = femto.fe.atm.ATMConfig()

    expected_u_kn, expected_n_k = numpy.array([0.0]), numpy.array([1])
    expected_output_dir = tmp_cwd / "outputs"

    expected_offset = numpy.ones((1, 3)) * openmm.unit.angstrom * 22.0

    mock_topology, mock_system = build_mock_structure(["[Ar]"]), openmm.System()
    mock_coords = [openmm.State()] * len(mock_config.states.lambda_1)

    mock_prepare_system = mocker.patch(
        "femto.fe.atm._runner._prepare_system",
        autospec=True,
        return_value=(mock_topology, mock_system, expected_offset),
    )
    mock_equilibrate = mocker.patch(
        "femto.fe.atm._equilibrate.equilibrate_states",
        autospec=True,
        return_value=mock_coords,
    )
    mock_sample_system = mocker.patch(
        "femto.fe.atm._sample.run_hremd",
        autospec=True,
        return_value=mock_coords,
    )
    mock_load_u_kn = mocker.patch(
        "femto.fe.ddg.load_u_kn",
        autospec=True,
        return_value=(expected_u_kn, expected_n_k),
    )
    mock_compute_ddg = mocker.patch(
        "femto.fe.atm._analyze.compute_ddg",
        autospec=True,
        return_value=pandas.DataFrame(),
    )

    run_workflow_fn = functools.partial(
        run_workflow,
        mock_config,
        TEMOA_SYSTEM.ligand_1_coords,
        TEMOA_SYSTEM.ligand_1_params,
        TEMOA_SYSTEM.ligand_2_coords,
        TEMOA_SYSTEM.ligand_2_params,
        TEMOA_SYSTEM.receptor_coords,
        TEMOA_SYSTEM.receptor_params,
        expected_output_dir,
        expected_output_dir,
        expected_offset,
        TEMOA_SYSTEM.ligand_1_ref_atoms,
        TEMOA_SYSTEM.ligand_2_ref_atoms,
        TEMOA_SYSTEM.receptor_cavity_mask,
    )

    run_workflow_fn()

    mock_prepare_system.assert_called_once_with(
        mock_config.setup,
        TEMOA_SYSTEM.ligand_1_coords,
        TEMOA_SYSTEM.ligand_1_params,
        TEMOA_SYSTEM.ligand_2_coords,
        TEMOA_SYSTEM.ligand_2_params,
        TEMOA_SYSTEM.receptor_coords,
        TEMOA_SYSTEM.receptor_params,
        pytest.approx(expected_offset),
        TEMOA_SYSTEM.ligand_1_ref_atoms,
        TEMOA_SYSTEM.ligand_2_ref_atoms,
        TEMOA_SYSTEM.receptor_cavity_mask,
        expected_output_dir / "_setup",
    )
    mock_equilibrate.assert_called_once_with(
        mock_system,
        mock_topology,
        mock_config.states,
        mock_config.equilibrate,
        expected_offset,
        femto.md.constants.OpenMMPlatform.CUDA,
        mocker.ANY,
    )
    mock_sample_system.assert_called_once_with(
        mock_system,
        mock_topology,
        mock_coords,
        mock_config.states,
        mock_config.sample,
        expected_offset,
        femto.md.constants.OpenMMPlatform.CUDA,
        expected_output_dir / "_sample",
        mocker.ANY,
    )

    mock_load_u_kn.assert_called_once()
    mock_compute_ddg.assert_called_once()

    # caching
    run_workflow_fn()

    mock_equilibrate.assert_called_once()
    mock_sample_system.assert_called_once()
    mock_load_u_kn.assert_called_once()
    mock_compute_ddg.assert_called_once()


def test_submit_network(tmp_cwd, mocker):
    expected_job_id = "1234"

    mock_submit = mocker.patch(
        "femto.fe.utils.queue.submit_slurm_job",
        autospec=True,
        return_value=expected_job_id,
    )

    mock_config = femto.fe.atm.ATMConfig()

    mock_network = TEMOA_SYSTEM.rbfe_network
    mock_network.receptor.metadata["ref_atoms"] = TEMOA_SYSTEM.receptor_cavity_mask

    mock_queue_options = femto.fe.utils.queue.SLURMOptions(
        n_nodes=1,
        n_tasks=2,
        n_cpus_per_task=3,
        n_gpus_per_task=4,
        walltime="6-0",
        partition="partition1",
        job_name="job_name3",
    )

    expected_output_dir = tmp_cwd / "outputs"

    expected_mpi_cmd = ["mpirun", "-n"]

    job_ids = submit_network(
        mock_config,
        mock_network,
        expected_output_dir,
        mock_queue_options,
        expected_mpi_cmd,
    )
    assert job_ids == [expected_job_id]

    mock_submit.assert_called_once_with(
        [
            "mpirun",
            "-n",
            "femto",
            "atm",
            mocker.ANY,
            "run-workflow",
            f"--receptor-coords={TEMOA_SYSTEM.receptor_coords}",
            f"--receptor-params={TEMOA_SYSTEM.receptor_params}",
            "--receptor-ref-atoms",
            TEMOA_SYSTEM.receptor_cavity_mask,
            f"--ligand-1-coords={TEMOA_SYSTEM.ligand_1_coords}",
            f"--ligand-1-params={TEMOA_SYSTEM.ligand_1_params}",
            "--ligand-1-ref-atoms",
            *TEMOA_SYSTEM.ligand_1_ref_atoms,
            f"--ligand-2-coords={TEMOA_SYSTEM.ligand_2_coords}",
            f"--ligand-2-params={TEMOA_SYSTEM.ligand_2_params}",
            "--ligand-2-ref-atoms",
            *TEMOA_SYSTEM.ligand_2_ref_atoms,
            f"--output-dir={expected_output_dir}/g1~g4",
            f"--report-dir={expected_output_dir}/g1~g4",
        ],
        mock_queue_options,
        mocker.ANY,
    )
