import openmm

import femto.fe.inputs
import femto.fe.septop
import femto.fe.utils.queue
import femto.top
from femto.fe.septop._runner import (
    _prepare_complex_phase,
    _prepare_solution_phase,
    run_complex_phase,
    run_solution_phase,
    submit_network,
)
from femto.fe.tests.systems import CDK2_SYSTEM
from femto.md.tests.mocking import build_mock_structure


def test_prepare_solution_phase(mock_bfe_directory, mocker):
    mock_setup = mocker.patch(
        "femto.fe.septop._setup.setup_solution",
        autospec=True,
        return_value=(femto.top.Topology(), openmm.System()),
    )

    ligand_1_coords = mock_bfe_directory / "forcefield/1h1q/vacuum.mol2"
    ligand_1_params = mock_bfe_directory / "forcefield/1h1q/vacuum.parm7"

    ligand_2_coords = mock_bfe_directory / "forcefield/1oiu/vacuum.mol2"
    ligand_2_params = mock_bfe_directory / "forcefield/1oiu/vacuum.parm7"

    ligand_1_ref_atoms = ("@1", "@2", "@3")
    ligand_2_ref_atoms = ("@4", "@5", "@6")

    config = femto.fe.septop.SepTopConfig().solution

    topology, system = _prepare_solution_phase(
        config,
        ligand_1_coords,
        ligand_1_params,
        ligand_2_coords,
        ligand_2_params,
        ligand_1_ref_atoms,
        ligand_2_ref_atoms,
    )

    assert isinstance(system, openmm.System)
    assert isinstance(topology, femto.top.Topology)

    mock_setup.assert_called_once_with(
        config.setup, mocker.ANY, mocker.ANY, ligand_1_ref_atoms, ligand_2_ref_atoms
    )


def test_prepare_complex_phase(mock_bfe_directory, mocker):
    mock_setup = mocker.patch(
        "femto.fe.septop.setup_complex",
        autospec=True,
        return_value=(femto.top.Topology(), openmm.System()),
    )
    mock_parameterize = mocker.patch(
        "femto.md.utils.amber.parameterize_structure", autospec=True
    )

    receptor_coords = mock_bfe_directory / "proteins/cdk2/protein.pdb"
    receptor_params = None

    ligand_1_coords = mock_bfe_directory / "forcefield/1h1q/vacuum.mol2"
    ligand_1_params = mock_bfe_directory / "forcefield/1h1q/vacuum.parm7"

    ligand_2_coords = mock_bfe_directory / "forcefield/1oiu/vacuum.mol2"
    ligand_2_params = mock_bfe_directory / "forcefield/1oiu/vacuum.parm7"

    ligand_1_ref_atoms = ("@1", "@2", "@3")
    ligand_2_ref_atoms = ("@4", "@5", "@6")
    receptor_ref_atoms = ("@7", "@8", "@9")

    config = femto.fe.septop.SepTopConfig().complex

    topology, system = _prepare_complex_phase(
        config,
        ligand_1_coords,
        ligand_1_params,
        ligand_2_coords,
        ligand_2_params,
        receptor_coords,
        receptor_params,
        ligand_1_ref_atoms,
        ligand_2_ref_atoms,
        receptor_ref_atoms,
    )

    assert isinstance(system, openmm.System)
    assert isinstance(topology, femto.top.Topology)

    mock_setup.assert_called_once_with(
        config.setup,
        mocker.ANY,
        mocker.ANY,
        mocker.ANY,
        receptor_ref_atoms,
        ligand_1_ref_atoms,
        ligand_2_ref_atoms,
    )
    mock_parameterize.assert_called_once()


def test_run_solution_phase(tmp_cwd, mock_bfe_directory, mocker):
    mock_topology = build_mock_structure(["O"])

    config = femto.fe.septop.SepTopConfig()

    mock_setup = mocker.patch(
        "femto.fe.septop._runner._prepare_solution_phase",
        autospec=True,
        return_value=(mock_topology, openmm.System()),
    )
    mock_equilibrate = mocker.patch(
        "femto.fe.septop.equilibrate_states",
        autospec=True,
        return_value=[openmm.State()] * len(config.solution.states.lambda_vdw_ligand_1),
    )
    mock_sample = mocker.patch("femto.fe.septop.run_hremd", autospec=True)

    ligand_1_coords = mock_bfe_directory / "forcefield/1h1q/vacuum.mol2"
    ligand_1_params = mock_bfe_directory / "forcefield/1h1q/vacuum.parm7"
    ligand_2_coords = mock_bfe_directory / "forcefield/1oiu/vacuum.mol2"
    ligand_2_params = mock_bfe_directory / "forcefield/1oiu/vacuum.parm7"
    ligand_1_ref_atoms = ("@1", "@2", "@3")
    ligand_2_ref_atoms = ("@4", "@5", "@6")

    output_dir = tmp_cwd / "outputs"

    run_solution_phase(
        config,
        ligand_1_coords,
        ligand_1_params,
        ligand_2_coords,
        ligand_2_params,
        output_dir,
        None,
        ligand_1_ref_atoms,
        ligand_2_ref_atoms,
    )

    mock_setup.assert_called_once()
    mock_equilibrate.assert_called_once()
    mock_sample.assert_called_once()

    (output_dir / "_sample/samples.arrow").write_text("")

    run_solution_phase(
        config,
        ligand_1_coords,
        ligand_1_params,
        ligand_2_coords,
        ligand_2_params,
        output_dir,
        None,
        ligand_1_ref_atoms,
        ligand_2_ref_atoms,
    )

    # caching should have taken place.
    mock_setup.assert_called_once()
    mock_equilibrate.assert_called_once()


def test_run_complex_phase(tmp_cwd, mock_bfe_directory, mocker):
    mock_topology = build_mock_structure(["O"])  # same as above
    config = femto.fe.septop.SepTopConfig()

    mock_setup = mocker.patch(
        "femto.fe.septop._runner._prepare_complex_phase",
        autospec=True,
        return_value=(mock_topology, openmm.System()),
    )
    mock_equilibrate = mocker.patch(
        "femto.fe.septop.equilibrate_states",
        autospec=True,
        return_value=[openmm.State()] * len(config.complex.states.lambda_vdw_ligand_1),
    )
    mock_sample = mocker.patch("femto.fe.septop.run_hremd", autospec=True)

    receptor_coords = mock_bfe_directory / "proteins/cdk2/protein.pdb"
    receptor_params = None
    ligand_1_coords = mock_bfe_directory / "forcefield/1h1q/vacuum.mol2"
    ligand_1_params = mock_bfe_directory / "forcefield/1h1q/vacuum.parm7"
    ligand_2_coords = mock_bfe_directory / "forcefield/1oiu/vacuum.mol2"
    ligand_2_params = mock_bfe_directory / "forcefield/1oiu/vacuum.parm7"
    ligand_1_ref_atoms = ("@1", "@2", "@3")
    ligand_2_ref_atoms = ("@4", "@5", "@6")
    receptor_ref_atoms = ("@7", "@8", "@9")

    output_dir = tmp_cwd / "outputs"

    run_complex_phase(
        config,
        ligand_1_coords,
        ligand_1_params,
        ligand_2_coords,
        ligand_2_params,
        receptor_coords,
        receptor_params,
        output_dir,
        None,
        ligand_1_ref_atoms,
        ligand_2_ref_atoms,
        receptor_ref_atoms,
    )

    mock_setup.assert_called_once()
    mock_equilibrate.assert_called_once()
    mock_sample.assert_called_once()

    (output_dir / "_sample/samples.arrow").write_text("")

    run_complex_phase(
        config,
        ligand_1_coords,
        ligand_1_params,
        ligand_2_coords,
        ligand_2_params,
        receptor_coords,
        receptor_params,
        output_dir,
        None,
        ligand_1_ref_atoms,
        ligand_2_ref_atoms,
        receptor_ref_atoms,
    )

    # caching should have taken place.
    mock_setup.assert_called_once()
    mock_equilibrate.assert_called_once()


def test_submit_network(tmp_cwd, mock_bfe_directory, mocker):
    mock_config = femto.fe.septop.SepTopConfig()
    mock_network = CDK2_SYSTEM.rbfe_network

    expected_ids = ("id-a", "id-b", "id-c")

    mock_submit_options = femto.fe.utils.queue.SLURMOptions(
        n_nodes=1,
        n_tasks=2,
        n_cpus_per_task=3,
        n_gpus_per_task=4,
        walltime="6-0",
        partition="partition1",
        job_name="job_name3",
    )

    mock_submit = mocker.patch(
        "femto.fe.utils.queue.submit_slurm_job",
        autospec=True,
        side_effect=expected_ids,
    )

    expected_output_dir = tmp_cwd / "outputs"

    job_ids = submit_network(
        mock_config,
        mock_network,
        expected_output_dir,
        mock_submit_options,
    )
    assert job_ids == [expected_ids]

    mock_submit.assert_has_calls(
        [
            mocker.call(
                [
                    "srun",
                    "--mpi=pmix",
                    "femto",
                    "septop",
                    "--config",
                    mocker.ANY,
                    "run-solution",
                    f"--ligand-1-coords={CDK2_SYSTEM.ligand_1_coords}",
                    f"--ligand-1-params={CDK2_SYSTEM.ligand_1_params}",
                    "--ligand-1-ref-atoms",
                    *CDK2_SYSTEM.ligand_1_ref_atoms,
                    f"--ligand-2-coords={CDK2_SYSTEM.ligand_2_coords}",
                    f"--ligand-2-params={CDK2_SYSTEM.ligand_2_params}",
                    "--ligand-2-ref-atoms",
                    *CDK2_SYSTEM.ligand_2_ref_atoms,
                    f"--output-dir={expected_output_dir}/1h1q~1oiu/solution",
                    f"--report-dir={expected_output_dir}/1h1q~1oiu/solution",
                ],
                mock_submit_options,
                mocker.ANY,
            ),
            mocker.call(
                [
                    "srun",
                    "--mpi=pmix",
                    "femto",
                    "septop",
                    "--config",
                    mocker.ANY,
                    "run-complex",
                    f"--receptor-coords={CDK2_SYSTEM.receptor_coords}",
                    "--receptor-ref-atoms",
                    *CDK2_SYSTEM.receptor_ref_atoms,
                    f"--ligand-1-coords={CDK2_SYSTEM.ligand_1_coords}",
                    f"--ligand-1-params={CDK2_SYSTEM.ligand_1_params}",
                    "--ligand-1-ref-atoms",
                    *CDK2_SYSTEM.ligand_1_ref_atoms,
                    f"--ligand-2-coords={CDK2_SYSTEM.ligand_2_coords}",
                    f"--ligand-2-params={CDK2_SYSTEM.ligand_2_params}",
                    "--ligand-2-ref-atoms",
                    *CDK2_SYSTEM.ligand_2_ref_atoms,
                    f"--output-dir={expected_output_dir}/1h1q~1oiu/complex",
                    f"--report-dir={expected_output_dir}/1h1q~1oiu/complex",
                ],
                mock_submit_options,
                mocker.ANY,
            ),
            mocker.call(
                [
                    "femto",
                    "septop",
                    "--config",
                    mocker.ANY,
                    "analyze",
                    "--complex-samples",
                    expected_output_dir / "1h1q~1oiu/complex/_sample/samples.arrow",
                    "--complex-system",
                    expected_output_dir / "1h1q~1oiu/complex/_setup/system.xml",
                    "--solution-samples",
                    expected_output_dir / "1h1q~1oiu/solution/_sample/samples.arrow",
                    "--solution-system",
                    expected_output_dir / "1h1q~1oiu/solution/_setup/system.xml",
                    "--output",
                    expected_output_dir / "1h1q~1oiu/ddg.csv",
                ],
                mock_submit_options,
                mocker.ANY,
                ["id-a", "id-b"],
            ),
        ]
    )
