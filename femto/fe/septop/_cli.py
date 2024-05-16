"""Command line interface for femto septop."""

import logging
import pathlib
import shlex

import cloup
import openmm
import yaml

import femto.fe.ddg
import femto.fe.inputs
import femto.fe.utils.cli
import femto.fe.utils.queue
import femto.md.config
import femto.md.utils.mpi

_LOGGER = logging.getLogger(__name__)


_RECEPTOR_OPTIONS = [
    *femto.fe.utils.cli.DEFAULT_RECEPTOR_OPTIONS,
    cloup.option(
        "--receptor-ref-atoms",
        type=(str, str, str),
        help="Three (optional) AMBER style queries that select the atoms of the "
        "receptor to align the ligand using.",
    ),
]
_RECEPTOR_OPTIONS_GROUP = cloup.option_group(
    femto.fe.utils.cli.DEFAULT_RECEPTOR_OPTIONS_GROUP_NAME,
    *_RECEPTOR_OPTIONS,
    help=femto.fe.utils.cli.DEFAULT_RECEPTOR_OPTIONS_GROUP_HELP,
)

_ANALYZE_OPTIONS = [
    cloup.option(
        "--complex-system",
        "complex_system_path",
        type=femto.fe.utils.cli.INPUT_PATH,
        required=True,
        help="The path to the complex system (.xml).",
    ),
    cloup.option(
        "--complex-samples",
        "complex_samples_path",
        type=femto.fe.utils.cli.INPUT_PATH,
        required=True,
        help="The path to the energies (.arrow) of the complex sampled by HREMD.",
    ),
    cloup.option(
        "--solution-system",
        "solution_system_path",
        type=femto.fe.utils.cli.INPUT_PATH,
        required=True,
        help="The path to the solution system (.xml).",
    ),
    cloup.option(
        "--solution-samples",
        "solution_samples_path",
        type=femto.fe.utils.cli.INPUT_PATH,
        required=True,
        help="The path to the energies (.arrow) of the solution sampled by HREMD.",
    ),
    cloup.option(
        "--output",
        "output_path",
        type=femto.fe.utils.cli.OUTPUT_PATH,
        required=True,
        help="The path to store the computed free energies to.",
    ),
]

_SUBMIT_INPUTS_GROUP = cloup.option_group(
    femto.fe.utils.cli.DEFAULT_INPUTS_GROUP_NAME,
    *femto.fe.utils.cli.DEFAULT_INPUTS[:2],
    help=femto.fe.utils.cli.DEFAULT_INPUTS_GROUP_HELP,
)
_SUBMIT_OUTPUTS_GROUP = cloup.option_group(
    femto.fe.utils.cli.DEFAULT_OUTPUTS_GROUP_NAME,
    femto.fe.utils.cli.DEFAULT_OUTPUTS[0],
    help=femto.fe.utils.cli.DEFAULT_OUTPUTS_GROUP_HELP,
)
_SUBMIT_SLURM_OPTION_GROUP = femto.fe.utils.cli.generate_slurm_cli_options(
    "SepTop", True
)
_SUBMIT_OPTIONS = [
    *femto.fe.utils.cli.DEFAULT_SUBMIT_OPTIONS,
    cloup.option(
        "--mpi-cmd",
        "mpi_command",
        type=str,
        help="The MPI runner command (e.g. `mpiexec`) to use.",
        default="srun --mpi pmix",
        show_default=True,
    ),
]
_SUBMIT_OPTIONS_GROUP = cloup.option_group(
    femto.fe.utils.cli.DEFAULT_SUBMIT_OPTIONS_GROUP_NAME,
    *_SUBMIT_OPTIONS,
    help=femto.fe.utils.cli.DEFAULT_SUBMIT_OPTIONS_GROUP_HELP,
)


@cloup.group("septop")
@cloup.option(
    "-c",
    "--config",
    "config_path",
    type=femto.fe.utils.cli.INPUT_PATH,
    required=False,
    help="The path to the configuration to use.",
)
@cloup.pass_context
def main(context: cloup.Context, config_path: pathlib.Path):
    import femto.fe.septop

    config_dict = femto.md.config.merge_configs(
        femto.fe.septop.SepTopConfig(),
        *([] if config_path is None else [yaml.safe_load(config_path.read_text())]),
    )
    context.obj = femto.fe.septop.SepTopConfig(**config_dict)


@main.command("config")
@cloup.pass_context
def _print_config(context: cloup.Context):
    """Print the default configuration."""
    config = context.obj
    print(config.model_dump_yaml(sort_keys=False), flush=True)


@main.command("run-solution")
@femto.fe.utils.cli.add_options(femto.fe.utils.cli.DEFAULT_INPUTS_GROUP)
@femto.fe.utils.cli.add_options(femto.fe.utils.cli.DEFAULT_LIGAND_PATHS_GROUP)
@femto.fe.utils.cli.add_options(femto.fe.utils.cli.DEFAULT_LIGAND_OPTIONS_GROUP)
@femto.fe.utils.cli.add_options(femto.fe.utils.cli.DEFAULT_OUTPUTS_GROUP)
@cloup.pass_context
def _run_solution_cli(
    context: cloup.Context,
    root_dir: pathlib.Path | None,
    edges_path: pathlib.Path | None,
    ligand_1: str | None,
    ligand_2: str | None,
    ligand_1_coords: pathlib.Path | None,
    ligand_1_params: pathlib.Path | None,
    ligand_1_ref_atoms: tuple[str, str, str] | None,
    ligand_2_coords: pathlib.Path | None,
    ligand_2_params: pathlib.Path | None,
    ligand_2_ref_atoms: tuple[str, str, str] | None,
    output_dir: pathlib.Path,
    report_dir: pathlib.Path | None,
):
    import femto.fe.septop

    config = context.obj

    using_directory, _ = femto.fe.utils.cli.validate_mutually_exclusive_groups(
        context,
        femto.fe.utils.cli.DEFAULT_INPUTS_GROUP_NAME,
        [
            femto.fe.utils.cli.DEFAULT_LIGAND_PATHS_GROUP_NAME,
            femto.fe.utils.cli.DEFAULT_LIGAND_OPTIONS_GROUP_NAME,
        ],
        optional_fields={
            "edges_path",
            "ligand_1_ref_atoms",
            "ligand_2",
            "ligand_2_coords",
            "ligand_2_params",
            "ligand_2_ref_atoms",
            "report_dir",
        },
    )

    if using_directory:
        network = femto.fe.inputs.find_edges(root_dir, config_path=edges_path)

        edge = network.find_edge(ligand_1, ligand_2)

        ligand_1_coords = edge.ligand_1.coords
        ligand_1_params = edge.ligand_1.params
        # ligand_1_ref_atoms = edge.ligand_1.metadata["ref_atoms"]

        ligand_2_coords = edge.ligand_2.coords
        ligand_2_params = edge.ligand_2.params
        # ligand_2_ref_atoms = edge.ligand_2.metadata["ref_atoms"]

    # handle cases where multiple simulations should be stacked on a single GPU
    femto.md.utils.mpi.divide_gpus()

    femto.fe.septop.run_solution_phase(
        config,
        ligand_1_coords,
        ligand_1_params,
        ligand_2_coords,
        ligand_2_params,
        output_dir,
        report_dir,
        ligand_1_ref_atoms,
        ligand_2_ref_atoms,
    )


@main.command("run-complex")
@femto.fe.utils.cli.add_options(femto.fe.utils.cli.DEFAULT_INPUTS_GROUP)
@femto.fe.utils.cli.add_options(femto.fe.utils.cli.DEFAULT_LIGAND_PATHS_GROUP)
@femto.fe.utils.cli.add_options(femto.fe.utils.cli.DEFAULT_LIGAND_OPTIONS_GROUP)
@femto.fe.utils.cli.add_options(femto.fe.utils.cli.DEFAULT_RECEPTOR_PATHS_GROUP)
@femto.fe.utils.cli.add_options(_RECEPTOR_OPTIONS_GROUP)
@femto.fe.utils.cli.add_options(femto.fe.utils.cli.DEFAULT_OUTPUTS_GROUP)
@cloup.pass_context
def _run_complex_cli(
    context: cloup.Context,
    root_dir: pathlib.Path | None,
    edges_path: pathlib.Path | None,
    ligand_1: str | None,
    ligand_2: str | None,
    receptor_coords: pathlib.Path | None,
    receptor_params: pathlib.Path | None,
    receptor_ref_atoms: tuple[str, str, str] | None,
    ligand_1_coords: pathlib.Path | None,
    ligand_1_params: pathlib.Path | None,
    ligand_1_ref_atoms: tuple[str, str, str] | None,
    ligand_2_coords: pathlib.Path | None,
    ligand_2_params: pathlib.Path | None,
    ligand_2_ref_atoms: tuple[str, str, str] | None,
    output_dir: pathlib.Path,
    report_dir: pathlib.Path | None,
):
    import femto.fe.septop

    config = context.obj

    using_directory, _ = femto.fe.utils.cli.validate_mutually_exclusive_groups(
        context,
        femto.fe.utils.cli.DEFAULT_INPUTS_GROUP_NAME,
        [
            femto.fe.utils.cli.DEFAULT_LIGAND_PATHS_GROUP_NAME,
            femto.fe.utils.cli.DEFAULT_LIGAND_OPTIONS_GROUP_NAME,
            femto.fe.utils.cli.DEFAULT_RECEPTOR_PATHS_GROUP_NAME,
            femto.fe.utils.cli.DEFAULT_RECEPTOR_OPTIONS_GROUP_NAME,
        ],
        optional_fields={
            "edges_path",
            "ligand_1_ref_atoms",
            "ligand_2",
            "ligand_2_coords",
            "ligand_2_params",
            "ligand_2_ref_atoms",
            "receptor_params",
            "receptor_ref_atoms",
            "report_dir",
        },
    )

    if using_directory:
        network = femto.fe.inputs.find_edges(root_dir, config_path=edges_path)

        receptor_coords = network.receptor.coords
        receptor_params = network.receptor.params
        # receptor_ref_atoms = network.receptor.metadata["ref_atoms"]

        edge = network.find_edge(ligand_1, ligand_2)

        ligand_1_coords = edge.ligand_1.coords
        ligand_1_params = edge.ligand_1.params
        # ligand_1_ref_atoms = edge.ligand_1.metadata["ref_atoms"]

        ligand_2_coords = edge.ligand_2.coords
        ligand_2_params = edge.ligand_2.params
        # ligand_2_ref_atoms = edge.ligand_2.metadata["ref_atoms"]

    # handle cases where multiple simulations should be stacked on a single GPU
    femto.md.utils.mpi.divide_gpus()

    femto.fe.septop.run_complex_phase(
        config,
        ligand_1_coords,
        ligand_1_params,
        ligand_2_coords,
        ligand_2_params,
        receptor_coords,
        receptor_params,
        output_dir,
        report_dir,
        ligand_1_ref_atoms,
        ligand_2_ref_atoms,
        receptor_ref_atoms,
    )


@main.command("analyze")
@femto.fe.utils.cli.add_options(_ANALYZE_OPTIONS)
@cloup.pass_context
def _analyze_cli(
    context: cloup.Context,
    complex_system_path: pathlib.Path,
    complex_samples_path: pathlib.Path,
    solution_system_path: pathlib.Path,
    solution_samples_path: pathlib.Path,
    output_path: pathlib.Path,
):
    import femto.fe.septop

    config = context.obj

    result_frame = femto.fe.septop.compute_ddg(
        config,
        *femto.fe.ddg.load_u_kn(complex_samples_path),
        openmm.XmlSerializer.deserialize(complex_system_path.read_text()),
        *femto.fe.ddg.load_u_kn(solution_samples_path),
        openmm.XmlSerializer.deserialize(solution_system_path.read_text()),
    )
    print(result_frame, flush=True)

    output_path.parent.mkdir(exist_ok=True, parents=True)
    result_frame.to_csv(output_path, index=False)


@main.command("submit-workflows")
@femto.fe.utils.cli.add_options(_SUBMIT_INPUTS_GROUP)
@femto.fe.utils.cli.add_options(_SUBMIT_OUTPUTS_GROUP)
@femto.fe.utils.cli.add_options(_SUBMIT_SLURM_OPTION_GROUP)
@femto.fe.utils.cli.add_options(_SUBMIT_OPTIONS_GROUP)
@cloup.pass_context
def _submit_workflows_cli(
    context: cloup.Context,
    root_dir: pathlib.Path,
    output_dir: pathlib.Path,
    edges_path: pathlib.Path | None,
    slurm_n_nodes: int,
    slurm_n_tasks: int,
    slurm_n_cpus_per_task: int,
    slurm_n_gpus_per_task: int,
    slurm_walltime: str,
    slurm_partition: str,
    slurm_account: str | None,
    slurm_job_name: str,
    slurm_reservation: str | None,
    wait: bool,
    mpi_command: str,
):
    import femto.fe.septop

    config = context.obj

    network = femto.fe.inputs.find_edges(root_dir, config_path=edges_path)

    queue_options = femto.fe.utils.queue.SLURMOptions(
        n_nodes=slurm_n_nodes,
        n_tasks=slurm_n_tasks,
        n_cpus_per_task=slurm_n_cpus_per_task,
        n_gpus_per_task=slurm_n_gpus_per_task,
        walltime=slurm_walltime,
        partition=slurm_partition,
        account=slurm_account,
        job_name=slurm_job_name,
        reservation=slurm_reservation,
    )

    job_ids_per_pair = femto.fe.septop.submit_network(
        config, network, output_dir, queue_options, shlex.split(mpi_command)
    )

    for edge, (solution_id, complex_id, analyze_id) in zip(
        network.edges, job_ids_per_pair, strict=True
    ):
        edge_name = edge.ligand_1.name + (
            "" if edge.ligand_2 is None else f"~{edge.ligand_2.name}"
        )
        _LOGGER.info(
            f"submitted {edge_name} as "
            f"run-solution={solution_id} "
            f"run-complex={complex_id} "
            f"analyze={analyze_id}"
        )

    if not wait:
        return

    job_ids = [job_id for job_ids in job_ids_per_pair for job_id in job_ids]

    _LOGGER.info(f"waiting for {len(job_ids)} jobs to complete")
    femto.fe.utils.queue.wait_for_slurm_jobs(job_ids)


@main.command("submit-replicas")
@cloup.option(
    "--n-replicas", type=int, required=True, help="The number of replicas to submit."
)
@femto.fe.utils.cli.add_options(_SUBMIT_INPUTS_GROUP)
@femto.fe.utils.cli.add_options(_SUBMIT_OUTPUTS_GROUP)
@femto.fe.utils.cli.add_options(_SUBMIT_SLURM_OPTION_GROUP)
@femto.fe.utils.cli.add_options(_SUBMIT_OPTIONS_GROUP)
@cloup.pass_context
def _submit_replicas_cli(
    context: cloup.Context,
    n_replicas: int,
    root_dir: pathlib.Path,
    edges_path: pathlib.Path | None,
    output_dir: pathlib.Path,
    slurm_n_nodes: int,
    slurm_n_tasks: int,
    slurm_n_cpus_per_task: int,
    slurm_n_gpus_per_task: int,
    slurm_walltime: str,
    slurm_partition: str,
    slurm_account: str | None,
    slurm_job_name: str,
    slurm_reservation: str | None,
    wait: bool,
    mpi_command: str,
):
    import femto.fe.septop

    config = context.obj

    network = femto.fe.inputs.find_edges(root_dir, config_path=edges_path)

    queue_options = femto.fe.utils.queue.SLURMOptions(
        n_nodes=slurm_n_nodes,
        n_tasks=slurm_n_tasks,
        n_cpus_per_task=slurm_n_cpus_per_task,
        n_gpus_per_task=slurm_n_gpus_per_task,
        walltime=slurm_walltime,
        partition=slurm_partition,
        account=slurm_account,
        job_name=slurm_job_name,
        reservation=slurm_reservation,
    )

    job_ids = set()

    for i in range(n_replicas):
        job_ids_per_pair = femto.fe.septop.submit_network(
            config,
            network,
            output_dir / f"replica-{i}",
            queue_options,
            shlex.split(mpi_command),
        )

        for edge, (solution_id, complex_id, analyze_id) in zip(
            network.edges, job_ids_per_pair, strict=True
        ):
            edge_name = edge.ligand_1.name + (
                "" if edge.ligand_2 is None else f"~{edge.ligand_2.name}"
            )

            _LOGGER.info(
                f"submitted {edge_name} "
                f"replica={i} as "
                f"run-solution={solution_id} "
                f"run-complex={complex_id} "
                f"analyze={analyze_id}"
            )

        job_ids.update(job_id for job_ids in job_ids_per_pair for job_id in job_ids)

    if not wait:
        return

    _LOGGER.info(f"waiting for {len(job_ids)} jobs to complete")
    femto.fe.utils.queue.wait_for_slurm_jobs(job_ids)
