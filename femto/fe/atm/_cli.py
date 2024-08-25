"""Command line interface for femto atm."""

import logging
import pathlib
import shlex

import click
import cloup
import numpy
import openmm.unit

import femto.fe.atm
import femto.fe.atm._config
import femto.fe.atm._runner
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
        type=str,
        help="An (optional) AMBER style query that selects the receptor atoms that "
        "define the binding site. If unspecified the alpha carbon of those residues "
        "closest to the ligand(s) will be used.",
    ),
]
_RECEPTOR_OPTIONS_GROUP = cloup.option_group(
    femto.fe.utils.cli.DEFAULT_RECEPTOR_OPTIONS_GROUP_NAME,
    *_RECEPTOR_OPTIONS,
    help=femto.fe.utils.cli.DEFAULT_RECEPTOR_OPTIONS_GROUP_HELP,
)

_LIGAND_OPTIONS = [
    *femto.fe.utils.cli.DEFAULT_LIGAND_OPTIONS,
    cloup.option(
        "--ligand-displacement",
        type=(float, float, float),
        default=None,
        help="The displacement vector to use for the ligands in Å.",
    ),
]
_LIGAND_OPTIONS_GROUP = cloup.option_group(
    femto.fe.utils.cli.DEFAULT_LIGAND_OPTIONS_GROUP_NAME,
    *_LIGAND_OPTIONS,
    help=femto.fe.utils.cli.DEFAULT_LIGAND_OPTIONS_GROUP_HELP,
)

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
_SUBMIT_SLURM_OPTION_GROUP = femto.fe.utils.cli.generate_slurm_cli_options("ATM", True)
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

_DEV_OPTIONS = [
    cloup.option(
        "--with-timer",
        type=bool,
        default=False,
        is_flag=True,
        help="Whether to show timing information.",
    ),
]
_DEV_OPTIONS_GROUP = cloup.option_group(
    "Developer options",
    *_DEV_OPTIONS,
)


@cloup.group("atm")
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
    config_dict = femto.md.config.merge_configs(
        femto.fe.atm._config.ATMConfig(),
        *(
            []
            if config_path is None
            else [femto.fe.atm._config.load_config(config_path)]
        ),
    )
    context.obj = femto.fe.atm._config.ATMConfig(**config_dict)


@main.command("config")
@cloup.pass_context
def _print_config(context: cloup.Context):
    """Print the default configuration."""
    config = context.obj
    print(config.model_dump_yaml(sort_keys=False), flush=True)


@main.command("run-workflow")
@femto.fe.utils.cli.add_options(femto.fe.utils.cli.DEFAULT_INPUTS_GROUP)
@femto.fe.utils.cli.add_options(femto.fe.utils.cli.DEFAULT_LIGAND_PATHS_GROUP)
@femto.fe.utils.cli.add_options(_LIGAND_OPTIONS_GROUP)
@femto.fe.utils.cli.add_options(femto.fe.utils.cli.DEFAULT_RECEPTOR_PATHS_GROUP)
@femto.fe.utils.cli.add_options(_RECEPTOR_OPTIONS_GROUP)
@femto.fe.utils.cli.add_options(femto.fe.utils.cli.DEFAULT_OUTPUTS_GROUP)
@femto.fe.utils.cli.add_options(_DEV_OPTIONS_GROUP)
@cloup.pass_context
def _run_workflow_cli(
    context: cloup.Context,
    root_dir: pathlib.Path | None,
    edges_path: pathlib.Path | None,
    ligand_1: str | None,
    ligand_2: str | None,
    receptor_coords: pathlib.Path | None,
    receptor_params: pathlib.Path | None,
    receptor_ref_atoms: str | None,
    ligand_1_coords: pathlib.Path | None,
    ligand_1_params: pathlib.Path | None,
    ligand_1_ref_atoms: tuple[str, str, str] | None,
    ligand_2_coords: pathlib.Path | None,
    ligand_2_params: pathlib.Path | None,
    ligand_2_ref_atoms: tuple[str, str, str] | None,
    ligand_displacement: tuple[float, float, float] | None,
    output_dir: pathlib.Path,
    report_dir: pathlib.Path | None,
    with_timer: bool,
):
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
        network = femto.fe.inputs.find_edges(
            root_dir, femto.fe.atm._config.ATMNetwork, edges_path
        )

        receptor_coords = network.receptor.coords
        receptor_params = network.receptor.params
        receptor_ref_atoms = network.receptor.metadata["ref_atoms"]

        edge = network.find_edge(ligand_1, ligand_2)

        ligand_1_coords = edge.ligand_1.coords
        ligand_1_params = edge.ligand_1.params
        ligand_1_ref_atoms = edge.ligand_1.metadata["ref_atoms"]

        ligand_2_coords = edge.ligand_2.coords
        ligand_2_params = edge.ligand_2.params
        ligand_2_ref_atoms = edge.ligand_2.metadata["ref_atoms"]
    else:
        if receptor_coords is None:
            raise click.UsageError("The receptor coordinates must be provided")
        if ligand_1_coords is None:
            raise click.UsageError("The ligand coordinates must be provided")

    ligand_displacement = (
        ligand_displacement
        if ligand_displacement is None
        else numpy.array(ligand_displacement) * openmm.unit.angstrom
    )

    # handle cases where multiple simulations should be stacked on a single GPU
    femto.md.utils.mpi.divide_gpus()

    femto.fe.atm._runner.run_workflow(
        config,
        ligand_1_coords,
        ligand_1_params,
        ligand_2_coords,
        ligand_2_params,
        receptor_coords,
        receptor_params,
        output_dir,
        report_dir,
        ligand_displacement,
        ligand_1_ref_atoms,
        ligand_2_ref_atoms,
        receptor_ref_atoms,
        with_timer,
    )


@main.command("submit-workflows")
@femto.fe.utils.cli.add_options(_SUBMIT_INPUTS_GROUP)
@femto.fe.utils.cli.add_options(_SUBMIT_OUTPUTS_GROUP)
@femto.fe.utils.cli.add_options(_SUBMIT_SLURM_OPTION_GROUP)
@femto.fe.utils.cli.add_options(_SUBMIT_OPTIONS_GROUP)
@femto.fe.utils.cli.add_options(_DEV_OPTIONS_GROUP)
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
    with_timer: bool,
):
    config = context.obj

    network = femto.fe.inputs.find_edges(
        root_dir, femto.fe.atm._config.ATMNetwork, edges_path
    )

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

    job_ids = femto.fe.atm._runner.submit_network(
        config, network, output_dir, queue_options, shlex.split(mpi_command), with_timer
    )

    for edge, job_id in zip(network.edges, job_ids, strict=True):
        edge_name = edge.ligand_1.name + (
            "" if edge.ligand_2 is None else f"~{edge.ligand_2.name}"
        )
        _LOGGER.info(f"submitted {edge_name} as job={job_id}")

    if not wait:
        return

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
@femto.fe.utils.cli.add_options(_DEV_OPTIONS_GROUP)
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
    with_timer: bool,
):
    config = context.obj

    network = femto.fe.inputs.find_edges(
        root_dir, femto.fe.atm._config.ATMNetwork, edges_path
    )

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
        job_ids = femto.fe.atm._runner.submit_network(
            config,
            network,
            output_dir / f"replica-{i}",
            queue_options,
            shlex.split(mpi_command),
            with_timer,
        )

        for edge, job_id in zip(network.edges, job_ids, strict=True):
            edge_name = edge.ligand_1.name + (
                "" if edge.ligand_2 is None else f"~{edge.ligand_2.name}"
            )
            _LOGGER.info(f"submitted {edge_name} replica={i} as job={job_id}")

    if not wait:
        return

    _LOGGER.info(f"waiting for {len(job_ids)} jobs to complete")
    femto.fe.utils.queue.wait_for_slurm_jobs(job_ids)
