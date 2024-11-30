"""Helpers to run SepTop calculations."""

import datetime
import functools
import pathlib
import typing

import openmm

import femto.fe.inputs
import femto.fe.utils.queue
import femto.md.constants
import femto.md.prepare
import femto.md.reporting
import femto.md.utils.mpi
import femto.top

if typing.TYPE_CHECKING:
    import femto.fe.septop


@femto.md.utils.mpi.run_on_rank_zero
def _prepare_solution_phase(
    config: "femto.fe.septop.SepTopPhaseConfig",
    ligand_1_coords: pathlib.Path,
    ligand_1_params: pathlib.Path,
    ligand_2_coords: pathlib.Path | None,
    ligand_2_params: pathlib.Path | None,
    ligand_1_ref_atoms: tuple[str, str, str] | None = None,
    ligand_2_ref_atoms: tuple[str, str, str] | None = None,
) -> tuple[femto.top.Topology, openmm.System]:
    ligand_1, ligand_2 = femto.md.prepare.load_ligands(ligand_1_coords, ligand_2_coords)
    return femto.fe.septop._setup.setup_solution(
        config.setup,
        ligand_1,
        ligand_2,
        ligand_1_ref_atoms,
        ligand_2_ref_atoms,
        ligand_1_params,
        ligand_2_params,
    )


@femto.md.utils.mpi.run_on_rank_zero
def _prepare_complex_phase(
    config: "femto.fe.septop.SepTopPhaseConfig",
    ligand_1_coords: pathlib.Path,
    ligand_1_params: pathlib.Path,
    ligand_2_coords: pathlib.Path | None,
    ligand_2_params: pathlib.Path | None,
    receptor_coords: pathlib.Path,
    receptor_params: pathlib.Path | None,
    ligand_1_ref_atoms: tuple[str, str, str] | None = None,
    ligand_2_ref_atoms: tuple[str, str, str] | None = None,
    receptor_ref_atoms: tuple[str, str, str] | None = None,
) -> tuple[femto.top.Topology, openmm.System]:
    import femto.fe.septop

    receptor = femto.md.prepare.load_receptor(receptor_coords)

    ligand_1, ligand_2 = femto.md.prepare.load_ligands(ligand_1_coords, ligand_2_coords)

    return femto.fe.septop.setup_complex(
        config.setup,
        receptor,
        ligand_1,
        ligand_2,
        receptor_ref_atoms,
        ligand_1_ref_atoms,
        ligand_2_ref_atoms,
        receptor_params,
        ligand_1_params,
        ligand_2_params,
    )


@femto.md.utils.mpi.run_on_rank_zero
def _cache_setup_outputs(
    topology: femto.top.Topology,
    topology_path: pathlib.Path,
    system: openmm.System,
    system_path: pathlib.Path,
):
    topology.to_file(topology_path)
    system_path.write_text(openmm.XmlSerializer.serialize(system))


@femto.md.utils.mpi.run_on_rank_zero
def _cache_equilibrate_outputs(coords: list[openmm.State], paths: list[pathlib.Path]):
    for path, coord in zip(paths, coords, strict=True):
        path.write_text(openmm.XmlSerializer.serialize(coord))


def _run_phase(
    config: "femto.fe.septop.SepTopPhaseConfig",
    prepare_fn: typing.Callable[[], tuple[femto.top.Topology, openmm.System]],
    output_dir: pathlib.Path,
    report_dir: pathlib.Path | None = None,
):
    """Run the setup, equilibration, and sampling phases.

    Args:
        config: The configuration for the phase.
        prepare_fn: A function that prepares the system and topology.
        output_dir: The directory to store all outputs in.
        report_dir: The directory to store the report in.
    """
    import femto.fe.septop

    reporter = (
        femto.md.reporting.NullReporter()
        if report_dir is None or not femto.md.utils.mpi.is_rank_zero()
        else femto.md.reporting.TensorboardReporter(report_dir)
    )

    setup_dir = output_dir / "_setup"
    setup_dir.mkdir(exist_ok=True, parents=True)

    topology_path, system_path = setup_dir / "system.pdb", setup_dir / "system.xml"

    if not topology_path.exists() or not system_path.exists():
        topology, system = prepare_fn()
        _cache_setup_outputs(topology, topology_path, system, system_path)
    else:
        topology = femto.top.Topology.from_file(topology_path)
        system = openmm.XmlSerializer.deserialize(system_path.read_text())

    equilibrate_dir = output_dir / "_equilibrate"
    equilibrate_dir.mkdir(exist_ok=True, parents=True)

    coord_paths = [
        equilibrate_dir / f"state_{state_idx}.xml"
        for state_idx in range(len(config.states.lambda_vdw_ligand_1))
    ]

    if any(not path.exists() for path in coord_paths):
        coords = femto.fe.septop.equilibrate_states(
            system,
            topology,
            config.states,
            config.equilibrate,
            femto.md.constants.OpenMMPlatform.CUDA,
            reporter,
        )
        _cache_equilibrate_outputs(coords, coord_paths)
    else:
        coords = [
            openmm.XmlSerializer.deserialize(path.read_text()) for path in coord_paths
        ]

    sample_dir = output_dir / "_sample"
    sample_dir.mkdir(exist_ok=True, parents=True)

    femto.fe.septop.run_hremd(
        system,
        topology,
        coords,
        config.states,
        config.sample,
        femto.md.constants.OpenMMPlatform.CUDA,
        sample_dir,
        reporter,
    )


def run_solution_phase(
    config: "femto.fe.septop.SepTopConfig",
    ligand_1_coords: pathlib.Path,
    ligand_1_params: pathlib.Path,
    ligand_2_coords: pathlib.Path | None,
    ligand_2_params: pathlib.Path | None,
    output_dir: pathlib.Path,
    report_dir: pathlib.Path | None = None,
    ligand_1_ref_atoms: tuple[str, str, str] | None = None,
    ligand_2_ref_atoms: tuple[str, str, str] | None = None,
):
    """Run the solution phase of the SepTop calculation.

    Args:
        config: The configuration.
        ligand_1_coords: The coordinates of the first ligand.
        ligand_1_params: The parameters of the first ligand.
        ligand_2_coords: The coordinates of the second ligand.
        ligand_2_params: The parameters of the second ligand.
        output_dir: The directory to store all outputs in.
        report_dir: The directory to store the report in.
        ligand_1_ref_atoms: The AMBER style query masks that select the first ligands
            reference atoms.
        ligand_2_ref_atoms: The AMBER style query masks that select the second ligands
            reference atoms.
    """

    prepare_fn = functools.partial(
        _prepare_solution_phase,
        config.solution,
        ligand_1_coords,
        ligand_1_params,
        ligand_2_coords,
        ligand_2_params,
        ligand_1_ref_atoms,
        ligand_2_ref_atoms,
    )
    _run_phase(config.solution, prepare_fn, output_dir, report_dir)


def run_complex_phase(
    config: "femto.fe.septop.SepTopConfig",
    ligand_1_coords: pathlib.Path,
    ligand_1_params: pathlib.Path,
    ligand_2_coords: pathlib.Path | None,
    ligand_2_params: pathlib.Path | None,
    receptor_coords: pathlib.Path,
    receptor_params: pathlib.Path | None,
    output_dir: pathlib.Path,
    report_dir: pathlib.Path | None = None,
    ligand_1_ref_atoms: tuple[str, str, str] | None = None,
    ligand_2_ref_atoms: tuple[str, str, str] | None = None,
    receptor_ref_atoms: tuple[str, str, str] | None = None,
):
    """Run the complex phase of the SepTop calculation.

    Args:
        config: The configuration.
        ligand_1_coords: The coordinates of the first ligand.
        ligand_1_params: The parameters of the first ligand.
        ligand_2_coords: The coordinates of the second ligand.
        ligand_2_params: The parameters of the second ligand.
        receptor_coords: The coordinates of the receptor.
        receptor_params: The parameters of the receptor.
        output_dir: The directory to store all outputs in.
        report_dir: The directory to store the logs / reports in.
        ligand_1_ref_atoms: The AMBER style query masks that select the first ligands
            reference atoms.
        ligand_2_ref_atoms: The AMBER style query masks that select the second ligands
            reference atoms.
        receptor_ref_atoms: The AMBER style query mask that selects the receptor atoms
            used to align the ligand.
    """

    prepare_fn = functools.partial(
        _prepare_complex_phase,
        config.complex,
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
    _run_phase(config.complex, prepare_fn, output_dir, report_dir)


def _create_run_flags(
    structure: femto.fe.inputs.Structure | None, prefix: str
) -> list[str]:
    """Create flags for the run CLI from an input structure containing paths.

    Args:
        structure: The input structure.
        prefix: The flag prefix (e.g. ``"ligand-1"`` yields ``"--ligand-1-coords"``).

    Returns:
        The flags.
    """

    if structure is None:
        return []

    ref_atoms = structure.metadata.get("ref_atoms", None)

    return [
        f"--{prefix}-coords={structure.coords}",
        *(
            []
            if structure.params is None
            else [f"--{prefix}-params={structure.params}"]
        ),
        *([] if (ref_atoms is None) else [f"--{prefix}-ref-atoms", *ref_atoms]),
    ]


def submit_network(
    config: "femto.fe.septop.SepTopConfig",
    network: femto.fe.inputs.Network,
    output_dir: pathlib.Path,
    queue_options: femto.fe.utils.queue.SLURMOptions,
    mpi_command: list[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Submits a set of SepTop calculations to an HPC queueing manager.

    Args:
        config: The configuration.
        network: The network of edges to run.
        output_dir: The directory to store any outputs in.
        queue_options: The options to use when submitting the jobs.
        mpi_command: The mpi runner command to use. The default is
            ``"srun --mpi=pmix"``.

    Returns:
        The ids of the submitted jobs.
    """

    mpi_command = mpi_command if mpi_command is not None else ["srun", "--mpi=pmix"]

    output_dir.mkdir(exist_ok=True, parents=True)

    date_str = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    config_path = output_dir / f"config-{date_str}.yaml"
    config_path.write_text(config.model_dump_yaml(sort_keys=False))

    femto_command = ["femto", "septop", "--config", config_path]

    slurm_job_ids = []

    for edge in network.edges:
        edge_dir = output_dir / f"{edge.ligand_1.name}~{edge.ligand_2.name}"

        complex_output_dir = edge_dir / "complex"
        solution_output_dir = edge_dir / "solution"

        ligand_args = [
            *_create_run_flags(edge.ligand_1, "ligand-1"),
            *_create_run_flags(edge.ligand_2, "ligand-2"),
        ]

        run_solution_id = femto.fe.utils.queue.submit_slurm_job(
            [
                *mpi_command,
                *femto_command,
                "run-solution",
                *ligand_args,
                f"--output-dir={solution_output_dir}",
                f"--report-dir={solution_output_dir}",
            ],
            queue_options,
            edge_dir / f"run-solution-{date_str}.out",
        )
        run_complex_id = femto.fe.utils.queue.submit_slurm_job(
            [
                *mpi_command,
                *femto_command,
                "run-complex",
                *_create_run_flags(network.receptor, "receptor"),
                *ligand_args,
                f"--output-dir={complex_output_dir}",
                f"--report-dir={complex_output_dir}",
            ],
            queue_options,
            edge_dir / f"run-complex-{date_str}.out",
        )

        analyze_id = femto.fe.utils.queue.submit_slurm_job(
            [
                *femto_command,
                "analyze",
                "--complex-samples",
                complex_output_dir / "_sample/samples.arrow",
                "--complex-system",
                complex_output_dir / "_setup/system.xml",
                "--solution-samples",
                solution_output_dir / "_sample/samples.arrow",
                "--solution-system",
                solution_output_dir / "_setup/system.xml",
                "--output",
                edge_dir / "ddg.csv",
            ],
            queue_options,
            edge_dir / f"analyze-{date_str}.out",
            [run_solution_id, run_complex_id],
        )

        slurm_job_ids.append((run_solution_id, run_complex_id, analyze_id))

    return slurm_job_ids
