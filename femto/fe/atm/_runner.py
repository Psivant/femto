"""Helpers to run ATM calculations."""

import datetime
import logging
import pathlib
import typing

import numpy
import openmm
import parmed
import yaml

import femto.fe.ddg
import femto.fe.inputs
import femto.fe.utils.queue
import femto.md.constants
import femto.md.reporting
import femto.md.system
import femto.md.utils.mpi

if typing.TYPE_CHECKING:
    import femto.fe.atm

_LOGGER = logging.getLogger(__name__)


@femto.md.utils.mpi.run_on_rank_zero
def _prepare_system(
    config: "femto.fe.atm.ATMSetupStage",
    ligand_1_coords: pathlib.Path,
    ligand_1_params: pathlib.Path,
    ligand_2_coords: pathlib.Path | None,
    ligand_2_params: pathlib.Path | None,
    receptor_coords: pathlib.Path,
    receptor_params: pathlib.Path | None,
    displacement: openmm.unit.Quantity | None,
    ligand_1_ref_atoms: tuple[str, str, str],
    ligand_2_ref_atoms: tuple[str, str, str],
    receptor_ref_atoms: str | None,
    output_dir: pathlib.Path,
) -> tuple[parmed.Structure, openmm.System, openmm.unit.Quantity]:
    """Prepare the system for running the ATM method, caching the topology and
    system."""
    import femto.fe.atm._setup

    output_dir.mkdir(exist_ok=True, parents=True)

    topology_path = output_dir / "system.pdb"
    system_path = output_dir / "system.xml"

    displacement_path = output_dir / "displacement.yaml"

    if topology_path.exists() and system_path.exists() and displacement_path.exists():
        topology = parmed.load_file(str(topology_path), structure=True)
        system = openmm.XmlSerializer.deserialize(system_path.read_text())

        displacement = (
            numpy.array(yaml.safe_load(displacement_path.read_text()))
            * openmm.unit.angstrom
        )

        return topology, system, displacement

    receptor = femto.md.system.load_receptor(
        receptor_coords,
        receptor_params,
        config.solvent.tleap_sources,
    )
    ligand_1, ligand_2 = femto.md.system.load_ligands(
        ligand_1_coords, ligand_1_params, ligand_2_coords, ligand_2_params
    )

    if displacement is None and isinstance(config.displacement, openmm.unit.Quantity):
        _LOGGER.info("selecting ligand displacement vector")
        displacement = femto.fe.atm._setup.select_displacement(
            receptor, ligand_1, ligand_2, config.displacement
        )
    elif displacement is None:
        displacement = numpy.array(
            [v.value_in_unit(openmm.unit.angstrom) for v in config.displacement]
        )
        displacement = displacement * openmm.unit.angstrom

    _LOGGER.info(f"ligands will be displaced by {displacement}")

    topology, system = femto.fe.atm._setup.setup_system(
        config,
        receptor,
        ligand_1,
        ligand_2,
        displacement,
        receptor_ref_atoms,
        ligand_1_ref_atoms,
        ligand_2_ref_atoms,
    )

    topology.save(str(topology_path), overwrite=True)
    system_path.write_text(openmm.XmlSerializer.serialize(system))

    displacement_path.write_text(
        yaml.safe_dump(displacement.value_in_unit(openmm.unit.angstrom).tolist())
    )

    return topology, system, displacement


@femto.md.utils.mpi.run_on_rank_zero
def _cache_equilibrate_outputs(coords: list[openmm.State], paths: list[pathlib.Path]):
    for path, coord in zip(paths, coords, strict=True):
        path.write_text(openmm.XmlSerializer.serialize(coord))


@femto.md.utils.mpi.run_on_rank_zero
def _analyze_results(
    config: "femto.fe.atm.ATMConfig",
    sample_dir: pathlib.Path,
    output_path: pathlib.Path,
):
    import femto.fe.atm._analyze

    u_kn, n_k = femto.fe.ddg.load_u_kn(sample_dir / "samples.arrow")

    ddg_frame = femto.fe.atm._analyze.compute_ddg(
        config.sample, config.states, u_kn, n_k
    )
    ddg_frame.to_csv(output_path, index=False)


def run_workflow(
    config: "femto.fe.atm.ATMConfig",
    ligand_1_coords: pathlib.Path,
    ligand_1_params: pathlib.Path,
    ligand_2_coords: pathlib.Path | None,
    ligand_2_params: pathlib.Path | None,
    receptor_coords: pathlib.Path,
    receptor_params: pathlib.Path | None,
    output_dir: pathlib.Path,
    report_dir: pathlib.Path | None = None,
    displacement: openmm.unit.Quantity | None = None,
    ligand_1_ref_atoms: tuple[str, str, str] | None = None,
    ligand_2_ref_atoms: tuple[str, str, str] | None = None,
    receptor_ref_atoms: str | None = None,
):
    """Run the setup, equilibration, and sampling phases.

    Args:
        config: The configuration.
        ligand_1_coords: The path to the first ligand coordinates.
        ligand_1_params: The path to the first ligand parameters.
        ligand_2_coords: The path to the second ligand coordinates.
        ligand_2_params: The path to the second ligand parameters.
        receptor_coords: The path to the receptor coordinates.
        receptor_params: The path to the receptor parameters.
        report_dir: The directory to write any statistics to.
        output_dir: The directory to store all outputs in.
        displacement: The displacement to offset the ligands by.
        ligand_1_ref_atoms: The AMBER style query masks that select the first ligands'
            reference atoms.
        ligand_2_ref_atoms: The AMBER style query masks that select the second ligands'
            reference atoms.
        receptor_ref_atoms: The AMBER style query mask that selects the receptor atoms
            that form the binding site.
    """
    import femto.fe.atm._equilibrate
    import femto.fe.atm._sample

    reporter = (
        femto.md.reporting.NullReporter()
        if report_dir is None or not femto.md.utils.mpi.is_rank_zero()
        else femto.md.reporting.TensorboardReporter(report_dir)
    )

    topology, system, displacement = _prepare_system(
        config.setup,
        ligand_1_coords,
        ligand_1_params,
        ligand_2_coords,
        ligand_2_params,
        receptor_coords,
        receptor_params,
        displacement,
        ligand_1_ref_atoms,
        ligand_2_ref_atoms,
        receptor_ref_atoms,
        output_dir / "_setup",
    )
    topology.symmetry = None  # needed as attr is lost after pickling by MPI

    equilibrate_dir = output_dir / "_equilibrate"
    equilibrate_dir.mkdir(exist_ok=True, parents=True)

    coord_paths = [
        equilibrate_dir / f"state_{state_idx}.xml"
        for state_idx in range(len(config.states.lambda_1))
    ]

    if any(not path.exists() for path in coord_paths):
        coords = femto.fe.atm._equilibrate.equilibrate_states(
            system,
            topology,
            config.states,
            config.equilibrate,
            displacement,
            femto.md.constants.OpenMMPlatform.CUDA,
            reporter,
        )
        _cache_equilibrate_outputs(coords, coord_paths)
    else:
        coords = [
            openmm.XmlSerializer.deserialize(path.read_text()) for path in coord_paths
        ]

    sample_dir = output_dir / "_sample"
    result_path = output_dir / "ddg.csv"

    if not result_path.exists():
        femto.fe.atm._sample.run_hremd(
            system,
            topology,
            coords,
            config.states,
            config.sample,
            displacement,
            femto.md.constants.OpenMMPlatform.CUDA,
            sample_dir,
            reporter,
        )
        _analyze_results(config, sample_dir, result_path)


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
    ref_atoms = [ref_atoms] if isinstance(ref_atoms, str) else ref_atoms

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
    config: "femto.fe.atm.ATMConfig",
    network: femto.fe.inputs.Network,
    output_dir: pathlib.Path,
    queue_options: femto.fe.utils.queue.SLURMOptions,
    mpi_command: list[str] | None = None,
) -> list[str]:
    """Submits a set of ATM calculations to the SLURM queueing manager.

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

    femto_command = [
        "femto",
        "atm",
        f"--config={config_path}",
        "run-workflow",
    ]

    slurm_job_ids = []

    for edge in network.edges:
        edge_dir = output_dir / f"{edge.ligand_1.name}~{edge.ligand_2.name}"

        job_id = femto.fe.utils.queue.submit_slurm_job(
            [
                *mpi_command,
                *femto_command,
                *_create_run_flags(network.receptor, "receptor"),
                *_create_run_flags(edge.ligand_1, "ligand-1"),
                *_create_run_flags(edge.ligand_2, "ligand-2"),
                f"--output-dir={edge_dir}",
                f"--report-dir={edge_dir}",
            ],
            queue_options,
            edge_dir / f"run-{date_str}.out",
        )

        slurm_job_ids.append(job_id)

    return slurm_job_ids
