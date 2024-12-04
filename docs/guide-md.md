# Molecular Dynamics

The `femto.md` module exposes a number of utilities for setting up and running MD simulations. These include solvating the
system, applying hydrogen mass repartitioning (HMR), preparing a system for REST2 sampling, and running Hamiltonian
replica exchange MD (HREMD) sampling across multiple processes.

## Preparing a System

Most utilities within the framework in general expected an OpenMM `System` object and a [femto.top.Topology][]. While these
can be loaded and generated from a variety of sources, the framework provides built-in utilities for loading
ligands from MOL2 and SDF files and 'receptors' (e.g. proteins with crystallographic waters) from PDB, MOL2 and SDF files.

Single ligands can be easily loaded using [femto.md.prepare.load_ligand][]

```python
import pathlib

import femto.md.constants
import femto.md.prepare

eralpha_dir = pathlib.Path("eralpha")

ligand = femto.md.prepare.load_ligand(
    eralpha_dir / "forcefield/2d/vacuum.mol2",
    residue_name=femto.md.constants.LIGAND_1_RESIDUE_NAME
)
```

while two ligands (e.g. for use in an RBFE calculation) can be loaded using [femto.md.prepare.load_ligands][]

```python
ligand_1, ligand_2 = femto.md.prepare.load_ligands(
    eralpha_dir / "forcefield/2d/vacuum.mol2",
    eralpha_dir / "forcefield/2e/vacuum.mol2",
)
```

in the latter case the ligands will have their residue names overwritten to [femto.md.constants.LIGAND_1_RESIDUE_NAME][]
and [femto.md.constants.LIGAND_2_RESIDUE_NAME][] respectively.

No modifications will be made to the ligands, so they should already be in the correct protonation state and tautomeric
form of interest.

The 'receptor' (namely anything that can be stored in a PDB file such as a protein and
crystallographic waters, or something pre-parameterized using a 'host' molecule) can be loaded using
[femto.md.prepare.load_receptor][].

Either the parameters should be explicitly specified:

```python
temoa_dir = pathlib.Path("temoa")

receptor = femto.md.prepare.load_receptor(
    temoa_dir / "host.mol2",
)
```

### Prepare the System

Once the ligand and / or receptor have been loaded, they can be solvated and parameterized using [femto.md.prepare.prepare_system][].
This step also includes neutralizing the system with counter ions, as well as optionally adding a salt concentration.

```python
import openmm.unit

import femto.md.prepare

solvent_config = femto.md.config.Prepare(
    ionic_strength=0.15 * openmm.unit.molar,
    neutralize=True,
    cation="Na+",
    anion="Cl-",
    water_model="tip3p",
    box_padding=10.0 * openmm.unit.angstrom,
)

topology, system = femto.md.prepare.prepare_system(
    receptor=receptor,  # or None if no receptor
    ligand_1=ligand_1,
    ligand_2=None,      # or `ligand_2` if setting up FEP for example
    solvent=solvent_config,
)
```

By default, an OpenFF force field will be used to parameterize the ligands / any cofactors. The
exact force field can be specified in the [femto.md.config.Prepare][] configuration.

If the ligands / receptor has already been parameterized, the OpenMM XML or AMBER prmtop files can additionally be
specified:

```python
extra_params = [
    eralpha_dir / "forcefield/2d/vacuum.parm7",
    eralpha_dir / "forcefield/2e/vacuum.parm7",
]

topology, system = femto.md.prepare.prepare_system(
    receptor=receptor,  # or None if no receptor
    ligand_1=ligand_1,
    ligand_2=None,      # or `ligand_2` if setting up FEP for example
    solvent=solvent_config,
    extra_params=extra_params
)
```

### HMR and REST2

HMR can be applied to the system using [femto.md.prepare.apply_hmr][]:

```python
femto.md.prepare.apply_hmr(system, topology)
```

This modifies the system in-place.

Similarly, the system can be prepared for REST2 sampling using [femto.md.rest.apply_rest][]:

```python
import femto.md.rest

rest_config = femto.md.config.REST(scale_torsions=True, scale_nonbonded=True)

solute_idxs = topology.select(f"resn {femto.md.constants.LIGAND_1_RESIDUE_NAME}")
femto.md.rest.apply_rest(system, solute_idxs, rest_config)
```

Currently only the torsions and non-bonded interactions (electrostatic and vdW) can be scaled, but this may be extended
in the future. Again, this modifies the system in-place.

???+ warning
    Any alchemical modifications to the system (e.g. using [femto.fe.fep.apply_fep][]) should be applied *before* trying
    to apply REST2.

REST2 is implemented by introducing global context parameters that represent $\frac{\beta_m}{\beta_0}$ and
$\sqrt{\frac{\beta_m}{\beta_0}}$ which can easily be set and modified on an OpenMM `Context`.

???+ tip
    See [femto.md.rest.REST_CTX_PARAM][] and [femto.md.rest.REST_CTX_PARAM_SQRT][] for the names of these parameters, and
    later on in this guide for convenience functions for setting and modifying them.

### Saving the System

The prepared inputs are most easily stored as a coordinate file and an OpenMM XML system file:

```python
import openmm

topology.to_file("system.pdb")
pathlib.Path("system.xml").write_text(openmm.XmlSerializer.serialize(system))
```

## Running MD

The `femto.md.simulate` modules provide convenience functions for simulating prepared systems. This includes chaining
together multiple 'stages' such as minimization, anealing, and molecular dynamics.

The simulation protocol is defined as a list of 'stage' configurations:

```python
import openmm.unit

import femto.md.simulate

kcal_per_mol = openmm.unit.kilocalorie_per_mole
angstrom = openmm.unit.angstrom

temperature = 300.0 * openmm.unit.kelvin

ligand_mask = f":{femto.md.constants.LIGAND_1_RESIDUE_NAME}"

restraints = {
    # each key should be an Amber style selection mask that defines which
    # atoms in the system should be restrained
    ligand_mask: femto.md.config.FlatBottomRestraint(
        k=25.0 * kcal_per_mol / angstrom**2, radius=1.5 * angstrom
    )
}

stages = [
    femto.md.config.Minimization(restraints=restraints),
    femto.md.config.Anneal(
        integrator=femto.md.config.LangevinIntegrator(
            timestep=1.0 * openmm.unit.femtosecond,
        ),
        restraints=restraints,
        temperature_initial=50.0 * openmm.unit.kelvin,
        temperature_final=temperature,
        n_steps=50000,
        # the frequency, in number of steps, with which to increase
        # the temperature
        frequency=100,
    ),
    femto.md.config.Simulation(
        integrator=femto.md.config.LangevinIntegrator(
            timestep=1.0 * openmm.unit.femtosecond,
        ),
        restraints=restraints,
        temperature=temperature,
        pressure=None,
        n_steps=50000,
    ),
    femto.md.config.Simulation(
        integrator=femto.md.config.LangevinIntegrator(
            timestep=4.0 * openmm.unit.femtosecond,
        ),
        temperature=temperature,
        pressure=1.0 * openmm.unit.bar,
        n_steps=150000,
    )
]
```

The restraints dictionary is optional, but can be used to place position restraints on atoms during the equilibration
stages. The reference positions for the restraints are taken as the output from the previous stage, or the inital
positions if it is the first stage.

The [femto.md.simulate.simulate_state][] function can then be used to run each stage sequentially:

```python
import femto.md.simulate

state = {femto.md.rest.REST_CTX_PARAM: 1.0}

final_coords = femto.md.simulate.simulate_state(
    system, topology, state, stages, femto.md.constants.OpenMMPlatform.CUDA
)
```

The initial coordinates and box vectors are taken from the `topology` object.

The `state` dictionary is used to set OpenMM global context parameters. If your system does not use any global context
parameters (e.g. it hasn't been prepared for REST2), or your happy to use the defaults that were set, then you can
simply pass an empty dictionary.

???+ note
    By default the REST context parameters are set to 1.0, i.e., there is no scaling, but we set it explicitly here as
    an example.

You may notice here that we have only set $\frac{\beta_m}{\beta_0}$ and not $\sqrt{\frac{\beta_m}{\beta_0}}$ even though
both are 'required'. The `simulate_state` will automatically set $\sqrt{\frac{\beta_m}{\beta_0}}$ based on the value
of $\frac{\beta_m}{\beta_0}$. See [femto.md.utils.openmm.evaluate_ctx_parameters][] for more details.

## Running HREMD

Hamiltonian replica exchange MD (HREMD) can be run using the [femto.md.hremd][] module. It expects a system that
has been prepared to expose global context parameters. These commonly include parameters for alchemically
scaling the vdW and electrostatic interactions, as well as parameters for REST2 sampling.

Each individual 'replica' (i.e. a simulation run at a given state as defined by a set of global context parameters)
can either be run in a single process, or in parallel across multiple processes using MPI. In the case of the former,
each state is run sequentially prior to proposing swaps, while in the latter case states are run in parallel.

???+ note
    When running in parallel, the number of processes does not need to match the number of states. In this case, each
    process will be assigned a subset of states to run sequentially.

### Running in a Single Process

The [femto.md.hremd.run_hremd][] function can be used directly as part of another script if running HREMD in a single
process:

```python
import pathlib

import openmm.unit

import femto.md.config
import femto.md.constants
import femto.md.hremd
import femto.md.utils.openmm
import femto.md.rest

output_dir = pathlib.Path("hremd-outputs")

# define the REST2 temperatures to sample at
rest_temperatures = [300.0, 310.0, 320.0] * openmm.unit.kelvin
rest_betas = [
    1.0 / (openmm.unit.MOLAR_GAS_CONSTANT_R * rest_temperature)
    for rest_temperature in rest_temperatures
]

states = [
    {femto.md.rest.REST_CTX_PARAM: rest_beta / rest_betas[0]}
    for rest_beta in rest_betas
]
# REST requires both beta_m / beta_0 and sqrt(beta_m / beta_0) to be defined
# we can use a helper to compute the later from the former for each state
states = [
    femto.md.utils.openmm.evaluate_ctx_parameters(state, system)
    for state in states
]

# create the OpenMM simulation object
intergrator_config = femto.md.config.LangevinIntegrator(
    timestep=2.0 * openmm.unit.femtosecond,
)
integrator = femto.md.utils.openmm.create_integrator(
    intergrator_config, rest_temperatures[0]
)

simulation = femto.md.utils.openmm.create_simulation(
    system,
    topology,
    final_coords,  # or None to use the coordinates / box in topology
    integrator=integrator,
    state=states[0],
    platform=femto.md.constants.OpenMMPlatform.CUDA,
)

# define how the HREMD should be run
hremd_config = femto.md.config.HREMD(
    # the number of steps to run each replica for before starting to
    # propose swaps
    n_warmup_steps=150000,
    # the number of steps to run before proposing swaps
    n_steps_per_cycle=500,
    # the number of 'swaps' to propose - the total simulation length
    # will be n_warmup_steps + n_steps * n_cycles
    n_cycles=2000,
    # the frequency with which to store trajectories of each replica.
    # set to None to not store trajectories
    trajectory_interval=10  # store every 10 * 500 steps.
)
femto.md.hremd.run_hremd(
    simulation,
    states,
    hremd_config,
    # the directory to store sampled reduced potentials and trajectories to
    output_dir=output_dir
)
```

If successful, you should see a `hremd-outputs/samples.arrow` file being written to and a `hremd-outputs/trajectories`
directory being created. The former contains the reduced potentials for each replica at each cycle, as well as
statistics such as the number of swaps proposed and accepted.

```python
import pyarrow

with pyarrow.OSFile("hremd-outputs/samples.arrow", "rb") as file:
    with pyarrow.RecordBatchStreamReader(file) as reader:
        output_table = reader.read_all()

print("HREMD Schema:", output_table.schema, flush=True)
print("HREMD Data:", flush=True)

print(output_table.to_pandas().head(), flush=True)
```

???+ tip
    See also the [femto.fe.ddg.load_u_kn][] utility for extracting decorrelated reduced potentials in a form that can be
    used with `pymbar`

### Running in Parallel

The above snipped for running HREMD in a single process can be easily modified to run in parallel across multiple
processes using MPI. The main difference is that

1. the snippet should be saved as a standalone script
2. the following should (optionally) be added to the beginning of the script:

```python
import femto.md.utils.mpi

femto.md.utils.mpi.divide_gpus()
```

This optional extra will attempt to crudely set the visible CUDA devices based on the current MPI rank. The script can
then be run using `mpirun` (or `srun` etc.) as normal.
