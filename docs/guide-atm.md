# ATM

The Alchemical Transfer Method (ATM), unlike traditional FEP methods, does not require setting up any kind of hybrid
topology, or even alchemically transforming the ligand. Instead, ligands are 'translated' along a displacement vector
from the binding site to the bulk solvent and vice-versa. Not only does this greatly simplify the setup, but also easily
allows for computing the binding free energy between two ligands that do not share a common core.

It is highly recommended to read the original [ABFE](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00266) and
[RBFE](https://pubs.acs.org/doi/full/10.1021/acs.jcim.1c01129) ATM publications for a more detailed description of
the method, as well as the fantastic documentation [being put together by the Gallicchio lab](https://www.compmolbiophysbc.org/atom-openmm)

Here the general approach and python API for running ATM calculations with `femto` is described. See the [overview](guide-fe.md)
for instructions on running using the CLI.

???+ tip
    This guide will walk through setting up all calculations serially, however `femto` supports parallelisation using
    MPI. See [femto.fe.atm.run_workflow][] and its source for an MPI ready runner / example implementation.

## Procedure

The implementation of ATM in `femto` generally proceeds by setting up the complex in solvation from the pre-prepared
receptor and ligand structures, running equilibration simulation at each lambda state, running Hamiltonian replica
exchange (HREMD) between each window, and finally computing the final free energy using MBAR or UWHAM.

The full procedure is configured using the [femto.fe.atm.ATMConfig][] class:

```python
import femto.fe.atm

config = femto.fe.atm.ATMConfig()
```

### Setup

The setup procedure is responsible for combining the ligand(s) and receptor structures into a single complex, solvating
it, selecting any reference atoms if not provided, applying restraints, and finally creating the main OpenMM system.

It begins with the already prepared (correct binding pose, parameterized, etc.) structures of the ligand(s) and
receptor:

```python
import pathlib

import femto.md.system

eralpha_dir = pathlib.Path("eralpha")

ligand_1, ligand_2 = femto.md.system.load_ligands(
    ligand_1_coords=eralpha_dir / "forcefield/2d/vacuum.mol2",
    ligand_1_params=eralpha_dir / "forcefield/2d/vacuum.parm7",
    ligand_2_coords=eralpha_dir / "forcefield/2e/vacuum.mol2",
    ligand_2_params=eralpha_dir / "forcefield/2e/vacuum.parm7",
)
receptor = femto.md.system.load_receptor(
    coord_path=eralpha_dir / "proteins/eralpha/protein.pdb",
    param_path=None,
    tleap_sources=config.setup.solvent.tleap_sources,
)
```

The ligands must **both** be in reasonable binding poses. The setup code will handle translating the ligands along the
displacement vector as required.

The vector along which the ligands will be translated must also be defined. `femto` provides an
[experimental utility][femto.fe.atm.select_displacement] to automatically compute this:

```python
displacement = femto.fe.atm.select_displacement(
    receptor, ligand_1, ligand_2, config.setup.displacement
)
```

At present, it will attempt to place the ligands in each corner of the simulation box at a pre-specified distance from
the binding site, and select the vector that maximizes the distance between the ligand and receptor atoms.

???+ warning
    This method is still experimental and may not always select the optimal vector.

When running RBFE calculations, the ligand 'reference' atoms (i.e. those that will be used for the alignment restraint)
can be optionally specified:

```python
ligand_1_ref_query = ["@7", "@11", "@23"]  # OR None
ligand_2_ref_query = ["@12", "@7", "@21"]  # OR None
```

If not specified, these will be automatically selected. By default (`ligand_method='chen'`), the distance between each
atom in the first ligand and each atom in the second ligand is calculated. Pairs of atoms with the smallest distance
will be greedily selected, ignoring pairs that would lead to all atoms being co-linear.

???+ warning
    This method is still experimental and may not always select the optimal reference atoms.

Similarly, the receptor atoms that define the binding site can be optionally specified:

```python
receptor_ref_query = [
    ":36,39,40,42,43,77,80,81,84,95,97,111,113,114,117,118,214,215,217,218 & @CA"
]  # OR None
```

If not specified, these will be automatically selected. By default, these will include all alpha carbons within a
[defined cutoff][femto.fe.atm.ATMReferenceSelection.receptor_cutoff] from the ligand.

The full complex structure (ParmEd) and OpenMM system can then be created:

```python
complex_structure, complex_system = femto.fe.atm.setup_system(
    config.setup,
    receptor,
    ligand_1,
    ligand_2,
    displacement,
    receptor_ref_query,
    ligand_1_ref_query,
    ligand_2_ref_query,
)
```

At this point, the system object will contain the fully parameterized system, a [center-of-mass restraint][femto.fe.atm.ATMRestraints.com]
on each ligand, [an 'alignment' restraint][femto.fe.atm.ATMRestraints.alignment] between the two ligands, and a
light [position restraint][femto.fe.atm.ATMRestraints.receptor] on (by default) the
[alpha carbons][femto.fe.atm.ATMRestraints.receptor_query] of the receptor. When solvating, a cavity will be created
where the ligands will be translated to.

If HMR or REST2 were enabled in the config, the system will also contain the appropriate modifications enabling these.

???+ tip
    The solvated system can be saved for easier inspection and checkpointing:

    ```python
    complex_structure.save("system.pdb")
    ```

### Equilibration

The equilibration procedure is reasonably flexible, and is almost fully specified by the [configuration][femto.fe.atm.ATMEquilibrateStage].
It comprises the list of 'stages' to run sequentially _for each state_, including minimization, temperature annealing,
and either NVT or NPT MD simulation.

The default procedure is to:

1. energy minimize the system
2. anneal the temperature from 50 K to 300 K over 100 ps
3. run 300 ps of NPT simulation at 300 K
4. set the box vectors to the maximum across all states as HREMD will be run at constant volume

where at each stage light flat bottom position restraints are applied to any protein and ligand atoms.

The equilibration across all states can be run using [femto.fe.atm.equilibrate_states][]:

```python
import femto.md.constants

coords = femto.fe.atm.equilibrate_states(
    complex_system,
    complex_structure,
    config.states,
    config.equilibrate,
    displacement,
    femto.md.constants.OpenMMPlatform.CUDA,
    reporter=None
)
```

The results are returned as a list of OpenMM `State` objects.

### HREMD

The HREMD procedure is also reasonably flexible, and is almost fully specified by the [configuration][femto.fe.atm.ATMSamplingStage].
By default, each step is propagated for 600 ps as a warmup phase. All statistics collected during the warmup are
discarded.

Following this, 10ns of replica exchange is performed with a timestep of 4 fs, with exchanges attempted every 4 ps.

```python
import femto.md.reporting

output_dir = pathlib.Path("eralpha/outputs-atm")

reporter = femto.md.reporting.TensorboardReporter(output_dir)  # OR None
config.sample.analysis_interval = 10

femto.fe.atm.run_hremd(
    complex_system,
    complex_structure,
    coords,
    config.states,
    config.sample,
    displacement,
    femto.md.constants.OpenMMPlatform.CUDA,
    output_dir,
    reporter
)
```

We have passed an optional [reporter][femto.md.reporting.TensorboardReporter] to the `run_hremd` function. As the name
suggests, this reporter will log statistics to a TensorBoard run file in the output directory. At present, these
include online estimates of the total free energy and the free energy of each 'leg'.

The statistics collected such as reduced potentials and acceptance rates are stored in a specified output directory
as a convenient Arrow parquet file (`"eralpha/outputs-atm/samples.arrow"`).

### Analysis

The final step is to compute the free energy using MBAR or UWHAM. This can be easily done using the [femto.fe.atm.compute_ddg][]
convience function:

```python
import femto.fe.ddg

u_kn, n_k = femto.fe.ddg.load_u_kn(output_dir/ "samples.arrow")

ddg_df = femto.fe.atm.compute_ddg(config.sample, config.states, u_kn, n_k)
print(ddg_df)
```

`femto` also offers lower level methods for computing the free energy, which can be useful for debugging or if you
want to compute things like overlap matrices:

```python
import femto.fe.ddg

n_states_leg_1 = config.states.direction.index(-1)
n_states_leg_2 = len(config.states.lambda_1) - n_states_leg_1

state_groups = [(n_states_leg_1, 1.0), (n_states_leg_2, 1.0)]

estimated, overlap = femto.fe.ddg.estimate_ddg(
    u_kn, n_k, config.sample.temperature, state_groups
)
```

## Edges

The ATM CLI provides an option to specify [a YAML file containing edge data][femto.fe.atm.ATMNetwork]. This file not only
includes the edges that need to be computed, but can also optionally specify the 'reference' atoms for each ligand and
the receptor atoms that form the binding cavity:

```yaml
receptor: eralpha
receptor_ref_query: ":36,39,40,42,43,77,80,81,84,95,97,111,113,114,117,118,214,215,217,218 & @CA"

edges:
  - {ligand_1: 2d, ligand_2: 2e, ligand_1_ref_atoms: ["@7", "@11", "@23"], ligand_2_ref_atoms: ["@12", "@7", "@21"]}
  - {ligand_1: 2d, ligand_2: 3a, ligand_1_ref_atoms: ["@7", "@11", "@23"], ligand_2_ref_atoms: ["@15", "@10", "@5"]}
  - ...
```

The `receptor` field optionally specifies the name of the receptor, and should match the name of a subdirectory in your
`proteins` directory. The optional `receptor_ref_query` field defines the AMBER style selection mask to use to manually
select the receptor atoms that define the binding site. If unspecified, such atoms will be selected automatically as
described above.

At minimum, the `edges` field must define pairs of ligands to compute the binding free energy between:

```yaml
edges:
  - {ligand_1: 2d, ligand_2: 2e}
  - {ligand_1: 2d, ligand_2: 3a}
  - ...
```

where each name should match the name of a subdirectory in your `forcefield` directory. In this case where no reference
atoms are specified, the reference atoms will be selected [based on the configuration][femto.fe.atm.ATMReferenceSelection]
as defined above.
