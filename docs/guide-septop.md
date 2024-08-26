# SepTop

The Seperated Topology (SepTop) method, unlike traditional FEP methods and similar to ATM, does not require setting up
any kind of hybrid topology [which can be tricky to do rigorously](https://pubs.acs.org/doi/10.1021/acs.jctc.0c01328).
It proceeds instead by applying light restraints in the complex and solution phases to correctly orientate the ligands.
Unlike ATM which can be run in a single shot, it currently requires computing the free energy of the ligand in the bound
state, and the free energy of the ligand in the solution.

It is highly recommended to read the [original](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3598757/) and
[updated](https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c00282) SepTop publications for a more detailed description of
the method. The implementation here was also inspired by the SepTop implementation implemented by
[Baumann et al](https://github.com/MobleyLab/SeparatedTopologies).

Here the general approach and python API for running SepTop calculations with `femto` is described. See the [overview](guide-fe.md)
for instructions on running using the CLI.

???+ tip
    This guide will walk through setting up all calculations serially, however `femto` supports parallelisation using
    MPI. See [femto.fe.septop.run_complex_phase][], [femto.fe.septop.run_solution_phase][] and their respective sources for MPI ready
    runners / example implementations.

## Procedure

The implementation of SepTop in `femto` generally splits the setup and running of calculations into those required by
the complex phase, and those required by the solution phase.

Both phases however follow roughly the same steps, which include building and solvating the systems from the
pre-prepared receptor and ligand structures, running equilibration simulation at each lambda state, running Hamiltonian
replica exchange (HREMD) between each window, and finally computing the final free energy using MBAR or UWHAM.

The full procedure is configured using the [femto.fe.septop.SepTopConfig][] class:

```python
import femto.fe.septop

config = femto.fe.septop.SepTopConfig()
```

and partitions the options into those [complex][femto.fe.septop.SepTopConfig.complex] and those for the
[solution][femto.fe.septop.SepTopConfig.solution] phase.

### Setup Complex

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
    tleap_sources=config.complex.setup.solvent.tleap_sources,
)
```

The ligand and receptor 'reference' atoms (i.e. those that will be used for the Boresch style restraints used to align
the ligands) can be optionally specified:

```python
ligand_1_ref_query = ["...", "...", "..."]  # OR None
ligand_2_ref_query = ["...", "...", "..."]  # OR None

receiver_ref_query = ["...", "...", "..."]  # OR None
```

If not specified, these will be automatically selected. By default (`ligand_method='baumann'`), this follows the
procedure described in the [Baumann et al. publication](https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c00282).

???+ note
    Currently the selection procedure only considers a single configuration of the complex rather than a trajectory,
    and so does not include the variance criteria.

The full complex structure (ParmEd) and OpenMM system can then be created:

```python
complex_structure, complex_system = femto.fe.septop.setup_complex(
    config.complex.setup,
    receptor,
    ligand_1,
    ligand_2,
    receptor_ref_query,
    ligand_1_ref_query,
    ligand_2_ref_query,
)
```

At this point, the system object will contain the fully parameterized system, and a Boresch style restraint on each
ligand. If HMR or REST2 were enabled in the config, the system will also contain the appropriate modifications enabling
these.

???+ tip
    The solvated system can be saved for easier inspection and checkpointing:

    ```python
    complex_structure.save("system.pdb")
    ```

### Setup Solution

The solution phase calculation follows the 'two ligands separated by a distance restraint' approach outlined by
Baumann _et al_, rather than performing two absolute hydration free energy (HFE) calculations. This is mostly as it
makes it easier to horizontally scale edge calculations independantly, but the framework does very much contain the
machinery to perform absolute HFE calculations if desired.

The setup procedure is responsible for combining the ligand(s) and at a fixed distance, solvating,
selecting any reference atoms if not provided, applying restraints, and finally creating the main OpenMM system.


```python
import femto.fe.septop

solution_structure, solution_system = femto.fe.septop.setup_solution(
    config.solution.setup,
    ligand_1,
    ligand_2,
    ligand_1_ref_query,
    ligand_2_ref_query,
)
```

The distance restraint between the two ligands is applied between the first reference atom of each ligand.

At this point, the system object will contain the fully parameterized system, and a distance restraint between
ligands. If HMR or REST2 were enabled in the config, the system will also contain the appropriate modifications enabling
these.

### Equilibration

The equilibration procedure is reasonably flexible, and is almost fully specified by the [configuration][femto.fe.septop.SepTopEquilibrateStage].
It comprises the list of 'stages' to run sequentially _for each state_, including minimization, temperature annealing,
and either NVT or NPT MD simulation.

The default procedure is to:

1. energy minimize the system
2. anneal the temperature from 50 K to 300 K over 100 ps
3. run 300 ps of NPT simulation at 300 K

where at each stage light flat bottom position restraints are applied to any protein and ligand atoms.

=== "Complex"

    ```python
    import femto.md.constants

    coords = femto.fe.septop.equilibrate_states(
        complex_system,
        complex_structure,
        config.complex.states,
        config.complex.equilibrate,
        femto.md.constants.OpenMMPlatform.CUDA,
        reporter=None
    )
    ```

=== "Solution"

    ```python
    import femto.md.constants

    coords = femto.fe.septop.equilibrate_states(
        solution_system,
        solution_structure,
        config.solution.states,
        config.solution.equilibrate,
        femto.md.constants.OpenMMPlatform.CUDA,
        reporter=None
    )
    ```

The results are returned as a list of OpenMM `State` objects.

### HREMD

The HREMD procedure is also reasonably flexible, and is almost fully specified by the [configuration][femto.fe.septop.SepTopSamplingStage].
By default, each step is propagated for 600 ps as a warmup phase. All statistics collected during the warmup are
discarded.

Following this, 10ns of replica exchange is performed with a timestep of 4 fs, with exchanges attempted every 4 ps.

=== "Complex"

    ```python
    import femto.md.reporting

    reporter = femto.md.reporting.TensorboardReporter(output_dir)  # OR None
    config.sample.analysis_interval = 10

    femto.fe.septop.run_hremd(
        complex_system,
        complex_structure,
        coords,
        config.complex.states,
        config.complex.sample,
        femto.md.constants.OpenMMPlatform.CUDA,
        output_dir / "complex",
        reporter
    )
    ```

=== "Solution"

    ```python
    import femto.md.reporting

    reporter = femto.md.reporting.TensorboardReporter(output_dir)  # OR None
    config.sample.analysis_interval = 10

    femto.fe.septop.run_hremd(
        solution_system,
        solution_structure,
        coords,
        config.solution.states,
        config.solution.sample,
        femto.md.constants.OpenMMPlatform.CUDA,
        output_dir / "solution",
        reporter
    )
    ```

We have passed an optional [reporter][femto.md.reporting.TensorboardReporter] to the `run_hremd` function. As the name
suggests, this reporter will log statistics to a TensorBoard run file in the output directory. At present, these
include online estimates of the total free energy and the free energy of each 'leg'.

The statistics collected such as reduced potentials and acceptance rates are stored in a specified output directory
as a convenient Arrow parquet file (`"eralpha/outputs-septop/<phase>/samples.arrow"`).

### Analysis

The final step is to compute the free energy using MBAR or UWHAM. This can be easily done using the [femto.fe.septop.compute_ddg][]
convience function:

```python
import femto.fe.ddg
import femto.fe.septop

u_kn_complex, n_k_complex = femto.fe.ddg.load_u_kn(
    output_dir / "complex/samples.arrow"
)
u_kn_solution, n_k_solution = femto.fe.ddg.load_u_kn(
    output_dir / "solution/samples.arrow"
)

ddg_df = femto.fe.septop.compute_ddg(
    config,
    u_kn_complex,
    n_k_complex,
    complex_system,
    u_kn_solution,
    n_k_solution,
    solution_system
)
print(ddg_df)
```

## Edges

The SepTop CLI provides an option to specify [a YAML file containing edge data][femto.fe.config.Network]. At present, this
file can only be used to specify the edges that need to be computed:

```yaml
edges:
  - {ligand_1: 2d, ligand_2: 2e}
  - {ligand_1: 2d, ligand_2: 3a}
  - ...
```

where each name should match the name of a subdirectory in your `forcefield` directory.
