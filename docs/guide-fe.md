# Free Energy

`femto` provides a simple interface for estimating relative and absolute binding free energies using different methods
(currently ATM and SepTop). If offers a full Python API as well as a convenient command-line interface.

## Preparing the Inputs

In general, the same set of inputs can be used for all-methods supported by this framework; these include
pre-parameterized and docked ligands, and pre-prepared protein structures.

The easiest way to prepare the inputs for `femto`, especially when using the CLI, is to structure them in the
'standard directory structure' expected by the framework:

```text
.
├─ forcefield/
│  ├─ <ligand 1>/
│  │  ├─ vacuum.mol2
│  │  └─ vacuum.xml
│  └─ ...
├─ proteins/
│  └─ <target>/
│     └─ protein.pdb
├─ config.yaml
└─ edges.yaml
```

In particular, it should contain:

<table>
    <thead>
        <tr>
            <th></th>
            <th>Contents</th>
        </tr>
    </thead>
    <tr>
        <td><b>forcefield</b></td>
        <td>A subdirectory for each ligand of interest, which must include:
            <ul>
                <li><code>vacuum.[mol2,sdf]</code>: The ligand file. The ligand should already be in the correct docked pose, with the correct protonation state and tautomeric form.</li>
                <li><code>vacuum.[xml,parm7]</code> (optional): The parameter file for the ligand.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td><b>proteins</b></td>
        <td>A <em>single</em> subdirectory named after the protein target, which must include:
            <ul>
                <li><code>protein.[pdb,mol2,sdf]</code>: A file containing the target protein and any crystallographic waters in the correct pose.</li>
                <li><code>protein.[xml,parm7]</code> (optional): The parameter file for the protein.</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td><b>config.yaml</b> (optional)</td>
        <td>A YAML file containing configuration settings, such as the equilibration protocol, lambda states, and HREMD settings.</td>
    </tr>
    <tr>
        <td><b>edges.yaml</b> or <b>Morph.in</b></td>
        <td>These files define the edges (ligand pairs) for which you want to compute the binding free energies.
            <ul>
                <li><code>edges.yaml</code>: A YAML file that specifies which edges to compute. It can include extra and per-edge settings depending on the method, such as masks for manually selecting reference atoms.</li>
                <li><code>Morph.in</code>: A simpler text file format where each line represents an edge in the format <code>&lt;ligand 1&gt;~&lt;ligand 2&gt;</code>.</li>
            </ul>
        </td>
    </tr>
</table>

???+ tip
    See the [`examples`](https://github.com/Psivant/femto/tree/main/examples) directory for examples of this structure.

Most methods define a default configuration that will be used if one isn't specified. These can be accessed using the
`config` command:

```shell
femto atm    config > default-atm-config.yaml
femto septop config > default-septop-config.yaml
```

## Running the Calculations

If running on a SLURM cluster (recommended), `femto` provides helpers for running all edges in parallel. Otherwise,
you can run each edge individually.

### All Edges

If running on a SLURM cluster, all the edges can be run using the `femto <method> submit-workflows` or
`femto <method> submit-replicas` commands:

=== "ATM"

    ```shell
    femto atm         --config              "eralpha/config-atm.yaml"  \
                                                                       \
      submit-replicas  --slurm-nodes         2                         \
                       --slurm-tasks         8                         \
                       --slurm-gpus-per-task 1                         \
                       --slurm-cpus-per-task 4                         \
                       --slurm-partition     "project-gpu"             \
                       --slurm-walltime      "48:00:00"                \
                                                                       \
                       --root-dir            "eralpha"                 \
                       --output-dir          "eralpha/outputs-atm"     \
                       --edges               "eralpha/edges-atm.yaml"  \
                       --n-replicas          5
    ```

=== "SepTop"

    ```shell
    femto septop      --config              "eralpha/config-septop.yaml"  \
                                                                          \
      submit-replicas  --slurm-nodes         5                            \
                       --slurm-tasks         19                           \
                       --slurm-gpus-per-task 1                            \
                       --slurm-cpus-per-task 4                            \
                       --slurm-partition     "project-gpu"                \
                       --slurm-walltime      "48:00:00"                   \
                                                                          \
                       --root-dir            "eralpha"                    \
                       --output-dir          "eralpha/outputs-septop"     \
                       --edges               "eralpha/edges-septop.yaml"  \
                       --n-replicas          5
    ```

???+ tip
    The [`examples`](https://github.com/Psivant/femto/tree/main/examples) directory also contains configuration files
    for enabling REST2 (`config-<method>-rest.yaml`).

The ERα ATM example uses 22 lambda windows (as defined in `"eralpha/config-atm.yaml"`). We have only requested a total
of 8 GPUs here, however. `femto` will split the lambda windows across the available GPUs, and simulate each in turn
until all windows have been simulated. In principle then the calculation can be run using anywhere between 1 and
`n_lambda` GPUs.

???+ note
    If more GPUs are requested than there are lambda windows, the extra GPUs will remain idle.

As calculations finish successfully, the estimated free energies will be written to the output directory as a CSV file
(`eralpha/outputs-<method>/ddg.csv`). If the `analysis_interval` is set in the `sample` section of the config, you
should also see TensorBoard run files being created in the output directory. These can be used to monitor the progress
of the simulations, especially online estimates of the free energies.

### Single Edges

If you want to run a specific edge rather than submitting all edges defined in `"eralpha/edges-<method>.yaml"`, you can
instead run:

=== "ATM"

    ```shell
    srun --mpi=pmix -n <N_LAMBDAS>                        \
                                                          \
    femto atm     --config     "eralpha/config-atm.yaml"  \
                                                          \
      run-workflow --ligand-1   "2d"                      \
                   --ligand-2   "2e"                      \
                   --root-dir   "eralpha"                 \
                   --output-dir "eralpha/outputs-atm"     \
                   --edges       "eralpha/edges-atm.yaml"
    ```

=== "SepTop"

    ```shell
    srun --mpi=pmix -n <N_COMLEX_LAMBDAS>                       \
                                                                \
    femto septop  --config    "eralpha/config-septop.yaml"      \
                                                                \
      run-complex --ligand-1   "2d"                             \
                  --ligand-2   "2e"                             \
                  --root-dir   "eralpha"                        \
                  --output-dir "eralpha/outputs-septop/complex" \
                  --edges      "eralpha/edges-septop.yaml"

    srun --mpi=pmix -n <N_SOLUTION_LAMBDAS>                       \
                                                                  \
    femto septop  --config     "eralpha/config-septop.yaml"       \
                                                                  \
      run-solution --ligand-1   "2d"                              \
                   --ligand-2   "2e"                              \
                   --root-dir   "eralpha"                         \
                   --output-dir "eralpha/outputs-septop/solution" \
                   --edges      "eralpha/edges-septop.yaml"

    femto septop  --config     "eralpha/config-septop.yaml"                            \
                                                                                       \
      analyze --complex-system   eralpha/outputs-septop/complex/_setup/system.xml      \
              --complex-samples  eralpha/outputs-septop/complex/_sample/samples.arrow  \
              --solution-system  eralpha/outputs-septop/solution/_setup/system.xml     \
              --solution-samples eralpha/outputs-septop/solution/_sample/samples.arrow \
              --output           eralpha/outputs-septop/ddg.csv
    ```

or using ``mpirun`` if not running using SLURM.

???+ note
    By default `femto` will try and assign each process on a node to a different GPU based on the local rank and
    `CUDA_VISIBLE_DEVICES`, i.e. `gpu_device = local_rank % len(CUDA_VISIBLE_DEVICES)`. Depending on your setup, you
    may need to create a hostfile to ensure the correct GPUs are used.

Here we have still specified the path to the edges file
(`edges-atm.yaml`) even though we are running a specific edge. This is optional, but will ensure `femto` uses the
reference atoms defined within rather than trying to automatically select them.

The `--report-dir` option can be used to optionally specify a directory to write tensorboard run files to. This can be
useful for monitoring the progress of the simulations, especially online estimates of the free energies.
