<h1 align="center">Free Energy and MD Toolkit using OpenMM</h1>

<p align="center">A comprehensive toolkit for predicting free energies</p>

<p align="center">
  <a href="https://github.com/psivant/femto/actions?query=workflow%3Aci">
    <img alt="ci" src="https://github.com/Psivant/femto/actions/workflows/ci.yaml/badge.svg" />
  </a>
  <a href="https://codecov.io/gh/psivant/femto/branch/main">
    <img alt="coverage" src="https://codecov.io/gh/psivant/femto/branch/main/graph/badge.svg" />
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="license" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>

---

The `femto` framework aims to offer not only a compact and comprehensive toolkit for predicting binding free energies
using methods including `ATM` and `SepTop`, but also a full suite of utilities for running advanced simulations using
OpenMM, including support for HREMD and REST2.

_**Warning:** From version 0.3.0 onwards, the codebase was re-written to completely remove the dependency on `parmed`,
allowing easy use of any force field parameters in OpenFF, Amber, and OpenMM FFXML formats. This re-write also introduced
a number of neccessary API changes. See the [migration guide for more details](https://psivant.github.io/femto/latest/migration/)._

_Further, the default protocols selected for the ATM and SepTop methods are still being tested and optimized, and may
not be optimal. It is recommended that you run a few test calculations to ensure that the results are reasonable._

## Installation

This package can be installed using `conda` (or `mamba`, a faster version of `conda`):

```shell
mamba install -c conda-forge femto
```

If you are running with MPI on an HPC cluster, you may need to instruct conda to use your local installation
depending on your setup

```shell
mamba install -c conda-forge femto "openmpi=4.1.5=*external*"
```

where in this example you should change `4.1.5` to match the version of OpenMPI installed on your cluster / machine.

## Getting Started

To get started, see the [usage guide](https://psivant.github.io/femto/latest/guide-md).

## Acknowledgements

This framework benefited hugely from the work of Psivant's [Open Science Fellows](https://psivant.com/company/open-science-fellows/).

### ATM

The ATM implementation is based upon the work and advice of E. Gallicchio _et al_:

* [Wu, Joe Z., et al. "Alchemical transfer approach to absolute binding free energy estimation." Journal of Chemical Theory and Computation 17.6 (2021): 3309-3319.](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00266)
* [Azimi, Solmaz, et al. "Relative binding free energy calculations for ligands with diverse scaffolds with the alchemical transfer method." Journal of Chemical Information and Modeling 62.2 (2022): 309-323.](https://pubs.acs.org/doi/full/10.1021/acs.jcim.1c01129)

### SepTop

The SepTop implementation is based upon the work and advice of H. M. Baumann, D. L. Mobley _et al_:

* [Rocklin, Gabriel J., David L. Mobley, and Ken A. Dill. "Separated topologies—A method for relative binding free energy calculations using orientational restraints." The Journal of chemical physics 138.8 (2013).](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3598757/)
* [Baumann, Hannah M., et al. "Broadening the scope of binding free energy calculations using a Separated Topologies approach." Journal of Chemical Theory and Computation 19.15 (2023): 5058-5076.](https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c00282)
