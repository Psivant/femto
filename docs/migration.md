# Migration Guide

This document outlines the major API and behaviour changes made to the `femto` codebase between versions, and provides
guidance on how to update your code to the latest version.

## From pre-0.3.0 to 0.3.0

The `femto` codebase was re-written to completely remove the dependency on `parmed`,
allowing easy use of any force field in OpenFF, Amber, and OpenMM FFXML formats. This re-write also
introduced a number of neccessary API changes.

### Behaviour Changes

#### MD

* Atom selection should now be performed using the `PyMol` atom selection language. This is more powerful and flexible than the
  previous `Amber` atom selection language, but may require some changes to your configuration files. Amber atom selection
  is currently still supported, but will be removed in a future version.
* Parameterization is now performed by [`openmmforcefields`](https://github.com/openmm/openmmforcefields?tab=readme-ov-file)
  rather than a combination of `parmed` and `tleap`. This allows for more flexibility in the force fields that can be used,
  and should give near identical parameters to those from `tleap`.
* Ligand force field parameters no longer need to be provided. The [femto.md.config.Prepare][] configuration now exposes
  a `default_ligand_ff` field that can be used to automatically parameterize ligands with an OpenFF based force field.
* The labelling of atoms in Boresch restraints has been updated to be more consistent with the literature, and also properly
  documented. See the [femto.md.restraints.create_boresch_restraint][] for more information on the exact definition of the
  distances, angles, and dihedrals that will be restrained.

#### FE

* A bug was fixed whereby the R2-R3-L1 angle, rather than the P1−L1−L2 angle force constant was being scaled by distance
  when running SepTop.
* Support has been added for co-factors, and force fields with virtual sites.

### API Changes

* [parmed.Structure][] is no longer used to store topological information. Instead, [mdtop.Topology][] is used.
  See the [mdtop documentation](https://simonboothroyd.github.io/mdtop/latest/) for more information.
* `femto.md.config.Solvent` has been renamed to [femto.md.config.Prepare][] to better reflect that it is used to
  more generally configure the system for simulation, not just solvation.
* The `tleap_sources` field of the old `femto.md.config.Solvent` configuration has been replaced by
  [femto.md.config.Prepare.default_protein_ff][], which now stores the paths to OpenMM force field XML files.
  See [`openmmforcefields`](https://github.com/openmm/openmmforcefields?tab=readme-ov-file#using-the-amber-and-charmm-biopolymer-force-fields)
  for details.
* A new `default_ligand_ff` field has been added to [femto.md.config.Prepare][]. This will accept the name / path to
  an OpenFF force field (e.g. `'openff-2.0.0.offxml'`) to use to automatically parameterize ligands.
* The `femto.md.solvate` and `femto.md.system` modules have been combined into a single `femto.md.prepare` module.
* The `femto.md.solvate.solvate_system` function has been replaced by the [femto.md.prepare.prepare_system][] function. The syntax
  is similar, but now also accepts cofactors, and an optional list of force field files to use for the receptor, ligand,
  cofactors, and solvent / ions.
