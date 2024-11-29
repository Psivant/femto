# Topology

The `femto.top` module exposes a `Topology` object which is used throughout the package.
It is meant as a more flexible replacement for OpenMM's `Topology` object, and ParmEd's
`Structure` object which have been employed in the past.

???+ warning
    `Topology` objects are still a work in progress, and are subject to change. Please
    report any issues or suggestions you may have on the [GitHub](https://github.com/Psivant/femto)

## Creating a Topology
Topologies can be created from a variety of sources. However, the most robust workflows
typically involve:

* **Loading a topology from OpenMM**: Use the `Topology.from_openmm()` method to import
  an existing OpenMM topology containing, for example, a protein structure.
* **Creating a topology from an RDKit molecule**: Use the `Topology.from_rdkit()` method
  to create a topology from an RDKit molecule representation.

```python
from openmm.app import PDBFile
from rdkit import Chem

from femto.top import Topology

# Load a protein topology from OpenMM
protein_top_omm = PDBFile("protein.pdb").topology
protein_top = Topology.from_openmm(protein_top_omm)

# Load a ligand using RDKit
ligand_rd = Chem.MolFromMolFile("ligand.sdf")
ligand_top = Topology.from_rdkit(ligand_rd)

# Merge the protein and ligand topologies
system_top = Topology.merge(protein_top, ligand_top)
# OR
system_top = protein_top + ligand_top
```

## Atom Selection

Subsets of atoms can be selected using a (for now) subset of the
[PyMol atom selection language]((https://pymolwiki.org/index.php/Selection_Algebra)).

For example, to select all atoms in chain A:

```python
selection = system_top.select("chain A")
```

or all atoms within 5 Å of the ligand:

```python
atom_idxs = system_top.select("all within 5 of resn LIG")
```

A subset of the topology can then be created using the `subset()` method:

```python
subset = system_top.subset(atom_idxs)
```

## Exporting Topologies

Topologies can be converted back into OpenMM or RDKit formats for further analysis or
simulation.

```python
# Export to OpenMM topology
system_top_omm = system_top.to_openmm()

# Export to RDKit molecule - this currently only works for small molecules
mol_rd = ligand_top.to_rdkit()
```
