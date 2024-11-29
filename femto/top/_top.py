"""Simple topology representations"""

import copy
import typing
import warnings

import numpy
import openmm.app
from rdkit import Chem


class Atom:
    """Represents atoms and virtual sites stored in a topology."""

    def __init__(
        self, name: str, atomic_num: int, formal_charge: int | None, serial: int
    ):
        self.name: str = name
        """The name of the atom."""

        self.atomic_num: int = atomic_num
        """The atomic number, or 0 if this is a virtual site."""
        self.formal_charge: int | None = formal_charge
        """The formal charge on the atom."""

        self.serial: int = serial
        """The index of this atom in its original source (e.g. the serial defined
        in a PDB). This may not be zero-index or contiguous."""

        self._residue: typing.Optional["Residue"] = None
        self._index: int | None = None

    @property
    def symbol(self):
        """The chemical symbol of the atom, or 'X' if this is a virtual site."""
        return (
            "X"
            if self.atomic_num == 0
            else openmm.app.Element.getByAtomicNumber(self.atomic_num).symbol
        )

    @property
    def residue(self) -> typing.Optional["Residue"]:
        """The residue that the atom belongs to."""
        return self._residue

    @property
    def chain(self) -> typing.Optional["Chain"]:
        """The chain that the atom belongs to."""
        return None if self.residue is None else self._residue.chain

    @property
    def index(self) -> int | None:
        """The index of the atom in the parent topology"""
        return self._index

    def __repr__(self):
        return (
            f"Atom("
            f"name={self.name}, "
            f"atomic_num={self.atomic_num}, "
            f"formal_charge={self.formal_charge}, "
            f"serial={self.serial})"
        )


class Bond:
    """Represents a bond between two atoms."""

    def __init__(self, idx_1: int, idx_2: int, order: int | None):
        self._idx_1 = idx_1
        self._idx_2 = idx_2
        self.order = order
        """The formal bond order"""

    @property
    def idx_1(self) -> int:
        """The index of the first atom."""
        return self._idx_1

    @property
    def idx_2(self) -> int:
        """The index of the second atom."""
        return self._idx_2

    def __repr__(self):
        return f"Bond(idx_1={self.idx_1}, idx_2={self.idx_2}, order={self.order})"


class Residue:
    """Represents residues stored in a topology."""

    def __init__(self, name: str, seq_num: int):
        self.name = name
        """The name of the residue."""
        self.seq_num = seq_num
        """The sequence number of the residue."""

        self._chain: typing.Optional["Chain"] = None
        self._atoms: list[Atom] = []

        self._index: int | None = None

    @property
    def chain(self) -> typing.Optional["Chain"]:
        """The chain the residue belongs to (if any)."""
        return self._chain

    @property
    def topology(self) -> typing.Optional["Topology"]:
        """The topology the residue belongs to (if any)."""
        return None if self._chain is None else self._chain.topology

    @property
    def atoms(self) -> tuple[Atom, ...]:
        """The atoms associated with the residue."""
        return tuple(self._atoms)

    @property
    def n_atoms(self) -> int:
        """The number of atoms in the residue."""
        return len(self._atoms)

    @property
    def index(self) -> int | None:
        """The index of the residue in the parent topology"""
        return self._index

    def __repr__(self):
        return f"Residue(name={self.name}, seq_num={self.seq_num})"


class Chain:
    """Represents chains stored in a topology."""

    def __init__(self, id_: str):
        self.id = id_
        """The ID of the chain."""

        self._topology: typing.Optional["Topology"] | None = None
        self._residues: list[Residue] = []

        self._index: int | None = None

    @property
    def topology(self) -> typing.Optional["Topology"]:
        """The topology the chain belongs to (if any)."""
        return self._topology

    @property
    def residues(self) -> tuple[Residue, ...]:
        """The residues associated with the chain."""
        return tuple(self._residues)

    @property
    def n_residues(self) -> int:
        """The number of chains in the chain."""
        return len(self._residues)

    @property
    def atoms(self) -> tuple[Atom, ...]:
        """The atoms associated with the chain."""
        return tuple(atom for residue in self._residues for atom in residue.atoms)

    @property
    def n_atoms(self) -> int:
        """The number of atoms in the chain."""
        return sum(residue.n_atoms for residue in self._residues)

    @property
    def index(self) -> int | None:
        """The index of the chain in the parent topology"""
        return self._index

    def __repr__(self):
        return f"Chain(id={self.id})"


class Topology:
    """Topological information about a system."""

    def __init__(self):
        self._chains: list[Chain] = []
        self._bonds: list[Bond] = []

        self._n_atoms: int = 0
        self._n_residues: int = 0

        self._xyz: openmm.unit.Quantity | None = None
        self._box: openmm.unit.Quantity | None = None

    @property
    def chains(self) -> tuple[Chain, ...]:
        """The chains associated with the topology."""
        return tuple(self._chains)

    @property
    def n_chains(self) -> int:
        """The number of chains in the topology."""
        return len(self.chains)

    @property
    def residues(self) -> tuple[Residue, ...]:
        """The residues associated with the topology."""
        return tuple(residue for chain in self.chains for residue in chain.residues)

    @property
    def n_residues(self) -> int:
        """The number of residues in the topology."""
        return self._n_residues

    @property
    def atoms(self) -> tuple[Atom, ...]:
        """The atoms associated with the topology."""
        return tuple(atom for residue in self.residues for atom in residue.atoms)

    @property
    def n_atoms(self) -> int:
        """The number of atoms in the topology."""
        return self._n_atoms

    @property
    def bonds(self) -> tuple[Bond, ...]:
        """The bonds associated with the topology."""
        return tuple(self._bonds)

    @property
    def n_bonds(self) -> int:
        """The number of bonds in the topology."""
        return len(self.bonds)

    @property
    def xyz(self) -> openmm.unit.Quantity | None:
        """The coordinates of the atoms in the topology."""
        return self._xyz

    @xyz.setter
    def xyz(self, value: openmm.unit.Quantity):
        if value is not None and value.shape != (self.n_atoms, 3):
            raise ValueError(f"expected shape {(self.n_atoms, 3)}, got {value.shape}")

        self._xyz = value

    @property
    def box(self) -> openmm.unit.Quantity | None:
        """The box vectors of the simulation box."""
        return self._box

    @box.setter
    def box(self, value: openmm.unit.Quantity):
        if value is not None and value.shape != (3, 3):
            raise ValueError(f"expected shape (3, 3), got {value.shape}")

        self._box = value

    def add_chain(self, id_: str) -> Chain:
        """Add a new chain to the topology.

        Args:
            id_: The ID of the chain to add.

        Returns:
             The newly created chain.
        """
        chain = Chain(id_=id_)
        chain._topology = self
        chain._index = self.n_chains

        self._chains.append(chain)

        self._n_atoms += chain.n_atoms
        self._n_residues += chain.n_residues

        return chain

    def add_residue(self, name: str, seq_num: int | None, chain: Chain):
        """Add a new residue to the topology.

        Args:
            name: The name of the residue to add
            seq_num: The sequence number of the residue. If ``None``, the index in the
                topology will be used.
            chain: The parent chain to add to.

        Returns:
             The newly created residue.
        """

        if chain.topology != self:
            raise ValueError(f"{chain} does not belong to this topology.")

        seq_num = int(self.n_residues if seq_num is None else seq_num)

        residue = Residue(name=name, seq_num=seq_num)
        residue._chain = chain
        residue._index = self.n_residues

        chain._residues.append(residue)

        self._n_atoms += residue.n_atoms
        self._n_residues += 1

        return residue

    def add_atom(
        self,
        name: str,
        atomic_num: int,
        formal_charge: int | None,
        serial: int | None,
        residue: Residue,
    ):
        """Add a new atom to the topology.

        Args:
            name: The name of the atom to add
            atomic_num: The atomic number of the atom to add, or 0 for virtual sites.
            formal_charge: The formal charge on the atom (if defined).
            serial: The index of this atom in its original source (e.g. the serial
                defined in a PDB), which may not be zero-index or contiguous. If
                ``None``, the index in the topology will be used.
            residue: The parent residue to add to.

        Returns:
            The newly created atom
        """
        if residue.topology != self:
            raise ValueError(f"{residue} does not belong to this topology.")

        serial = int(self.n_atoms if serial is None else serial)

        atom = Atom(
            name=name, atomic_num=atomic_num, formal_charge=formal_charge, serial=serial
        )
        atom._residue = residue
        atom._index = self.n_atoms

        residue._atoms.append(atom)

        self._n_atoms += 1

        return atom

    def add_bond(self, idx_1: int, idx_2: int, order: int | None):
        """Add a new bond to the topology.

        Args:
            idx_1: The index of the first atom.
            idx_2: The index of the second atom.
            order: The formal bond order (if defined).

        Returns:
            The newly created bond.
        """

        if idx_1 >= self.n_atoms:
            raise ValueError("Index 1 is out of range.")
        if idx_2 >= self.n_atoms:
            raise ValueError("Index 2 is out of range.")

        bond = Bond(idx_1=idx_1, idx_2=idx_2, order=order)
        self._bonds.append(bond)

    @classmethod
    def from_openmm(cls, topology_omm: openmm.app.Topology) -> "Topology":
        """Create a topology from an OpenMM topology.

        Args:
            topology_omm: The OpenMM topology to convert.

        Returns:
            The converted topology.
        """
        topology = cls()

        for chain_omm in topology_omm.chains():
            chain = topology.add_chain(chain_omm.id)

            for residue_omm in chain_omm.residues():
                residue = topology.add_residue(residue_omm.name, residue_omm.id, chain)

                for atom_omm in residue_omm.atoms():
                    is_v_site = atom_omm.element is None

                    topology.add_atom(
                        atom_omm.name,
                        atom_omm.element.atomic_number if not is_v_site else 0,
                        None if is_v_site else getattr(atom_omm, "formalCharge", None),
                        atom_omm.id,
                        residue,
                    )

        for bond_omm in topology_omm.bonds():
            order = bond_omm.order

            if order is None and bond_omm.type is not None:
                raise NotImplementedError

            topology.add_bond(bond_omm.atom1.index, bond_omm.atom2.index, order)

        if topology_omm.getPeriodicBoxVectors() is not None:
            box = topology_omm.getPeriodicBoxVectors().value_in_unit(
                openmm.unit.angstrom
            )
            topology.box = numpy.array(box) * openmm.unit.angstrom

        return topology

    def to_openmm(self) -> openmm.app.Topology:
        """Convert the topology to an OpenMM topology.

        Returns:
            The OpenMM topology.
        """
        topology_omm = openmm.app.Topology()

        atoms_omm = []

        for chain in self.chains:
            chain_omm = topology_omm.addChain(chain.id)

            for residue in chain.residues:
                residue_omm = topology_omm.addResidue(
                    residue.name, chain_omm, str(residue.seq_num)
                )

                for atom in residue.atoms:
                    element = (
                        None
                        if atom.atomic_num == 0
                        else openmm.app.Element.getByAtomicNumber(atom.atomic_num)
                    )

                    atom_omm = topology_omm.addAtom(
                        atom.name, element, residue_omm, str(atom.serial)
                    )

                    if hasattr(atom_omm, "formalCharge"):
                        atom_omm.formalCharge = atom.formal_charge

                    atoms_omm.append(atom_omm)

        bond_order_to_type = {
            1: openmm.app.Single,
            2: openmm.app.Double,
            3: openmm.app.Triple,
        }

        for bond in self.bonds:
            topology_omm.addBond(
                atoms_omm[bond.idx_1],
                atoms_omm[bond.idx_2],
                bond_order_to_type[bond.order] if bond.order is not None else None,
                bond.order,
            )

        if self.box is not None:
            topology_omm.setPeriodicBoxVectors(self.box)

        return topology_omm

    @classmethod
    def from_rdkit(
        cls, mol: Chem.Mol, name: str = "LIG", chain: str = ""
    ) -> "Topology":
        """Create a topology from an RDKit molecule.

        Args:
            mol: The RDKit molecule to convert.
            name: The residue name to use for the ligand.
            chain: The chain ID to use for the ligand.

        Returns:
            The converted topology.
        """

        mol = Chem.AddHs(mol)
        Chem.Kekulize(mol)

        topology = cls()
        topology.add_chain(chain)
        residue = topology.add_residue(name, 1, topology.chains[0])

        for atom in mol.GetAtoms():
            topology.add_atom(
                name=name,
                atomic_num=atom.GetAtomicNum(),
                formal_charge=atom.GetFormalCharge(),
                serial=atom.GetIdx() + 1,
                residue=residue,
            )

        for bond in mol.GetBonds():
            topology.add_bond(
                idx_1=bond.GetBeginAtomIdx(),
                idx_2=bond.GetEndAtomIdx(),
                order=int(bond.GetBondTypeAsDouble()),
            )

        if mol.GetNumConformers() >= 1:
            xyz = mol.GetConformer().GetPositions()
            topology.xyz = numpy.array(xyz) * openmm.unit.angstrom

        return topology

    def to_rdkit(self) -> Chem.Mol:
        """Convert the Topology to an RDKit Mol object.

        Notes:
            * Currently this requires formal charges to be set on the atoms, and
              formal bond orders to be set on the bonds.

        Returns:
            The RDKit Mol object.
        """
        mol = Chem.RWMol()
        atoms_rd = []

        for atom in self.atoms:
            if atom.formal_charge is None:
                raise ValueError("Formal charges must be set on all atoms.")

            atom_rd = Chem.Atom(atom.atomic_num)
            atom_rd.SetFormalCharge(atom.formal_charge)

            atoms_rd.append(mol.AddAtom(atom_rd))

        bond_order_to_type = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
        }
        for bond in self.bonds:
            if bond.order is None:
                raise ValueError("Formal bond orders must be set on all bonds.")
            if bond.order not in bond_order_to_type:
                raise NotImplementedError(f"Bond order {bond.order} is not supported.")

            mol.AddBond(bond.idx_1, bond.idx_2, bond_order_to_type[bond.order])

        if self.xyz is not None:
            xyz = self.xyz.value_in_unit(openmm.unit.angstrom)
            conf = Chem.Conformer(len(atoms_rd))

            for idx, pos in enumerate(xyz):
                conf.SetAtomPosition(idx, pos)

            mol.AddConformer(conf, assignId=True)

        Chem.SanitizeMol(mol)
        return Chem.Mol(mol)

    def _select_amber(self, expr: str) -> numpy.ndarray | None:
        try:
            import parmed.amber
        except ImportError:
            return None

        try:
            topology_pmd = parmed.openmm.load_topology(self.to_openmm())
            result = parmed.amber.AmberMask(topology_pmd, expr).Selection()

            warnings.warn(
                "Using an Amber style selection mask is deprecated. Please use the "
                "PyMol style selection language instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            return numpy.array(tuple(i for i, matches in enumerate(result) if matches))
        except parmed.exceptions.MaskError:
            return

    def select(self, expr: str) -> numpy.ndarray:
        """Select atoms from the topology using a selection expression.

        The selection expression should be expressed in terms of the PyMol
        selection language. For example, to select all atoms in chain A:

        ```python
        selection = topology.select("chain A")
        ```

        or all atoms within 5 Å of the ligand:

        ```python
        selection = topology.select("all within 5 of resn LIG")
        ```

        Notes:
            An Amber-style selection mask can also be used, but this is deprecated
            and will be removed in a future version.

        Args:
            expr: The selection expression.
        """
        from femto.top._sel import select

        idxs = self._select_amber(expr)

        if idxs is not None:
            return idxs

        return select(self, expr)

    def subset(self, idxs: typing.Iterable[int]) -> "Topology":
        """Create a subset of the topology.

        Args:
            idxs: The indices of the atoms to include in the subset. Note the order of
                the atoms in the subset will be the same as in the original topology,
                regardless of the order of the indices.

        Returns:
            The subset of the topology.
        """
        idxs = numpy.array(idxs)
        idxs_unique = set(idxs)

        if len(idxs_unique) != len(idxs):
            raise ValueError("Indices are not unique.")

        subset = Topology()

        idx_old_to_new = {}

        for chain in self.chains:
            has_chain = any(
                atom.index in idxs_unique
                for residue in chain.residues
                for atom in residue.atoms
            )

            if not has_chain:
                continue

            chain_new = subset.add_chain(chain.id)

            for residue in chain.residues:
                has_residue = any(atom.index in idxs_unique for atom in residue.atoms)

                if not has_residue:
                    continue

                residue_new = subset.add_residue(
                    residue.name, residue.seq_num, chain_new
                )

                for atom in residue.atoms:
                    if atom.index not in idxs_unique:
                        continue

                    atom_new = subset.add_atom(
                        atom.name,
                        atom.atomic_num,
                        atom.formal_charge,
                        atom.serial,
                        residue_new,
                    )
                    idx_old_to_new[atom.index] = atom_new.index

        for bond in self.bonds:
            if bond.idx_1 not in idxs_unique or bond.idx_2 not in idxs_unique:
                continue

            subset.add_bond(
                idx_old_to_new[bond.idx_1], idx_old_to_new[bond.idx_2], bond.order
            )

        return subset

    @classmethod
    def merge(cls, *topologies: "Topology") -> "Topology":
        """Merge multiple topologies.

        Notes:
            * The box vectors of the first topology will be used.
            * Topologies without coordinates will be treated as if they have all zero
              coordinates.

        Args:
            topologies: The topologies to merge together.

        Returns:
            The merged topology.
        """

        if len(topologies) == 0:
            return cls()

        merged = copy.deepcopy(topologies[0])

        for topology in topologies[1:]:
            merged += topology

        return merged

    def __iadd__(self, other: "Topology"):
        if not isinstance(other, Topology):
            raise TypeError("Can only combine topologies.")

        idx_offset = self.n_atoms

        xyz_a = (
            None if self.xyz is None else self.xyz.value_in_unit(openmm.unit.angstrom)
        )
        xyz_b = (
            None if other.xyz is None else other.xyz.value_in_unit(openmm.unit.angstrom)
        )

        if xyz_a is None and xyz_b is not None:
            xyz_a = numpy.zeros((self.n_atoms, 3), dtype=float)
        if xyz_b is None and xyz_a is not None:
            xyz_b = numpy.zeros((other.n_atoms, 3), dtype=float)

        self._chains.extend(other.chains)
        self._n_atoms += other.n_atoms
        self._n_residues += other.n_residues

        for idx, atom in enumerate(self.atoms):
            atom._index = idx
        for idx, residue in enumerate(self.residues):
            residue._index = idx

        if xyz_a is not None and xyz_b is not None:
            self.xyz = numpy.vstack((xyz_a, xyz_b)) * openmm.unit.angstrom

        for bond in other.bonds:
            self.add_bond(bond.idx_1 + idx_offset, bond.idx_2 + idx_offset, bond.order)

        return self

    def __add__(self, other: "Topology") -> "Topology":
        combined = copy.deepcopy(self)
        combined += other
        return combined

    def __repr__(self):
        return (
            f"Topology("
            f"n_chains={self.n_chains}, "
            f"n_residues={self.n_residues}, "
            f"n_atoms={self.n_atoms})"
        )
