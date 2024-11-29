import numpy
import openmm
import openmm.app
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from femto.top import Atom, Bond, Chain, Residue, Topology


def test_atom_properties():
    atom = Atom(name="C", atomic_num=6, formal_charge=0, serial=1)
    assert atom.name == "C"
    assert atom.atomic_num == 6
    assert atom.formal_charge == 0
    assert atom.serial == 1
    assert atom.symbol == "C"


def test_bond_properties():
    bond = Bond(idx_1=0, idx_2=1, order=1)
    assert bond.idx_1 == 0
    assert bond.idx_2 == 1
    assert bond.order == 1


def test_residue_properties():
    residue = Residue(name="ALA", seq_num=1)
    assert residue.name == "ALA"
    assert residue.seq_num == 1
    assert residue.n_atoms == 0
    assert len(residue.atoms) == 0

    atom = Atom(name="C", atomic_num=6, formal_charge=0, serial=1)
    residue._atoms.append(atom)
    assert residue.n_atoms == 1


def test_chain_properties():
    chain = Chain(id_="A")
    assert chain.id == "A"
    assert chain.n_residues == 0
    assert len(chain.residues) == 0
    assert chain.n_atoms == 0

    residue = Residue(name="ALA", seq_num=1)
    chain._residues.append(residue)
    assert chain.n_residues == 1

    atom = Atom(name="C", atomic_num=6, formal_charge=0, serial=1)
    residue._atoms.append(atom)
    assert chain.n_atoms == 1


def test_topology_add_chain():
    topology = Topology()
    chain = topology.add_chain(id_="A")
    assert len(topology.chains) == 1
    assert chain.id == "A"
    assert topology.n_chains == 1


def test_topology_add_residue():
    topology = Topology()
    chain = topology.add_chain(id_="A")
    residue = topology.add_residue(name="ALA", seq_num=1, chain=chain)
    assert len(chain.residues) == 1
    assert residue.name == "ALA"
    assert residue.seq_num == 1
    assert topology.n_residues == 1


def test_topology_add_atom():
    topology = Topology()
    chain = topology.add_chain(id_="A")
    residue = topology.add_residue(name="ALA", seq_num=1, chain=chain)
    atom = topology.add_atom(
        name="C", atomic_num=6, formal_charge=0, serial=1, residue=residue
    )
    assert len(residue.atoms) == 1
    assert atom.name == "C"
    assert topology.n_atoms == 1


def test_topology_add_bond():
    topology = Topology()
    chain = topology.add_chain(id_="A")
    residue1 = topology.add_residue(name="ALA", seq_num=1, chain=chain)
    atom1 = topology.add_atom(
        name="C", atomic_num=6, formal_charge=0, serial=1, residue=residue1
    )
    residue2 = topology.add_residue(name="GLY", seq_num=2, chain=chain)
    atom2 = topology.add_atom(
        name="N", atomic_num=7, formal_charge=0, serial=2, residue=residue2
    )
    topology.add_bond(idx_1=atom1.index, idx_2=atom2.index, order=1)
    assert topology.n_bonds == 1
    assert len(topology.bonds) == 1


def test_topology_invalid_add_bond():
    topology = Topology()
    topology.add_chain("A")
    topology.add_residue("ALA", seq_num=1, chain=topology.chains[0])
    topology.add_atom("C", 6, 0, 1, topology.residues[0])

    with pytest.raises(ValueError, match="Index 1 is out of range."):
        topology.add_bond(idx_1=10, idx_2=0, order=1)

    with pytest.raises(ValueError, match="Index 2 is out of range."):
        topology.add_bond(idx_1=0, idx_2=10, order=1)


def test_topology_omm_roundtrip(test_data_dir):
    pdb = openmm.app.PDBFile(str(test_data_dir / "protein.pdb"))
    topology_original = Topology.from_openmm(pdb.topology)

    topology_omm = topology_original.to_openmm()
    topology_roundtrip = Topology.from_openmm(topology_omm)

    assert topology_original.n_chains == topology_roundtrip.n_chains
    assert topology_original.n_residues == topology_roundtrip.n_residues
    assert topology_original.n_atoms == topology_roundtrip.n_atoms
    assert topology_original.n_bonds == topology_roundtrip.n_bonds

    for chain_orig, chain_rt in zip(
        topology_original.chains, topology_roundtrip.chains, strict=True
    ):
        assert chain_orig.id == chain_rt.id
        assert chain_orig.n_residues == chain_rt.n_residues
        assert chain_orig.n_atoms == chain_rt.n_atoms

        for residue_orig, residue_rt in zip(
            chain_orig.residues, chain_rt.residues, strict=True
        ):
            assert residue_orig.name == residue_rt.name
            assert residue_orig.seq_num == residue_rt.seq_num
            assert residue_orig.n_atoms == residue_rt.n_atoms

            for atom_orig, atom_rt in zip(
                residue_orig.atoms, residue_rt.atoms, strict=True
            ):
                assert atom_orig.name == atom_rt.name
                assert atom_orig.atomic_num == atom_rt.atomic_num
                assert atom_orig.formal_charge == atom_rt.formal_charge
                assert atom_orig.serial == atom_rt.serial

    for bond_orig, bond_rt in zip(
        topology_original.bonds, topology_roundtrip.bonds, strict=True
    ):
        assert bond_orig.idx_1 == bond_rt.idx_1
        assert bond_orig.idx_2 == bond_rt.idx_2
        assert bond_orig.order == bond_rt.order


def test_topology_rdkit_roundtrip():
    mol = Chem.AddHs(Chem.MolFromSmiles("O=Cc1ccccc1[N+](=O)[O-]"))
    assert mol is not None, "Failed to create RDKit molecule from SMILES."
    expected_smiles = Chem.MolToSmiles(mol, canonical=True)

    AllChem.EmbedMolecule(mol)
    expected_coords = numpy.array(mol.GetConformer().GetPositions())

    topology = Topology.from_rdkit(mol)

    roundtrip_mol = topology.to_rdkit()
    roundtrip_smiles = Chem.MolToSmiles(roundtrip_mol, canonical=True)
    roundtrip_coords = numpy.array(roundtrip_mol.GetConformer().GetPositions())

    assert expected_smiles == roundtrip_smiles

    assert expected_coords.shape == roundtrip_coords.shape
    assert numpy.allclose(expected_coords, roundtrip_coords)


def test_topology_select():
    topology = Topology()

    chain_a = topology.add_chain("A")
    res_a = topology.add_residue("ALA", 1, chain_a)
    topology.add_atom("C1", 6, 0, 1, res_a)

    selection = topology.select("c. A and r. ALA")
    assert numpy.allclose(selection, numpy.array([0]))


def test_topology_select_amber():
    topology = Topology()

    chain_a = topology.add_chain("A")
    res_a = topology.add_residue("ACE", 1, chain_a)
    topology.add_atom("H1", 1, 0, 1, res_a)
    topology.add_atom("CH3", 6, 0, 2, res_a)
    topology.add_atom("H2", 1, 0, 3, res_a)
    topology.add_atom("H3", 1, 0, 4, res_a)
    topology.add_atom("C", 6, 0, 5, res_a)
    topology.add_atom("O", 8, 0, 6, res_a)

    selection = topology.select(":ACE & !@/H")
    assert numpy.allclose(selection, numpy.array([1, 4, 5]))


def test_topology_subset():
    topology = Topology()

    chain_a = topology.add_chain("A")
    res_a = topology.add_residue("ALA", 1, chain_a)
    topology.add_atom("C1", 6, 0, 1, res_a)
    topology.add_atom("C2", 6, 0, 2, res_a)
    res_b = topology.add_residue("MET", 2, chain_a)
    topology.add_atom("C3", 6, 0, 3, res_b)
    topology.add_atom("C4", 6, 0, 4, res_b)
    topology.add_residue("TYR", 2, chain_a)

    chain_b = topology.add_chain("B")
    res_d = topology.add_residue("GLY", 3, chain_b)
    topology.add_atom("C5", 6, 0, 5, res_d)
    chain_c = topology.add_chain("C")
    res_e = topology.add_residue("SER", 4, chain_c)
    topology.add_atom("C6", 6, 0, 6, res_e)

    topology.add_bond(0, 1, 1)
    topology.add_bond(0, 5, 1)

    subset = topology.subset([0, 3, 5])

    assert subset.n_chains == 2
    assert [c.id for c in subset.chains] == ["A", "C"]

    assert subset.n_residues == 3
    assert [r.name for r in subset.residues] == ["ALA", "MET", "SER"]

    assert subset.n_atoms == 3
    assert [a.name for a in subset.atoms] == ["C1", "C4", "C6"]

    assert subset.n_bonds == 1
    assert subset.bonds[0].idx_1 == 0
    assert subset.bonds[0].idx_2 == 2


def test_topology_xyz_setter():
    topology = Topology()
    chain = topology.add_chain("A")
    residue = topology.add_residue("ALA", 1, chain)
    topology.add_atom("C", 6, 0, 1, residue)

    valid_xyz = numpy.array([[0.0, 0.0, 0.0]]) * openmm.unit.angstrom
    topology.xyz = valid_xyz
    assert (topology.xyz == valid_xyz).all()

    invalid_xyz = numpy.array([[0.0, 0.0]]) * openmm.unit.angstrom
    with pytest.raises(ValueError, match="expected shape"):
        topology.xyz = invalid_xyz


def test_topology_box_setter():
    topology = Topology()

    valid_box = numpy.eye(3) * openmm.unit.angstrom
    topology.box = valid_box
    assert (topology.box == valid_box).all()

    invalid_box = numpy.array([[0.0, 0.0]]) * openmm.unit.angstrom
    with pytest.raises(ValueError, match="expected shape"):
        topology.box = invalid_box


def test_topology_merge():
    topology1 = Topology()
    topology1.add_chain(id_="A")
    topology1.add_residue("ALA", 1, topology1.chains[0])
    topology1.add_atom("C", 6, 0, 1, topology1.residues[-1])
    topology1.add_residue("GLY", 2, topology1.chains[0])
    topology1.add_atom("N", 7, 0, 2, topology1.residues[-1])
    topology1.add_chain(id_="B")
    topology1.add_residue("SER", 1, topology1.chains[1])
    topology1.add_atom("O", 8, 0, 3, topology1.residues[-1])
    topology1.add_bond(0, 1, 1)
    topology1.xyz = (
        numpy.arange(topology1.n_atoms * 3).reshape(-1, 3) * openmm.unit.angstrom
    )

    topology2 = Topology()
    topology2.add_chain(id_="C")
    topology2.add_residue("VAL", 1, topology2.chains[0])
    topology2.add_atom("CA", 6, 0, 1, topology2.residues[-1])
    topology2.add_residue("GLU", 2, topology2.chains[0])
    topology2.add_atom("CB", 6, 0, 2, topology2.residues[-1])
    topology2.add_bond(0, 1, 1)
    topology2.xyz = None

    merged_topology = Topology.merge(topology1, topology2)

    assert topology1.n_chains == 2
    assert topology2.n_chains == 1

    assert merged_topology.n_chains == 3
    assert merged_topology.n_chains == 3
    assert merged_topology.n_residues == topology1.n_residues + topology2.n_residues
    assert merged_topology.n_atoms == topology1.n_atoms + topology2.n_atoms
    assert merged_topology.n_bonds == topology1.n_bonds + topology2.n_bonds

    assert merged_topology.chains[0].id == "A"
    assert merged_topology.chains[1].id == "B"
    assert merged_topology.chains[2].id == "C"

    chain_a_residues = merged_topology.chains[0].residues
    assert chain_a_residues[0].name == "ALA"
    assert chain_a_residues[1].name == "GLY"
    chain_b_residues = merged_topology.chains[1].residues
    assert chain_b_residues[0].name == "SER"
    chain_c_residues = merged_topology.chains[2].residues
    assert chain_c_residues[0].name == "VAL"
    assert chain_c_residues[1].name == "GLU"
    assert merged_topology.bonds[0].idx_1 == 0
    assert merged_topology.bonds[0].idx_2 == 1
    assert merged_topology.bonds[1].idx_1 == topology1.n_atoms
    assert merged_topology.bonds[1].idx_2 == topology1.n_atoms + 1

    expected_xyz = numpy.vstack(
        [
            numpy.arange(topology1.n_atoms * 3).reshape(-1, 3),
            numpy.zeros((topology2.n_atoms, 3)),
        ]
    )
    assert merged_topology.xyz.shape == expected_xyz.shape
    assert numpy.allclose(merged_topology.xyz, expected_xyz)

    for i, atom in enumerate(merged_topology.atoms):
        assert atom.index == i
    for i, residue in enumerate(merged_topology.residues):
        assert residue.index == i


def test_topology_merge_with_b_coords():
    topology1 = Topology()
    topology1.add_chain(id_="A")
    topology1.add_residue("ALA", 1, topology1.chains[0])
    topology1.add_atom("C", 6, 0, 1, topology1.residues[-1])
    topology1.add_atom("N", 7, 0, 2, topology1.residues[-1])
    topology1.xyz = None

    topology2 = Topology()
    topology2.add_chain(id_="C")
    topology2.add_residue("ALA", 1, topology2.chains[0])
    topology2.add_atom("CA", 6, 0, 1, topology2.residues[-1])
    topology2.xyz = numpy.array([[1.0, 2.0, 3.0]]) * openmm.unit.angstrom

    merged_topology = Topology.merge(topology1, topology2)

    expected_xyz = (
        numpy.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        * openmm.unit.angstrom
    )
    assert merged_topology.xyz.shape == expected_xyz.shape
    assert numpy.allclose(merged_topology.xyz, expected_xyz)


def test_topology_add():
    topology1 = Topology()
    topology1.add_chain(id_="A")
    topology1.add_residue("ALA", 1, topology1.chains[0])
    topology1.add_atom("C", 6, 0, 1, topology1.residues[-1])
    topology1.xyz = numpy.array([[1.0, 2.0, 3.0]]) * openmm.unit.angstrom

    topology2 = Topology()
    topology2.add_chain(id_="C")
    topology2.add_residue("ALA", 1, topology2.chains[0])
    topology2.add_atom("CA", 6, 0, 1, topology2.residues[-1])
    topology2.xyz = numpy.array([[4.0, 5.0, 6.0]]) * openmm.unit.angstrom

    merged_topology = topology1 + topology2

    expected_xyz = (
        numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) * openmm.unit.angstrom
    )
    assert merged_topology.xyz.shape == expected_xyz.shape
    assert numpy.allclose(merged_topology.xyz, expected_xyz)
