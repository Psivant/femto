import numpy
import openmm
import openmm.app
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from femto.top import Atom, Bond, Chain, Residue, Topology


def compare_topologies(top_a: Topology, top_b: Topology):
    assert top_a.n_chains == top_b.n_chains
    assert top_a.n_residues == top_b.n_residues
    assert top_a.n_atoms == top_b.n_atoms
    assert top_a.n_bonds == top_b.n_bonds

    for chain_orig, chain_rt in zip(top_a.chains, top_b.chains, strict=True):
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

    for bond_orig, bond_rt in zip(top_a.bonds, top_b.bonds, strict=True):
        assert bond_orig.idx_1 == bond_rt.idx_1
        assert bond_orig.idx_2 == bond_rt.idx_2
        assert bond_orig.order == bond_rt.order

    assert (top_a.xyz is None) == (top_b.xyz is None)

    if top_a.xyz is not None:
        assert numpy.allclose(
            top_a.xyz.value_in_unit(openmm.unit.angstrom),
            top_b.xyz.value_in_unit(openmm.unit.angstrom),
        )

    assert (top_a.box is None) == (top_b.box is None)

    if top_a.box is not None:
        assert numpy.allclose(
            top_a.box.value_in_unit(openmm.unit.angstrom),
            top_b.box.value_in_unit(openmm.unit.angstrom),
        )


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

    compare_topologies(topology_roundtrip, topology_original)


def test_topology_rdkit_roundtrip():
    mol = Chem.AddHs(Chem.MolFromSmiles("O=Cc1ccccc1[N+](=O)[O-]"))
    assert mol is not None, "Failed to create RDKit molecule from SMILES."
    expected_smiles = Chem.MolToSmiles(mol, canonical=True)

    AllChem.EmbedMolecule(mol)
    expected_coords = numpy.array(mol.GetConformer().GetPositions())

    topology = Topology.from_rdkit(mol, "ABC", "E")

    roundtrip_mol = topology.to_rdkit()
    roundtrip_smiles = Chem.MolToSmiles(roundtrip_mol, canonical=True)
    roundtrip_coords = numpy.array(roundtrip_mol.GetConformer().GetPositions())

    assert expected_smiles == roundtrip_smiles

    assert expected_coords.shape == roundtrip_coords.shape
    assert numpy.allclose(expected_coords, roundtrip_coords)


def test_topology_file_roundtrip(tmp_path, test_data_dir):
    topology_original = Topology.from_file(test_data_dir / "protein.pdb")

    topology_original.to_file(tmp_path / "protein.pdb")
    topology_roundtrip = Topology.from_file(tmp_path / "protein.pdb")

    compare_topologies(topology_roundtrip, topology_original)


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


@pytest.mark.parametrize(
    "xyz",
    [
        numpy.arange(6).reshape(-1, 3) * openmm.unit.angstrom,
        numpy.arange(6).reshape(-1, 3).tolist() * openmm.unit.angstrom,
        numpy.arange(6).reshape(-1, 3),
        numpy.arange(6).reshape(-1, 3).tolist(),
    ],
)
def test_topology_xyz_setter(xyz):
    topology = Topology()
    topology.add_chain("A")
    topology.add_residue("ALA", 1, topology.chains[-1])
    topology.add_atom("C", 6, 0, 1, topology.residues[-1])
    topology.add_atom("C", 6, 0, 2, topology.residues[-1])
    topology.xyz = xyz

    expected_xyz = numpy.arange(topology.n_atoms * 3).reshape(-1, 3)

    assert isinstance(topology.xyz, openmm.unit.Quantity)

    xyz_array = topology.xyz.value_in_unit(openmm.unit.angstrom)
    assert isinstance(xyz_array, numpy.ndarray)
    assert xyz_array.shape == expected_xyz.shape
    assert numpy.allclose(xyz_array, expected_xyz)

    with pytest.raises(ValueError, match="expected shape"):
        topology.xyz = numpy.zeros((0, 3))


def test_topology_xyz_setter_none():
    topology = Topology()
    topology.xyz = numpy.zeros((0, 3)) * openmm.unit.angstrom
    topology.xyz = None
    assert topology.xyz is None


@pytest.mark.parametrize(
    "box",
    [
        numpy.eye(3) * openmm.unit.angstrom,
        numpy.eye(3).tolist() * openmm.unit.angstrom,
        numpy.eye(3),
    ],
)
def test_topology_box_setter(box):
    topology = Topology()
    topology.box = box

    expected_box = numpy.eye(3)

    assert isinstance(topology.box, openmm.unit.Quantity)

    box_array = topology.box.value_in_unit(openmm.unit.angstrom)
    assert isinstance(box_array, numpy.ndarray)
    assert box_array.shape == expected_box.shape
    assert numpy.allclose(box_array, expected_box)

    with pytest.raises(ValueError, match="expected shape"):
        topology.box = numpy.zeros((0, 3))


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


@pytest.mark.parametrize(
    "item, expected",
    [
        ("idx. 1 5", numpy.array([0, 4])),
        (3, numpy.array([3])),
        (slice(0, 4), numpy.array([0, 1, 2, 3])),
        (numpy.array([1, 3, 5]), numpy.array([1, 3, 5])),
        ([1, 3, 5], numpy.array([1, 3, 5])),
    ],
)
def test_topology_slice(mocker, item, expected):
    mock_subset = mocker.patch("femto.top.Topology.subset", autospec=True)

    topology = Topology()
    topology.add_chain("A")
    topology.add_residue("ALA", 1, topology.chains[-1])

    for i in range(10):
        topology.add_atom("C", 6, 0, i + 1, topology.residues[-1])

    topology.__getitem__(item)

    mock_subset.assert_called_once()

    idxs = mock_subset.call_args.args[1]
    assert isinstance(idxs, numpy.ndarray)

    assert idxs.shape == expected.shape
    assert numpy.allclose(idxs, expected)
