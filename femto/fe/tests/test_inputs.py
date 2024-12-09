import pytest

import femto.fe.config
from femto.fe.inputs import _find_config, _find_edge, _find_receptor, find_edges
from femto.fe.tests.systems import (
    CDK2_SYSTEM,
    create_cdk2_input_directory,
    create_temoa_input_directory,
)


def test_find_config_legacy(tmp_cwd):
    (tmp_cwd / "Morph.in").write_text("l1~l2\nl1~l1")

    expected_edges = [
        femto.fe.config.Edge(ligand_1="l1", ligand_2="l2"),
        femto.fe.config.Edge(ligand_1="l1", ligand_2=None),
    ]

    found_network = _find_config(tmp_cwd)

    assert found_network.receptor is None
    assert found_network.edges == expected_edges


def test_find_receptor(tmp_cwd):
    create_temoa_input_directory(tmp_cwd)

    receptor = _find_receptor(tmp_cwd)

    assert receptor.name == "temoa"
    assert receptor.coords == tmp_cwd / "proteins" / "temoa" / "protein.sdf"
    assert receptor.params == tmp_cwd / "proteins" / "temoa" / "protein.xml"


def test_find_receptor_no_params(tmp_cwd):
    create_cdk2_input_directory(tmp_cwd)

    receptor = _find_receptor(tmp_cwd)

    assert receptor.name == "cdk2"
    assert receptor.coords == tmp_cwd / "proteins" / "cdk2" / "protein.pdb"
    assert receptor.params is None


def test_find_receptor_missing(tmp_cwd):
    (tmp_cwd / "proteins").mkdir()

    with pytest.raises(RuntimeError, match="Expected to find exactly one receptor"):
        _find_receptor(tmp_cwd, "temoa")


def test_find_edge(tmp_cwd):
    create_temoa_input_directory(tmp_cwd)

    edge_config = femto.fe.config.Edge(ligand_1="g1", ligand_2="g4")

    found_edge = _find_edge(tmp_cwd, edge_config)

    assert found_edge.ligand_1.name == "g1"
    assert found_edge.ligand_1.coords == tmp_cwd / "forcefield" / "g1" / "vacuum.mol2"
    assert found_edge.ligand_1.params == tmp_cwd / "forcefield" / "g1" / "vacuum.xml"

    assert found_edge.ligand_2.name == "g4"
    assert found_edge.ligand_2.coords == tmp_cwd / "forcefield" / "g4" / "vacuum.mol2"
    assert found_edge.ligand_2.params == tmp_cwd / "forcefield" / "g4" / "vacuum.xml"


@pytest.mark.parametrize(
    "ligand_1, ligand_2, expected_match",
    [
        ("g2", "g4", "Could not find files for g2"),
        ("g1", "g3", "Could not find files for g3"),
    ],
)
def test_find_edge_missing(ligand_1, ligand_2, expected_match, tmp_cwd):
    create_temoa_input_directory(tmp_cwd)

    edge_config = femto.fe.config.Edge(ligand_1=ligand_1, ligand_2=ligand_2)

    with pytest.raises(RuntimeError, match=expected_match):
        _find_edge(tmp_cwd, edge_config)


def test_find_edges(tmp_cwd):
    create_cdk2_input_directory(tmp_cwd)
    (tmp_cwd / "Morph.in").write_text(
        f"{CDK2_SYSTEM.ligand_1_name}~{CDK2_SYSTEM.ligand_1_name}"
    )

    network = find_edges(tmp_cwd)

    assert network.receptor.name == "cdk2"
    assert network.receptor.coords == tmp_cwd / "proteins" / "cdk2" / "protein.pdb"
    assert network.receptor.params is None

    assert len(network.edges) == 1
    found_edge = network.edges[0]

    assert found_edge.ligand_1.name == "1h1q"
    assert found_edge.ligand_1.coords == tmp_cwd / "forcefield" / "1h1q" / "vacuum.sdf"
    assert found_edge.ligand_1.params == tmp_cwd / "forcefield" / "1h1q" / "vacuum.xml"
