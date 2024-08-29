import re
import xml.etree.ElementTree as ElementTree

import openff.toolkit
import openmm
import parmed
import pytest
from openmm.app import ForceField

from femto.md._params import (
    _add_atom_types,
    _add_residue,
    _convert_angle_force,
    _convert_bond_force,
    _convert_nonbonded_force,
    _convert_torsion_force,
    _find_bonds,
    _has_bond,
    _quantity_to_str,
    convert_system_to_ffxml,
)


@pytest.fixture
def mock_molecule() -> openff.toolkit.Molecule:
    molecule = openff.toolkit.Molecule.from_smiles("C(=O)O")
    return molecule


@pytest.fixture
def mock_topology(mock_molecule) -> parmed.Structure:
    return parmed.openmm.load_topology(mock_molecule.to_topology().to_openmm())


@pytest.fixture
def mock_system(mock_molecule) -> openmm.System:
    ff = openff.toolkit.ForceField("openff-2.0.0.offxml")
    return ff.create_openmm_system(mock_molecule.to_topology())


@pytest.fixture
def mock_types(mock_molecule) -> list[str]:
    mock_molecule.generate_unique_atom_names()
    return [f"LIG-{a.name}" for a in mock_molecule.atoms]


@pytest.fixture
def mock_bonds(mock_molecule) -> set[tuple[int, int]]:
    return {(bond.atom1_index, bond.atom2_index) for bond in mock_molecule.bonds}


def compare_xml(actual: str, expected: str):
    actual = re.sub(r"\s+", "", actual)
    expected = re.sub(r"\s+", "", expected)
    assert actual == expected


def get_force(system: openmm.System, type_: type[openmm.Force]) -> openmm.Force:
    forces = [force for force in system.getForces() if isinstance(force, type_)]
    assert len(forces) == 1

    return forces[0]


def test_quantity_to_str():
    quantity = openmm.unit.Quantity(1.0, openmm.unit.angstrom)
    assert _quantity_to_str(quantity) == "0.1"


def test_find_bonds():
    force = openmm.HarmonicBondForce()
    force.addBond(0, 1, 1.0, 100.0)
    force.addBond(1, 0, 1.0, 100.0)
    force.addBond(2, 3, 1.0, 100.0)
    assert _find_bonds(force) == {(0, 1), (2, 3)}


def test_has_bond():
    bonds = {(0, 1), (1, 2)}
    assert _has_bond(0, 1, bonds) is True
    assert _has_bond(1, 0, bonds) is True
    assert _has_bond(2, 3, bonds) is False


def test_add_atom_types(mock_topology, mock_types):
    root = ElementTree.Element("ForceField")

    _add_atom_types(root, mock_topology, mock_types)

    actual = ElementTree.tostring(root, encoding="unicode")
    expected = (
        "<ForceField>"
        "  <AtomTypes>"
        '    <Type name="C1x" class="LIG-C1x" element="C" mass="12.01078" />'
        '    <Type name="O1x" class="LIG-O1x" element="O" mass="15.99943" />'
        '    <Type name="O2x" class="LIG-O2x" element="O" mass="15.99943" />'
        '    <Type name="H1x" class="LIG-H1x" element="H" mass="1.007947" />'
        '    <Type name="H2x" class="LIG-H2x" element="H" mass="1.007947" />'
        "  </AtomTypes>"
        "</ForceField>"
    )
    compare_xml(actual, expected)


def test_add_residue(mock_molecule, mock_topology, mock_types, mock_bonds):
    mock_names = [a.name for a in mock_molecule.atoms]

    root = ElementTree.Element("ForceField")

    _add_residue(root, "LIG", mock_names, mock_types, mock_bonds)

    actual = ElementTree.tostring(root, encoding="unicode")
    expected = (
        "<ForceField>"
        "  <Residues>"
        '    <Residue name="LIG">'
        '      <Atom name="C1x" type="LIG-C1x" />'
        '      <Atom name="O1x" type="LIG-O1x" />'
        '      <Atom name="O2x" type="LIG-O2x" />'
        '      <Atom name="H1x" type="LIG-H1x" />'
        '      <Atom name="H2x" type="LIG-H2x" />'
        '      <Bond atomName1="C1x" atomName2="O1x" />'
        '      <Bond atomName1="C1x" atomName2="O2x" />'
        '      <Bond atomName1="C1x" atomName2="H1x" />'
        '      <Bond atomName1="O2x" atomName2="H2x" />'
        "    </Residue>"
        "  </Residues>"
        "</ForceField>"
    )
    compare_xml(actual, expected)


def test_convert_nonbonded_force(mock_system, mock_topology, mock_types):
    force = get_force(mock_system, openmm.NonbondedForce)
    root = ElementTree.Element("ForceField")

    _convert_nonbonded_force(force, root, mock_types, 0.123, 0.456)

    actual = ElementTree.tostring(root, encoding="unicode")
    expected = ()

    compare_xml(actual, expected)


def test_convert_bond_force(mock_system, mock_topology, mock_types):
    force = get_force(mock_system, openmm.HarmonicBondForce)
    root = ElementTree.Element("ForceField")

    _convert_bond_force(force, root, mock_types)

    actual = ElementTree.tostring(root, encoding="unicode")
    expected = ()

    compare_xml(actual, expected)


def test_convert_angle_force(mock_system, mock_topology, mock_types):
    force = get_force(mock_system, openmm.HarmonicAngleForce)
    root = ElementTree.Element("ForceField")

    _convert_angle_force(force, root, mock_types)

    actual = ElementTree.tostring(root, encoding="unicode")
    expected = ()
    compare_xml(actual, expected)


def test_convert_torsion_force(mock_system, mock_topology, mock_types):
    root = ElementTree.Element("Root")
    force = openmm.PeriodicTorsionForce()
    force.addTorsion(0, 1, 2, 3, 1, 180.0, 10.0)
    types = ["C", "H", "O", "N"]
    bonds = {(0, 1), (1, 2), (2, 3)}
    _convert_torsion_force(force, root, types, bonds)
    assert len(root.findall(".//Proper")) == 1


def test_convert_system_to_ffxml(mock_system, mock_topology, tmp_path):
    ff_xml = convert_system_to_ffxml(mock_system, mock_topology, 0.123, 0.456)
    ff_path = tmp_path / "ff.xml"

    with ff_path.open("w") as file:
        file.write(ff_xml)

    ff = ForceField(ff_path)

    system = ff.createSystem(mock_topology.topology)
    system_xml = openmm.XmlSerializer.serializeSystem(system)

    expected_xml = openmm.XmlSerializer.serializeSystem(mock_system)

    assert system_xml == expected_xml
