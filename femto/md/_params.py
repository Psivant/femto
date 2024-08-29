"""Parameterize systems."""

import collections
import typing
import xml.etree.ElementTree as ElementTree

import openmm
import parmed

_SUPPORTED_FORCES = (
    openmm.NonbondedForce,
    openmm.HarmonicBondForce,
    openmm.HarmonicAngleForce,
    openmm.PeriodicTorsionForce,
)


def _quantity_to_str(quantity: openmm.unit.Quantity) -> str:
    """Cast a unit wrapped quantity to a string in the default MD unit system."""

    return str(quantity.value_in_unit_system(openmm.unit.md_unit_system))


def _find_bonds(force: openmm.HarmonicBondForce) -> set[tuple[int, int]]:
    """Find all bonds in a HarmonicBondForce."""
    bonds = set()

    for bond_index in range(force.getNumBonds()):
        idx_1, idx_2, *_ = force.getBondParameters(bond_index)
        bonds.add(tuple(sorted((idx_1, idx_2))))

    return typing.cast(set[tuple[int, int]], bonds)


def _has_bond(idx_1: int, idx_2: int, bonds: set[tuple[int, int]]) -> bool:
    """Check if a bond exists between two atoms."""
    return (idx_1, idx_2) in bonds or (idx_2, idx_1) in bonds


def _add_atom_types(
    root: ElementTree.Element, topology: parmed.Structure, types: list[str]
):
    """Add atom types to the XML tree."""
    atom_types_xml = ElementTree.SubElement(root, "AtomTypes")

    for idx, atom in enumerate(topology.atoms):
        ElementTree.SubElement(
            atom_types_xml,
            "Type",
            attrib={
                "name": atom.name,
                "class": types[idx],
                "element": atom.element_name,
                "mass": str(atom.mass),
            },
        )


def _add_residue(
    root: ElementTree.Element,
    name: str,
    names: list[str],
    types: list[str],
    bonds: set[tuple[int, int]],
):
    """Add the residue matcher to the XML tree."""
    residues_xml = ElementTree.SubElement(root, "Residues")
    residue_xml = ElementTree.SubElement(residues_xml, "Residue", name=name)

    for idx in range(len(types)):
        ElementTree.SubElement(
            residue_xml, "Atom", attrib={"name": names[idx], "type": types[idx]}
        )

    for idx_1, idx_2 in sorted(bonds):
        ElementTree.SubElement(
            residue_xml,
            "Bond",
            attrib={"atomName1": names[idx_1], "atomName2": names[idx_2]},
        )


def _convert_nonbonded_force(
    force: openmm.NonbondedForce,
    root: ElementTree.Element,
    types: list[str],
    scale_lj: float,
    scale_q: float,
):
    """Convert a NonbondedForce to an XML element."""
    force_xml = ElementTree.SubElement(
        root,
        "NonbondedForce",
        attrib={"coulomb14scale": str(scale_q), "lj14scale": str(scale_lj)},
    )

    for atom_idx in range(force.getNumParticles()):
        charge, sigma, epsilon = force.getParticleParameters(atom_idx)

        ElementTree.SubElement(
            force_xml,
            "Atom",
            attrib={
                "class": types[atom_idx],
                "charge": _quantity_to_str(charge),
                "sigma": _quantity_to_str(sigma),
                "epsilon": _quantity_to_str(epsilon),
            },
        )


def _convert_bond_force(
    force: openmm.HarmonicBondForce, root: ElementTree.Element, types: list[str]
):
    """Convert a HarmonicBondForce to an XML element."""
    force_xml = ElementTree.SubElement(root, "HarmonicBondForce")

    for bond_idx in range(force.getNumBonds()):
        idx_1, idx_2, length, k = force.getBondParameters(bond_idx)

        ElementTree.SubElement(
            force_xml,
            "Bond",
            attrib={
                "class1": types[idx_1],
                "class2": types[idx_2],
                "length": _quantity_to_str(length),
                "k": _quantity_to_str(k),
            },
        )


def _convert_angle_force(
    force: openmm.HarmonicAngleForce, root: ElementTree.Element, types: list[str]
):
    """Convert a HarmonicAngleForce to an XML element."""
    force_xml = ElementTree.SubElement(root, "HarmonicAngleForce")

    for angle_idx in range(force.getNumAngles()):
        idx_1, idx_2, idx_3, angle, k = force.getAngleParameters(angle_idx)

        ElementTree.SubElement(
            force_xml,
            "Angle",
            attrib={
                "class1": types[idx_1],
                "class2": types[idx_2],
                "class3": types[idx_3],
                "angle": _quantity_to_str(angle),
                "k": _quantity_to_str(k),
            },
        )


def _convert_torsion_force(
    force: openmm.PeriodicTorsionForce,
    root: ElementTree.Element,
    types: list[str],
    bonds: set[tuple[int, int]],
    order: typing.Literal["smirnoff"] = "smirnoff",
):
    """Convert a PeriodicTorsionForce to an XML element."""
    params = collections.defaultdict(list)

    for idx in range(force.getNumTorsions()):
        idx_1, idx_2, idx_3, idx_4, n, phase, k = force.getTorsionParameters(idx)
        params[(idx_1, idx_2, idx_3, idx_4)].append((n, phase, k))

    force_xml = ElementTree.SubElement(
        root, "PeriodicTorsionForce", ordering="smirnoff"
    )

    for (idx_1, idx_2, idx_3, idx_4), terms in params.items():
        is_proper = (
            _has_bond(idx_1, idx_2, bonds)
            and _has_bond(idx_2, idx_3, bonds)
            and _has_bond(idx_3, idx_4, bonds)
        )

        tag = "Proper" if is_proper else "Improper"

        attrib = {
            "class1": types[idx_1],
            "class2": types[idx_2],
            "class3": types[idx_3],
            "class4": types[idx_4],
            "ordering": order,
        }

        for idx, (n, phase, k) in enumerate(terms):
            attrib.update(
                {
                    f"periodicity{idx + 1}": str(n),
                    f"phase{idx + 1}": _quantity_to_str(phase),
                    f"k{idx + 1}": _quantity_to_str(k),
                }
            )

        ElementTree.SubElement(force_xml, tag, attrib=attrib)


def convert_system_to_ffxml(
    system: openmm.System,
    topology: parmed.Structure,
    scale_lj: float = 0.5,
    scale_q: float = 0.833333,
) -> str:
    """Convert an OpenMM System an XML serialized OpenMM ForceField residue template.

    Notes:
        This function only supports systems with a single residue / molecule.

    Args:
        system: The system to convert.
        topology: The associated topology.
        scale_lj: Scaling factor for 1-4 Lennard-Jones interactions.
        scale_q: Scaling factor for 1-4 electrostatic interactions.

    Returns:
        XML serialized OpenMM ForceField residue template.
    """

    if len(topology.residues) != 1:
        raise NotImplementedError("only single residue topologies are supported")
    if len(topology.atoms) != system.getNumParticles():
        raise ValueError("topology and system have different number of particles")

    residue: parmed.Residue = topology.residues[0]

    names = [atom.name for atom in topology.atoms]
    types = [f"{residue.name}-{name}" for name in names]

    forces = {force.__class__ for force in system.getForces()}

    if len(forces) != system.getNumForces():
        raise NotImplementedError("multiple forces of the same type is not supported")

    bond_forces = [
        force
        for force in system.getForces()
        if isinstance(force, openmm.HarmonicBondForce)
    ]
    bonds = set() if len(bond_forces) == 0 else _find_bonds(bond_forces[0])

    root = ElementTree.Element("ForceField")

    _add_atom_types(root, topology, types)
    _add_residue(root, residue.name, names, types, bonds)

    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            _convert_nonbonded_force(force, root, types, scale_lj, scale_q)
        elif isinstance(force, openmm.HarmonicBondForce):
            _convert_bond_force(force, root, types)
        elif isinstance(force, openmm.HarmonicAngleForce):
            _convert_angle_force(force, root, types)
        elif isinstance(force, openmm.PeriodicTorsionForce):
            _convert_torsion_force(force, root, types, bonds)
        else:
            raise NotImplementedError(f"{force.__class__} is not supported")

    v_site_idxs = [
        idx for idx in range(system.getNumParticles()) if system.isVirtualSite(idx)
    ]

    if len(v_site_idxs) > 0:
        raise NotImplementedError("virtual sites are not supported")

    return ElementTree.tostring(root, encoding="unicode")
