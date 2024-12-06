import numpy
import openmm
import openmm.unit
import pytest
from scipy.spatial.transform import Rotation

import femto.md.config
import femto.md.constants
import femto.md.utils.openmm
from femto.md.restraints import (
    create_boresch_restraint,
    create_flat_bottom_restraint,
    create_position_restraints,
)
from femto.md.tests.mocking import build_mock_structure

_KCAL = openmm.unit.kilocalorie_per_mole

_KCAL_PER_ANG_SQR = openmm.unit.kilocalorie_per_mole / openmm.unit.angstrom**2
_KCAL_PER_RAD_SQR = openmm.unit.kilocalorie_per_mole / openmm.unit.radians**2

_ANGSTROM = openmm.unit.angstrom

_MD_UNITS = openmm.unit.md_unit_system


def create_dihedral_rotation(coord_a, coord_b, angle) -> Rotation:
    """Creates a rotation object that rotates around the ab vector."""
    vector_ab = coord_b - coord_a
    vector_ab /= numpy.linalg.norm(vector_ab)

    return Rotation.from_rotvec(vector_ab * angle)


@pytest.fixture
def mock_boresch_config():
    return femto.md.config.BoreschRestraint(
        k_distance=1.0 * _KCAL_PER_ANG_SQR,
        k_angle_a=2.0 * _KCAL_PER_RAD_SQR,
        k_angle_b=3.0 * _KCAL_PER_RAD_SQR,
        k_dihedral_a=4.0 * _KCAL_PER_RAD_SQR,
        k_dihedral_b=5.0 * _KCAL_PER_RAD_SQR,
        k_dihedral_c=6.0 * _KCAL_PER_RAD_SQR,
    )


def mock_boresch_coords(
    dist: float = 1.0,
    theta_a: float = 135.0,
    theta_b: float = 135.0,
    dihedral_a: float = 180.0,
    dihedral_b: float = 0.0,
    dihedral_c: float = 180.0,
) -> numpy.ndarray:
    """Create coords for 3 'receptor' and 3 ligand atoms in a ``\\/ \\/`` shape ."""
    # initial theta_a and theta_b = 135 degrees
    theta_initial = 135.0

    coords_a = numpy.array([[-2.0, 0.0, 0.0], [-1.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
    coords_b = numpy.array(
        [[dist, 0.0, 0.0], [dist + 1.0, -1.0, 0.0], [dist + 2.0, 0.0, 0.0]]
    )

    delta_theta_a = numpy.radians(theta_a - theta_initial)
    delta_theta_b = numpy.radians(theta_b - theta_initial)

    delta_dihedral_a = numpy.radians(dihedral_a - 180.0)
    delta_dihedral_b = numpy.radians(dihedral_b - 0.0)
    delta_dihedral_c = numpy.radians(dihedral_c - 180.0)

    rotation = Rotation.from_rotvec(numpy.array([0.0, 0.0, delta_theta_a]))
    coords_b = rotation.apply(coords_b)

    offset = coords_b[0, :].copy()
    coords_b -= offset

    rotation = Rotation.from_rotvec(numpy.array([0.0, 0.0, delta_theta_b]))
    coords_b = rotation.apply(coords_b) + offset

    coords = [coords_a]

    rotation = create_dihedral_rotation(
        coords_a[1, :], coords_a[2, :], delta_dihedral_a
    )
    coords_b = rotation.apply(coords_b - coords_a[2, :]) + coords_a[2, :]
    coords.append(coords_b[0, :].copy())

    rotation = create_dihedral_rotation(
        coords_a[2, :], coords_b[0, :], delta_dihedral_b
    )
    coords_b = rotation.apply(coords_b - coords[-1]) + coords[-1]
    coords.append(coords_b[1, :].copy())

    rotation = create_dihedral_rotation(
        coords_b[0, :], coords_b[1, :], delta_dihedral_c
    )
    coords_b = rotation.apply(coords_b - coords[-1]) + coords[-1]
    coords.append(coords_b[2, :].copy())

    return numpy.vstack(coords)


@pytest.mark.parametrize(
    "distance, expected_energy",
    [
        (0.0, 0.0 * openmm.unit.kilocalorie_per_mole),
        (1.0, 0.0 * openmm.unit.kilocalorie_per_mole),
        # distance - radius ** 2
        (2.0, (2.0 - 1.5) ** 2 * openmm.unit.kilocalorie_per_mole),
    ],
)
def test_create_flat_bottom_restraint(distance, expected_energy):
    particle_idx = 1
    particle_coords = {particle_idx: openmm.Vec3(0.0, 0.0, 0.0) * _ANGSTROM}

    force = create_flat_bottom_restraint(
        config=femto.md.config.FlatBottomRestraint(
            k=2.0 * _KCAL_PER_ANG_SQR, radius=1.5 * _ANGSTROM
        ),
        coords=particle_coords,
    )
    assert force.getNumParticles() == len(particle_coords)
    assert force.getParticleParameters(0)[0] == particle_idx

    assert (
        force.getForceGroup() == femto.md.constants.OpenMMForceGroup.POSITION_RESTRAINT
    )
    assert force.getName() == femto.md.constants.OpenMMForceName.POSITION_RESTRAINT

    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    system.addForce(force)

    energy = femto.md.utils.openmm.compute_energy(
        system,
        [openmm.Vec3(5.0, 0.0, 0.0), openmm.Vec3(distance, 0.0, 0.0)] * _ANGSTROM,
        None,
        platform=femto.md.constants.OpenMMPlatform.REFERENCE,
    )

    assert femto.md.utils.openmm.is_close(energy, expected_energy)


def test_create_position_restraints():
    topology = build_mock_structure(["CO", "O", "C", "[Na+]", "O"])

    topology.residues[0].name = femto.md.constants.LIGAND_1_RESIDUE_NAME
    topology.residues[2].name = femto.md.constants.LIGAND_2_RESIDUE_NAME

    topology.xyz = (
        numpy.array([[float(i), 0.0, 0.0] for i in range(len(topology.atoms))])
        * _ANGSTROM
    )

    expected_k = 25.0 * openmm.unit.kilocalorie_per_mole / _ANGSTROM**2
    expected_radius = 1.5 * _ANGSTROM

    # ligand 1 and 2 excluding hydrogens
    restraint_mask = "not (water or ion or elem H)"

    restraint = create_position_restraints(
        topology,
        restraint_mask,
        femto.md.config.FlatBottomRestraint(k=expected_k, radius=expected_radius),
    )
    assert isinstance(restraint, openmm.CustomExternalForce)

    restraint_parameters = [
        restraint.getParticleParameters(i) for i in range(restraint.getNumParticles())
    ]
    expected_parameters = [
        (0, (0.0 * _ANGSTROM, 0.0 * _ANGSTROM, 0.0 * _ANGSTROM)),  # CO carbon
        (1, (1.0 * _ANGSTROM, 0.0 * _ANGSTROM, 0.0 * _ANGSTROM)),  # CO oxygen
        (9, (9.0 * _ANGSTROM, 0.0 * _ANGSTROM, 0.0 * _ANGSTROM)),  # C  carbon
    ]

    assert len(restraint_parameters) == len(expected_parameters)

    for (_, values), (_, expected_values) in zip(
        restraint_parameters, expected_parameters, strict=True
    ):
        x, y, z, k, tol = values
        expected_x, expected_y, expected_z = expected_values

        assert numpy.isclose(x, expected_x.value_in_unit_system(_MD_UNITS))
        assert numpy.isclose(y, expected_y.value_in_unit_system(_MD_UNITS))
        assert numpy.isclose(z, expected_z.value_in_unit_system(_MD_UNITS))

        assert numpy.isclose(k, expected_k.value_in_unit_system(_MD_UNITS))
        assert numpy.isclose(tol, expected_radius.value_in_unit_system(_MD_UNITS))


def test_create_boresch_restraint():
    """Make sure boresch restraints get the correct parameters and initial geometry."""
    expected_distance = numpy.random.uniform(2.0, 3.0) * _ANGSTROM
    expected_angle_a = numpy.random.uniform(45.0, 180.0) * openmm.unit.degree
    expected_angle_b = numpy.random.uniform(45.0, 180.0) * openmm.unit.degree
    expected_dihedral_a = numpy.random.uniform(-180.0, 180.0) * openmm.unit.degree
    expected_dihedral_b = numpy.random.uniform(-180.0, 180.0) * openmm.unit.degree
    expected_dihedral_c = numpy.random.uniform(-180.0, 180.0) * openmm.unit.degree

    expected_k_distance = 1.0 * _KCAL_PER_ANG_SQR
    expected_k_angle_a = 2.0 * _KCAL_PER_RAD_SQR
    expected_k_angle_b = 3.0 * _KCAL_PER_RAD_SQR
    expected_k_dihedral_a = 4.0 * _KCAL_PER_RAD_SQR
    expected_k_dihedral_b = 5.0 * _KCAL_PER_RAD_SQR
    expected_k_dihedral_c = 6.0 * _KCAL_PER_RAD_SQR

    coords = mock_boresch_coords(
        expected_distance.value_in_unit(_ANGSTROM),
        expected_angle_a.value_in_unit(openmm.unit.degrees),
        expected_angle_b.value_in_unit(openmm.unit.degrees),
        expected_dihedral_a.value_in_unit(openmm.unit.degrees),
        expected_dihedral_b.value_in_unit(openmm.unit.degrees),
        expected_dihedral_c.value_in_unit(openmm.unit.degrees),
    )

    force = create_boresch_restraint(
        femto.md.config.BoreschRestraint(
            k_distance=expected_k_distance,
            k_angle_a=expected_k_angle_a,
            k_angle_b=expected_k_angle_b,
            k_dihedral_a=expected_k_dihedral_a,
            k_dihedral_b=expected_k_dihedral_b,
            k_dihedral_c=expected_k_dihedral_c,
        ),
        (0, 1, 2),
        (3, 4, 5),
        coords * _ANGSTROM,
    )

    expected_parameters = [
        ("k_dist_a", expected_k_distance),
        ("k_theta_a", expected_k_angle_a),
        ("k_theta_b", expected_k_angle_b),
        ("k_phi_a", expected_k_dihedral_a),
        ("k_phi_b", expected_k_dihedral_b),
        ("k_phi_c", expected_k_dihedral_c),
        ("dist_0", expected_distance),
        ("theta_a_0", expected_angle_a),
        ("theta_b_0", expected_angle_b),
        ("phi_a_0", expected_dihedral_a),
        ("phi_b_0", expected_dihedral_b),
        ("phi_c_0", expected_dihedral_c),
    ]

    assert force.getNumBonds() == 1

    assert force.getNumPerBondParameters() == len(expected_parameters)
    actual_idxs, actual_values = force.getBondParameters(0)

    assert actual_idxs == (0, 1, 2, 3, 4, 5)

    for i in range(force.getNumPerBondParameters()):
        expected_key, expected_value = expected_parameters[i]

        found_key = force.getPerBondParameterName(i)
        assert found_key == expected_key

        expected_value = expected_value.value_in_unit_system(openmm.unit.md_unit_system)
        actual_value = actual_values[i]
        assert numpy.isclose(actual_value, expected_value)


def test_create_boresch_restraint_ctx_param():
    coords = mock_boresch_coords(1, 1, 1, 1, 1, 1)

    expected_ctx_parameter = "lambda_boresch"

    force = create_boresch_restraint(
        femto.md.config.BoreschRestraint(
            k_distance=1.0 * _KCAL_PER_ANG_SQR,
            k_angle_a=1.0 * _KCAL_PER_RAD_SQR,
            k_angle_b=1.0 * _KCAL_PER_RAD_SQR,
            k_dihedral_a=1.0 * _KCAL_PER_RAD_SQR,
            k_dihedral_b=1.0 * _KCAL_PER_RAD_SQR,
            k_dihedral_c=1.0 * _KCAL_PER_RAD_SQR,
        ),
        (0, 1, 2),
        (3, 4, 5),
        coords * _ANGSTROM,
        expected_ctx_parameter,
    )

    assert force.getEnergyFunction().startswith(f"{expected_ctx_parameter} * 0.5")
    assert force.getNumGlobalParameters() == 1
    assert expected_ctx_parameter in force.getGlobalParameterName(0)


@pytest.mark.parametrize(
    "coords, expected_energy",
    [
        (mock_boresch_coords(dist=3.0), 0.5 * 1.0 * (3.0 - 1.0) ** 2 * _KCAL),
        (
            mock_boresch_coords(theta_a=150.0),
            0.5 * 2.0 * numpy.radians(150.0 - 135.0) ** 2 * _KCAL,
        ),
        (
            mock_boresch_coords(theta_b=145.0),
            0.5 * 3.0 * numpy.radians(145.0 - 135.0) ** 2 * _KCAL,
        ),
        (
            mock_boresch_coords(dihedral_a=175.0),
            0.5 * 4.0 * numpy.radians(175 - 180.0) ** 2 * _KCAL,
        ),
        (
            mock_boresch_coords(dihedral_a=-175.0),
            0.5 * 4.0 * numpy.radians(175 - 180.0) ** 2 * _KCAL,
        ),
        (
            mock_boresch_coords(dihedral_b=10.0),
            0.5 * 5.0 * numpy.radians(10.0 - 0.0) ** 2 * _KCAL,
        ),
        (
            mock_boresch_coords(dihedral_b=-10.0),
            0.5 * 5.0 * numpy.radians(10.0 - 0.0) ** 2 * _KCAL,
        ),
        (
            mock_boresch_coords(dihedral_c=175.0),
            0.5 * 6.0 * numpy.radians(175 - 180.0) ** 2 * _KCAL,
        ),
        (
            mock_boresch_coords(dihedral_c=-175.0),
            0.5 * 6.0 * numpy.radians(175 - 180.0) ** 2 * _KCAL,
        ),
    ],
)
def test_boresch_restraint_energy(coords, expected_energy, mock_boresch_config):
    """Make sure the end points of the restraint do not produce NaNs."""

    coords_0 = mock_boresch_coords() * _ANGSTROM
    force = create_boresch_restraint(
        mock_boresch_config, (0, 1, 2), (3, 4, 5), coords_0
    )

    system = openmm.System()
    for _ in range(6):
        system.addParticle(1.0)
    system.addForce(force)

    energy_0 = femto.md.utils.openmm.compute_energy(
        system, coords_0, None, platform=femto.md.constants.OpenMMPlatform.REFERENCE
    )
    assert femto.md.utils.openmm.is_close(energy_0, 0.0 * _KCAL)

    energy = femto.md.utils.openmm.compute_energy(
        system,
        coords * _ANGSTROM,
        None,
        platform=femto.md.constants.OpenMMPlatform.REFERENCE,
    )

    assert femto.md.utils.openmm.is_close(energy, expected_energy)
