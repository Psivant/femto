import numpy
import pytest

from femto.md.utils.geometry import (
    compute_angles,
    compute_bond_vectors,
    compute_dihedrals,
    compute_distances,
)


@pytest.mark.parametrize(
    "geometry_function", [compute_angles, compute_dihedrals, compute_bond_vectors]
)
def test_compute_geometry_no_atoms(geometry_function):
    valence_terms = geometry_function(numpy.array([]), numpy.array([]))

    if not isinstance(valence_terms, tuple):
        valence_terms = (valence_terms,)

    assert all(term.shape == () for term in valence_terms)


def test_compute_bond_vectors():
    coords = numpy.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    idxs = numpy.array([[2, 0], [0, 1]])

    bond_vectors, bond_norms = compute_bond_vectors(coords, idxs)

    assert bond_vectors.shape == (len(idxs), 3)
    assert bond_norms.shape == (len(idxs),)

    assert numpy.allclose(
        bond_vectors, numpy.array([[0.0, -3.0, 0.0], [2.0, 0.0, 0.0]])
    )
    assert numpy.allclose(bond_norms, numpy.array([3.0, 2.0]))


def test_compute_distances():
    coords = numpy.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    idxs = numpy.array([[2, 0], [0, 1]])

    distances = compute_distances(coords, idxs)

    assert distances.shape == (len(idxs),)
    assert numpy.allclose(distances, numpy.array([3.0, 2.0]))


def test_compute_angles():
    coords = numpy.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    idxs = numpy.array([[0, 1, 2], [1, 0, 2], [0, 1, 3]])

    angles = compute_angles(coords, idxs)

    assert angles.shape == (len(idxs),)
    assert numpy.allclose(angles, numpy.array([numpy.pi / 2, numpy.pi / 4, numpy.pi]))


@pytest.mark.parametrize("phi_sign", [-1.0, 1.0])
def test_compute_dihedrals(phi_sign):
    coords = numpy.array(
        [[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, phi_sign]]
    )
    idxs = numpy.array([[0, 1, 2, 3]])

    dihedrals = compute_dihedrals(coords, idxs)

    assert dihedrals.shape == (len(idxs),)
    assert numpy.allclose(dihedrals, numpy.array([phi_sign * numpy.pi / 4.0]))
