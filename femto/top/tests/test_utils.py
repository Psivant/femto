import numpy

from femto.top._utils import compute_pairwise_distances


def test_compute_pairwise_distances():
    xyz_a = numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    xyz_b = numpy.array([[2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    box = numpy.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])

    dists = compute_pairwise_distances(xyz_a, xyz_b, box)
    expected_dists = numpy.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    assert dists.shape == expected_dists.shape
    assert numpy.allclose(dists, expected_dists)

    dists = compute_pairwise_distances(xyz_a, xyz_b, None)
    expected_dists = numpy.array([[2.0, 1.0], [1.0, 0.0], [3.0, 2.0]])

    assert dists.shape == expected_dists.shape
    assert numpy.allclose(dists, expected_dists)
