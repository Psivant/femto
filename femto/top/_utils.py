import numpy
import scipy


def compute_pairwise_distances(
    xyz_a: numpy.ndarray,
    xyz_b: numpy.ndarray,
    box: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """
    Computes all pairwise distances between particles in a periodic simulation box.

    Args:
        xyz_a: The coordinates of the first set of particles with
            ``shape=(n_atoms_a, 3)``.
        xyz_b: The coordinates of the second set of particles with
            ``shape=(n_atoms_a, 3)``.
        box: The box vectors of the simulation box with ``shape=(3, 3)``.

    Returns:
        The pairwise distances with ``shape=(n_atoms_a, n_atoms_b)``.
    """
    if box is None:
        return scipy.spatial.distance.cdist(xyz_a, xyz_b, "euclidean")

    if not numpy.allclose(box, numpy.diag(numpy.diagonal(box))):
        raise NotImplementedError("only orthogonal boxes are supported.")

    box_flat = numpy.diag(box)
    box_inv = 1.0 / box_flat

    delta = xyz_a[:, None, :] - xyz_b[None, :, :]
    delta -= numpy.floor(delta * box_inv[None, None, :] + 0.5) * box_flat[None, None, :]

    return numpy.linalg.norm(delta, axis=-1)
