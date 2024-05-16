"""Estimate free energies using the UWHAM method [1]

References:
    [1]: Zhiqiang Tan, Emilio Gallicchio, Mauro Lapelosa, and Ronald M. Levy,
         "Theory of binless multi-state free energy estimation with applications
         to protein-ligand binding", J. Chem. Phys. 136, 144102 (2012)
"""

import functools

import numpy
import scipy.optimize
import scipy.special


def _compute_weights(ln_z: numpy.ndarray, ln_q: numpy.ndarray, factor: numpy.ndarray):
    q_ij = numpy.exp(ln_q - ln_z)
    return q_ij / (factor * q_ij).sum(axis=-1, keepdims=True)


def _compute_kappa_hessian(
    ln_z: numpy.ndarray, ln_q: numpy.ndarray, factor: numpy.ndarray, n: int
) -> numpy.ndarray:
    ln_z = numpy.insert(ln_z, 0, 0.0)

    w = (factor * _compute_weights(ln_z, ln_q, factor))[:, 1:]
    return -w.T @ w / n + numpy.diag(w.sum(axis=0) / n)


def _compute_kappa(
    ln_z: numpy.ndarray, ln_q: numpy.ndarray, factor: numpy.ndarray, n: int
) -> tuple[numpy.ndarray, numpy.ndarray]:
    ln_z = numpy.insert(ln_z, 0, 0.0)

    ln_q_ij_sum = scipy.special.logsumexp(a=ln_q - ln_z, b=factor, axis=1)
    kappa = ln_q_ij_sum.sum() / n + (factor * ln_z).sum()

    w = factor * _compute_weights(ln_z, ln_q, factor)
    grad = -w[:, 1:].sum(axis=0) / n + factor[1:]

    return kappa, grad


def _compute_variance(
    ln_z: numpy.ndarray, w: numpy.ndarray, factor: numpy.ndarray, n: int
) -> numpy.ndarray:
    o = w.T @ w / n

    b = o * factor - numpy.eye(len(ln_z))
    b = b[1:, 1:]

    b_inv_a = -o + o[0, :]
    b_inv_a = b_inv_a[1:, 1:]

    var_matrix = (b_inv_a @ numpy.linalg.inv(b.T)) / n
    return numpy.insert(numpy.diag(var_matrix), 0, 0.0)


def estimate_f_i(
    u_kn: numpy.ndarray, n_k: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Estimates the free energies of a set of *sampled* states.

    Args:
        u_kn: The uncorrelated reduced potentials sampled at ``k`` states with
            ``shape=(n_states, n_samples)``.
        n_k: The number of samples at state ``k``.

    Returns:
        The estimated reduced free energies and their estimated variance.
    """

    u_kn = numpy.array(u_kn)
    n_k = numpy.array(n_k)

    ln_q = -u_kn.T

    n_samples, n_states = ln_q.shape

    if n_states != len(n_k):
        raise RuntimeError("The number of states do not match")
    if n_samples != n_k.sum():
        raise RuntimeError("The number of samples do not match")

    ln_z = numpy.zeros(len(n_k) - 1)  # ln_z_0 is always fixed at 0.0
    ln_q -= ln_q[:, :1]

    n = n_k.sum()
    factor = n_k / n

    result = scipy.optimize.minimize(
        functools.partial(_compute_kappa, ln_q=ln_q, n=n, factor=factor),
        ln_z,
        method="trust-ncg",
        jac=True,
        hess=functools.partial(_compute_kappa_hessian, ln_q=ln_q, n=n, factor=factor),
    )

    if not result.success:
        raise RuntimeError("The UWHAM minimization failed to converge.")

    f_i = numpy.insert(-result.x, 0, 0.0)
    ln_z = numpy.insert(result.x, 0, 0.0)

    weights = _compute_weights(ln_z, ln_q, factor)

    if not numpy.allclose(weights.sum(axis=0) / n, 1.0, atol=1e-3):
        raise RuntimeError("The UWHAM weights do not sum to 1.0")

    df_i = _compute_variance(ln_z, weights, factor, n)

    return f_i, df_i, weights / n
