from scipy.spatial import ConvexHull
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, LinearOperator, lobpcg
import numpy as np
import igl


def diagonal_inverse(M, regularizer=1E-5):
    inv = M.copy()
    print("TYPE INV: ", inv.dtype)
    for i in range(inv.shape[0]):
        if np.isclose(inv[i, i], 0.0):
            print("CLOSE TO ZERO: {}".format(inv[i, i]))
            inv[i, i] += regularizer
        inv[i, i] = 1.0 / inv[i, i]
    return inv


def compute_cotan_laplacian(points):
    """
    compute mass matrix and cotan laplacian matrix
    :param points: shaped as (N, 3), point cloud sampled on S^2
    :return L, M, in that order
    """
    assert points.shape[-1] == 3
    ch = ConvexHull(points)
    assert points.shape[0] == ch.points.shape[0]
    L = -csr_matrix(igl.cotmatrix(ch.points, ch.simplices))  # positive semidefiniteness
    M = csr_matrix(igl.massmatrix(ch.points, ch.simplices, igl.MASSMATRIX_TYPE_VORONOI))
    return L, M


def power_iteration(ML, tol, max_steps):
    x = np.random.normal(size=ML.shape[0])
    x = x / np.linalg.norm(x)
    for step in range(max_steps):
        y = ML @ x
        y = y / np.linalg.norm(y)
        # convergence criterion
        cosine = x @ y  # should converge to one
        if (step + 1) % 1000 == 0:
            print("Loss={} at step {}".format(1-cosine, step))
        if 1 - cosine < tol:
            print("Successfully converged after {} steps".format(step))
            return y
        if step >= max_steps:
            print("Failed to converge within {} steps with loss = {}".format(max_steps, 1 - cosine))
            return y
        x = y
