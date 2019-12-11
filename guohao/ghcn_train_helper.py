"""
this file contains utilities that would be useful to carry out training algorithm
"""
from scipy.spatial import ConvexHull
from scipy.sparse import csr_matrix
import numpy as np
import igl

from ghcn_helper import df2points


def diagonal_inverse(M, regularizer=1E-7):
    """
    will probably throw the "change sparsity pattern" warning.
    this is because a few entries in M are so small that igl takes them as zeros
    and doesn't store them in the sparse M.
    :param M: mass matrix given by igl
    :param regularizer: my quick hack to solve divide-bby-zero-error
    :return:
    """
    inv = M.copy()
    for i in range(inv.shape[0]):
        if np.isclose(inv[i, i], 0.0):
            print("{}th AREA CLOSE TO ZERO: {}".format(i, inv[i, i]))
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
    # ISSUE: some areas are way too small such that igl ends up considering it exactly zero.
    # as a result, it's not even stored in the sparse matrix.
    # QUICK HACK/FIX: add these small ones by 1E-7
    M = csr_matrix(igl.massmatrix(ch.points, ch.simplices, igl.MASSMATRIX_TYPE_VORONOI))
    return L, M


def power_iteration(ML, tol, max_steps):
    """
    a quick and dirty implementation of power iteration to find lmax
    :param ML: actually M^-1 @ L
    :param tol: absolute tolerance
    :param max_steps: self-explanatory
    :return: the largest eigenpair (val, vec)
    """
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
            eigval = y.T @ (ML @ y)
            return eigval, y
        x = y
    print("Failed to converge within {} steps with loss = {}".format(max_steps, 1 - cosine))
    return None, y


def make_subset_generator(whole_set, gen_type, num_iter, spec):
    """
    :param whole_set: the whole set from which we sample subset
    :param gen_type: types of subset generator
    :param num_iter: number of iterations wanted
    :param spec: other specifications
    """
    assert gen_type in ["ALL"]
    if gen_type == "ALL":
        def subset_generator():
            for _ in range(num_iter):
                yield whole_set
    else:
        raise NotImplemented("Type not implemented yet")
    return subset_generator()


def compute_lmax(df):
    """
    :param df: pandas data frame
    :return: lmax
    """
    points = df2points(df).T
    L, M = compute_cotan_laplacian(points)
    ML = diagonal_inverse(M) @ L
    lmax, eigvec = power_iteration(ML, tol=1E-15, max_steps=10000)
    assert lmax is not None  # power iter has to converge
    # check how good it is
    lhs = ML @ eigvec
    rhs = lmax * eigvec
    diff = np.linalg.norm(lhs - rhs)
    print("Difference between ML @ v and lambda * v: ", diff)
    return lmax

