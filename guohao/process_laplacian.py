"""
this file helps us work with the graph Laplacian
"""
from scipy.spatial import ConvexHull
from scipy import sparse
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
import igl
import torch

from ghcn_helper import df2points


def compute_cotan_laplacian(df):
    """
    compute mass matrix and cotan laplacian matrix
    :param df: pandas data frame
    :return L, M, in that order, CSR-sparse matrix
    """
    points = df2points(df).T
    assert points.shape[-1] == 3
    ch = ConvexHull(points)
    assert points.shape[0] == ch.points.shape[0]
    L = -csr_matrix(igl.cotmatrix(ch.points, ch.simplices))  # positive semidefiniteness
    # ISSUE: some areas are way too small such that igl ends up considering it exactly zero.
    # as a result, it's not even stored in the sparse matrix. coarsening needed
    M = csr_matrix(igl.massmatrix(ch.points, ch.simplices, igl.MASSMATRIX_TYPE_VORONOI))
    return L, M


def scale_laplacian(laplacian, lmax):
    """
    scale Laplacian to make all eigenvalues between [-1, 1]
    :param laplacian: sparse matrix of format CSR
    :param lmax: externally provided approximated largest eigenvalue
    :return sparse matrix format COO
    """
    I = sparse.identity(laplacian.shape[0], format=laplacian.format, dtype=laplacian.dtype)
    scale = 1
    laplacian *= 2 * scale / lmax
    laplacian -= I
    return coo_matrix(laplacian)


def mask_laplacian(laplacian, column_indices):
    """
    THIS creates a deep copy of L before doing inplace masking
    :param laplacian: COO-format sparse matrix
    :param column_indices: vertices to mask
    :return a COO-format sparse Laplacian
    """
    assert column_indices.dtype == np.long
    assert laplacian.format == "coo"
    laplacian_sparse = laplacian.tocsr()  # this actually creates a deep copy if L is not already csr
    assert laplacian_sparse is not laplacian
    I = sparse.identity(laplacian.shape[0], format="csr")
    I[column_indices, column_indices] = 0.0
    laplacian_sparse = laplacian_sparse @ I
    laplacian_sparse = laplacian_sparse.tolil()
    laplacian_sparse.setdiag(0.0)
    laplacian_sparse.setdiag(-np.asarray(laplacian_sparse.sum(axis=1)).squeeze())
    laplacian = laplacian_sparse.tocoo()
    return laplacian


def scipy2torch(laplacian):
    """
    :param laplacian: COO-format sparse matrix
    """
    # PyTorch wants a LongTensor (int64) as indices (it'll otherwise convert).
    indices = np.empty((2, laplacian.nnz), dtype=np.int64)
    np.stack((laplacian.row, laplacian.col), axis=0, out=indices)
    indices = torch.from_numpy(indices)

    laplacian = torch.sparse_coo_tensor(indices, laplacian.data, laplacian.shape)
    laplacian = laplacian.coalesce()  # More efficient subsequent operations.
    return laplacian


def combine_cotan_mass(L, M):
    """
    will probably throw the "change sparsity pattern" warning.
    this is because a few entries in M are so small that igl takes them as zeros
    and doesn't store them in the sparse M.
    :param L, CSR-sparse
    :param M: CSR-sparse
    :return inv(M) @ L, CSR-sparse
    """
    inv = M.copy()
    for i in range(inv.shape[0]):
        if np.isclose(inv[i, i], 0.0):
            raise Exception("WAY TOO SMALL AREA; CONSIDER PRE-COARSENING!")
        inv[i, i] = 1.0 / inv[i, i]
    return inv @ L


def _power_iteration(laplacian, tol, max_steps):
    """
    a quick and dirty implementation of power iteration to find lmax
    :param laplacian: actually M^-1 @ L, csr-sparse matrix
    :param tol: absolute tolerance
    :param max_steps: self-explanatory
    :return: the largest eigenpair (val, vec)
    """
    x = np.random.normal(size=laplacian.shape[0])
    x = x / np.linalg.norm(x)
    for step in range(max_steps):
        y = laplacian @ x
        y = y / np.linalg.norm(y)
        # convergence criterion
        cosine = x @ y  # should converge to one
        if (step + 1) % 1000 == 0:
            print("Loss={} at step {}".format(1-cosine, step))
        if 1 - cosine < tol:
            print("Successfully converged after {} steps".format(step))
            eigval = y.T @ (laplacian @ y)
            return eigval, y
        x = y
    print("Failed to converge within {} steps with loss = {}".format(max_steps, 1 - cosine))
    return None, None


def compute_lmax(laplacian):
    """
    :param laplacian: inv(M) @ L, CSR-sparse matrix
    :return: lmax
    """
    lmax, eigvec = _power_iteration(laplacian, tol=1E-15, max_steps=10000)
    assert (lmax is not None), "Power iteration fails to converge"  # power iter has to converge
    # check how good it is
    lhs = laplacian @ eigvec
    rhs = lmax * eigvec
    diff = np.linalg.norm(lhs - rhs)
    print("Difference between ML @ v and lambda * v: ", diff)
    return lmax


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

