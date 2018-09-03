import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix, csc_matrix, hstack, csr_matrix
from typing import List


def fano_factor(interval_vector):
    """
    Calculates Fano factor according to
    :param interval_vector:
    :return:
    """
    raise NotImplementedError


def coefficient_of_variation(interval_vector):
    """
    Calculates coefficient_of_variation according to $$\frac{E[\tau]}{Var[\tau]}
    :param interval_vector:
    :return:
    """
    return np.mean(interval_vector.flatten()) / np.std(interval_vector.flatten())


def sparse_tensor_convolution(a: List[dok_matrix], b: List[dok_matrix]) -> np.ndarray:
    """

    :param a: list of sparse
    :param b: list of sparse
    :return:
    """
    a_cpy = a.copy()[::-1]
    out = np.empty((len(a)))
    depth_reduced_b = hstack(b).tocsr()
    a_cpy += [csr_matrix(np.zeros(b[0].shape)) for k in range(len(b))]
    for k in range(len(out)):
        depth_reduced_a = hstack(a_cpy[k:(k + len(b))]).tocsr()
        out[k] = np.sum(depth_reduced_a * depth_reduced_b.T)
    return out[::-1]


def trans_to_symm(A: np.ndarray, diag_value=None):
    out = np.maximum(A, A.transpose())
    if diag_value is None:
        return out
    else:
        diag_ind = np.arange(A.shape[0])
        out[diag_ind, diag_ind] = diag_value
        return out

# a = tf.SparseTensor(indices=[[0,2,0],[1,1,0],[0,0,1],[1,1,1],[2,2,2],[1,1,2]]
#                       , values=[1,1,1,1,1,1],
#                         dense_shape=[3,3,3])
# b = tf.SparseTensor(indices=[[1,2,0],[1,1,0],[2,0,1],[1,1,1]]
#                       , values=[1,1,1,1],
#                         dense_shape=[3,3,2])

# a = [dok_matrix(np.random.randint(0, 2, (3, 3))) for x in range(4)]
# b = [dok_matrix(np.random.randint(0, 3, (3, 3))) for x in range(2)]
# out = np.empty((len(a)))
# depth_reduced_b = hstack(b).tocsr()
# a += [csr_matrix(np.zeros(b[0].shape)) for k in range(len(b))]
# for k in range(len(out)):
#     depth_reduced_a = hstack(a[k:(k + len(b))]).tocsr()
#     out[k] = np.sum(depth_reduced_a * depth_reduced_b.T)
