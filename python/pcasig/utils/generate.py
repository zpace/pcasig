# encoding: utf-8
#
# @Author: Zachary Pace
# @Date: 23 Oct 2019
# @Filename: gls.py
# @License: BSD
# @Copyright: 2019

import numpy as np
from sklearn.datasets import make_spd_matrix

def gen_cov(n):
    """make random covariance matrix
    
    Parameters
    ----------
    n : {int}
        dimension of covariance

    Returns
    -------
    :class:`~numpy:numpy.ndarray`
        shape `(n, n)` covariance array
    """
    
    return make_spd_matrix(n)

def gen_pca_system(l, q):
    """make random PCA linear system
    
    Parameters
    ----------
    l : {int}
        dimension of observations
    q : {int}
        dimension of PC system

    Returns
    -------
    :class:`~numpy:numpy.ndarray`
        shape `(l, )` array of eigenvalues
    :class:`~numpy:numpy.ndarray`
        shape `(l, q)` array of `l` features and `q` eigenvectors
    """

    Ktot = gen_cov(l)
    evals_, evecs_ = np.linalg.eigh(Ktot)
    # return top q eigenvalues and eigenvectors
    return evals_[::-1][:q], evecs_[:, ::-1][:, :q]