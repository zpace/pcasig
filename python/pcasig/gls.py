# encoding: utf-8
#
# @Author: Zachary Pace
# @Date: 23 Oct 2019
# @Filename: gls.py
# @License: BSD
# @Copyright: 2019

import numpy as np
import scipy.linalg as spla

def cov_chol_lower(sig):
    """find lower-triangular Cholesky decomposition of covariance matrix
    
    compute Cholesky decomposition, eschewing all the normal checks
    
    Parameters
    ----------
    sig : :class:`~numpy:numpy.ndarray`
        array shape (l, l)
    """

    return spla.cholesky(sig, lower=True, check_finite=False)

def solve_downproject_gls(y, X, sig):
    """solve PCA downprojection from data-space to PC-space with generalized least-squares
    
    project data down from an observation vector to a principal component amplitude
    vector, subject to covariate uncertainties on data, using GLS

    see https://web.stanford.edu/class/stats253/lectures/lect3.pdf
    
    Parameters
    ----------
    y : :class:`~numpy:numpy.ndarray`
        array shape (l, ), representing observations
    X : :class:`~numpy:numpy.ndarray`
        array shape (l, q) describing principal component vectors (q << l)
    sig : :class:`~numpy:numpy.ndarray`
        array shape (l, l) describing covariance of measurements
    """
    l, q = X.shape
    
    # compute lower-triangular cholesky decomposition of cov
    L = cov_chol_lower(sig)
    # and L's inverse by back-substitution
    L_inv = spla.solve_triangular(L, np.eye(l), lower=True, check_finite=False)

    # solve L z = y by back-substitution
    z = spla.solve_triangular(L, y, lower=True, check_finite=False)

    # and solve L W_i = X_i for elements of W, rows of X
    # again by back-substitution
    W = spla.solve_triangular(L, X, lower=True, check_finite=False)

    # and regress z on W
    a, *_ = spla.lstsq(W, z, check_finite=False)

    # Covariance of solution (a) comes from residuals whitened by L...
    # and inverse of whitened transformation vectors.
    # This is what statsmodels does for its HC0 robust covariance
    resid = (X @ a) - y
    wresid = np.dot(L_inv, resid)  # whitened residuals
    a_sig_scale = wresid**2.
    pinv_wX = spla.pinv(np.dot(L_inv, X))  # inverse of whitened evecs
    a_sig = np.dot(pinv_wX, a_sig_scale[:, None] * pinv_wX.T)

    return a, a_sig, resid