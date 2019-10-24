# encoding: utf-8
#
# gls.py


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from pytest import mark, raises

from pcasig.gls import solve_downproject_gls, cov_chol_lower
from pcasig.utils import *

import numpy as np
from scipy import linalg as spla

class TestDownproject(object):
    """Tests for downprojection
    """
    @mark.parametrize(('l', 'q'), [(100, 30), (200, 31)])
    def test_result_shapes(self, l, q):
        """test shapes of results
        
        Parameters
        ----------
        l : int
            number of observations
        q : int
            number of PCs
        """

        K = gen_cov(l)
        eigvals, X = gen_pca_system(l, q)
        assert eigvals.shape == (q, ), \
               'should have q = {} eigenvalues: is {}'.format(
                    q, eigvals.shape)
        assert X.shape == (l, q), \
               'should have (l, q) = ({}, {}) eigenvalue system: is {}'.format(
                    l, q, X.shape)

        a, a_sig, resid = solve_downproject_gls(
            y=np.zeros(l), X=X, sig=K)

        assert a.shape == (q, ), \
               'should have q = {} solution: is {}'.format(
                    q, a.shape)
        assert a_sig.shape == (q, q), \
               'should have (q, q) = ({}, {}) covariance: is {}'.format(
                    q, q, a_sig.shape)
        assert resid.shape == (l, ), \
               'should have l = {} residual: is {}'.format(
                    l, resid.shape)

class TestCovCholLower(object):
    """Tests for cov_chol_lower
    """

    @mark.parametrize(('l', 'result'), [(10, np.eye(10)), (100, np.eye(100))])
    def test_eye(self, l, result):
        '''Cholesky of identity should be identity
        
        Parameters
        ----------
        l : {int}
            size of vector
        '''
        assert np.allclose(cov_chol_lower(np.eye(l)), result)

    @mark.parametrize(('K', 'exc'), 
                      [(np.ones((10, 10)), spla.LinAlgError),
                       (np.eye(10, 9), ValueError)])
    def test_nonspd(self, K, exc):
        with raises(exc):
            res = cov_chol_lower(K)
