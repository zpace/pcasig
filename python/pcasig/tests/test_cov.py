# encoding: utf-8
#
# cov.py


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

from pytest import mark, raises

from pcasig.utils import *
from pcasig.cov import CovarianceWithAddedVariance

import numpy as np
from scipy import linalg as spla

class TestCovarianceWithAddedVariance(object):
    """Tests for covariance
    """
    
    @mark.parametrize(('K', 'var'), [(np.ones(10), np.ones((10, 2, 2))),
                                     (np.eye(10, 9), np.ones((10, 2, 2))),
                                     (np.eye(10), np.ones((9, 2, 2))),
                                     (np.eye(10), np.ones(10))])
    def test_invalid_input_shapes(self, K, var):
        with raises(AssertionError):
            cov = CovarianceWithAddedVariance(K, var)

    @mark.parametrize(('K', 'var'), [(np.eye(10), np.ones((10, 2, 3)))])
    def test_valid_input_shapes(self, K, var):
        cov = CovarianceWithAddedVariance(K, var)
        assert tuple(cov.spatial_shape) == (2, 3)
        assert cov.n == 10
        assert cov[:, 0, 0].shape == (10, 10)

    def test_invalid_slice_access(self):
        ktot = CovarianceWithAddedVariance(np.eye(10), np.ones((10, 2, 2)))
        with raises(AssertionError):
            kitem = ktot[:, 2:5, 2:5]
