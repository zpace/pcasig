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
    
    @mark.parametrize(('K', 'var'), [(np.ones(10), np.ones(10)),
                                     (np.eye(10, 9), np.ones(10)),
                                     (np.eye(10), np.ones(9))])
    def test_input_shapes(self, K, var):
        with raises(AssertionError):
            cov = CovarianceWithAddedVariance(K, var)