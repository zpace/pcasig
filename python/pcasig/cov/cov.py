# encoding: utf-8
#
# @Author: Zachary Pace
# @Date: 23 Oct 2019
# @Filename: gls.py
# @License: BSD
# @Copyright: 2019

import numpy as np
from dataclasses import dataclass

@dataclass
class CovarianceWithAddedVariance:
    """class of covariance matrices with an added diagonal component
    
    provides convenience methods for accessing total covariance matrix
    for sum of some common covariance matrix and an additional diagonal term
    
    Attributes
    ----------
    K: np.ndarray : float
        common covariance matrix
    var: np.ndarray : float
        additional diagonal component (first dim must be same as K)
    """
    K: np.ndarray
    var: np.ndarray

    def __post_init__(self):
        K, var = self.K, self.var
        assert len(K.shape) == 2, \
               'K must be 2d'
        assert K.shape[0] == K.shape[1], \
               'K must be square'
        assert K.shape[0] == var.shape[0], \
               'var must have channel incremented on axis 0'

        self.n, *self.spatial_shape = var.shape

    def __getitem__(self, *args, **kwargs):
        diag = np.diag(self.var.__getitem(*args, **kwargs))
        return self.K + diag