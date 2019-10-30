# encoding: utf-8
#
# @Author: Zachary Pace
# @Date: 23 Oct 2019
# @Filename: gls.py
# @License: BSD
# @Copyright: 2019

import numpy as np
import dataclasses

@dataclasses.dataclass
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

        assert len(self.spatial_shape) >= 1, \
               'spatial_shape should be non-negligible'

    def __getitem__(self, *args, **kwargs):
        diag_select = self.var.__getitem__(*args, **kwargs)
        assert np.array(diag_select).ndim == 1, \
               'view of variance must be 1d'
        diag = np.diag(diag_select)
        return self.K + diag