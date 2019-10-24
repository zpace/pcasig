# encoding: utf-8
#
# @Author: Zachary Pace
# @Date: 23 Oct 2019
# @Filename: gls.py
# @License: BSD
# @Copyright: 2019

import numpy as np
import matplotlib.pyplot as plt

import pcasig.gls as gls
import pcasig.generate as gen

def example_gls(l, q, perturb_prob=0., perturb_std=10.):
    """Example randomized GLS downprojection with outliers
    
    An example of downprojection, with randomly-initialized outliers
    
    Parameters
    ----------
    l : int
        dimension of observations
    q : int
        dimension of PC system
    perturb_prob : {number}, optional
        probability that a given observation dimension is an outlier
        (the default is 0., which excludes outliers)
    perturb_std : {number}, optional
        standard-deviation of outlier distribution
        (the default is 10.)
    """
    K = gen.gen_cov(l)
    pc_eigv, X = gen.gen_pca_system(l, q)
    a0 = np.random.randn(q) * pc_eigv
    y0 = a0 @ X.T
    y = np.random.multivariate_normal(y0, Kobs)

    is_perturbed = np.random.rand(l) < perturb_prob
    y[is_perturbed] = perturb_std * \
                      np.random.randn(l)[is_perturbed]

    sol, asol_sig, asol_resid = gls.solve_downproject_gls(
        y, X, Kobs)
    yrecon = asol @ X.T

    plt.scatter(y0, y, edgecolor='None', label='obs')
    plt.scatter(y0[is_perturbed], y[is_perturbed],
                edgecolor='None', label='perturbed',
                marker='x')
    plt.scatter(y0, yrecon, edgecolor='None', label='recon')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    l, q = 200, 5
    p, s = .03, 10.
    example_gls(l, q, p, s)