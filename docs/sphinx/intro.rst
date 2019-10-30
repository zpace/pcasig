
.. _intro:

Introduction to pcasig
===============================

`pcasig` can be used to swiftly transform between coordinate systems, allowing for covariate uncertainty in measurements. 

A simple example
^^^^^^^^^^^^^^^^

Given some covariance matrix `K` (n-by-n), an observation vector `y` (length n), and a linear transformation `X` (n-by-q), generalized least-squares produces the maximum-likelihood solution and its coviariance.

.. code-block:: python3
    
    from pcasig import gls

    a, a_sig, resid = gls.downproject(
        y, # length-n observation vector
        X, # n-by-q linear transofmration system
        K) # n-by-n covariance matrix of observations

A more complete example
^^^^^^^^^^^^^^^^^^^^^^^

Here's a more detailed example, where the true PC solution is known.

.. code-block:: python3

    from pcasig.utils import generate as gen
    from pcasig import gls
    import numpy as np

    # set dimension of underlying system and number of PCs to keep
    n, q = 500, 4

    # generate observational covariance matrix
    K = gen.gen_cov(n)
    
    # generate full set of eigenvectors, and then truncate them
    evals_, evecs_ = gen.gen_pca_system(n, n)
    evals, evecs = evals_[:q], evecs_[:, :q]

    a0_ = np.random.randn(n) * evals_
    a0 = a0_[:q]

    y0 = evecs_ @ a0_
    y_obs = np.random.multivariate_normal(y0, K)

    a_sol, a_sol_sig, resid = gls.downproject(y_obs, evecs, K)

    print(f'RMS={np.std(resid):.2e}')
