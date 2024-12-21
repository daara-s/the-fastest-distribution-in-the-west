import numpy as np
from numba import njit


@njit
def numba_poisson(lam, size):
    return np.random.poisson(lam, size)

@njit
def numba_rng_poisson(rng, lam, size):
    return rng.poisson(lam, size)
