import numpy as np
from numba import njit, prange


@njit(parallel=True)
def numba_negative_binomial(n, p, size):
    out = np.empty(size, dtype=np.int64)
    for idx in prange(out.size):
        out[idx] = np.random.negative_binomial(n, p)
    return out


@njit
def numba_rng_negative_binomial(rng, n, p, size):
    return rng.negative_binomial(n, p, size)


@njit(parallel=True)
def numba_gamma_poisson(n, p, size):
    assert n > 0
    assert p > 0.0 and p < 1.0

    def _impl(n, p, size):
        out = np.empty(size, dtype=np.int64)
        for idx in prange(out.size):
            Y = np.random.gamma(n, (1.0 - p) / p)
            out[idx] = np.random.poisson(Y)
        return out

    return _impl(n, p, size)
