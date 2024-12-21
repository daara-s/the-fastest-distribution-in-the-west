from scipy.stats import poisson as scipy_poisson
from functools import partial
import pytest
import numpy as np

from the_fastest_distribution_in_the_west.poisson.numba_poisson import (
    numba_poisson,
    numba_rng_poisson,
)

rng = np.random.default_rng()

pytestmark = pytest.mark.parametrize(
    ("sampling_func"),
    [
        np.random.poisson,
        numba_poisson,
        rng.poisson,
        partial(numba_rng_poisson, rng),
        scipy_poisson.rvs,
    ],
    ids=[
        "np poisson",
        "numba poisson",
        "np rng poisson",
        "numba rng poisson",
        "scipy poisson",
    ],
)



def test_benchmark_large_lambda(benchmark, sampling_func):
    lam = 20
    size = 10_000
    sampling_func(lam, size)  # warmup

    benchmark(sampling_func, lam, size)

def test_benchmark_large_size(benchmark, sampling_func):
    lam = 1
    size = 10_000_000
    sampling_func(lam, size)  # warmup

    benchmark(sampling_func, lam, size)
