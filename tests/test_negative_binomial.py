from scipy.stats import nbinom
from functools import partial
import pytest
import numpy as np

from the_fastest_distribution_in_the_west.negative_binomial.numba_negative_binomial import (
    numba_gamma_poisson,
    numba_negative_binomial,
    numba_rng_negative_binomial,
)

rng = np.random.default_rng()

pytestmark = pytest.mark.parametrize(
    ("sampling_func"),
    [
        np.random.negative_binomial,
        numba_negative_binomial,
        rng.negative_binomial,
        partial(numba_rng_negative_binomial, rng),
        nbinom.rvs,
        numba_gamma_poisson,
    ],
    ids=[
        "np negative_binomial",
        "numba negative_binomial",
        "np rng negative_binomial",
        "numba rng negative_binomial",
        "scipy negative_binomial",
        "numba gamma poisson",
    ],
)


def test_benchmark_large_size(benchmark, sampling_func):
    n = 2
    p = 0.5
    size = 10_000
    sampling_func(n, p, size)  # warmup

    benchmark(sampling_func, n, p, size)
