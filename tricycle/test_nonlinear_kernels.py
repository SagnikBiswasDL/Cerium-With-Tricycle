import numpy as np
import pytest
from numpy.polynomial import chebyshev as C

from examples.primes import Primes
from tricycle.kernels import NumpyBackend
from tricycle.nonlinear import chebyshev_fit, gelu
from tricycle.nonlinear_kernels import BLOCK_SLOTS, make_kernel, run_numpy
from tricycle.poly import Evaluator, Value, chebyshev_divide


def test_chebyshev_divide_reconstructs_the_series():
    c = np.random.default_rng(0).normal(size=120)
    q, r = chebyshev_divide(c, 64)
    x = np.linspace(-1, 1, 101)
    tn = C.chebval(x, [0] * 64 + [1])
    assert np.allclose(C.chebval(x, c), C.chebval(x, q) * tn + C.chebval(x, r), atol=1e-10)


@pytest.mark.parametrize("limbs", [2, 3])
def test_degree_119_series_keeps_its_scale_exact(limbs):
    coeffs = chebyshev_fit(gelu, -8.0, 8.0, 119)
    ev = Evaluator(NumpyBackend({}), Primes, limbs, 28 * limbs, "t")
    xs = np.random.default_rng(1).uniform(-8, 8, 4096)
    scale = 2.0 ** (28 * limbs)
    out = ev.chebyshev(Value(xs, scale, 40), coeffs, -8.0, 8.0)
    assert out.scale == scale
    assert out.level == 40 - 8 * limbs
    assert np.max(np.abs(out.handle - gelu(xs))) < 1e-9


def test_gelu_kernel_matches_gelu_on_the_data_slots():
    kernel = make_kernel("gelu")
    inputs = kernel.sample(np.random.default_rng(0))
    outputs, _ = run_numpy(kernel, inputs, Primes, 2, 21)
    reference = kernel.reference(inputs)
    for name, value in outputs.items():
        assert np.max(np.abs(value.handle[:BLOCK_SLOTS] - reference[name][:BLOCK_SLOTS])) < 1e-3


def test_layernorm_kernel_matches_layer_norm():
    kernel = make_kernel("layernorm")
    inputs = kernel.sample(np.random.default_rng(0))
    outputs, ev = run_numpy(kernel, inputs, Primes, 2, 12)
    reference = kernel.reference(inputs)
    for name, value in outputs.items():
        assert np.max(np.abs(value.handle[:BLOCK_SLOTS] - reference[name][:BLOCK_SLOTS])) < 1e-8
    assert ev.counts["rotations"] == 182
