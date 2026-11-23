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
    outputs, ev = run_numpy(kernel, inputs, Primes, 2, 13)
    reference = kernel.reference(inputs)
    for name, value in outputs.items():
        assert np.max(np.abs(value.handle[:BLOCK_SLOTS] - reference[name][:BLOCK_SLOTS])) < 1e-8
    assert ev.counts["rotations"] == 182


@pytest.mark.parametrize("spec,level,bootstraps", [("layernorm:fused=1", 13, 1),
                                                   ("softmax:fused=1", 21, 20)])
def test_fused_kernels_bootstrap_in_the_program(spec, level, bootstraps):
    kernel = make_kernel(spec)
    inputs = kernel.sample(np.random.default_rng(0))
    assert set(inputs) == set(kernel.cipher_inputs())
    outputs, ev = run_numpy(kernel, inputs, Primes, 2, level)
    assert set(outputs) == set(kernel.reference(inputs))
    assert ev.counts["bootstraps"] == bootstraps
    staged = make_kernel(spec.split(":")[0])
    staged_outputs, _ = run_numpy(staged, staged.sample(np.random.default_rng(0)), Primes, 2, level)
    for name, value in outputs.items():
        want = staged_outputs[name].handle[:BLOCK_SLOTS]
        assert np.max(np.abs(value.handle[:BLOCK_SLOTS] - want)) < 1e-9
        assert value.level >= 3


@pytest.mark.parametrize("spec,level", [("gelu:vector=4", 21), ("layernorm", 13), ("softmax", 21)])
def test_constants_stay_near_the_nominal_scale(spec, level):
    kernel = make_kernel(spec)
    outputs, ev = run_numpy(kernel, kernel.sample(np.random.default_rng(0)), Primes, 2, level)
    exponents = [np.log2(scale) for _, scale in ev.scalars.values()]
    assert 48 < min(exponents) and max(exponents) < 85
    assert all(v.level >= 3 for v in outputs.values())
