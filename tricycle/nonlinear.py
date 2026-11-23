"""Softmax, GELU and LayerNorm.

The exact functions are what a plaintext BERT layer computes. The
approximations are the ones ``nonlinear_heaan.cpp`` evaluates on ciphertexts
in https://bitbucket.org/tricyclepaper/tricycle (commit 83c4294):

- exponential, ``(1 + x / 2^k) ^ (2^k)``, done as ``k`` squarings
  (``k = 7`` in the original Tricycle softmax, ``k = 8`` in the SST-2 fork);
- GELU as a Chebyshev series of ``0.5 z (1 + erf(z / sqrt(2)))`` with
  ``z = x / mult_factor`` (``gelu_precomp``);
- inverse square root as a Chebyshev series on a positive interval, the
  polynomial inside LayerNorm and inside the deeper softmax normalization.

Homomorphic range reduction, bootstrapping and the approximate max-subtraction
used to keep the exponential input negative are not reproduced here. Those
need the HEaaN evaluator. What is here is the polynomial each of those layers
is built around, so the nonlinear side can be checked before it is lowered.
"""

from math import erf, sqrt

import numpy as np


def softmax(logits, axis=-1):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=axis, keepdims=True)


def gelu(values):
    """``gelu_precomp``'s target, ``0.5 z (1 + erf(z / sqrt(2)))``."""
    values = np.asarray(values, dtype=float)
    return 0.5 * values * (1.0 + np.vectorize(erf)(values / sqrt(2.0)))


def layer_norm(values, weight, bias, eps=1e-12, axis=-1):
    """BERT LayerNorm. Variance divides by ``N``, matching ``torch.nn.LayerNorm``."""
    values = np.asarray(values, dtype=float)
    weight = np.asarray(weight, dtype=float)
    bias = np.asarray(bias, dtype=float)
    mean = np.mean(values, axis=axis, keepdims=True)
    var = np.mean((values - mean) ** 2, axis=axis, keepdims=True)
    normed = (values - mean) / np.sqrt(var + eps)
    return normed * weight + bias


def exp_power(values, iterations):
    """``(1 + x / 2^k) ^ (2^k)`` by ``k`` squarings, the softmax exponential.

    ``iterations`` is ``k``. The HEaaN layer squares a ciphertext this many
    times after adding the plaintext ``2^{-k}``. Inputs must stay above
    ``-2^k`` or the base crosses zero and the approximation leaves the
    exponential.
    """
    values = np.asarray(values, dtype=float)
    base = 1.0 + values / (2.0 ** iterations)
    for _ in range(int(iterations)):
        base = base * base
    return base


def _chebyshev_nodes(degree):
    k = np.arange(degree + 1, dtype=float)
    return np.cos(np.pi * (k + 0.5) / (degree + 1))


def chebyshev_fit(fn, lower, upper, degree):
    """Chebyshev coefficients of ``fn`` on ``[lower, upper]``, in NumPy's basis.

    Degree 27 is the BERT-Tiny inverse-square-root in ``layer_norm_precomp``.
    Degree 119 is the BERT-Base / GELU degree. Coefficients are for the
    variable mapped from ``[lower, upper]`` onto ``[-1, 1]``.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if upper <= lower:
        raise ValueError("upper bound must exceed lower bound")
    nodes = _chebyshev_nodes(degree)
    xs = 0.5 * (lower + upper) + 0.5 * (upper - lower) * nodes
    values = np.asarray(fn(xs), dtype=float)
    return np.polynomial.chebyshev.chebfit(nodes, values, degree)


def chebyshev_eval(coefficients, values, lower, upper):
    values = np.asarray(values, dtype=float)
    mapped = (2.0 * values - lower - upper) / (upper - lower)
    return np.polynomial.chebyshev.chebval(mapped, coefficients)


def gelu_chebyshev(values, degree, mult_factor, lower=-1.0, upper=1.0):
    """Evaluate the ``gelu_precomp`` polynomial.

    ``values`` are the Chebyshev inputs on ``[lower, upper]``. The polynomial
    approximates ``gelu(values / mult_factor)``. With the original Tiny
    settings, ``degree`` is 59 and ``mult_factor`` is ``1/15``.
    """
    if mult_factor == 0:
        raise ValueError("mult_factor must be nonzero")

    def target(x):
        return gelu(np.asarray(x, dtype=float) / mult_factor)

    coefficients = chebyshev_fit(target, lower, upper, degree)
    return chebyshev_eval(coefficients, values, lower, upper), coefficients


def inv_sqrt_chebyshev(values, degree, initial_lower, initial_upper, lower=-1.0, upper=1.0):
    """Evaluate the ``inverse_sqrt_preprocess`` polynomial.

    The series lives on ``[lower, upper]`` (the ciphertext after the linear
    map into that interval). ``values`` passed here are on the original
    interval ``[initial_lower, initial_upper]``; they are mapped the same way
    the plaintext masks in LayerNorm map a variance into ``[-1, 1]`` before
    the series, and the series returns ``1/sqrt``.
    """
    if initial_upper <= initial_lower:
        raise ValueError("inverse-sqrt interval is empty")
    if initial_lower <= 0:
        raise ValueError("inverse-sqrt interval must stay positive")
    span = initial_upper - initial_lower
    mult_factor = (upper - lower) / span

    def target(x):
        argument = (np.asarray(x, dtype=float) - lower) / mult_factor + initial_lower
        return argument ** -0.5

    coefficients = chebyshev_fit(target, lower, upper, degree)
    mapped = lower + (np.asarray(values, dtype=float) - initial_lower) * mult_factor
    return chebyshev_eval(coefficients, mapped, lower, upper), coefficients
