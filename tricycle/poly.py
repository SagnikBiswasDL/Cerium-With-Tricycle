"""Chebyshev series on ciphertexts with exact CKKS scales.

``Evaluator`` carries every ciphertext as ``Value(handle, scale, level)``:
``handle`` is whatever the backend returns (a NumPy slot vector or a Cerium
DSL expression), ``scale`` the numeric CKKS scale, ``level`` the number of RNS
primes left. One multiplicative level drops ``limbs`` primes (Cerium's BERT
uses two primes of ~27.6 bits per level at a 2^56 scale).

Scales are never left to drift: a scalar plaintext that multiplies a
ciphertext is encoded at the scale that makes the rescaled product land on a
chosen target, the way Cerium's ``bert_runner.py`` picks its polynomial
coefficient scales. Additions therefore always see equal scales, which the
evaluator checks. The scalar values and scales are collected in ``scalars``
for the runtime's raw inputs.

Backend calls used here: ``mul``, ``relinearize``, ``rescale``, ``add``,
``sub``, ``modswitch`` and ``scalar(name, value, level, bits)``.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Value:
    handle: object
    scale: float
    level: int


def _same_scale(a, b):
    return abs(a - b) <= 1e-9 * max(abs(a), abs(b))


class Evaluator:
    def __init__(self, backend, primes, limbs, bits, prefix):
        self.b = backend
        self.primes = list(primes)
        self.limbs = limbs
        self.bits = bits
        self.prefix = prefix
        self.scalars = {}
        self.levels = {}
        self.inputs = {}
        self.counts = {"ct_ct": 0, "ct_scalar": 0}

    # -- primitive bookkeeping ------------------------------------------------

    def dropped(self, level):
        """Product of the primes a rescale from ``level`` removes."""
        product = 1.0
        for i in range(self.limbs):
            product *= self.primes[level - 1 - i]
        return product

    def input(self, name, level):
        """A ciphertext input encrypted at the scale its first rescale removes.

        Starting every value at that prime product keeps the scales of repeated
        squarings (giant steps, the softmax exponential) near it; any other
        start doubles its excess with every squaring.
        """
        scale = self.dropped(level)
        self.inputs[name] = scale
        return Value(self.b.cipher(name, level), scale, level)

    def _name(self, tag):
        name = f"{self.prefix}_{tag}_{len(self.scalars)}"
        return name

    def _scalar(self, tag, value, scale, level):
        name = self._name(tag)
        self.scalars[name] = (float(value), float(scale))
        self.levels[name] = level
        return self.b.scalar(name, float(value), level, self.bits)

    def rescale(self, x):
        if x.level - self.limbs < 1:
            raise ValueError(f"{self.prefix}: out of levels (rescale at level {x.level})")
        h = x.handle
        for _ in range(self.limbs):
            h = self.b.rescale(h)
        return Value(h, x.scale / self.dropped(x.level), x.level - self.limbs)

    def down(self, x, level):
        if level > x.level:
            raise ValueError(f"cannot raise level {x.level} to {level}")
        h = x.handle
        for _ in range(x.level - level):
            h = self.b.modswitch(h)
        return Value(h, x.scale, level)

    def add(self, a, b):
        level = min(a.level, b.level)
        a, b = self.down(a, level), self.down(b, level)
        if not _same_scale(a.scale, b.scale):
            raise ValueError(f"add of scales {a.scale} and {b.scale}")
        return Value(self.b.add(a.handle, b.handle), a.scale, level)

    def sub(self, a, b):
        level = min(a.level, b.level)
        a, b = self.down(a, level), self.down(b, level)
        if not _same_scale(a.scale, b.scale):
            raise ValueError(f"sub of scales {a.scale} and {b.scale}")
        return Value(self.b.sub(a.handle, b.handle), a.scale, level)

    def double(self, x):
        return Value(self.b.add(x.handle, x.handle), x.scale, x.level)

    def add_const(self, x, value):
        c = self._scalar("c", value, x.scale, x.level)
        return Value(self.b.add(x.handle, c), x.scale, x.level)

    def mul_const(self, x, value, target_scale):
        """``value * x``, rescaled, at exactly ``target_scale``."""
        scale = target_scale * self.dropped(x.level) / x.scale
        c = self._scalar("k", value, scale, x.level)
        self.counts["ct_scalar"] += 1
        return self.rescale(Value(self.b.mul(x.handle, c), x.scale * scale, x.level))

    def mul_vector(self, x, values, target_scale, tag="m"):
        """Slot-wise ``values * x`` (a plaintext vector), rescaled to ``target_scale``."""
        scale = target_scale * self.dropped(x.level) / x.scale
        name = self._name(tag)
        values = np.asarray(values, dtype=float)
        self.scalars[name] = (values, float(scale))
        self.levels[name] = x.level
        p = self.b.vector_plain(name, values, x.level, self.bits)
        self.counts["ct_scalar"] += 1
        return self.rescale(Value(self.b.mul(x.handle, p), x.scale * scale, x.level))

    def add_vector(self, x, values, tag="a"):
        name = self._name(tag)
        values = np.asarray(values, dtype=float)
        self.scalars[name] = (values, float(x.scale))
        self.levels[name] = x.level
        p = self.b.vector_plain(name, values, x.level, self.bits)
        return Value(self.b.add(x.handle, p), x.scale, x.level)

    def rotate(self, x, k):
        self.counts["rotations"] = self.counts.get("rotations", 0) + (1 if k % len_slots(self) else 0)
        return Value(self.b.rotate(x.handle, k), x.scale, x.level)

    def mul(self, a, b):
        level = min(a.level, b.level)
        a, b = self.down(a, level), self.down(b, level)
        self.counts["ct_ct"] += 1
        h = self.b.relinearize(self.b.mul(a.handle, b.handle))
        return self.rescale(Value(h, a.scale * b.scale, level))

    def square(self, a):
        return self.mul(a, a)

    # -- Chebyshev series -------------------------------------------------------

    def chebyshev_basis(self, t1, count):
        """T_0..T_{count-1}; ``T[0]`` is None (the constant is added as a scalar)."""
        T = [None, t1]
        for n in range(2, count):
            a = 1 << (n.bit_length() - 1)
            if a == n:
                a //= 2
            b = n - a
            product = self.mul(self.double(T[a]), T[b])
            if a == b:
                T.append(self.add_const(product, -1.0))
            else:
                rest = self.mul_const(T[a - b], 1.0, product.scale)
                T.append(self.sub(product, rest))
        return T[:count]

    def giant_steps(self, base, count):
        """T_m, T_2m, T_4m, ... from ``base`` = T_m."""
        G = [base]
        for _ in range(count - 1):
            G.append(self.add_const(self.mul(self.double(G[-1]), G[-1]), -1.0))
        return G

    def _leaf_level(self, coeffs, T):
        levels = [T[i].level - self.limbs for i in range(1, len(coeffs)) if coeffs[i] != 0]
        return min(levels) if levels else None

    def _plan_level(self, coeffs, T, G):
        """Output level of ``_series`` for these coefficients (no ops emitted)."""
        if not G or len(coeffs) <= len(T):
            return self._leaf_level(coeffs, T)
        g = G[-1]
        n = len(T) << (len(G) - 1)
        if len(coeffs) <= n:
            return self._plan_level(coeffs, T, G[:-1])
        q, r = chebyshev_divide(coeffs, n)
        lq = self._plan_level(q, T, G[:-1])
        product = (min(lq, g.level) if lq is not None else g.level) - self.limbs
        lr = self._plan_level(r, T, G[:-1])
        return product if lr is None else min(product, lr)

    def _series(self, coeffs, T, G, target, level):
        """Sum c_i T_i at exactly ``target`` scale and ``level``."""
        if not G or len(coeffs) <= len(T):
            terms = [self.mul_const(T[i], c, target) for i, c in enumerate(coeffs) if i and c != 0]
            if not terms:
                raise ValueError("constant-only series")
            total = terms[0]
            for term in terms[1:]:
                total = self.add(total, term)
            total = self.down(total, level)
            return self.add_const(total, coeffs[0]) if coeffs[0] != 0 else total
        n = len(T) << (len(G) - 1)
        if len(coeffs) <= n:
            return self._series(coeffs, T, G[:-1], target, level)
        g = G[-1]
        q, r = chebyshev_divide(coeffs, n)
        if len(q) == 1 or not any(q[1:]):
            product = self.down(self.mul_const(g, q[0], target), level)
        else:
            lq = self._plan_level(q, T, G[:-1])
            product_level = min(lq, g.level)
            q_target = target * self.dropped(product_level) / g.scale
            qv = self._series(q, T, G[:-1], q_target, lq)
            product = self.down(self.mul(qv, g), level)
        if not any(r[1:]):
            return self.add_const(product, r[0]) if r[0] != 0 else product
        rv = self._series(r, T, G[:-1], target, level)
        return self.add(product, rv)

    def chebyshev(self, x, coeffs, lower, upper, baby=8):
        """Evaluate ``sum c_i T_i((2x - lower - upper) / (upper - lower))``."""
        coeffs = np.asarray(coeffs, dtype=float)
        a = 2.0 / (upper - lower)
        b = -(lower + upper) / (upper - lower)
        t1 = x if a == 1 else self.mul_const(x, a, x.scale)
        if b != 0:
            t1 = self.add_const(t1, b)
        T = self.chebyshev_basis(t1, baby)
        giants = 0
        while (baby << giants) <= len(coeffs) - 1:
            giants += 1
        G = []
        if giants:
            h = T[baby // 2]
            G = self.giant_steps(self.add_const(self.mul(self.double(h), h), -1.0), giants)
        level = self._plan_level(coeffs, T, G)
        return self._series(coeffs, T, G, x.scale, level)


def len_slots(ev):
    return getattr(ev.b, "num_slots", 1 << 15)


def chebyshev_divide(coeffs, n):
    """``p = q * T_n + r`` for a Chebyshev series ``p``; ``deg r < n``."""
    c = np.array(coeffs, dtype=float)
    d = len(c) - 1
    if d < n:
        return np.zeros(1), c
    q = np.zeros(d - n + 1)
    for i in range(d, n, -1):
        q[i - n] += 2.0 * c[i]
        c[abs(2 * n - i)] -= c[i]
        c[i] = 0.0
    q[0] += c[n]
    c[n] = 0.0
    return q, c[:n]
