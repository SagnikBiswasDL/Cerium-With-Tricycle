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

    def dropped(self, level, primes=None):
        """Product of the ``primes`` (default ``limbs``) primes a rescale from ``level`` removes."""
        product = 1.0
        for i in range(self.limbs if primes is None else primes):
            product *= self.primes[level - 1 - i]
        return product

    def input(self, name, level, primes=None):
        """A ciphertext input encrypted at the scale its first rescale removes.

        Starting every value at that prime product keeps the scales of repeated
        squarings (giant steps, the softmax exponential) near it; any other
        start doubles its excess with every squaring. ``primes=limbs + 1`` gives
        a high-scale input for rotations (see ``high``).
        """
        scale = self.dropped(level, primes)
        self.inputs[name] = scale
        return Value(self.b.cipher(name, level, self._bits(scale)), scale, level)

    def _bits(self, scale):
        """The DSL's bit count for ``scale``: 28 per prime the scale amounts to.

        Cerium's DSL tracks scales as bit counts (a rescale subtracts 28 and
        additions need equal counts), so every input is declared at the number
        of primes its numeric scale is worth.
        """
        prime_bits = np.mean(np.log2(self.primes))
        return 28 * max(0, int(round(np.log2(scale) / prime_bits)))

    def _name(self, tag):
        name = f"{self.prefix}_{tag}_{len(self.scalars)}"
        return name

    def _scalar(self, tag, value, scale, level):
        name = self._name(tag)
        self.scalars[name] = (float(value), float(scale))
        self.levels[name] = level
        return self.b.scalar(name, float(value), level, self._bits(scale))

    @property
    def high(self):
        """Primes the multiply consuming a rotated value rescales.

        Cerium's rotations add ~2^52 of absolute key-switch noise, 0.1 at a
        2^55 scale. Values that get rotated are therefore kept one prime higher
        (~2^83): the multiply producing them rescales ``limbs - 1`` primes and
        the multiply consuming them ``limbs + 1``, so no level is spent.
        """
        return self.limbs + 1

    @property
    def low(self):
        """Primes the multiply producing a value for rotation rescales."""
        return self.limbs - 1

    def rescale(self, x, primes=None):
        primes = self.limbs if primes is None else primes
        if x.level - primes < 1:
            raise ValueError(f"{self.prefix}: out of levels (rescale at level {x.level})")
        h = x.handle
        for _ in range(primes):
            h = self.b.rescale(h)
        return Value(h, x.scale / self.dropped(x.level, primes), x.level - primes)

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

    def mul_const(self, x, value, target_scale, primes=None):
        """``value * x``, rescaled by ``primes`` primes, at exactly ``target_scale``."""
        scale = target_scale * self.dropped(x.level, primes) / x.scale
        c = self._scalar("k", value, scale, x.level)
        self.counts["ct_scalar"] += 1
        return self.rescale(Value(self.b.mul(x.handle, c), x.scale * scale, x.level), primes)

    def mul_vector(self, x, values, target_scale, tag="m", primes=None):
        """Slot-wise ``values * x`` (a plaintext vector), rescaled to ``target_scale``."""
        scale = target_scale * self.dropped(x.level, primes) / x.scale
        name = self._name(tag)
        values = np.asarray(values, dtype=float)
        self.scalars[name] = (values, float(scale))
        self.levels[name] = x.level
        p = self.b.vector_plain(name, values, x.level, self._bits(scale))
        self.counts["ct_scalar"] += 1
        return self.rescale(Value(self.b.mul(x.handle, p), x.scale * scale, x.level), primes)

    def add_vector(self, x, values, tag="a"):
        name = self._name(tag)
        values = np.asarray(values, dtype=float)
        self.scalars[name] = (values, float(x.scale))
        self.levels[name] = x.level
        p = self.b.vector_plain(name, values, x.level, self._bits(x.scale))
        return Value(self.b.add(x.handle, p), x.scale, x.level)

    def bootstrap(self, x, level, headroom=5):
        """Refresh ``x`` to ``level`` limbs at the scale ``input`` gives that level.

        The bootstrap takes a 2-limb ciphertext and returns its message at the
        same numeric scale, so the input is brought to ``dropped(level) /
        2^headroom`` (the headroom keeps the message well below the first
        modulus) and the output is multiplied by the exact integer
        2^headroom without rescaling, as Cerium's BERT does.
        """
        prime_bits = np.mean(np.log2(self.primes))
        primes = max(1, int(round(np.log2(x.scale) / prime_bits)))
        if x.level < 2 + primes:
            raise ValueError(f"{self.prefix}: bootstrap input at level {x.level} is too low")
        inner = self.dropped(level) / 2 ** headroom
        y = self.mul_const(self.down(x, 2 + primes), 1.0, inner, primes)
        h = self.b.bootstrap(y.handle, level)
        self.counts["bootstraps"] = self.counts.get("bootstraps", 0) + 1
        c = self._scalar("bs", 1.0, 2 ** headroom, level)
        return Value(self.b.mul(h, c), inner * 2 ** headroom, level)

    def rotate(self, x, k):
        self.counts["rotations"] = self.counts.get("rotations", 0) + (1 if k % len_slots(self) else 0)
        return Value(self.b.rotate(x.handle, k), x.scale, x.level)

    def mul(self, a, b, primes=None):
        level = min(a.level, b.level)
        a, b = self.down(a, level), self.down(b, level)
        self.counts["ct_ct"] += 1
        h = self.b.relinearize(self.b.mul(a.handle, b.handle))
        return self.rescale(Value(h, a.scale * b.scale, level), primes)

    def square(self, a, primes=None):
        return self.mul(a, a, primes)

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

    def _plan_level(self, coeffs, T, G, primes=None):
        """Output level of ``_series`` for these coefficients (no ops emitted)."""
        if not G or len(coeffs) <= len(T):
            return self._leaf_level(coeffs, T)
        g = G[-1]
        n = len(T) << (len(G) - 1)
        if len(coeffs) <= n:
            return self._plan_level(coeffs, T, G[:-1], primes)
        q, r = chebyshev_divide(coeffs, n)
        lq = self._plan_level(q, T, G[:-1])
        drop = self.limbs if primes is None else primes
        product = (min(lq, g.level) if lq is not None else g.level) - drop
        lr = self._plan_level(r, T, G[:-1], primes)
        return product if lr is None else min(product, lr)

    def _scaled(self, x, c, target, mask):
        if mask is None:
            return self.mul_const(x, c, target)
        return self.mul_vector(x, mask * c, target, "mk")

    def _plus(self, x, c, mask):
        if mask is None:
            return self.add_const(x, c)
        return self.add_vector(x, mask * c, "mc")

    def _series(self, coeffs, T, G, target, level, mask=None, primes=None):
        """Sum c_i T_i at exactly ``target`` scale and ``level``.

        ``mask`` multiplies the result slot-wise at no level cost (it scales
        the leaf coefficients); ``primes`` is the rescale of the products that
        end at the output (the top one and those of the remainders).
        """
        if not G or len(coeffs) <= len(T):
            terms = [self._scaled(T[i], c, target, mask) for i, c in enumerate(coeffs) if i and c != 0]
            if not terms:
                raise ValueError("constant-only series")
            total = terms[0]
            for term in terms[1:]:
                total = self.add(total, term)
            total = self.down(total, level)
            return self._plus(total, coeffs[0], mask) if coeffs[0] != 0 else total
        n = len(T) << (len(G) - 1)
        if len(coeffs) <= n:
            return self._series(coeffs, T, G[:-1], target, level, mask, primes)
        g = G[-1]
        q, r = chebyshev_divide(coeffs, n)
        if len(q) == 1 or not any(q[1:]):
            if mask is None:
                product = self.mul_const(g, q[0], target, primes)
            else:
                product = self.mul_vector(g, mask * q[0], target, "mk", primes)
            product = self.down(product, level)
        else:
            lq = self._plan_level(q, T, G[:-1])
            product_level = min(lq, g.level)
            q_target = target * self.dropped(product_level, primes) / g.scale
            qv = self._series(q, T, G[:-1], q_target, lq, mask)
            product = self.down(self.mul(qv, g, primes), level)
        if not any(r[1:]):
            return self._plus(product, r[0], mask) if r[0] != 0 else product
        rv = self._series(r, T, G[:-1], target, level, mask, primes)
        return self.add(product, rv)

    def chebyshev(self, x, coeffs, lower, upper, baby=8, masks=None, high=False):
        """Evaluate ``sum c_i T_i((2x - lower - upper) / (upper - lower))``.

        With ``masks``, returns one series per mask, each multiplied slot-wise
        by its mask, sharing the basis. ``high`` puts the result at the high
        scale (``Evaluator.high``) one prime above the usual output level.
        """
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
        primes = self.low if high else None
        level = self._plan_level(coeffs, T, G, primes)
        target = x.scale
        if high:
            target = x.scale * self.dropped(level + self.low) / self.dropped(level + self.low, self.low)
        if masks is None:
            return self._series(coeffs, T, G, target, level, None, primes)
        return [self._series(coeffs, T, G, target, level, m, primes) for m in masks]


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
