"""Emit ``tricycle.kernels`` programs as Cerium DSL.

Use inside ``with CeriumProgram(...)`` and ``with CeriumFunction(...)``. Scales
are the DSL's bit counts: ciphertext inputs at ``cipher_bits``, every
plaintext at one RNS prime (``rns_bits``), so each rescale returns a
ciphertext-plaintext product to its input scale. The numeric scales the
runtime needs come from ``tricycle.kernels.track_scales``.
"""

from cerium.dsl import (
    BreakVector,
    BsgsMultiplyAccumulate,
    CiphertextInput,
    HoistedRotate,
    MakeVector,
    PlaintextInput,
    RotateAccumulateMany,
)


class CeriumBackend:
    def __init__(self, cipher_bits, cipher_level, rns_bits=28, num_slots=1 << 15):
        self.cipher_bits = cipher_bits
        self.cipher_level = cipher_level
        self.rns_bits = rns_bits
        self.num_slots = num_slots
        self._plaintexts = {}
        self._ciphertexts = {}

    def _norm(self, k):
        return int(k) % self.num_slots

    def cipher(self, name, level=None, bits=None):
        if name not in self._ciphertexts:
            level = self.cipher_level if level is None else level
            bits = self.cipher_bits if bits is None else bits
            self._ciphertexts[name] = CiphertextInput(name, bits, level)
        return self._ciphertexts[name]

    def plain(self, name, like):
        # One input per name: a weight diagonal is shared by every row block.
        cached = self._plaintexts.get(name)
        if cached is not None:
            if cached.level() != like.level():
                raise ValueError(f"plaintext {name} used at two levels")
            return cached
        value = PlaintextInput(name, self.rns_bits, like.level())
        self._plaintexts[name] = value
        return value

    def rotate(self, x, k):
        k = self._norm(k)
        return x if k == 0 else x << k

    def hoisted(self, x, ks):
        ks = [self._norm(k) for k in ks]
        if len(ks) < 2:
            return [self.rotate(x, k) for k in ks]
        return HoistedRotate(x, ks)

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def mul(self, a, b):
        return a * b

    def scalar(self, name, value, level, bits):
        return PlaintextInput(name, bits, level, True)

    def vector_plain(self, name, values, level, bits):
        return PlaintextInput(name, bits, level)

    def bootstrap(self, x, level):
        """Cerium BERT's real 33-level bootstrap (``benchmarks/bert_base/bootstrap.py``,
        imported as ``bootstrap``), called as a function; returns ``level`` limbs."""
        from bootstrap import bootstrap

        y = bootstrap(x, "b33", level - 2)
        if y.level() != level:
            raise ValueError(f"bootstrap returned level {y.level()}, not {level}")
        return y

    def modswitch(self, x):
        return x.modswitch()

    def relinearize(self, x):
        return x.relinearize()

    def rescale(self, x):
        return x.rescale()

    def double_rescale(self, x):
        return x.doubleRescale()

    def rotate_accumulate(self, xs, ks):
        ks = [self._norm(k) for k in ks]
        if len(xs) == 1:
            return self.rotate(xs[0], ks[0])
        return RotateAccumulateMany(list(xs), ks)

    def vector(self, xs):
        return xs[0] if len(xs) == 1 else MakeVector(list(xs))

    def unvector(self, x, count):
        return [x] if count == 1 else BreakVector(x)

    def bsgs(self, x, plains, babies, giants):
        return BsgsMultiplyAccumulate(x, list(plains), [self._norm(k) for k in babies],
                                      [self._norm(k) for k in giants])
