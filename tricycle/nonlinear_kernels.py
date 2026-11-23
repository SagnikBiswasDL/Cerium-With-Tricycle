"""Tricycle's nonlinear layers as CKKS programs at BERT-Base shape.

Each kernel names its ciphertext inputs, emits its program through a
``tricycle.poly.Evaluator`` (so the NumPy and Cerium backends run the same
operations), makes test inputs, and gives the exact function the
approximation should reproduce.

Shapes follow the 128-token BERT-Base run of ``refs/tricycle-sst2-checkpoint``:
4 row blocks of 33 tokens; a hidden-state block is the tricyclic tensor
``(1, 33, 769)`` (25377 slots); FF1's real output is 4 x 4 such blocks.
"""

from dataclasses import dataclass, field

import numpy as np

from tricycle.nonlinear import chebyshev_fit, gelu
from tricycle.poly import Evaluator, Value

SLOTS = 1 << 15
TOKENS_PER_BLOCK = 33
D_TRICYCLIC = 769
BLOCK_SLOTS = TOKENS_PER_BLOCK * D_TRICYCLIC

# configs/bert_base_gelu_ranges.txt, layers 1..12.
GELU_RANGES = [(-30, 30), (-30, 30), (-27, 64), (-65, 15), (-65, 15), (-55, 15),
               (-30, 30), (-30, 30), (-30, 30), (-50, 100), (-35, 150), (-30, 30)]


@dataclass
class Gelu:
    """GELU on FF1's real output blocks, one degree-``degree`` Chebyshev series.

    ``count`` ciphertexts are evaluated as one Cerium vector of width
    ``vector`` at a time (MakeVector), so independent blocks share kernels.
    """

    layer: int = 1
    count: int = 16
    degree: int = 119
    vector: int = 16
    name: str = "gelu"
    outputs: dict = field(default_factory=dict)

    @property
    def interval(self):
        return GELU_RANGES[self.layer - 1]

    def cipher_inputs(self):
        return [f"{self.name}_x{i}" for i in range(self.count)]

    def emit(self, backend, ev, level, scale):
        lower, upper = self.interval
        coeffs = chebyshev_fit(gelu, lower, upper, self.degree)
        names = self.cipher_inputs()
        out = {}
        for start in range(0, self.count, self.vector):
            group = names[start:start + self.vector]
            x = backend.vector([backend.cipher(n) for n in group])
            y = ev.chebyshev(Value(x, scale, level), coeffs, lower, upper)
            for n, h in zip(group, backend.unvector(y.handle, len(group))):
                out[n.replace("_x", "_y")] = Value(h, y.scale, y.level)
        return out

    def sample(self, rng):
        lower, upper = self.interval
        inputs = {}
        for n in self.cipher_inputs():
            v = np.zeros(SLOTS)
            v[:BLOCK_SLOTS] = rng.uniform(lower, upper, BLOCK_SLOTS)
            inputs[n] = v
        return inputs

    def reference(self, inputs):
        return {n.replace("_x", "_y"): gelu(v) for n, v in inputs.items()}


def sum_rotations(num_summations, every_other, num_slots=SLOTS):
    """``sum_preprocess2``: forward, fixup, backward and redistribute rotations."""
    ell = int(np.floor(np.log2(max(1, num_summations)))) + 1
    forward = [every_other * (1 << (i - 1)) % num_slots for i in range(1, ell)]
    fixup = []
    for i in range(ell - 1):
        if (num_summations >> i) & 1:
            n_mod = num_summations % (1 << (i + 1))
            fixup.append((i, every_other * (num_summations - n_mod) % num_slots))
    backward = [-every_other * (1 << (i - 1)) % num_slots for i in range(1, ell)]
    redistribute = [(i, -every_other * (1 << i) % num_slots)
                    for i in range(ell - 2, -1, -1) if (num_summations >> i) & 1]
    return forward, fixup, backward, redistribute


def sum_part1(ev, x, rotations, mask, target):
    """``sum_inplace2_part1``: per-group sums land in the first ``every_other`` slots."""
    forward, fixup, _, _ = rotations
    U = [x]
    u = x
    for r in forward:
        u = ev.add(u, ev.rotate(u, r))
        U.append(u)
    s = U[-1]
    for i, r in fixup:
        s = ev.add(s, ev.rotate(U[i], r))
    return ev.mul_vector(s, mask, target, "mask")


def sum_part2(ev, x, rotations):
    """``sum_inplace2_part2``: spread the first ``every_other`` slots over the block."""
    _, _, backward, redistribute = rotations
    V = [x]
    v = x
    for r in backward:
        v = ev.add(v, ev.rotate(v, r))
        V.append(v)
    t = V[-1]
    for i, r in redistribute:
        t = ev.add(ev.rotate(t, r), V[i])
    return t


# configs/bert_base_layernorm_scaling.txt, attention LayerNorm of layer 1:
# input scale 1, physical variance interval [.1, 10] times Tricycle's 0.1^2 prescale.
LN_VARIANCE = (0.1 * 0.01, 10 * 0.01)
LN_EPSILON = 1e-14


@dataclass
class LayerNorm:
    """``matrix_deeper_layer_norm`` on 4 hidden-state blocks of ``(1, 33, 769)``.

    Before the bootstrap (inputs ``x*`` at ``level``): per-token sums by
    rotation ladders, mean, centred values times gamma, and the variance mapped
    onto [-1, 1], with the 4 blocks' 33 variances coalesced into one
    ciphertext. That ciphertext is the bootstrap input. After it (input
    ``var_boot`` at ``boot_level``, the exact coalesced variance): the
    degree-``degree`` inverse square root, decoalescing, the per-token spread,
    the product with the centred values, and beta.
    """

    rows: int = 4
    degree: int = 119
    boot_level: int = 21
    features: int = 768
    name: str = "ln"

    def __post_init__(self):
        self.rotations = sum_rotations(D_TRICYCLIC, TOKENS_PER_BLOCK)
        rng = np.random.default_rng(5)
        self.gamma = rng.uniform(0.5, 1.5, self.features)
        self.beta = rng.uniform(-0.2, 0.2, self.features)

    def cipher_inputs(self):
        return [f"{self.name}_x{i}" for i in range(self.rows)] + [f"{self.name}_var_boot"]

    def _block_vector(self, per_feature):
        """A per-feature vector laid out like a hidden block (feature 768 is padding)."""
        padded = np.zeros(D_TRICYCLIC)
        padded[:self.features] = per_feature
        v = np.zeros(SLOTS)
        v[:BLOCK_SLOTS] = padded[np.arange(BLOCK_SLOTS) % D_TRICYCLIC]
        return v

    def _token_mask(self, value):
        v = np.zeros(SLOTS)
        v[:TOKENS_PER_BLOCK] = value
        return v

    def _map(self):
        lower, upper = LN_VARIANCE
        mult = 2.0 / (upper - lower)
        add = -1.0 - lower * mult
        return mult, add

    def emit(self, backend, ev, level, scale):
        n = self.features
        mult, add = self._map()
        ones = self._token_mask(1.0)
        xs = [Value(backend.cipher(f"{self.name}_x{i}", level), scale, level)
              for i in range(self.rows)]
        diffs, variances = [], []
        for x in xs:
            s1 = sum_part1(ev, x, self.rotations, ones, scale)
            remapped_mean = ev.mul_vector(s1, self._token_mask(np.sqrt(mult) / n), scale, "rmean")
            mean_square = ev.square(remapped_mean)
            division = np.zeros(SLOTS)
            division[:BLOCK_SLOTS] = 1.0 / n
            mean = ev.mul_vector(sum_part2(ev, s1, self.rotations), division, scale, "mdiv")
            squares = ev.square(x)
            diff = ev.sub(x, mean)
            diffs.append(ev.mul_vector(diff, self._block_vector(self.gamma), scale, "gamma"))
            ss = sum_part1(ev, squares, self.rotations, ones, scale)
            p = ev.mul_vector(ss, self._token_mask(mult / n), mean_square.scale, "coefP")
            var = ev.sub(p, mean_square)
            variances.append(ev.add_vector(var, self._token_mask(add + mult * LN_EPSILON), "rconst"))
        coalesced = variances[0]
        for v in variances[1:]:
            coalesced = ev.add(ev.rotate(coalesced, -TOKENS_PER_BLOCK), v)
        out = {f"{self.name}_var": coalesced}

        boot = Value(backend.cipher(f"{self.name}_var_boot", self.boot_level), scale, self.boot_level)
        coeffs = self._inv_sqrt_coeffs()
        inv = ev.chebyshev(boot, coeffs, -1.0, 1.0)
        for j in range(self.rows):
            row = self.rows - 1 - j
            piece = ev.mul_vector(inv, ones, scale, "demask")
            inv_row = sum_part2(ev, piece, self.rotations)
            y = ev.mul(diffs[row], inv_row)
            out[f"{self.name}_y{row}"] = ev.add_vector(y, self._block_vector(self.beta), "beta")
            if j + 1 < self.rows:
                inv = ev.rotate(inv, TOKENS_PER_BLOCK)
        return out

    def _inv_sqrt_coeffs(self):
        lower, upper = LN_VARIANCE
        mult, _ = self._map()
        return chebyshev_fit(lambda y: ((y + 1.0) / mult + lower) ** -0.5, -1.0, 1.0, self.degree)

    def _hidden(self, rng):
        """Per-token hidden states whose (prescaled) variance spans LN_VARIANCE."""
        lower, upper = LN_VARIANCE
        blocks = []
        for _ in range(self.rows):
            var = rng.uniform(lower * 1.05, upper * 0.95, TOKENS_PER_BLOCK)
            h = rng.normal(size=(TOKENS_PER_BLOCK, self.features))
            h = (h - h.mean(1, keepdims=True)) / h.std(1, keepdims=True)
            h = h * np.sqrt(var)[:, None] + rng.uniform(-0.1, 0.1, TOKENS_PER_BLOCK)[:, None]
            blocks.append(h)
        return blocks

    @staticmethod
    def _encode(h):
        padded = np.zeros((TOKENS_PER_BLOCK, D_TRICYCLIC))
        padded[:, :h.shape[1]] = h
        k = np.arange(BLOCK_SLOTS)
        v = np.zeros(SLOTS)
        v[:BLOCK_SLOTS] = padded[k % TOKENS_PER_BLOCK, k % D_TRICYCLIC]
        return v

    def sample(self, rng):
        blocks = self._hidden(rng)
        self._blocks = blocks
        mult, add = self._map()
        inputs = {f"{self.name}_x{i}": self._encode(h) for i, h in enumerate(blocks)}
        coalesced = np.zeros(SLOTS)
        for i, h in enumerate(blocks):
            start = (self.rows - 1 - i) * TOKENS_PER_BLOCK
            coalesced[start:start + TOKENS_PER_BLOCK] = mult * (h.var(1) + LN_EPSILON) + add
        inputs[f"{self.name}_var_boot"] = coalesced
        return inputs

    def reference(self, inputs):
        from tricycle.nonlinear import layer_norm

        out = {f"{self.name}_var": inputs[f"{self.name}_var_boot"]}
        for i, h in enumerate(self._blocks):
            y = layer_norm(h, self.gamma, self.beta, eps=LN_EPSILON)
            out[f"{self.name}_y{i}"] = self._encode(y)
        return out


KERNELS = {"gelu": Gelu, "layernorm": LayerNorm}


def make_kernel(spec):
    """``gelu`` or ``gelu:layer=10,count=16,vector=8``."""
    name, _, args = spec.partition(":")
    kwargs = {}
    for item in filter(None, args.split(",")):
        key, value = item.split("=")
        kwargs[key] = int(value)
    return KERNELS[name](**kwargs)


def run_numpy(kernel, inputs, primes, limbs, level):
    """Execute ``kernel`` on slot vectors; returns (outputs, evaluator)."""
    from tricycle.kernels import NumpyBackend

    backend = NumpyBackend(inputs, SLOTS)
    ev = Evaluator(backend, primes, limbs, 28 * limbs, kernel.name)
    outputs = kernel.emit(backend, ev, level, float(2 ** (28 * limbs)))
    return outputs, ev
