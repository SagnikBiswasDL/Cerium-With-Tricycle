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

    The inputs are FF1's outputs already mapped from the layer's GELU range
    onto [-1, 1]: the map is affine, so FF1's weights and bias absorb it and
    the series spends no level on it. The outputs are at the high scale
    (``Evaluator.high``), ready for FF2's rotations. ``count`` ciphertexts are
    evaluated as one Cerium vector of width ``vector`` at a time (MakeVector),
    so independent blocks share kernels.
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

    def emit(self, backend, ev, level):
        lower, upper = self.interval
        coeffs = chebyshev_fit(gelu, lower, upper, self.degree)
        names = self.cipher_inputs()
        out = {}
        for start in range(0, self.count, self.vector):
            group = names[start:start + self.vector]
            xs = [ev.input(n, level) for n in group]
            x = Value(backend.vector([v.handle for v in xs]), xs[0].scale, level)
            y = ev.chebyshev(x, coeffs, -1.0, 1.0, high=True)
            for n, h in zip(group, backend.unvector(y.handle, len(group))):
                out[n.replace("_x", "_y")] = Value(h, y.scale, y.level)
        return out

    def sample(self, rng):
        lower, upper = self.interval
        inputs, self._x = {}, {}
        for n in self.cipher_inputs():
            x = np.zeros(SLOTS)
            x[:BLOCK_SLOTS] = rng.uniform(lower, upper, BLOCK_SLOTS)
            self._x[n] = x
            inputs[n] = (2.0 * x - lower - upper) / (upper - lower)
        return inputs

    def reference(self, inputs):
        return {n.replace("_x", "_y"): gelu(x) for n, x in self._x.items()}


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


def sum_part1(ev, x, rotations, mask):
    """``sum_inplace2_part1``: per-group sums land in the first ``every_other`` slots.

    The masking multiply keeps ``x``'s scale, so a high-scale ``x`` stays high.
    """
    forward, fixup, _, _ = rotations
    U = [x]
    u = x
    for r in forward:
        u = ev.add(u, ev.rotate(u, r))
        U.append(u)
    s = U[-1]
    for i, r in fixup:
        s = ev.add(s, ev.rotate(U[i], r))
    return ev.mul_vector(s, mask, s.scale, "mask")


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
    boot_level: int = 19
    headroom: int = 5
    features: int = 768
    fused: int = 0
    name: str = "ln"

    def __post_init__(self):
        self.rotations = sum_rotations(D_TRICYCLIC, TOKENS_PER_BLOCK)
        rng = np.random.default_rng(5)
        self.gamma = rng.uniform(0.5, 1.5, self.features)
        self.beta = rng.uniform(-0.2, 0.2, self.features)

    def cipher_inputs(self):
        boot = [] if self.fused else [f"{self.name}_var_boot"]
        return [f"{self.name}_x{i}" for i in range(self.rows)] + boot

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

    def emit(self, backend, ev, level):
        n = self.features
        mult, add = self._map()
        ones = self._token_mask(1.0)
        hi, lo = ev.high, ev.low
        xs = [ev.input(f"{self.name}_x{i}", level, hi) for i in range(self.rows)]
        diffs, variances = [], []
        for x in xs:
            s1 = sum_part1(ev, x, self.rotations, ones)
            rm_level = s1.level - hi
            remapped_mean = ev.mul_vector(s1, self._token_mask(np.sqrt(mult) / n),
                                          ev.dropped(rm_level), "rmean", hi)
            mean_square = ev.square(remapped_mean, lo)
            division = np.zeros(SLOTS)
            division[:BLOCK_SLOTS] = 1.0 / n
            mean = ev.mul_vector(sum_part2(ev, s1, self.rotations), division, x.scale, "mdiv")
            squares = ev.square(x, hi)
            diff = ev.sub(x, mean)
            diffs.append(ev.mul_vector(diff, self._block_vector(self.gamma),
                                       ev.dropped(diff.level - hi), "gamma", hi))
            ss = sum_part1(ev, squares, self.rotations, ones)
            p = ev.mul_vector(ss, self._token_mask(mult / n), mean_square.scale, "coefP")
            var = ev.sub(p, mean_square)
            variances.append(ev.add_vector(var, self._token_mask(add + mult * LN_EPSILON), "rconst"))
        coalesced = variances[0]
        for v in variances[1:]:
            coalesced = ev.add(ev.rotate(coalesced, -TOKENS_PER_BLOCK), v)
        out = {} if self.fused else {f"{self.name}_var": coalesced}

        if self.fused:
            boot = ev.bootstrap(coalesced, self.boot_level, self.headroom)
        else:
            boot = ev.input(f"{self.name}_var_boot", self.boot_level)
        masks = [np.roll(ones, j * TOKENS_PER_BLOCK) for j in range(self.rows)]
        pieces = ev.chebyshev(boot, self._inv_sqrt_coeffs(), -1.0, 1.0, masks=masks, high=True)
        for j, piece in enumerate(pieces):
            row = self.rows - 1 - j
            piece = ev.rotate(piece, j * TOKENS_PER_BLOCK)
            inv_row = sum_part2(ev, piece, self.rotations)
            y = ev.mul(diffs[row], inv_row, hi)
            out[f"{self.name}_y{row}"] = ev.add_vector(y, self._block_vector(self.beta), "beta")
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
        self._coalesced = coalesced
        if not self.fused:
            inputs[f"{self.name}_var_boot"] = coalesced
        return inputs

    def reference(self, inputs):
        from tricycle.nonlinear import layer_norm

        out = {} if self.fused else {f"{self.name}_var": self._coalesced}
        for i, h in enumerate(self._blocks):
            y = layer_norm(h, self.gamma, self.beta, eps=LN_EPSILON)
            out[f"{self.name}_y{i}"] = self._encode(y)
        return out


HEADS = 13
KEY_DIM = 35
SCORE_DIMS = (HEADS, TOKENS_PER_BLOCK, KEY_DIM)
SCORE_SLOTS = HEADS * TOKENS_PER_BLOCK * KEY_DIM
# configs/bert_base_softmax_128/layer1.txt: inverse interval, then per-head
# (y intercept, slope) of the max estimate; the 13th (padding) head keeps
# softmax_patch's defaults.
SOFTMAX_LAYER1 = {
    "inverse": (0.65386292885478059, 12.271508273663168),
    "heads": [(1.2343408395303446, 0.12944402995983414), (1.2929836866789246, 0.11691981024872976),
              (1.654407398058098, 0.12104710442444402), (1.0177349536454126, 0.21301689896110862),
              (1.1771101547185165, 0.22051445633382644), (1.3294663321520297, 0.12131507174231135),
              (1.0782124752185727, 0.2204501208034112), (1.3009164561463809, 0.069303970814902499),
              (1.4011011549102417, 0.14297172475466191), (1.1707610632799224, 0.26070312883685176),
              (1.572036892130352, 0.14679916180793046), (1.4508370684923215, 0.087037368141924032),
              (1.25, 0.1)],
}


def score_rotations(num_slots=SLOTS):
    """``sum_preprocess(35, 13 * 33)``: sums over the key axis of a score block."""
    every_other, n = HEADS * TOKENS_PER_BLOCK, KEY_DIM
    rounds = int(np.floor(np.log2(n)))
    left_tail = n - (1 << rounds)
    right_tail = left_tail - 1
    ceil_log2 = lambda v: int(np.ceil(np.log2(v)))
    steps = [(1 << i) * every_other % num_slots for i in range(rounds)]
    reverse = -every_other * (1 << rounds) % num_slots
    left = (ceil_log2(left_tail), (1 << rounds) * every_other % num_slots) if left_tail > 0 else None
    right = None
    if right_tail > 0:
        loop = ceil_log2(right_tail)
        right = (loop, -every_other * ((n - right_tail) + (1 << loop) - 1) % num_slots)
    return steps, reverse, left, right


def sum_keys(ev, cts, rotations):
    """``sum_multiple``: add the column blocks, then sum over keys in place."""
    steps, reverse, left, right = rotations
    c = cts[0]
    for ct in cts[1:]:
        c = ev.add(c, ct)
    saved = {0: c}
    for i, r in enumerate(steps, start=1):
        c = ev.add(c, ev.rotate(c, r))
        saved[i] = c
    c = ev.add(c, ev.rotate(c, reverse))
    if left:
        c = ev.add(c, ev.rotate(saved[left[0]], left[1]))
    if right:
        c = ev.add(c, ev.rotate(saved[right[0]], right[1]))
    return c


def score_mask(per_head, keys_only=True):
    """A tricyclic (13, 33, 35) mask: ``per_head[h]`` on real keys, 0 on padding keys."""
    k = np.arange(SCORE_SLOTS)
    values = np.asarray(per_head, dtype=float)[k % HEADS]
    if keys_only:
        values = np.where(k % KEY_DIM < TOKENS_PER_BLOCK, values, 0.0)
    v = np.zeros(SLOTS)
    v[:SCORE_SLOTS] = values
    return v


@dataclass
class Softmax:
    """Direct-reciprocal ``softmax`` (layers 1, 2, 10, 12) on the 4 x 4 score blocks.

    Before the 4-way bootstrap (inputs ``s*`` at ``level``): the max estimate
    from row sums and sums of squares, ``1 + x / 256`` and ``before`` of the 8
    squarings. After it (inputs ``e*_boot`` at ``boot_level``): the remaining
    squarings, the row sums mapped onto [-1, 1]; that is the 2-way bootstrap's
    input. After that (``d*_boot``): the degree-``degree`` reciprocal and the
    product with every exponential.
    """

    rows: int = 4
    before: int = 3
    degree: int = 119
    boot_level: int = 19
    headroom: int = 5
    fused: int = 0
    name: str = "sm"
    iterations: int = 8

    def __post_init__(self):
        self.rotations = score_rotations()
        self.config = SOFTMAX_LAYER1
        self.n_size = KEY_DIM * self.rows

    def cipher_inputs(self):
        grid = [f"{i}{j}" for i in range(self.rows) for j in range(self.rows)]
        if self.fused:
            return [f"{self.name}_s{g}" for g in grid]
        return ([f"{self.name}_s{g}" for g in grid] + [f"{self.name}_e{g}_boot" for g in grid]
                + [f"{self.name}_d{i}_boot" for i in range(self.rows)])

    def _coefficients(self):
        inv = 1.0 / 2 ** self.iterations
        f = np.sqrt(2 * np.log(self.n_size))
        intercepts = [y for y, _ in self.config["heads"]]
        slopes = [s for _, s in self.config["heads"]]
        sum_coeff = inv / self.n_size
        sos = [s * f * inv / self.n_size for s in slopes]
        sts = [-c / self.n_size for c in sos]
        shift = [y * f * inv for y in intercepts]
        return inv, sum_coeff, sos, sts, shift

    def _prescale(self):
        lower, upper = self.config["inverse"]
        mult = 2.0 / (upper - lower)
        add = -(upper + lower) / (upper - lower)
        k = np.arange(SCORE_SLOTS)
        real = k % KEY_DIM < TOKENS_PER_BLOCK
        m = np.zeros(SLOTS)
        m[:SCORE_SLOTS] = np.where(real, mult, 0.0)
        a = np.full(SLOTS, 1.1 * mult + add)
        a[:SCORE_SLOTS] = np.where(real, add, 1.1 * mult + add)
        return m, a

    def _reciprocal_coeffs(self):
        lower, upper = self.config["inverse"]
        return chebyshev_fit(lambda y: 1.0 / (lower + (y + 1.0) * (upper - lower) / 2.0),
                             -1.0, 1.0, self.degree)

    def emit(self, backend, ev, level):
        inv, sum_coeff, sos, sts, shift = self._coefficients()
        heads = np.ones(HEADS)
        out, partial, sums = {}, {}, {}
        n = self.rows
        hi, lo = ev.high, ev.low
        for i in range(n):
            s = [ev.input(f"{self.name}_s{i}{j}", level, hi) for j in range(n)]
            total = sum_keys(ev, s, self.rotations)
            mean = ev.mul_vector(total, score_mask(heads / self.n_size),
                                 ev.dropped(total.level - hi), "cmean", hi)
            mean_sq = ev.square(mean)
            squares = [ev.square(x, hi) for x in s]
            total_sq = sum_keys(ev, squares, self.rotations)
            target = mean_sq.scale
            shift_v = ev.add(ev.mul_vector(total, score_mask(heads * sum_coeff), target, "scoef", hi),
                             ev.mul_vector(total_sq, score_mask(sos), target, "sscoef", hi))
            sts_term = ev.mul_vector(ev.square(total, hi), score_mask(sts), target, "stscoef", hi)
            shift_v = ev.add_vector(ev.add(shift_v, sts_term), score_mask(shift), "shift")
            for j, x in enumerate(s):
                u = ev.sub(ev.mul_vector(x, score_mask(heads * inv), target, "inv2k", hi), shift_v)
                e = ev.add_vector(u, score_mask(heads), "one")
                for _ in range(self.before):
                    e = ev.square(e)
                partial[i, j] = e
                if not self.fused:
                    out[f"{self.name}_e{i}{j}"] = e

        boot = self.boot_level
        mult, add = self._prescale()
        exps = {}
        for i in range(n):
            row = []
            for j in range(n):
                if self.fused:
                    e = ev.bootstrap(partial[i, j], boot, self.headroom)
                else:
                    e = ev.input(f"{self.name}_e{i}{j}_boot", boot)
                for _ in range(self.iterations - self.before - 1):
                    e = ev.square(e)
                row.append((ev.square(e), ev.square(e, lo)))
            exps[i] = [low for low, _ in row]
            total = sum_keys(ev, [high for _, high in row], self.rotations)
            pre = ev.add_vector(ev.mul_vector(total, mult, ev.dropped(total.level - hi), "pmul", hi),
                                add, "padd")
            sums[i] = pre
            if not self.fused:
                out[f"{self.name}_d{i}"] = pre

        coeffs = self._reciprocal_coeffs()
        for i in range(n):
            if self.fused:
                d = ev.bootstrap(sums[i], boot, self.headroom)
            else:
                d = ev.input(f"{self.name}_d{i}_boot", boot)
            r = ev.chebyshev(d, coeffs, -1.0, 1.0, high=True)
            for j in range(n):
                out[f"{self.name}_p{i}{j}"] = ev.mul(exps[i][j], r, hi)
        return out

    # -- test data ------------------------------------------------------------

    def _scores(self, rng):
        """Per-row scores (heads x 132 queries x 132 keys) whose exp-sum fits the interval."""
        lower, upper = self.config["inverse"]
        q = self.rows * TOKENS_PER_BLOCK
        scores = np.zeros((HEADS, q, q))
        for h in range(HEADS - 1):
            for t in range(q):
                for _ in range(100):
                    row = rng.normal(0, rng.uniform(0.5, 3.0), q) + rng.uniform(-5, 5)
                    total = np.sum(self._exp_approx(row, h))
                    if lower * 1.05 < total < upper * 0.95:
                        break
                else:
                    raise RuntimeError("could not draw a score row inside the inverse interval")
                scores[h, t] = row
        return scores

    def _exp_approx(self, row, h):
        """What the program computes for one row of 132 real keys (padding keys are zero)."""
        inv, sum_coeff, sos, sts, shift = self._coefficients()
        total, total_sq = row.sum(), (row ** 2).sum()
        s = sum_coeff * total + sos[h] * total_sq + sts[h] * total ** 2 + shift[h]
        return (1.0 + row * inv - s) ** (2 ** self.iterations)

    def _blocks(self, scores):
        """Encode (heads, 132, 132) into the 4 x 4 grid of (13, 33, 35) blocks."""
        k = np.arange(SCORE_SLOTS)
        h, t, key = k % HEADS, k % TOKENS_PER_BLOCK, k % KEY_DIM
        real = key < TOKENS_PER_BLOCK
        blocks = {}
        for i in range(self.rows):
            for j in range(self.rows):
                v = np.zeros(SLOTS)
                vals = scores[h, i * TOKENS_PER_BLOCK + t, j * TOKENS_PER_BLOCK + np.minimum(key, 32)]
                v[:SCORE_SLOTS] = np.where(real, vals, 0.0)
                blocks[(i, j)] = v
        return blocks

    def sample(self, rng):
        scores = self._scores(rng)
        self._scores_cache = scores
        inv, sum_coeff, sos, sts, shift = self._coefficients()
        q = self.rows * TOKENS_PER_BLOCK
        partial = np.zeros_like(scores)
        exps = np.zeros_like(scores)
        for h in range(HEADS):
            for t in range(q):
                row = scores[h, t]
                total, total_sq = row.sum(), (row ** 2).sum()
                s = sum_coeff * total + sos[h] * total_sq + sts[h] * total ** 2 + shift[h]
                base = 1.0 + row * inv - s
                partial[h, t] = base ** (2 ** self.before)
                exps[h, t] = base ** (2 ** self.iterations)
        inputs = {}
        for (i, j), v in self._blocks(scores).items():
            inputs[f"{self.name}_s{i}{j}"] = v
        self._exps = exps
        self._partial = partial
        if self.fused:
            return inputs
        for (i, j), v in self._blocks(partial).items():
            inputs[f"{self.name}_e{i}{j}_boot"] = v
        mult, add = self._prescale()
        sums = exps.sum(axis=2, keepdims=True) * np.ones_like(exps)
        for i in range(self.rows):
            d = self._blocks(sums)[(i, 0)] * mult + add
            inputs[f"{self.name}_d{i}_boot"] = d
        return inputs

    def reference(self, inputs):
        from tricycle.nonlinear import softmax

        out = {}
        probs = softmax(self._scores_cache, axis=-1)
        exps = self._exps
        mult, add = self._prescale()
        for (i, j), v in self._blocks(probs).items():
            out[f"{self.name}_p{i}{j}"] = v
        if self.fused:
            return out
        for (i, j), v in self._blocks(self._partial).items():
            out[f"{self.name}_e{i}{j}"] = v
        sums = exps.sum(axis=2, keepdims=True) * np.ones_like(exps)
        for i in range(self.rows):
            out[f"{self.name}_d{i}"] = self._blocks(sums)[(i, 0)] * mult + add
        return out


@dataclass
class Probe:
    """One of each operation the kernels use, on one ciphertext, to attribute CKKS error."""

    level: int = 21
    name: str = "probe"

    def cipher_inputs(self):
        return [f"{self.name}_x"]

    def _mask(self):
        return np.random.default_rng(3).uniform(-1, 1, SLOTS)

    def emit(self, backend, ev, level):
        x = ev.input(f"{self.name}_x", level)
        mask = self._mask()
        return {
            f"{self.name}_rot": ev.rotate(x, 33),
            f"{self.name}_rot_neg": ev.rotate(x, -429),
            f"{self.name}_mask": ev.mul_vector(x, mask, x.scale, "mask"),
            f"{self.name}_mask_low": ev.mul_vector(x, mask, x.scale / 2 ** 27, "masklow"),
            f"{self.name}_const": ev.mul_const(x, 0.37, x.scale),
            f"{self.name}_square": ev.square(x),
            f"{self.name}_rot_mask": ev.mul_vector(ev.rotate(x, 33), mask, x.scale, "rmask"),
        }

    def sample(self, rng):
        return {f"{self.name}_x": rng.uniform(-1, 1, SLOTS)}

    def reference(self, inputs):
        x, mask = inputs[f"{self.name}_x"], self._mask()
        return {f"{self.name}_rot": np.roll(x, -33), f"{self.name}_rot_neg": np.roll(x, 429),
                f"{self.name}_mask": x * mask, f"{self.name}_mask_low": x * mask,
                f"{self.name}_const": 0.37 * x, f"{self.name}_square": x * x,
                f"{self.name}_rot_mask": np.roll(x, -33) * mask}


KERNELS = {"gelu": Gelu, "layernorm": LayerNorm, "softmax": Softmax, "probe": Probe}


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
    outputs = kernel.emit(backend, ev, level)
    return outputs, ev
