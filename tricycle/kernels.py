"""Tricycle's matrix products as CKKS programs over block grids.

Each emitter here is one of the four products the Tricycle layer runs
(``transformer.cpp``), written once against a small backend interface:

``cipher(name)``, ``plain(name, like)``, ``rotate(x, k)``, ``hoisted(x, ks)``,
``add``, ``mul``, ``relinearize``, ``rescale``, ``double_rescale`` and
``rotate_accumulate(xs, ks)`` (sum of ``xs[i]`` rotated left by ``ks[i]``).

``NumpyBackend`` runs the program on slot vectors, ``ScaleBackend`` replays it
on (scale, level) pairs with Cerium's real RNS primes, and
``tricycle.cerium_backend.CeriumBackend`` emits it as Cerium DSL. Plaintext and
input names are shared, so the vectors built by ``*_plaintexts`` feed all three.

Level budget per product, starting from ciphertexts at level ``L``:
ciphertext-plaintext and plaintext-ciphertext end at ``L - 1``; masked
ciphertext-ciphertext ends at ``L - 2`` (one double rescale after the mask and
the product), unmasked at ``L - 1``.
"""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from tricycle.encoding import as_slot_array, concatenated_matmul, decode, dense_matmul
from tricycle.matmul import (
    DEFAULT_SLOTS,
    ct_ct_plan,
    ct_pt_plan,
    extract_diagonal,
    output_mask,
    pt_ct_plan,
    put_encoded,
)
from tricycle.slots import left_rotate


def _bits_lsb_first(number):
    bits = []
    while number > 0:
        bits.append(number & 1)
        number >>= 1
    return bits


def _norm(rotation, num_slots):
    return int(rotation) % num_slots


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class NumpyBackend:
    """Executes a kernel on slot vectors and counts the homomorphic work."""

    def __init__(self, inputs, num_slots=DEFAULT_SLOTS):
        self.inputs = inputs
        self.num_slots = num_slots
        self.counts = Counter()

    def cipher(self, name, level=None):
        return np.asarray(self.inputs[name])

    def plain(self, name, like):
        return np.asarray(self.inputs[name])

    def rotate(self, x, k):
        k = _norm(k, self.num_slots)
        if k == 0:
            return x
        self.counts["rotations"] += 1
        return left_rotate(x, k)

    def hoisted(self, x, ks):
        ks = [_norm(k, self.num_slots) for k in ks]
        self.counts["hoisted_rotations"] += sum(1 for k in ks if k)
        return [left_rotate(x, k) for k in ks]

    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return a - b

    def mul(self, a, b):
        self.counts["multiplies"] += 1
        return a * b

    def scalar(self, name, value, level, bits):
        return value

    def vector_plain(self, name, values, level, bits):
        return np.asarray(values)

    def modswitch(self, x):
        return x

    def relinearize(self, x):
        self.counts["relinearizations"] += 1
        return x

    def rescale(self, x):
        return x

    def double_rescale(self, x):
        return x

    def rotate_accumulate(self, xs, ks):
        total = None
        for x, k in zip(xs, ks):
            term = self.rotate(x, k)
            total = term if total is None else total + term
        return total

    def vector(self, xs):
        return np.stack(xs)

    def unvector(self, x, count):
        return list(x)

    def bsgs(self, x, plains, babies, giants):
        babies = [_norm(k, self.num_slots) for k in babies]
        giants = [_norm(k, self.num_slots) for k in giants]
        self.counts["hoisted_rotations"] += sum(1 for k in babies if k) * len(x)
        self.counts["rotations"] += sum(1 for k in giants if k) * len(x)
        self.counts["multiplies"] += len(plains) * len(x)
        rotated = [np.roll(x, -k, axis=-1) for k in babies]
        total = 0
        for g, giant in enumerate(giants):
            acc = sum(rotated[y] * plains[g * len(babies) + y] for y in range(len(babies)))
            total = total + np.roll(acc, -giant, axis=-1)
        return total


@dataclass(frozen=True)
class Tracked:
    """Exact CKKS bookkeeping: numeric scale, level, and ciphertext degree."""

    scale: Fraction
    level: int
    degree: int = 1
    is_plain: bool = False


class ScaleBackend:
    """Replays a kernel on scales and levels the way Cerium's runtime will.

    Rescaling at level ``l`` divides by ``primes[l - 1]``. Plaintexts get the
    scale of the prime the following rescale removes, so a
    ciphertext-plaintext product comes back at its input scale exactly. The scales chosen for every plaintext are kept in
    ``plaintext_scales`` for the runtime's raw inputs.
    """

    def __init__(self, primes, cipher_scales):
        self.primes = list(primes)
        self.cipher_scales = cipher_scales
        self.plaintext_scales = {}

    def cipher(self, name):
        scale, level = self.cipher_scales[name]
        return Tracked(Fraction(scale), level)

    def plain(self, name, like):
        scale = self.primes[like.level - 1]
        previous = self.plaintext_scales.get(name)
        if previous is not None and previous != (scale, like.level):
            raise ValueError(f"plaintext {name} used at two levels")
        self.plaintext_scales[name] = (scale, like.level)
        return Tracked(Fraction(scale), like.level, 0, True)

    def rotate(self, x, k):
        if x.degree != 1:
            raise ValueError("rotating a ciphertext that is not relinearized")
        return x

    def hoisted(self, x, ks):
        return [self.rotate(x, k) for k in ks]

    def add(self, a, b):
        if a.level != b.level or a.scale != b.scale:
            raise ValueError(f"add of mismatched operands {a} + {b}")
        return Tracked(a.scale, a.level, max(a.degree, b.degree))

    def mul(self, a, b):
        if a.level != b.level:
            raise ValueError(f"mul at different levels {a.level} and {b.level}")
        if a.degree > 1 or b.degree > 1:
            raise ValueError("multiplying a ciphertext that is not relinearized")
        return Tracked(a.scale * b.scale, a.level, a.degree + b.degree)

    def relinearize(self, x):
        return Tracked(x.scale, x.level, 1)

    def rescale(self, x):
        if x.level < 2:
            raise ValueError("out of levels")
        return Tracked(x.scale / self.primes[x.level - 1], x.level - 1, x.degree)

    def double_rescale(self, x):
        return self.rescale(self.rescale(x))

    def rotate_accumulate(self, xs, ks):
        total = xs[0]
        for x in xs[1:]:
            total = self.add(total, x)
        return self.rotate(total, 0)

    def vector(self, xs):
        for x in xs[1:]:
            self.add(xs[0], x)
        return xs[0]

    def unvector(self, x, count):
        return [x] * count

    def bsgs(self, x, plains, babies, giants):
        if len(plains) != len(babies) * len(giants):
            raise ValueError("bsgs needs one plaintext per baby and giant step")
        return self.rotate(_sum(self, [self.mul(x, p) for p in plains]), 0)


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------


def emit_replicate(b, x, num_elements, num_replications, num_slots):
    """``replicate_ctxt``: doubling ladder of left rotations by ``-num_elements * 2^i``."""
    if num_replications == 1:
        return x
    bits = _bits_lsb_first(num_replications)
    last = x
    rotation_number = -1
    total = x if bits[0] else None
    for i in range(1, len(bits)):
        last = b.add(last, b.rotate(last, _norm(rotation_number * num_elements, num_slots)))
        rotation_number *= 2
        if bits[i]:
            if total is None:
                total = last
            else:
                shift = _norm(rotation_number * num_elements, num_slots)
                total = b.add(b.rotate(total, shift), last)
    return total


def _baby_steps(b, x, count, step, num_slots):
    return b.hoisted(x, [_norm(y * step, num_slots) for y in range(count)])


def _sum(b, terms):
    total = terms[0]
    for term in terms[1:]:
        total = b.add(total, term)
    return total


def _giant(b, accs, bases):
    return b.rotate_accumulate(accs, bases)


def emit_ct_pt(b, plan, cipher_grid, plain_name, cols, num_slots=DEFAULT_SLOTS):
    """``ctxt_ptxt_block_matmul``: ``out[i][k] = sum_j C[i][j] @ W[j][k]``.

    ``plain_name(j, k, g, y)`` names the diagonal for weight block ``(j, k)``,
    giant step ``g``, baby step ``y``. Products accumulate unrescaled across
    the inner block axis and every chunk; one rescale per output block.
    """
    rows, inner = len(cipher_grid), len(cipher_grid[0])
    babies = [[
        _baby_steps(b, emit_replicate(b, cipher_grid[i][j], plan.cipher_elements,
                                      plan.num_repetitions, num_slots),
                    plan.num_baby_steps, plan.baby_step, num_slots)
        for j in range(inner)] for i in range(rows)]
    bases = [_norm(base, num_slots) for base, _ in plan.chunks]
    out = []
    for i in range(rows):
        row = []
        for k in range(cols):
            accs = []
            for g, (_, extracts) in enumerate(plan.chunks):
                terms = []
                for j in range(inner):
                    for y in range(len(extracts)):
                        baby = babies[i][j][y]
                        terms.append(b.mul(baby, b.plain(plain_name(j, k, g, y), baby)))
                accs.append(_sum(b, terms))
            row.append(b.rescale(_giant(b, accs, bases)))
        out.append(row)
    return out


def emit_pt_ct(b, plan, cipher_grid, plain_name, rows, num_slots=DEFAULT_SLOTS):
    """``ptxt_ctxt_block_matmul``: ``out[i][k] = sum_j P[i][j] @ C[j][k]``.

    ``plain_name(i, j, g, y)`` names the diagonal of plaintext block ``(i, j)``.
    """
    inner, cols = len(cipher_grid), len(cipher_grid[0])
    babies = [[
        _baby_steps(b, emit_replicate(b, cipher_grid[j][k], plan.cipher_elements,
                                      plan.num_repetitions, num_slots),
                    plan.num_baby_steps, plan.baby_step, num_slots)
        for k in range(cols)] for j in range(inner)]
    bases = [_norm(base, num_slots) for base, _ in plan.chunks]
    out = []
    for i in range(rows):
        row = []
        for k in range(cols):
            accs = []
            for g, (_, extracts) in enumerate(plan.chunks):
                terms = []
                for j in range(inner):
                    for y in range(len(extracts)):
                        baby = babies[j][k][y]
                        terms.append(b.mul(baby, b.plain(plain_name(i, j, g, y), baby)))
                accs.append(_sum(b, terms))
            row.append(b.rescale(_giant(b, accs, bases)))
        out.append(row)
    return out


def emit_fused_bsgs(b, plan, cipher_grid, plain_name, zero_name, plain_side,
                    num_slots=DEFAULT_SLOTS):
    """:func:`emit_ct_pt` (``plain_side="right"``) or :func:`emit_pt_ct` (``"left"``)
    as Cerium's ``BsgsMultiplyAccumulate`` over a ciphertext vector.

    The ciphertext blocks that share a weight block are one vector: the rows
    ``C[:][j]`` of a ciphertext-plaintext product, the columns ``C[j][:]`` of a
    plaintext-ciphertext one. Every diagonal multiplies the whole vector, and
    each vector gets its own hoisted baby steps per output block. Chunks
    shorter than ``num_baby_steps`` are padded with the zero plaintext.
    """
    if plain_side == "right":
        count, inner = len(cipher_grid), len(cipher_grid[0])
        blocks = [[cipher_grid[i][j] for i in range(count)] for j in range(inner)]
    else:
        inner, count = len(cipher_grid), len(cipher_grid[0])
        blocks = [list(cipher_grid[j]) for j in range(inner)]
    vectors = [b.vector([emit_replicate(b, x, plan.cipher_elements, plan.num_repetitions, num_slots)
                         for x in row]) for row in blocks]
    babies = [_norm(y * plan.baby_step, num_slots) for y in range(plan.num_baby_steps)]
    giants = [_norm(base, num_slots) for base, _ in plan.chunks]

    def products(other):
        total = None
        for j, vector in enumerate(vectors):
            a, c = (j, other) if plain_side == "right" else (other, j)
            names = [plain_name(a, c, g, y) if y < len(extracts) else zero_name
                     for g, (_, extracts) in enumerate(plan.chunks)
                     for y in range(plan.num_baby_steps)]
            term = b.bsgs(vector, [b.plain(name, vector) for name in names], babies, giants)
            total = term if total is None else b.add(total, term)
        return b.unvector(b.rescale(total), count)

    return products


def emit_ct_ct(b, plan, left_grid, right_grid, mask_name=None, num_slots=DEFAULT_SLOTS):
    """``ctxt_ctxt_{right,left}_bsgs_block_matmul``: ``out[i][k] = sum_j L[i][j] @ R[j][k]``.

    The ``plan.baby_side`` operand gets hoisted baby steps; the other operand
    gets one hoisted rotation per pre-giant-step amount, multiplied by the
    rotated output mask ``mask_name(g)`` when the plan has a mask. Ciphertext
    products are summed per chunk, relinearized once per chunk, rotated by the
    chunk base and summed. HEaaN rescales after every mask multiply; here the
    mask stays unrescaled and the sum is double-rescaled once, which reaches
    the same scale and level without rescaling each pre-rotation or
    mod-switching the baby steps (Cerium's register allocator rejects a
    mod-switched program input fused with mod-switched intermediates).
    """
    rows, inner = len(left_grid), len(left_grid[0])
    cols = len(right_grid[0])
    left = [[emit_replicate(b, left_grid[i][j], plan.left_elements, plan.repetitions_left, num_slots)
             for j in range(inner)] for i in range(rows)]
    right = [[emit_replicate(b, right_grid[j][k], plan.right_elements, plan.repetitions_right,
                             num_slots)
              for k in range(cols)] for j in range(inner)]
    baby_grid, other_grid = (right, left) if plan.baby_side == "right" else (left, right)
    masked = plan.mask_value is not None

    def baby_steps(x):
        return _baby_steps(b, x, plan.num_baby_steps, plan.baby_step, num_slots)

    distinct = sorted({rotation for _, before in plan.chunks for rotation in before})

    def pre_rotations(x):
        rotated = dict(zip(distinct, b.hoisted(x, distinct) if len(distinct) > 1
                           else [b.rotate(x, distinct[0])]))
        per_chunk = []
        for g, (_, before) in enumerate(plan.chunks):
            chunk = []
            for rotation in before:
                y = rotated[rotation]
                if masked:
                    y = b.mul(y, b.plain(mask_name(g), y))
                chunk.append(y)
            per_chunk.append(chunk)
        return per_chunk

    babies = [[baby_steps(x) for x in row] for row in baby_grid]
    others = [[pre_rotations(x) for x in row] for row in other_grid]
    bases = [_norm(base, num_slots) for base, _ in plan.chunks]
    out = []
    for i in range(rows):
        row = []
        for k in range(cols):
            accs = []
            for g, (_, before) in enumerate(plan.chunks):
                terms = []
                for j in range(inner):
                    for y in range(len(before)):
                        if plan.baby_side == "right":
                            terms.append(b.mul(babies[j][k][y], others[i][j][g][y]))
                        else:
                            terms.append(b.mul(babies[i][j][y], others[j][k][g][y]))
                accs.append(b.relinearize(_sum(b, terms)))
            total = _giant(b, accs, bases)
            row.append(b.double_rescale(total) if masked else b.rescale(total))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Plaintext data
# ---------------------------------------------------------------------------


def ct_pt_plaintexts(plan, weight_blocks, plain_name, num_slots=DEFAULT_SLOTS):
    """Diagonals for :func:`emit_ct_pt` (``weight_blocks[j][k]``) or :func:`emit_pt_ct`
    (``weight_blocks[i][j]``): each extract, rotated by minus its chunk base."""
    values = {}
    for a, row in enumerate(weight_blocks):
        for c, block in enumerate(row):
            block = as_slot_array(block)
            for g, (base, extracts) in enumerate(plan.chunks):
                for y, params in enumerate(extracts):
                    diagonal = extract_diagonal(block, params, num_slots)
                    values[plain_name(a, c, g, y)] = left_rotate(diagonal, -base)
    return values


def mask_plaintexts(plan, mask_name, num_slots=DEFAULT_SLOTS):
    if plan.mask_value is None:
        return {}
    return {mask_name(g): output_mask(plan, base, num_slots)
            for g, (base, _) in enumerate(plan.chunks)}


# ---------------------------------------------------------------------------
# The seven products of one Tricycle encoder layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerShape:
    """Tricyclic block sizes of one layer.

    ``tokens`` per row block (``num_tokens_per_block``), ``d_model`` and
    ``d_k`` after tricyclic padding, ``heads`` per head block, ``d_ff`` per
    feed-forward block. ``SHAPES`` has Tricycle's ``transformer_params.h``
    values for 128 tokens. Only ciphertext and output shapes must be pairwise
    coprime; the square ``(1, d_model, d_ff)`` weights are diagonal sources.
    """

    tokens: int
    d_model: int
    heads: int
    d_k: int
    d_ff: int
    row_blocks: int = 1
    ff_blocks: int = 1
    reshape_pad: int = 2

    @property
    def padded_tokens(self):
        return self.tokens + self.reshape_pad


SHAPES = {
    # Every product of the layer, small enough to compile and run in seconds.
    "mini": LayerShape(tokens=5, d_model=7, heads=3, d_k=4, d_ff=11, row_blocks=2, ff_blocks=2),
    # BERT-Tiny: 128 tokens in 2 blocks of 89; d_ff 512 complex-packed into 2 blocks of 131.
    "tiny": LayerShape(tokens=89, d_model=131, heads=2, d_k=33, d_ff=131, row_blocks=2,
                       ff_blocks=2),
    # BERT-Base: 128 tokens in 4 blocks of 33; 12 heads of 64 complex-packed into 13 x 34;
    # d_ff 3072 complex-packed into 2 blocks of 769.
    "base": LayerShape(tokens=33, d_model=769, heads=13, d_k=34, d_ff=769, row_blocks=4,
                       ff_blocks=2),
}


@dataclass
class Matmul:
    """One Tricycle product: plan, block grid, input names and dense reference."""

    name: str
    kind: str
    plan: object
    left_grid: tuple
    right_grid: tuple
    left_dims: tuple
    right_dims: tuple
    cipher_side: str
    concatenated: bool = False
    description: str = ""
    mask_value: object = None
    # Plaintext products only: emit through ``emit_fused_bsgs``.
    fused: bool = False

    @property
    def out_grid(self):
        return (self.left_grid[0], self.right_grid[1])

    def input_name(self, side, a, c):
        return f"{self.name}_{side}_{a}_{c}"

    def plain_name(self, a, c, g, y):
        return f"{self.name}_diag_{a}_{c}_g{g}_y{y}"

    def mask_name(self, g):
        return f"{self.name}_mask_g{g}"

    @property
    def zero_name(self):
        return f"{self.name}_zero"

    def output_name(self, i, k):
        return f"{self.name}_out_{i}_{k}"

    def cipher_inputs(self):
        sides = ("left", "right") if self.kind == "ct_ct" else (self.cipher_side,)
        names = []
        for side in sides:
            rows, cols = self.left_grid if side == "left" else self.right_grid
            names.extend(self.input_name(side, a, c) for a in range(rows) for c in range(cols))
        return names

    def emit(self, b, num_slots=DEFAULT_SLOTS):
        """Emit the product on backend ``b``. Returns ``{output_name: value}``."""
        def grid(side):
            rows, cols = self.left_grid if side == "left" else self.right_grid
            return [[b.cipher(self.input_name(side, a, c)) for c in range(cols)]
                    for a in range(rows)]

        # BsgsMultiplyAccumulate needs a rotated giant step (one ModDown to
        # aggregate into); a single unrotated chunk fails in Cerium's compiler.
        fused = self.fused and len(self.plan.chunks) > 1
        if fused and self.kind == "ct_pt":
            products = emit_fused_bsgs(b, self.plan, grid("left"), self.plain_name,
                                       self.zero_name, "right", num_slots)
            columns = [products(k) for k in range(self.right_grid[1])]
            out = [list(row) for row in zip(*columns)]
        elif fused and self.kind == "pt_ct":
            products = emit_fused_bsgs(b, self.plan, grid("right"), self.plain_name,
                                       self.zero_name, "left", num_slots)
            out = [products(i) for i in range(self.left_grid[0])]
        elif self.kind == "ct_pt":
            out = emit_ct_pt(b, self.plan, grid("left"), self.plain_name, self.right_grid[1],
                             num_slots)
        elif self.kind == "pt_ct":
            out = emit_pt_ct(b, self.plan, grid("right"), self.plain_name, self.left_grid[0],
                             num_slots)
        else:
            out = emit_ct_ct(b, self.plan, grid("left"), grid("right"), self.mask_name,
                             num_slots)
        return {self.output_name(i, k): out[i][k]
                for i in range(len(out)) for k in range(len(out[0]))}

    def random_blocks(self, rng, scale=0.5):
        """Random dense blocks for both operands, ``[rows][cols]`` of tensors."""
        def make(grid_shape, dims):
            rows, cols = grid_shape
            return [[rng.uniform(-scale, scale, dims) for _ in range(cols)] for _ in range(rows)]
        return make(self.left_grid, self.left_dims), make(self.right_grid, self.right_dims)

    def inputs(self, left_blocks, right_blocks, num_slots=DEFAULT_SLOTS):
        """Slot vectors for every named input: ciphertexts, diagonals, masks.

        A ciphertext block is a tensor of the operand's dims, or already a
        length-``num_slots`` slot vector (another product's output).
        """
        values = {}
        for side, blocks in (("left", left_blocks), ("right", right_blocks)):
            if self.kind != "ct_ct" and side != self.cipher_side:
                plain_blocks = blocks
                continue
            for a, row in enumerate(blocks):
                for c, block in enumerate(row):
                    slots = np.asarray(block) if np.ndim(block) == 1 else put_encoded(block, num_slots)
                    values[self.input_name(side, a, c)] = slots
        if self.kind != "ct_ct":
            values.update(ct_pt_plaintexts(self.plan, plain_blocks, self.plain_name, num_slots))
            if self.fused and len(self.plan.chunks) > 1 and any(len(e) < self.plan.num_baby_steps for _, e in self.plan.chunks):
                values[self.zero_name] = np.zeros(num_slots)
        else:
            values.update(mask_plaintexts(self.plan, self.mask_name, num_slots))
        return values

    def reference(self, left_blocks, right_blocks):
        """Dense ``out[i][k] = sum_j left[i][j] @ right[j][k]`` (batch-summed if concatenated)."""
        product = concatenated_matmul if self.concatenated else dense_matmul
        rows, inner = self.left_grid
        cols = self.right_grid[1]
        out = {}
        for i in range(rows):
            for k in range(cols):
                total = sum(product(left_blocks[i][j], right_blocks[j][k]) for j in range(inner))
                if self.mask_value is not None:
                    total = total * self.mask_value
                out[self.output_name(i, k)] = total
        return out

    def decode(self, outputs):
        return {name: decode(np.asarray(value), self.plan.result_dims)
                for name, value in outputs.items()}


def tricycle_layer_matmuls(shape, num_slots=DEFAULT_SLOTS):
    """The seven products of ``evaluate_layer`` in ``transformer.cpp``, in order.

    The baby/giant-step weights passed to each plan are the ones
    ``transformer_layer_agnostic_preprocess`` passes: baby steps are paid once
    per input block column, giant steps once per output block column.
    """
    s = shape
    T, Tp, D, H, K, F = s.tokens, s.padded_tokens, s.d_model, s.heads, s.d_k, s.d_ff
    R = s.row_blocks
    matmuls = []

    matmuls.append(Matmul(
        "qkv", "ct_pt", ct_pt_plan((1, T, D), (H, D, K), num_slots,
                                   baby_weight=1, giant_weight=3),
        left_grid=(R, 1), right_grid=(1, 3), left_dims=(1, T, D), right_dims=(H, D, K),
        cipher_side="left",
        description="X (1,T,D) @ [Wq | Wk | Wv] (H,D,K) -> Q, K, V (H,T,K)"))

    matmuls.append(Matmul(
        "reshape", "pt_ct", pt_ct_plan((1, Tp, T), (H, T, K), num_slots),
        left_grid=(1, 1), right_grid=(1, 2 * R), left_dims=(1, Tp, T), right_dims=(H, T, K),
        cipher_side="right",
        description="pad (1,T+2,T) @ [K_blocks | V_blocks] (H,T,K) -> (H,T+2,K); "
                    "K's slots read as K^T (H,K,T+2)"))

    matmuls.append(Matmul(
        "qk", "ct_ct", ct_ct_plan((H, T, K), (H, K, Tp), num_slots, "right", 1.0 / np.sqrt(K),
                                  baby_weight=1, giant_weight=2),
        left_grid=(R, 1), right_grid=(1, R), left_dims=(H, T, K), right_dims=(H, K, Tp),
        cipher_side="both", mask_value=1.0 / np.sqrt(K),
        description="Q (H,T,K) @ K^T (H,K,T+2) / sqrt(d_k) -> scores (H,T,T+2), right BSGS"))

    matmuls.append(Matmul(
        "pv", "ct_ct", ct_ct_plan((H, T, Tp), (H, Tp, K), num_slots, "left", 1.0,
                                  baby_weight=2, giant_weight=1),
        left_grid=(R, R), right_grid=(R, 1), left_dims=(H, T, Tp), right_dims=(H, Tp, K),
        cipher_side="both", mask_value=1.0,
        description="softmax (H,T,T+2) @ V (H,T+2,K) -> (H,T,K), left BSGS"))

    matmuls.append(Matmul(
        "attn_out", "ct_pt", ct_pt_plan((H, T, K), (H, K, D), num_slots, concatenated=True,
                                        baby_weight=2, giant_weight=1),
        left_grid=(R, 1), right_grid=(1, 1), left_dims=(H, T, K), right_dims=(H, K, D),
        cipher_side="left", concatenated=True,
        description="attention (H,T,K) @ Wo (H,K,D) summed over heads -> (1,T,D)"))

    matmuls.append(Matmul(
        "ff1", "ct_pt", ct_pt_plan((1, T, D), (1, D, F), num_slots,
                                   baby_weight=1, giant_weight=s.ff_blocks),
        left_grid=(R, 1), right_grid=(1, s.ff_blocks), left_dims=(1, T, D), right_dims=(1, D, F),
        cipher_side="left",
        description="X (1,T,D) @ W1 (1,D,F) -> (1,T,F) per feed-forward block"))

    matmuls.append(Matmul(
        "ff2", "ct_pt", ct_pt_plan((1, T, F), (1, F, D), num_slots,
                                   baby_weight=s.ff_blocks, giant_weight=1),
        left_grid=(R, s.ff_blocks), right_grid=(s.ff_blocks, 1), left_dims=(1, T, F),
        right_dims=(1, F, D), cipher_side="left",
        description="H (1,T,F) @ W2 (1,F,D) summed over feed-forward blocks -> (1,T,D)"))
    return matmuls


def matmul_by_name(shape, name, num_slots=DEFAULT_SLOTS):
    for matmul in tricycle_layer_matmuls(shape, num_slots):
        if matmul.name == name:
            return matmul
    raise KeyError(name)


def block_matmul(kind, left_blocks, right_blocks, num_slots=DEFAULT_SLOTS, baby_side="right",
                 mask_value=None, concatenated=False):
    """Decoded ``out[i][k] = sum_j left[i][j] @ right[j][k]`` through one emitter.

    ``kind`` is ``ct_pt``, ``pt_ct`` or ``ct_ct``; every block shares one plan.
    """
    left_dims = np.shape(left_blocks[0][0])
    right_dims = np.shape(right_blocks[0][0])
    if kind == "ct_pt":
        plan = ct_pt_plan(left_dims, right_dims, num_slots, concatenated=concatenated)
        cipher_side = "left"
    elif kind == "pt_ct":
        plan = pt_ct_plan(left_dims, right_dims, num_slots, concatenated=concatenated)
        cipher_side = "right"
    else:
        plan = ct_ct_plan(left_dims, right_dims, num_slots, baby_side, mask_value)
        cipher_side = "both"
    matmul = Matmul(
        "block", kind, plan,
        left_grid=(len(left_blocks), len(left_blocks[0])),
        right_grid=(len(right_blocks), len(right_blocks[0])),
        left_dims=left_dims, right_dims=right_dims, cipher_side=cipher_side,
        concatenated=concatenated, mask_value=mask_value)
    if matmul.left_grid[1] != matmul.right_grid[0]:
        raise ValueError("block inner dimensions disagree")
    decoded, _, _ = simulate(matmul, left_blocks, right_blocks, num_slots)
    rows, cols = matmul.out_grid
    return [[decoded[matmul.output_name(i, k)] for k in range(cols)] for i in range(rows)]


def simulate(matmul, left_blocks, right_blocks, num_slots=DEFAULT_SLOTS):
    """Run ``matmul`` on the NumPy backend. Returns (decoded outputs, slot outputs, counts)."""
    backend = NumpyBackend(matmul.inputs(left_blocks, right_blocks, num_slots), num_slots)
    slots = matmul.emit(backend, num_slots)
    return matmul.decode(slots), slots, backend.counts


def track_scales(matmul, primes, cipher_scale, cipher_level, num_slots=DEFAULT_SLOTS):
    """Exact output scales/levels and the scale every plaintext must be encoded at."""
    backend = ScaleBackend(primes, {name: (cipher_scale, cipher_level)
                                    for name in matmul.cipher_inputs()})
    outputs = matmul.emit(backend, num_slots)
    return outputs, backend.plaintext_scales
