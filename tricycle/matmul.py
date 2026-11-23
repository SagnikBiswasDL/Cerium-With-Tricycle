"""Tricyclic matrix products, as slot programs.

Port of ``tricyclic_matmul.cpp`` from
https://bitbucket.org/tricyclepaper/tricyclic-matmul (commit 6f9fa57) and of the
HEaaN schedules in ``tricyclic_matmul_heaan.cpp`` from
https://bitbucket.org/tricyclepaper/tricycle, which is what the Tricycle
transformer layer runs.

A ciphertext is a length-``num_slots`` vector. Plaintext diagonals, baby-step
rotations and giant-step rotations are the same indices the C++ key-switches.
The BSGS grouping (``group_numbers``, ``optimal_BSGS_params``) is Lawrence's,
including the tie break that keeps the largest baby-step count when costs match.

This module builds the plans and evaluates single products directly on slot
vectors. ``direct_*`` evaluates one diagonal per contraction index and is the
specification the BSGS schedule is checked against. ``matmul_*`` evaluates the
BSGS schedule. Both decode to the same tensor as a dense batch matmul.
``tricycle.kernels`` lowers the same plans, over block grids, to a backend
(NumPy here, the Cerium DSL in ``tricycle.cerium_backend``).
"""

from dataclasses import dataclass, field
from math import gcd

import numpy as np

from tricycle.encoding import as_slot_array, decode, encode, require_coprime
from tricycle.slots import left_rotate, replicate

# CKKS ring dimension 2^16, the size Lawrence's tests and Cerium use.
DEFAULT_SLOTS = 1 << 15


def _cdiv(numerator, denominator):
    """Toward-zero integer division, matching C++ ``/`` on ints."""
    if numerator < 0:
        return -((-numerator) // denominator)
    return numerator // denominator


def mod_inverse(value, modulus):
    """Positive modular inverse. ``extended_gcd`` / ``mod_inverse`` in the C++."""
    value %= modulus
    x, y = 1, 0
    a, b = value, modulus
    while b:
        q = a // b
        a, b = b, a - q * b
        x, y = y, x - q * y
    if a != 1:
        raise ValueError(f"no inverse of {value} mod {modulus}")
    return x % modulus


def group_numbers(numbers, baby_step):
    """Group a sorted rotation list into runs whose gaps equal ``baby_step``.

    Groups come back longest-first, which is the order giant steps are emitted.
    """
    groups = []
    for number in numbers:
        placed = False
        for group in groups:
            if group and number - group[-1] == baby_step:
                group.append(number)
                placed = True
                break
        if not placed:
            groups.append([number])
    groups.sort(key=len, reverse=True)
    return groups


def optimal_bsgs(numbers, baby_step, baby_weight=1, giant_weight=1):
    """``optimal_BSGS_params``. Returns ``(baby_count, list of giant-step chunks)``."""
    groups = group_numbers(numbers, baby_step)
    largest = max((len(group) for group in groups), default=0)
    best_x = 1
    best_cost = giant_weight * len(numbers) + baby_weight
    if largest >= 2:
        for x in range(2, largest + 1):
            cost = baby_weight * x
            for group in groups:
                cost += giant_weight * ((len(group) + x - 1) // x)
            if cost <= best_cost:
                best_cost = cost
                best_x = x
    chunks = []
    for group in groups:
        for start in range(0, len(group), best_x):
            chunks.append(group[start:start + best_x])
    return best_x, chunks


@dataclass
class Extract:
    dims: tuple
    start: tuple
    zero_padding: int
    end: int


@dataclass
class CtPtPlan:
    """Rotation schedule for a ciphertext-plaintext or plaintext-ciphertext product.

    ``chunks[g]`` is ``(giant_base, [Extract, ...])``. Baby step ``y`` of that
    chunk pairs with extract ``y``. The ciphertext is rotated by
    ``y * baby_step``; the diagonal is then rotated by ``-giant_base``; the
    summed products of the chunk are rotated by ``giant_base``.
    """

    cipher_dims: tuple
    plain_dims: tuple
    result_dims: tuple
    baby_step: int
    num_baby_steps: int
    num_repetitions: int
    cipher_elements: int
    chunks: list
    rotations_needed: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    concatenated: bool = False
    cipher_is_left: bool = True

    @property
    def output_elements(self):
        return int(np.prod(self.result_dims))


@dataclass
class CtCtPlan:
    """Rotation schedule for a ciphertext-ciphertext product.

    ``pairs[i] = (left_rotation, right_rotation)`` is contraction index ``i``.
    ``baby_side`` names the operand that gets the ``y * baby_step`` baby steps.
    ``chunks[g] = (giant_base, [other_rotation_before_giant, ...])``: the other
    operand is rotated by those amounts before the giant step rotates the chunk
    by ``giant_base``. With ``mask_value`` set, each pre-rotated operand is
    multiplied by the output-region mask rotated by ``-giant_base`` (HEaaN's
    ``*_mask_preprocess``), so every slot past the output is zero.
    """

    left_dims: tuple
    right_dims: tuple
    result_dims: tuple
    baby_side: str
    baby_step: int
    num_baby_steps: int
    repetitions_left: int
    repetitions_right: int
    left_elements: int
    right_elements: int
    pairs: list
    chunks: list
    rotations_needed: list = field(default_factory=list)
    mask_value: object = None

    @property
    def output_elements(self):
        return int(np.prod(self.result_dims))


def extract_diagonal(tensor, params, num_slots):
    """Slots ``[zero_padding, end)`` of one generalized diagonal of ``tensor``."""
    tensor = as_slot_array(tensor)
    values = np.zeros(num_slots, dtype=tensor.dtype)
    dims = params.dims
    slots = np.arange(params.zero_padding, params.end)
    index = tuple((params.start[d] + slots) % dims[d] for d in range(3))
    values[slots] = tensor[index]
    return values


def output_mask(plan, base, num_slots):
    """``mask_value`` over the output slots, rotated by ``-base``."""
    values = np.zeros(num_slots, dtype=np.result_type(float, type(plan.mask_value)))
    values[:plan.output_elements] = plan.mask_value
    return left_rotate(values, -base)


def put_encoded(tensor, num_slots):
    encoded = encode(tensor)
    if encoded.shape[0] > num_slots:
        raise ValueError(f"{np.shape(tensor)} needs {encoded.shape[0]} slots, have {num_slots}")
    values = np.zeros(num_slots, dtype=encoded.dtype)
    values[:encoded.shape[0]] = encoded
    return values


def _check_slots(num_slots):
    num_slots = int(num_slots)
    if num_slots <= 0:
        raise ValueError("num_slots must be positive")
    return num_slots


def _check_product(dims1, dims2, concatenated):
    if concatenated:
        if dims1[0] != dims2[0] or dims1[2] != dims2[1]:
            raise ValueError(f"concatenated matmul needs equal batch and inner dims: {dims1} @ {dims2}")
        return
    # A left batch of 1 broadcasts against the right batch; the result has the
    # right batch, which is what the extracts below index.
    if dims1[0] not in (1, dims2[0]) or dims1[2] != dims2[1]:
        raise ValueError(f"inner dimensions disagree: {dims1} @ {dims2}")


def _diagonal_schedule(num_multiplications, increments, inner, starts, num_elements,
                       repetitions, num_elements3, num_slots, extract_dims):
    """Shared diagonal walk of both ciphertext-plaintext preprocessors."""
    increment1, increment2 = increments
    replicated = num_elements * repetitions
    by_rotation = {}
    steps = []
    for i in range(num_multiplications):
        total = (increment1 * i) % num_elements
        start = starts(i, None)
        if increment2 is not None:
            c = i % inner
            d = i // inner
            total = (d * increment1 + c * increment2) % num_elements
            start = starts(c, d)
        filled = 0
        first = True
        while filled < num_elements3:
            if first:
                capacity = replicated - total
                rotation = total
                first = False
            else:
                capacity = replicated
                rotation = num_slots - filled
            take = min(capacity, num_elements3 - filled)
            if take <= 0:
                raise ValueError(
                    "diagonal does not advance; raise the repetition count "
                    f"(rotation {rotation}, filled {filled} of {num_elements3})"
                )
            params = Extract(extract_dims, start, filled, take + filled)
            by_rotation[rotation] = params
            steps.append((rotation, params))
            filled += take
    return by_rotation, steps


def _finish_ct_pt(by_rotation, steps, baby_step, num_elements, repetitions, num_slots,
                  baby_weight, giant_weight):
    ordered = sorted(rotation for rotation, _ in steps)
    baby_count, chunks = optimal_bsgs(ordered, baby_step, baby_weight, giant_weight)
    plan_chunks = []
    rotations = []
    for chunk in chunks:
        plan_chunks.append((chunk[0], [by_rotation[rotation] for rotation in chunk]))
        rotations.append(chunk[0] % num_slots)
    power = 1
    while power < repetitions:
        rotations.append((-num_elements * power) % num_slots)
        power <<= 1
    for i in range(1, baby_count):
        rotations.append((baby_step * i) % num_slots)
    return baby_count, plan_chunks, rotations


def _repetitions(num_elements3, offset, num_elements, num_slots, force_repetitions):
    useful = 1 + _cdiv(num_elements3 - offset + num_elements - 1, num_elements)
    repetitions = min(useful, num_slots // num_elements)
    if force_repetitions != -1:
        repetitions = int(force_repetitions)
    if repetitions < 1 or repetitions * num_elements > num_slots:
        raise ValueError("repetition count does not fit in the ciphertext")
    return repetitions


def ct_pt_plan(left_dims, right_dims, num_slots=DEFAULT_SLOTS, force_repetitions=-1,
               concatenated=False, baby_weight=1, giant_weight=1):
    """``ctxt_ptxt_matmul_agnostic_preprocess``.

    ``left_dims`` is the ciphertext, ``right_dims`` the plaintext.
    """
    num_slots = _check_slots(num_slots)
    dims1 = require_coprime(left_dims)
    dims2 = require_coprime(right_dims)
    _check_product(dims1, dims2, concatenated)

    num_elements1 = dims1[0] * dims1[1] * dims1[2]
    increment1 = (dims1[0] * dims1[1] * mod_inverse(dims1[0] * dims1[1], dims1[2])) % num_elements1
    increment2 = None
    num_multiplications = dims1[2]
    num_elements3 = dims2[0] * dims1[1] * dims2[2]
    if concatenated:
        increment2 = increment1
        increment1 = (dims1[1] * dims1[2] * mod_inverse(dims1[1] * dims1[2], dims2[0])) % num_elements1
        num_multiplications = dims1[0] * dims1[2]
        num_elements3 = dims1[1] * dims2[2]

    repetitions = _repetitions(num_elements3, dims1[0] * dims1[1], num_elements1, num_slots,
                               force_repetitions)
    by_rotation, steps = _diagonal_schedule(
        num_multiplications, (increment1, increment2), dims1[2],
        lambda c, d: (0, c, 0) if d is None else (d, c, 0),
        num_elements1, repetitions, num_elements3, num_slots, dims2)
    baby_step = dims1[0] * dims1[1]
    baby_count, chunks, rotations = _finish_ct_pt(
        by_rotation, steps, baby_step, num_elements1, repetitions, num_slots,
        baby_weight, giant_weight)
    result_dims = (1, dims1[1], dims2[2]) if concatenated else (dims2[0], dims1[1], dims2[2])
    return CtPtPlan(
        cipher_dims=dims1,
        plain_dims=dims2,
        result_dims=result_dims,
        baby_step=baby_step,
        num_baby_steps=baby_count,
        num_repetitions=repetitions,
        cipher_elements=num_elements1,
        chunks=chunks,
        rotations_needed=rotations,
        steps=steps,
        concatenated=concatenated,
        cipher_is_left=True,
    )


def pt_ct_plan(left_dims, right_dims, num_slots=DEFAULT_SLOTS, force_repetitions=-1,
               concatenated=False, baby_weight=1, giant_weight=1):
    """``ptxt_ctxt_matmul_agnostic_preprocess``.

    ``left_dims`` is the plaintext, ``right_dims`` the ciphertext. Baby steps
    rotate the ciphertext.
    """
    num_slots = _check_slots(num_slots)
    dims1 = require_coprime(left_dims)
    dims2 = require_coprime(right_dims)
    _check_product(dims1, dims2, concatenated)

    num_elements2 = dims2[0] * dims2[1] * dims2[2]
    increment1 = (dims2[0] * dims2[2] * mod_inverse(dims2[0] * dims2[2], dims2[1])) % num_elements2
    increment2 = None
    num_multiplications = dims1[2]
    num_elements3 = dims2[0] * dims1[1] * dims2[2]
    if concatenated:
        # The ciphertext-plaintext path saves its contraction step and then
        # replaces the other step with a batch step. The published
        # plaintext-ciphertext path instead rebuilds the contraction step with
        # mod_inverse(..., dims1[1]), i.e. modulo the output rows. That walks
        # the contraction index by a unit other than 1, so the diagonal is
        # paired with the wrong column of the ciphertext. Saving the step
        # already computed above (inverse modulo the contraction length,
        # dims2[1]) is the version whose slots equal the batch sum.
        increment2 = increment1
        increment1 = (dims2[1] * dims2[2] * mod_inverse(dims2[1] * dims2[2], dims2[0])) % num_elements2
        num_multiplications = dims1[0] * dims1[2]
        num_elements3 = dims1[1] * dims2[2]

    repetitions = _repetitions(num_elements3, dims2[0] * dims2[2], num_elements2, num_slots,
                               force_repetitions)
    by_rotation, steps = _diagonal_schedule(
        num_multiplications, (increment1, increment2), dims1[2],
        lambda c, d: (0, 0, c) if d is None else (d, 0, c),
        num_elements2, repetitions, num_elements3, num_slots, dims1)
    baby_step = dims2[0] * dims2[2]
    baby_count, chunks, rotations = _finish_ct_pt(
        by_rotation, steps, baby_step, num_elements2, repetitions, num_slots,
        baby_weight, giant_weight)
    result_dims = (1, dims1[1], dims2[2]) if concatenated else (dims2[0], dims1[1], dims2[2])
    return CtPtPlan(
        cipher_dims=dims2,
        plain_dims=dims1,
        result_dims=result_dims,
        baby_step=baby_step,
        num_baby_steps=baby_count,
        num_repetitions=repetitions,
        cipher_elements=num_elements2,
        chunks=chunks,
        rotations_needed=rotations,
        steps=steps,
        concatenated=concatenated,
        cipher_is_left=False,
    )


def ct_ct_plan(left_dims, right_dims, num_slots=DEFAULT_SLOTS, baby_side="right",
               mask_value=None, baby_weight=1, giant_weight=1):
    """``ctxt_ctxt_{right,left}_bsgs_matmul_agnostic_preprocess``.

    Tricycle uses right-matrix baby steps for ``Q @ K^T`` (with mask
    ``1/sqrt(d_k)`` on ``Q``) and left-matrix baby steps for ``softmax @ V``
    (with mask ``1`` on ``V``).
    """
    if baby_side not in ("right", "left"):
        raise ValueError("baby_side is 'right' or 'left'")
    num_slots = _check_slots(num_slots)
    dims1 = require_coprime(left_dims)
    dims2 = require_coprime(right_dims)
    if dims1[0] != dims2[0] or dims1[2] != dims2[1]:
        raise ValueError(f"inner dimensions disagree: {dims1} @ {dims2}")

    num_elements1 = dims1[0] * dims1[1] * dims1[2]
    num_elements2 = dims2[0] * dims2[1] * dims2[2]
    increment1 = (dims1[0] * dims1[1] * mod_inverse(dims1[0] * dims1[1], dims1[2])) % num_elements1
    increment2 = (dims2[0] * dims2[2] * mod_inverse(dims2[0] * dims2[2], dims2[1])) % num_elements2
    num_elements3 = dims2[0] * dims1[1] * dims2[2]
    repetitions1 = 1 + _cdiv(num_elements3 - dims1[0] * dims1[1] + num_elements1 - 1, num_elements1)
    repetitions2 = 1 + _cdiv(num_elements3 - dims2[0] * dims2[2] + num_elements2 - 1, num_elements2)
    if repetitions1 < 1 or repetitions2 < 1:
        raise ValueError("negative repetition count")
    if repetitions1 * num_elements1 > num_slots or repetitions2 * num_elements2 > num_slots:
        raise ValueError("not enough slots for the required repetitions")

    pairs = []
    for i in range(dims1[2]):
        left_rotation = (increment1 * i) % num_elements1
        right_rotation = (increment2 * i) % num_elements2
        covered = min(
            repetitions1 * num_elements1 - left_rotation,
            repetitions2 * num_elements2 - right_rotation,
            num_elements3,
        )
        if covered < num_elements3:
            raise ValueError("one rotation does not cover the output; ct-ct has no second tile")
        pairs.append((left_rotation, right_rotation))

    if baby_side == "right":
        baby_step = dims2[0] * dims2[2]
        to_other = {right: left for left, right in pairs}
    else:
        baby_step = dims1[0] * dims1[1]
        to_other = {left: right for left, right in pairs}
    baby_count, chunks = optimal_bsgs(sorted(to_other), baby_step, baby_weight, giant_weight)

    rotations = []
    for count, elements in ((repetitions1, num_elements1), (repetitions2, num_elements2)):
        power = 1
        while power < count:
            rotations.append((-elements * power) % num_slots)
            power <<= 1
    for i in range(1, baby_count):
        rotations.append((baby_step * i) % num_slots)
    plan_chunks = []
    for chunk in chunks:
        base = chunk[0] % num_slots
        before = [(to_other[rotation] - base) % num_slots for rotation in chunk]
        rotations.extend(before)
        rotations.append(base)
        plan_chunks.append((base, before))

    return CtCtPlan(
        left_dims=dims1,
        right_dims=dims2,
        result_dims=(dims2[0], dims1[1], dims2[2]),
        baby_side=baby_side,
        baby_step=baby_step,
        num_baby_steps=baby_count,
        repetitions_left=repetitions1,
        repetitions_right=repetitions2,
        left_elements=num_elements1,
        right_elements=num_elements2,
        pairs=pairs,
        chunks=plan_chunks,
        rotations_needed=rotations,
        mask_value=mask_value,
    )


def _apply_ct_pt_plan(cipher, plaintext, plan):
    """BSGS product. ``plaintext`` supplies the diagonals, ``cipher`` is rotated."""
    num_slots = cipher.shape[0]
    tiled = replicate(cipher, plan.cipher_elements, plan.num_repetitions)
    babies = [left_rotate(tiled, y * plan.baby_step) for y in range(plan.num_baby_steps)]
    total = np.zeros(num_slots, dtype=np.result_type(cipher, plaintext))
    for base, extracts in plan.chunks:
        acc = np.zeros_like(total)
        for y, params in enumerate(extracts):
            diagonal = left_rotate(extract_diagonal(plaintext, params, num_slots), -base)
            acc += babies[y] * diagonal
        total += left_rotate(acc, base)
    return total


def _apply_steps(cipher, plaintext, plan):
    """One product per diagonal, no baby/giant split. Same linear combination."""
    num_slots = cipher.shape[0]
    tiled = replicate(cipher, plan.cipher_elements, plan.num_repetitions)
    total = np.zeros(num_slots, dtype=np.result_type(cipher, plaintext))
    for rotation, params in plan.steps:
        total += left_rotate(tiled, rotation) * extract_diagonal(plaintext, params, num_slots)
    return total


def matmul_ct_pt(left, right, num_slots=DEFAULT_SLOTS, force_repetitions=-1, concatenated=False):
    """Ciphertext ``left`` times plaintext ``right``. Returns the decoded tensor."""
    left, right = as_slot_array(left), as_slot_array(right)
    plan = ct_pt_plan(left.shape, right.shape, num_slots, force_repetitions, concatenated)
    return decode(_apply_ct_pt_plan(put_encoded(left, num_slots), right, plan), plan.result_dims)


def matmul_pt_ct(left, right, num_slots=DEFAULT_SLOTS, force_repetitions=-1, concatenated=False):
    """Plaintext ``left`` times ciphertext ``right``."""
    left, right = as_slot_array(left), as_slot_array(right)
    plan = pt_ct_plan(left.shape, right.shape, num_slots, force_repetitions, concatenated)
    return decode(_apply_ct_pt_plan(put_encoded(right, num_slots), left, plan), plan.result_dims)


def direct_ct_pt(left, right, num_slots=DEFAULT_SLOTS, force_repetitions=-1, concatenated=False):
    """Same product as :func:`matmul_ct_pt`, summed diagonal by diagonal."""
    left, right = as_slot_array(left), as_slot_array(right)
    plan = ct_pt_plan(left.shape, right.shape, num_slots, force_repetitions, concatenated)
    return decode(_apply_steps(put_encoded(left, num_slots), right, plan), plan.result_dims)


def direct_pt_ct(left, right, num_slots=DEFAULT_SLOTS, force_repetitions=-1, concatenated=False):
    left, right = as_slot_array(left), as_slot_array(right)
    plan = pt_ct_plan(left.shape, right.shape, num_slots, force_repetitions, concatenated)
    return decode(_apply_steps(put_encoded(right, num_slots), left, plan), plan.result_dims)


def _apply_ct_ct(left_slots, right_slots, plan):
    num_slots = left_slots.shape[0]
    left = replicate(left_slots, plan.left_elements, plan.repetitions_left)
    right = replicate(right_slots, plan.right_elements, plan.repetitions_right)
    baby_source, other = (right, left) if plan.baby_side == "right" else (left, right)
    babies = [left_rotate(baby_source, y * plan.baby_step) for y in range(plan.num_baby_steps)]
    total = np.zeros(num_slots, dtype=np.result_type(left, right))
    for base, before in plan.chunks:
        acc = np.zeros_like(total)
        mask = None if plan.mask_value is None else output_mask(plan, base, num_slots)
        for y, rotation in enumerate(before):
            rotated = left_rotate(other, rotation)
            if mask is not None:
                rotated = rotated * mask
            acc += babies[y] * rotated
        total += left_rotate(acc, base)
    return total


def _apply_ct_ct_direct(left_slots, right_slots, plan):
    num_slots = left_slots.shape[0]
    left = replicate(left_slots, plan.left_elements, plan.repetitions_left)
    right = replicate(right_slots, plan.right_elements, plan.repetitions_right)
    total = np.zeros(num_slots, dtype=np.result_type(left, right))
    for left_rotation, right_rotation in plan.pairs:
        total += left_rotate(left, left_rotation) * left_rotate(right, right_rotation)
    if plan.mask_value is not None:
        total = total * output_mask(plan, 0, num_slots)
    return total


def matmul_ct_ct(left, right, num_slots=DEFAULT_SLOTS, baby_side="right", mask_value=None):
    """Ciphertext-ciphertext product under the HEaaN right- or left-BSGS schedule."""
    left, right = as_slot_array(left), as_slot_array(right)
    plan = ct_ct_plan(left.shape, right.shape, num_slots, baby_side, mask_value)
    slots = _apply_ct_ct(put_encoded(left, num_slots), put_encoded(right, num_slots), plan)
    return decode(slots, plan.result_dims)


def direct_ct_ct(left, right, num_slots=DEFAULT_SLOTS, baby_side="right", mask_value=None):
    left, right = as_slot_array(left), as_slot_array(right)
    plan = ct_ct_plan(left.shape, right.shape, num_slots, baby_side, mask_value)
    slots = _apply_ct_ct_direct(put_encoded(left, num_slots), put_encoded(right, num_slots), plan)
    return decode(slots, plan.result_dims)


def schedule_summary(plan):
    """Counts a Cerium lowering pays for one product."""
    multiplies = sum(len(items) for _, items in plan.chunks)
    return {
        "baby_steps": plan.num_baby_steps,
        "giant_steps": len(plan.chunks),
        "diagonal_multiplies": multiplies,
        "rotations_in_keygen": len(set(plan.rotations_needed)),
        "result_dims": plan.result_dims,
    }


# gcd is re-exported for callers that want the same coprime test the C++ uses.
def dimensions_coprime(dims):
    g = 0
    for dim in dims:
        g = gcd(g, int(dim))
    return g == 1
