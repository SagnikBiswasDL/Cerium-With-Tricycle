"""Slot-vector rotations and the doubling replication from ``replicate_ctxt``.

``left_rotate(v, k)`` matches OpenFHE ``EvalRotate`` / ``EvalFastRotation`` and
``rotate_plaintext``: index ``i`` receives what was at index ``i+k``.
"""

import numpy as np


def left_rotate(values, steps):
    values = np.asarray(values)
    steps = int(steps) % values.shape[0]
    if steps == 0:
        return values.copy()
    return np.concatenate((values[steps:], values[:steps]))


def _bits_lsb_first(number):
    bits = []
    while number > 0:
        bits.append(number & 1)
        number >>= 1
    return bits


def replicate(values, num_elements, num_replications):
    """Tile the first ``num_elements`` slots, ``num_replications`` times.

    This is ``replicate_ctxt``: a doubling ladder on left rotations by
    ``-num_elements``, ``-2 num_elements``, and so on. ``num_replications == 1``
    returns the vector unchanged.
    """
    values = np.asarray(values)
    num_replications = int(num_replications)
    num_elements = int(num_elements)
    if num_replications < 1:
        raise ValueError("num_replications must be positive")
    if num_replications == 1:
        return values.copy()
    num_slots = values.shape[0]
    bits = _bits_lsb_first(num_replications)
    last = values.copy()
    rotation_number = -1
    total = None
    if bits[0] == 1:
        total = values.copy()
    for i in range(1, len(bits)):
        rotation_size = (rotation_number * num_elements) % num_slots
        last = last + left_rotate(last, rotation_size)
        rotation_number *= 2
        if bits[i] == 1:
            if total is None:
                total = last.copy()
            else:
                shift = (rotation_number * num_elements) % num_slots
                total = left_rotate(total, shift) + last
    return total
