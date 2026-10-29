# SPDX-License-Identifier: Apache-2.0

"""Slot layout and plaintext reference for BERT's attention matmul.

The kernel multiplies two attention heads at once: context_h = P_h @ V_h with
P_h 128x128 (softmax probabilities) and V_h 128x64, for h in {0, 1}. Row r of
the layout owns slots 256*r .. 256*r + 255; head h uses the 128 slots starting
at 256*r + 128*h.

  probs[256*r + 128*h + c] = P_h[r, (r + c) % 128]   (diagonal-shifted rows)
  value[256*r + 128*h + j] = V_h[r, j]                 (j < 64, rest zero)
  output uses the same layout as value.
"""

import numpy as np

SLOTS = 32 * 1024
ROWS = 128
HEAD_COLS = 64
HEADS = 2
_r = np.arange(ROWS)[:, None]
_c = np.arange(ROWS)[None, :]
_j = np.arange(HEAD_COLS)[None, :]


def pack_probs(probs):
    out = np.zeros(SLOTS)
    for h in range(HEADS):
        out[256 * _r + 128 * h + _c] = probs[h][_r, (_r + _c) % ROWS]
    return out


def pack_values(values):
    out = np.zeros(SLOTS)
    for h in range(HEADS):
        out[256 * _r + 128 * h + _j] = values[h]
    return out


def unpack_output(packed):
    return np.stack([packed[256 * _r + 128 * h + _j] for h in range(HEADS)])


def masks():
    """Plaintext masks named as in bert_params_torch.attention_masks."""
    lane = np.arange(SLOTS) % ROWS
    out = {}
    for shift in range(ROWS):
        out[f"attention_matmul_mask_128x128_rot_{shift * 128}"] = (lane == shift).astype(float)
        out[f"attention_matmul_maskL_128x128_{shift}"] = (lane < ROWS - shift).astype(float)
        out[f"attention_matmul_maskR_128x128_{shift}"] = (lane >= ROWS - shift).astype(float)
    return out


def reference(probs, value):
    """NumPy port of bert_torch_packed.packed_attention_product."""
    rotate = lambda vector, steps: np.roll(vector, -steps)
    lane = np.arange(SLOTS) % ROWS
    value_baby_steps = [rotate(value, 256 * baby) for baby in range(8)]
    diagonal = np.zeros(SLOTS)
    for giant in range(16):
        diagonal_baby = np.zeros(SLOTS)
        for baby, rotated in enumerate(value_baby_steps):
            diagonal_baby += rotated * (lane == baby + 8 * giant)
        diagonal += rotate(diagonal_baby, 2048 * giant) if giant else diagonal_baby

    result = np.zeros(SLOTS)
    for giant in range(16):
        probs_giant = rotate(probs, 8 * giant - 2048 * giant)
        baby_sum = np.zeros(SLOTS)
        for baby in range(8):
            left_mask = (lane < ROWS - (baby + 8 * giant)).astype(float)
            aligned = rotate(probs_giant, baby) * left_mask
            aligned += rotate(probs_giant, -128 + baby) * (1 - left_mask)
            baby_sum += rotate(diagonal, 256 * baby) * aligned
        result += rotate(baby_sum, 2048 * giant) if giant else baby_sum
    return result
