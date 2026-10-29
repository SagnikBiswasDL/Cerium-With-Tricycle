# SPDX-License-Identifier: Apache-2.0

"""Run BERT's attention matmul on random softmax rows and check P_h @ V_h."""

import math

import numpy as np

from examples.bert_attention_matmul.packing import (HEAD_COLS, HEADS, ROWS, masks, pack_probs,
                                                    pack_values, reference, unpack_output)
from examples.common import run_cli, run_program
from examples.primes import Primes

SHAPE = f"{HEADS}x({ROWS}x{ROWS} @ {ROWS}x{HEAD_COLS})"

# Scales follow bert_runner.get_attention_io and bert_params_torch.attention_masks.
PROBS_SCALE = Primes[7] * Primes[6] * Primes[5]
VALUES_SCALE = Primes[8] * Primes[7] * Primes[6]
ROT_MASK_SCALE = Primes[4] * math.sqrt(Primes[3])
LR_MASK_SCALE = math.sqrt(Primes[3]) * Primes[2]
OUTPUT_SCALE = Primes[4] * Primes[3] * Primes[2]


def make_case(verbose=True):
    rng = np.random.default_rng(13)
    logits = rng.normal(size=(HEADS, ROWS, ROWS))
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    values = rng.uniform(-1, 1, size=(HEADS, ROWS, HEAD_COLS))

    packed_probs, packed_values = pack_probs(probs), pack_values(values)
    expected = reference(packed_probs, packed_values)
    assert np.allclose(unpack_output(expected), probs @ values)
    if verbose:
        print(f"{SHAPE}: context[0][0, :4] = {(probs @ values)[0, 0, :4]}")

    raw_inputs = {"probs": (packed_probs, PROBS_SCALE), "values": (packed_values, VALUES_SCALE)}
    for name, mask in masks().items():
        raw_inputs[name] = (mask, ROT_MASK_SCALE if "_rot_" in name else LR_MASK_SCALE)
    return raw_inputs, expected, OUTPUT_SCALE


def main(gpus):
    run_program("bert_attention_matmul", gpus, *make_case())


if __name__ == "__main__":
    run_cli(main, __doc__)
