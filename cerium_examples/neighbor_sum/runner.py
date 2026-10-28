# SPDX-License-Identifier: Apache-2.0

"""Verify that every slot holds the sum of itself and the next slot."""

import numpy as np
from examples.common import run_cli, run_program, sample_vector


def main(gpus):
    x = sample_vector(1)
    scale = 1 << 84
    run_program("neighbor_sum", gpus, {"x": (x, scale)}, x + np.roll(x, -1), scale)


if __name__ == "__main__":
    run_cli(main, __doc__)
