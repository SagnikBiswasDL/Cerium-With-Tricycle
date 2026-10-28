# SPDX-License-Identifier: Apache-2.0

"""Encrypt x and y, encode weights a and b, and verify x * a + y * b."""

from examples.common import run_cli, run_program, sample_vector


def main(gpus):
    x, y = sample_vector(1), sample_vector(2)
    a, b = sample_vector(3) + 1, sample_vector(4) + 1
    scale = 1 << 28
    run_program("weighted_sum", gpus,
                {"x": (x, scale), "y": (y, scale), "a": (a, scale), "b": (b, scale)},
                x * a + y * b, scale * scale)


if __name__ == "__main__":
    run_cli(main, __doc__)
