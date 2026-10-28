# SPDX-License-Identifier: Apache-2.0

"""Encrypt two vectors and verify the decrypted value of x + x + y."""

from examples.common import run_cli, run_program, sample_vector


def main(gpus):
    x, y = sample_vector(1), sample_vector(2)
    scale = 1 << 56
    run_program("double_x_plus_y", gpus, {"x": (x, scale), "y": (y, scale)}, 2 * x + y, scale)


if __name__ == "__main__":
    run_cli(main, __doc__)
