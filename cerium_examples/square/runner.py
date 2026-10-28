# SPDX-License-Identifier: Apache-2.0

"""Encrypt x and verify the decrypted, relinearized square."""

from examples.common import run_cli, run_program, sample_vector


def main(gpus):
    x = sample_vector(1)
    scale = 1 << 56
    run_program("square", gpus, {"x": (x, scale)}, x * x, scale * scale)


if __name__ == "__main__":
    run_cli(main, __doc__)
