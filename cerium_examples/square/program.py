# SPDX-License-Identifier: Apache-2.0

"""Square an encrypted vector: z = relinearize(x * x)."""

from cerium.dsl import CeriumFunction, CeriumProgram, CiphertextInput, Output
from examples.common import LEVEL, RNS_BITS, compile_cli, main_name


def build_program(gpus):
    program = CeriumProgram("square", rns_bit_size=RNS_BITS, num_gpus=gpus)
    with program:
        with CeriumFunction(main_name(gpus), gpus, 0):
            x = CiphertextInput("x", scale=56, level=LEVEL)
            product = x * x
            print(f"x * x: scale={product.scale()} bits, level={product.level()}", flush=True)
            z = product.relinearize()
            print(f"after relinearize: scale={z.scale()} bits, level={z.level()}", flush=True)
            Output("z", z)
    return program


if __name__ == "__main__":
    compile_cli("square", build_program, __doc__)
