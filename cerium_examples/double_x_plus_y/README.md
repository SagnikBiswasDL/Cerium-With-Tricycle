# Reuse an encrypted input: z = x + x + y

A small variation on [addition](../addition/README.md). The DSL declares ciphertexts `x` and `y` at scale 2^56 and outputs `z = x + x + y`. The runner supplies `x` once and checks the decrypted output against `2 * x + y`.

## Run

From the `examples/` directory:

```bash
bash compile.sh double_x_plus_y 1
bash run_example.sh double_x_plus_y 1
```

Both additions preserve scale and level, so the program needs no evaluation keys and the output is decoded at 2^56, as in addition.
