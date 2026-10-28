# New Cerium examples

Six small programs written while learning Cerium ([CMU-CAOS/Cerium](https://github.com/CMU-CAOS/Cerium) at `6b48704`). Each folder follows the layout of Cerium's own `examples/addition/`: `program.py` defines the encrypted computation in the DSL, and `runner.py` supplies inputs, decrypts output `z`, and checks it against NumPy.

| Example | What it computes |
| --- | --- |
| `double_x_plus_y` | `x + x + y` |
| `weighted_sum` | `x * a + y * b` with plaintext weights |
| `neighbor_sum` | `x + (x << 1)` |
| `square` | `relinearize(x * x)` |
| `bicyclic_matmul` | encrypted `A (3×4) @ B (4×5)` with bicyclic packing (see `encoding.py`) |
| `diagonal_matvec` | plaintext `W (16×16) @` encrypted `x` with the diagonal method |

They import `examples.common`, so copy them into a Cerium checkout before running:

```bash
git clone --recurse-submodules https://github.com/CMU-CAOS/Cerium.git
cp -r cerium_examples/* Cerium/examples/
```

Then follow Cerium's setup and, from `Cerium/examples/`, run `bash compile.sh <name> 1 && bash run_example.sh <name> 1`. On a RunPod pod without Docker, use `familiarization/pod/setup_pod.sh` and `run_pod.sh <name> 1` instead.

Results and the full write-up: [`familiarization/NOTES.md`](../familiarization/NOTES.md). GPU logs: [`familiarization/logs/pod/`](../familiarization/logs/pod/).
