# Cerium familiarization notes

Upstream: https://github.com/CMU-CAOS/Cerium at commit `6b48704` ("Cerium Initial Release v1.0.0"). There is no `AGENTS.md` in the repository.

## Summary

All 7 repository examples and 6 new programs compiled, ran on an NVIDIA H200, decrypted correctly, and passed the runner's check (`np.allclose`, `rtol=1e-5`, `atol=1e-4`). Single GPU. The errors below are from the Tuesday Oct 27 2026 session, in which all 13 were run together.

| Example | What it computes | Output scale / level | Max abs error |
| --- | --- | --- | --- |
| `addition` | `x + y` | 56 bits / 12 | 5.540e-14 |
| `multiplication` | `x * factor` (plaintext) | 56 / 12 | 4.526e-05 |
| `rotation` | `x << 1` | 84 / 12 | 2.244e-10 |
| `relinearization` | `relinearize(x * y)` | 112 / 12 | 2.152e-13 |
| `rescale` | `rescale(x * factor)` | 56 / 11 | 4.580e-06 |
| `periodic_plaintext_matrix` | repeated 2×2 matrix on pairs | 112 / 12 | 2.251e-08 |
| `function_calling` | entry function calls a helper | n/a | 7.372e-14 |
| **`double_x_plus_y`** (new) | `x + x + y` | 56 / 12 | 8.767e-14 |
| **`weighted_sum`** (new) | `x * a + y * b` (plaintext weights) | 56 / 12 | 6.117e-05 |
| **`neighbor_sum`** (new) | `x + (x << 1)` | 84 / 12 | 2.244e-10 |
| **`square`** (new) | `relinearize(x * x)` | 112 / 12 | 2.697e-13 |
| **`bicyclic_matmul`** (new) | encrypted `A (3×4) @ B (4×5)`, bicyclic packing | 112 / 12 | 2.981e-13 |
| **`diagonal_matvec`** (new) | plaintext `W (16×16) @` encrypted `x`, diagonal packing | 112 / 12 | 1.234e-07 |

Some examples' errors change slightly between sessions (for example, `multiplication` gave 4.460e-05 on Oct 26 and 4.526e-05 on Oct 27), so part of the encryption noise is not fixed by the runners' seeds.

Example: addition decrypted `[6, 3, 3, 5, 2, 4, 6, 4, ...]`, the same as NumPy `x + y`. Each decrypted value has an imaginary part around 1e-14, which is normal CKKS noise. GPU execution time is reported as under 1 ms. Each compile and run takes about 5–7 seconds.

What the errors show: programs that use only additions at a large scale are at about 1e-13. Plaintext multiplication at 28-bit scales is at about 1e-5, because the plaintext is encoded with less precision. Rescale drops a prime, giving about 5e-6.

## New mock programs

Each is its own folder under `Cerium/examples/`, laid out like `addition/` (`program.py`, `runner.py`, `__init__.py`). None of the original examples were modified.

- `double_x_plus_y`: reuses one input twice. The kernel computes temporary `t5 = x + x`, then `z = t5 + y`, in a single fused kernel; `x` is encrypted only once.
- `weighted_sum`: two ciphertext×plaintext products, each 28 + 28 = 56 bits, so they can be added without rescaling.
- `neighbor_sum`: rotation plus addition. The compiler adds a rotation (Galois) key to `program_inputs`, and the runner generates it automatically.
- `square`: ciphertext×ciphertext product, so it needs relinearization; the scale doubles to 112 bits.

## Matrix packing proof of concept

How a matrix is "linearized" into a ciphertext is not a Cerium feature. The runner packs the matrix into the 32,768-slot vector with NumPy (CKKS then encodes that vector as a polynomial), and `program.py` performs the matching rotations, masks and additions.

- `bicyclic_matmul` (encrypted × encrypted). Entry (i, j) of an r×c matrix with gcd(r, c) = 1 goes into every slot k with k % r == i and k % c == j (Chinese remainder theorem; see `bicyclic_matmul/encoding.py`). For C = A·B with M, N, P pairwise coprime (3, 4, 5), slot k of `a * b` holds `A[k%M, k%N] * B[k%N, k%P]`. Adding N − 1 left rotations by multiples of M·P (15, 30, 45) sums over j, so slot k holds `C[k%M, k%P]`, which is C in bicyclic layout. Cost: 1 ciphertext×ciphertext multiply, 1 relinearization, 3 rotations, and 1 level of scale growth (56 → 112 bits).
  - Decrypted slots `[36, 6, 13, 39, 21, 12, 11, 24]` are C00, C11, C22, C03, C14, C20, C01, C12 of `A @ B`.
  - The NumPy model (`encoding.simulate`) was checked to reproduce `A @ B` exactly on 80 random cases for 3×4·4×5, 5×7·7×9, 7×8·8×9 and 4×9·9×5. The runner also re-checks it before encrypting.
  - Rotations wrap around all 32,768 slots, not the M·N·P block. Because the packing is defined for every slot, only the last (N − 1)·M·P slots are affected; the result is read from slots 0 .. M·P − 1.
- `diagonal_matvec` (Halevi–Shoup diagonal method, as used by Cerium's BERT benchmark). x (length 16) is repeated across all slots; `diag_k[i] = W[i, (i+k) % 16]` is a `PeriodicPlaintextInput` with period 16; `y = Σ_k (x << k) * diag_k`. Cost: 16 ciphertext×plaintext multiplies and 15 rotations (15 rotation keys).
- Rotations are applied at 84 or 112 bits on purpose. The `rotation` example's error (2.2e-10 at 2^84) means key-switching noise is about 2^52 in absolute terms, which would give an error of about 0.06 at a 2^56 scale.

## Where it ran

RunPod pod `royal_white_vole-migration`: `ssh root@103.196.86.136 -p <port> -i ~/.ssh/id_ed25519`. The port changes on restart (53034, then 46005); copy the current one from the pod's Connect tab.

After a restart, this template does not install the account's SSH key on the pod. Re-add it from the web terminal: `mkdir -p ~/.ssh && echo "<contents of ~/.ssh/id_ed25519.pub>" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys`. Then run `bash /workspace/cerium_work/setup_pod.sh 90`, which only reinstalls CUDA 13.3 (about 5 minutes).

- NVIDIA H200 (compute capability 9.0, Hopper), driver 595.91.07 (CUDA 13.2), Ubuntu 24.04 x86_64, 192 cores.
- No Docker inside a pod, so the Dockerfile and build scripts were reproduced natively by `familiarization/pod/setup_pod.sh`:
  - CUDA toolkit 13.3.1 via apt, beside the pod's CUDA 12.8; `/usr/local/cuda` still points to 12.8.
  - NCCL 2.30.7+cuda13.3 extracted to `/workspace/cerium_work/deps/`; the system NCCL 2.25 is untouched.
  - SEAL 4.1.1 built into `/workspace/cerium_work/deps/`.
  - Cerium built with CUDA arch `90`, Python 3.12 only.
- The CUDA 13.3 toolkit ran fine on the CUDA 13.2 driver, because code is compiled for the exact GPU (`sm_90`).

## Exact commands

On the pod:

```bash
cd /workspace/cerium_work
bash setup_pod.sh 90             # rerun after every pod restart; finished steps are skipped
bash run_pod.sh addition 1       # compile + run any example under Cerium/examples/
```

`run_pod.sh` runs the same steps as `examples/compile.sh` + `examples/run_example.sh` without Docker:
`python -m examples.<ex>.program --gpus 1`, then `make_example.sh <ex> 1`, then `python -m examples.<ex>.runner --gpus 1`.

To add a new example: create `Cerium/examples/<name>/` with `program.py`, `runner.py` and an empty `__init__.py`, copy it to the pod **without** an `outputs/` folder, then `bash run_pod.sh <name> 1`.

## Deviations and gotchas

- **Python environment is minimal.** The pod's `/workspace` volume is almost full (about 51 GB, mostly other projects). The full `requirements.txt` (PyTorch plus about 5 GB of CUDA wheels) failed with `Disk quota exceeded`. The venv has only the pinned `numpy==2.2.6`, `psutil==7.2.2` and `-e build/cerium/python`, which is all the examples import. The Llama and BERT benchmarks would need the full list and more disk.
- **Do not copy `outputs/` between machines.** `make` reuses stale object files; an arm64 `.o` from the Mac broke one link with `file in wrong format`. Fixed by deleting `outputs/`.
- **RunPod wipes the container disk on stop.** apt packages (CUDA 13.3, g++-12) disappear; everything under `/workspace/cerium_work` persists. Rerun `setup_pod.sh`.
- **Only 1 GPU**, so the 2/4/8-GPU variants were not run.

## Earlier Mac attempt (kept for reference)

On the Apple M5 Mac, the Docker image, Cerium build, venv and example compilation all worked natively on arm64. `run_example.sh` failed at `docker run --gpus all` with `failed to discover GPU vendor from CDI: no known GPU vendor found` (exit 125), because there is no NVIDIA GPU. Commands, from `Cerium/`: `docker build -t cerium/build-cu13.3:v1 docker/`, `bash build_cerium.sh "80"`, `bash activate_cerium_venv.sh`, then `cd examples && bash compile.sh addition 1`.

## Files

- New examples: `Cerium/examples/{double_x_plus_y,weighted_sum,neighbor_sum,square,bicyclic_matmul,diagonal_matvec}/`. The copy committed to this repo is in `cerium_examples/`, because `Cerium/` is an untracked clone of upstream.
- Pod scripts: `familiarization/pod/setup_pod.sh`, `familiarization/pod/run_pod.sh` (copies at `/workspace/cerium_work/` on the pod)
- GPU logs: `familiarization/logs/pod/<example>_{compile,run}_1gpus.log`, `setup.log`
- Compiled outputs on the pod: `/workspace/cerium_work/Cerium/examples/<ex>/outputs/<ex>/compiled/main_1gpus/`
- Generated CUDA: `.../outputs/<ex>/main_1gpus/partition_0/kernels_0.cu`
