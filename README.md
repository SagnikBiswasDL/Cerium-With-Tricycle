# Cerium-With-Tricycle
Integrating CMU's Cerium runtime with Tricycle.

`tricycle/` is a plaintext port of Lawrence's two trees, split the way the inference work is split:

- `tricycle/matmul.py` — tricyclic encoding and the ciphertext-plaintext, plaintext-ciphertext, and ciphertext-ciphertext products from [tricyclic-matmul](https://bitbucket.org/tricyclepaper/tricyclic-matmul/src/main/) (commit `6f9fa57`), including the BSGS schedule, plus the HEaaN right/left-BSGS ciphertext-ciphertext schedules and output masks from [tricycle](https://bitbucket.org/tricyclepaper/tricycle/src/main/).
- `tricycle/nonlinear.py` — softmax, GELU, and LayerNorm from [tricycle](https://bitbucket.org/tricyclepaper/tricycle/src/main/) (commit `83c4294`), exact and as the Chebyshev / power-of-two polynomials the encrypted layers evaluate.
- `tricycle/encoder.py` — one encoder block that calls both.
- `tricycle/kernels.py` — every matrix product of a Tricycle encoder layer as a CKKS program over block grids, written once and run on three backends: NumPy slot vectors, exact scale/level bookkeeping with Cerium's primes, and the Cerium DSL (`tricycle/cerium_backend.py`).

`python3 -m tricycle.test_tricycle` and `python3 -m tricycle.test_kernels` check everything against NumPy; neither needs HEaaN, OpenFHE, or Cerium.

## Tricycle's matmuls on Cerium

The seven products of `evaluate_layer` in Tricycle's `transformer.cpp`:

| Name | Product | Kind |
| --- | --- | --- |
| `qkv` | `X (1,T,D) @ [Wq\|Wk\|Wv] (H,D,K)` | ciphertext-plaintext, batch broadcast |
| `reshape` | `pad (1,T+2,T) @ [K\|V] (H,T,K)`; the padded K's slots are K^T | plaintext-ciphertext |
| `qk` | `Q (H,T,K) @ K^T (H,K,T+2)`, mask `1/sqrt(d_k)` | ciphertext-ciphertext, right BSGS |
| `pv` | `softmax (H,T,T+2) @ V (H,T+2,K)`, mask 1 | ciphertext-ciphertext, left BSGS |
| `attn_out` | `(H,T,K) @ Wo (H,K,D)` summed over heads | ciphertext-plaintext, concatenated |
| `ff1` | `(1,T,D) @ W1 (1,D,F)` | ciphertext-plaintext |
| `ff2` | `(1,T,F) @ W2 (1,F,D)` summed over FF blocks | ciphertext-plaintext |

`cerium_examples/tricycle_matmul/` compiles them into one Cerium program; `tricycle.sh` runs Cerium's Docker image with this whole repository mounted:

```bash
# Build the DSL graph only (checks scales and levels), then compile to CUDA.
bash cerium_examples/tricycle_matmul/tricycle.sh emit
bash cerium_examples/tricycle_matmul/tricycle.sh compile
# On a machine with an NVIDIA GPU: encrypt, run, decrypt, compare every slot.
bash cerium_examples/tricycle_matmul/tricycle.sh run
```

`TRICYCLE_SHAPE` picks the layer shape and `TRICYCLE_MATMULS` a subset (for example `qk,pv`). Compiler settings that worked here:

| Shape | Per block | Compile settings |
| --- | --- | --- |
| `mini` (default) | 5 tokens, d_model 7, 3 heads × 4, d_ff 11; 2 row blocks, 2 FF blocks | defaults (8192 vregs, 256 BCUs) |
| `tiny` (BERT-Tiny) | 89 tokens, 131, 2 × 33, 131; 2 row blocks, 2 FF blocks | `CERIUM_VREGS=16384 CERIUM_BCUS=1024` |
| `base` (BERT-Base) | 33 tokens, 769, 13 × 34, 769; 4 row blocks, 2 FF blocks | `CERIUM_VREGS=32768 CERIUM_BCUS=2048 MAKE_JOBS=3` |

`MAKE_JOBS` caps parallel nvcc jobs; without it BERT-Base's CUDA build is killed for memory in an 8 GB Docker VM.

Ciphertexts are encrypted at 2^84. Cerium's key-switching noise is about 2^52, so at 2^56 every rotation costs about 0.06 of error; the first GPU run at 2^56 was off by about 1e-2.

### Results on one NVIDIA H100 80 GB

Random inputs, one Cerium program per row, each decrypted output compared slot by slot with the NumPy execution of the same program. GPU time is one `program.run()`; key generation and encryption are excluded. Logs: [`familiarization/logs/pod/tricycle/`](familiarization/logs/pod/tricycle/).

| Shape | Products | Plaintexts | vregs | GPU time | Max abs error | Outputs |
| --- | --- | --- | --- | --- | --- | --- |
| mini | all 7 | 78 | 8192 | 10.9 ms | 1.6e-6 | 24 / 24 match |
| BERT-Tiny | all 7 | 1,084 | 16384 | 210.6 ms | 6.6e-6 | 24 / 24 match |
| BERT-Base | reshape, Q·Kᵀ, softmax·V, attention output | 812 | 24576 | 326.6 ms | 1.4e-5 | 32 / 32 match |
| BERT-Base | QKV | 3,630 | 24576 | 867.7 ms | 1.5e-5 | 12 / 12 match |
| BERT-Base | FF1 | 3,074 | 16384 | 311.4 ms | 1.6e-5 | 8 / 8 match |
| BERT-Base | FF2 | 3,074 | 16384 | 311.3 ms | 2.5e-5 | 4 / 4 match |

The BERT-Base rows use Tricycle's baby-step/giant-step weights from `transformer.cpp` and start the attention output, FF1 and FF2 at level 5 (`TRICYCLE_LEVELS=attn_out=5,ff1=5,ff2=5`), the level Cerium's BERT-Base feeds its own versions of those matmuls; QKV, the reshape and both ciphertext-ciphertext products start at level 12. The first run, with equal weights and every product at level 12, took 426.3, 890.8, 739.6 and 748.6 ms (2.81 s; logs in `familiarization/logs/pod/tricycle/`, the current ones in `v2_levels/`).

All seven BERT-Base products in one program compile (32768 vregs) but run out of device memory when the inputs are copied: the register file takes about 66 GB and the weight diagonals about 32 GB. `run_base_split.sh` runs the four programs above instead (1.82 s in total).

### Baseline: Cerium's own BERT-Base

Cerium's `benchmarks/bert_base` (encrypted BERT-Base fine-tuned on RTE: all 12 layers with its own packing, polynomial softmax/GELU/LayerNorm and bootstrapping, plus the classifier) on the same H100, one GPU, run with `familiarization/pod/run_bert_baseline.sh`:

| Samples | Time per sample | Accuracy | Compile (CUDA build) |
| --- | --- | --- | --- |
| 5 (the shipped `inputs/`) | 37.94 s (37.92–37.97) | 4 / 5 | 504 s |

That is about 3.2 s per layer for everything. `familiarization/pod/profile_bert.sh` splits one sample's GPU time by Cerium function with Nsight Systems (Cerium names every kernel `Cerium::<function>::...`): bootstrapping is 51%, and the matrix multiplications compare with the Tricycle ones like this, per layer:

| Product | Cerium BERT-Base | Tricycle on Cerium | |
| --- | --- | --- | --- |
| Q·Kᵀ + softmax·V + attention output | 637 ms | 327 ms (also includes the K/V reshape) | Tricycle 1.9× faster |
| QKV | 229 ms | 868 ms | Tricycle 3.8× slower |
| FF1 | 128 ms | 311 ms | Tricycle 2.4× slower |
| FF2 | 131 ms | 311 ms | Tricycle 2.4× slower |
| All matrix multiplications | 1,125 ms | 1,817 ms | Tricycle 1.6× slower |

Cerium's numbers are kernel time summed over the 12 layers and divided by 12 (`familiarization/logs/pod/bert_baseline/`). Cerium's plaintext-weight matmuls use 256-periodic compressed plaintexts and its fused baby-step giant-step operation; the Tricycle diagonals are full-length plaintexts multiplied one at a time, which is where the remaining gap is.

On a Runpod Pod (Ubuntu 24.04, x86_64), `cerium_examples/tricycle_matmul/bootstrap_pod.sh` builds Cerium at `6b48704` with `familiarization/pod/setup_pod.sh` and runs `run_pod.sh` for each shape; `run_base_split.sh` does BERT-Base.
