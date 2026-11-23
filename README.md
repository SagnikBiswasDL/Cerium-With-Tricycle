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
| `base` (BERT-Base) | 33 tokens, 769, 13 × 34, 769; 4 row blocks, 2 FF blocks | `CERIUM_VREGS=16384 CERIUM_BCUS=128 MAKE_JOBS=3`, one split program at a time (`run_base_split.sh`) |

`MAKE_JOBS` caps parallel nvcc jobs; without it BERT-Base's CUDA build is killed for memory in an 8 GB Docker VM.

`program.py` sets `CERIUM_BLOCK_SIZE=32` (Cerium's horizontal-fusion block size) unless it is already set. That is what Cerium's own BERT-Base uses for its weight matmuls, and it matters a lot here: with the default 1024, the plaintext multiply-accumulate kernels run about 10× slower (see below).

Ciphertexts are encrypted at 2^84. Cerium's key-switching noise is about 2^52, so at 2^56 every rotation costs about 0.06 of error; the first GPU run at 2^56 was off by about 1e-2.

### Results on one NVIDIA H100 80 GB

Random inputs, one Cerium program per row, each decrypted output compared slot by slot with the NumPy execution of the same program. GPU time is `program.run()` (it returns after `cudaDeviceSynchronize`): the median of 10 runs after 3 warmup runs; the first, cold run is within 1 ms of it in every row. Key generation and encryption are excluded. Logs: [`familiarization/logs/pod/tricycle/v3_blocksize/`](familiarization/logs/pod/tricycle/v3_blocksize/).

| Shape | Products | Plaintexts | vregs | GPU time | Max abs error | Outputs |
| --- | --- | --- | --- | --- | --- | --- |
| mini | all 7 | 78 | 8192 | 10.8 ms (block size 1024) | 1.3e-6 | 24 / 24 match |
| BERT-Tiny | all 7 | 1,084 | 16384 | 210.6 ms (block size 1024) | 6.6e-6 | 24 / 24 match |
| BERT-Base | reshape, Q·Kᵀ, softmax·V, attention output | 813 | 8192 | 52.7 ms | 1.2e-5 | 32 / 32 match |
| BERT-Base | QKV | 3,630 | 16384 | 97.6 ms (repeat: 97.4) | 1.5e-5 | 12 / 12 match |
| BERT-Base | FF1 | 3,074 | 8192 | 31.7 ms | 1.6e-5 | 8 / 8 match |
| BERT-Base | FF2 | 3,074 | 8192 | 21.6 ms | 2.4e-5 | 4 / 4 match |

The BERT-Base rows use Tricycle's baby-step/giant-step weights from `transformer.cpp` and start the attention output, FF1 and FF2 at level 5 (`TRICYCLE_LEVELS=attn_out=5,ff1=5,ff2=5`), the level Cerium's BERT-Base feeds its own versions of those matmuls; QKV, the reshape and both ciphertext-ciphertext products start at level 12. They take 204 ms in total per layer, all with `CERIUM_BLOCK_SIZE=32` and 128 BCUs.

How it got there, BERT-Base total per layer:

| Version | Total | QKV | FF1 | FF2 | Attention group | Logs |
| --- | --- | --- | --- | --- | --- | --- |
| Equal BSGS weights, every product at level 12, block size 1024 | 2.81 s | 890.8 ms | 739.6 ms | 748.6 ms | 426.3 ms | `tricycle/` |
| Tricycle's BSGS weights, attn_out/FF at level 5, block size 1024 | 1.82 s | 867.7 ms | 311.4 ms | 311.3 ms | 326.6 ms | `tricycle/v2_levels/` |
| Same, block size 32 | 0.20 s | 97.6 ms | 31.7 ms | 21.6 ms | 52.7 ms | `tricycle/v3_blocksize/bs32/` |

Why block size matters: with block size 1024, Nsight Systems puts 923 of QKV's 951 ms of kernel time in Cerium's fused element-wise kernels (the ciphertext-plaintext multiply-accumulates), and only 28 ms in key switching (NTT, base conversion, ModDown). Those kernels each take about 29 ms and reach about 3% of the H100's memory bandwidth. With block size 32 the element-wise time drops to 90 ms. FF1 shows the same pattern (398 → 31 ms element-wise). The block size is a Cerium compiler setting, not a change to the Tricycle algorithm. The 1024 runs are in `v3_blocksize/bs1024_default/`.

All seven BERT-Base products in one program compiled with block size 1024 and 32768 vregs, but ran out of device memory when the inputs were copied: the register file took about 66 GB and the weight diagonals about 32 GB. With block size 32 the split programs need 8192–16384 vregs; one program with all seven has not been tried again.

`TRICYCLE_FUSED` (for example `qkv,ff1,ff2` or `all`) emits the plaintext products with Cerium's `BsgsMultiplyAccumulate` over the row blocks instead, so each weight diagonal multiplies all row blocks in one operation. It matches NumPy on every shape (`python3 -m tricycle.test_kernels`) and on the GPU for `mini`, but at BERT-Base it has not compiled yet: "Called Evict Reg" at 16384 vregs and "Insufficient Base Conversion Units" at 128 BCUs. With block size 32 it is no longer needed to beat Cerium, so it has not been pursued further.

### Baseline: Cerium's own BERT-Base

Cerium's `benchmarks/bert_base` (encrypted BERT-Base fine-tuned on RTE: all 12 layers with its own packing, polynomial softmax/GELU/LayerNorm and bootstrapping, plus the classifier) on the same H100, one GPU, run with `familiarization/pod/run_bert_baseline.sh`:

| Samples | Time per sample | Accuracy | Compile (CUDA build) |
| --- | --- | --- | --- |
| 5 (the shipped `inputs/`) | 37.94 s (37.92–37.97) | 4 / 5 | 504 s |

That is about 3.2 s per layer for everything. `familiarization/pod/profile_bert.sh` splits one sample's GPU time by Cerium function with Nsight Systems (Cerium names every kernel `Cerium::<function>::...`): bootstrapping is 51%, and the matrix multiplications compare with the Tricycle ones like this, per layer:

| Product | Cerium BERT-Base | Tricycle on Cerium, wall time | Tricycle on Cerium, kernel time | Tricycle vs Cerium (kernel time) |
| --- | --- | --- | --- | --- |
| Q·Kᵀ + softmax·V + attention output | 637 ms | 52.7 ms | 77.3 ms (also includes the K/V reshape) | 8.2× faster |
| QKV | 229 ms | 97.6 ms | 105.9 ms | 2.2× faster |
| FF1 | 128 ms | 31.7 ms | 37.3 ms | 3.4× faster |
| FF2 | 131 ms | 21.6 ms | 24.9 ms | 5.3× faster |
| All matrix multiplications | 1,125 ms | 203.6 ms | 245.4 ms | 4.6× faster |

Cerium's numbers are kernel time summed over the 12 layers and divided by 12 (`familiarization/logs/pod/bert_baseline/`). The Tricycle kernel time is measured the same way (Nsight Systems, `--cuda-graph-trace=node`, per launch); it is higher than wall time because profiling each kernel inside the CUDA graph adds overhead. Comparing kernel time with kernel time is the fair comparison.

Caveats:

- Different packing, so not identical work: Cerium's weight matmuls use 256-periodic compressed plaintexts, which Tricycle's encoding cannot use (its slot vectors are built from pairwise-coprime dimensions and never repeat with a power-of-two period). Cerium's numbers also include the bias additions.
- About 128 tokens on both sides (Tricycle: 4 blocks of 33 = 132).
- Levels match for the attention output and FF products (5). Cerium's QKV starts at level 13, ours at 12.
- Both now use block size 32 for these products. Cerium's other settings for them are 8192 vregs and 128 BCUs; ours are the same except 16384 vregs for QKV.
- These are matrix multiplications only. Bootstrapping is 51% of Cerium's BERT time, and the Tricycle layer does not yet include nonlinear operations or bootstrapping.

### Toward a full layer: how many bootstraps Tricycle needs

Read from the source of `refs/tricycle-sst2-checkpoint` (128-token configuration from its README; not run, since it needs the HEaaN SDK). Every bootstrap is a full-slot HEaaN `Bootstrapper::bootstrap`, mostly complex-packed (two real ciphertexts per bootstrap), and the code's own counter (`GetBootstrapInvocationCount`, incremented before each call) would report the same numbers:

| Layers | Softmax normalization | Wide GELU features | Bootstraps per layer |
| --- | --- | --- | --- |
| 1, 2, 12 | direct reciprocal | no | 16 |
| 3 | reciprocal, then two inverse square roots | no | 20 |
| 4–9 | two inverse square roots | no | 18 |
| 10 | direct reciprocal | yes | 22 |
| 11 | two inverse square roots | yes | 24 |

That is 222 for the 12 layers, 18.5 per layer on average. Per layer: 4 for the softmax exponential (16 score ciphertexts, 4 per bootstrap), 2 to 6 for softmax normalization, 2 after softmax·V, 1 + 2 for each of the two LayerNorms (variance, then output), and 2 after FF2, or 8 after GELU instead in layers 10 and 11.

Cerium spends 1.75 s per layer bootstrapping in its BERT (kernel time, same profile as above). One Cerium bootstrap took 17.4 ms on the October H200 run (`benchmarks/results/2026-10-29_h200/`). 18.5 × 17.4 ms ≈ 0.32 s per layer is only a rough first estimate, not a measurement:

- The GPUs differ (H200 against H100). The cost per bootstrap still has to be measured on the H100.
- Levels may not line up. Tricycle assumes 12 levels after each bootstrap under HEaaN's FGb parameters, while Cerium's 32K bootstrap returns 18 limbs at scale 2^56. If that leaves fewer usable levels, the nonlinear functions need more bootstraps.

On a Runpod Pod (Ubuntu 24.04, x86_64), `cerium_examples/tricycle_matmul/bootstrap_pod.sh` builds Cerium at `6b48704` with `familiarization/pod/setup_pod.sh` and runs `run_pod.sh` for each shape; `run_base_split.sh` does BERT-Base.
