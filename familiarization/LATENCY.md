# Matrix multiplication latency in Cerium (1× NVIDIA H200)

Measured Wednesday Oct 28 2026 on RunPod pod `rouc2yjzcf491z`: NVIDIA H200, driver 595.91.07, CUDA 13.3 toolkit. Every number comes from a run whose decrypted output passed the correctness check against NumPy (24/24 passed). Raw data: [`logs/pod/latency/latency_results.jsonl`](logs/pod/latency/latency_results.jsonl).

## Headline

| Operation | Shape | Rotations | GPU latency per multiply | Max abs error |
| --- | --- | --- | --- | --- |
| Bicyclic, encrypted × encrypted, linear schedule | 3×4 @ 4×5 | 3 | 0.27 ms | 3.0e-13 |
| | 7×8 @ 8×9 | 7 | 0.51 ms | 3.9e-13 |
| | 15×16 @ 16×17 | 15 | 1.01 ms | 5.9e-13 |
| | 31×32 @ 32×33 | 31 | 2.03 ms | 7.3e-13 |
| Bicyclic, encrypted × encrypted, doubling schedule | 3×4 @ 4×5 | 2 | 0.25 ms | 3.0e-13 |
| | 7×8 @ 8×9 | 3 | 0.33 ms | 3.9e-13 |
| | 15×16 @ 16×17 | 4 | 0.41 ms | 5.9e-13 |
| | **31×32 @ 32×33** | 5 | **0.49 ms** | 7.3e-13 |
| Diagonal method, plaintext matrix × encrypted vector | 16×16 | 15 | 0.93 ms | 1.3e-07 |
| | 32×32 | 31 | 1.95 ms | 3.0e-07 |
| | 64×64 | 63 | 5.79 ms | 5.4e-07 |
| | 128×128 | 127 | 13.14 ms | 1.5e-06 |

"GPU latency per multiply" is the steady-state time: one `program.run()` replays the compiled CUDA graph 100 times back to back (`CERIUM_RUNTIME_NUM_ITERS=100`); the table shows the median wall time of 5 such calls divided by 100. A single isolated call (median of 50, one graph launch each) adds roughly 0.07–0.10 ms of host overhead; for example, 31×32 @ 32×33 doubling takes 0.56 ms per isolated call.

## What the numbers show

- **Rotations dominate.** For the linear bicyclic schedule, latency grows linearly with the number of rotations, about 0.063 ms per rotation (least-squares fit over the four shapes), on top of about 0.07 ms for the ciphertext × ciphertext multiply and relinearization. The diagonal method also follows its rotation count (15 → 127).
- **Fewer rotations means proportionally less time.** When the shared dimension N is a power of two, the N − 1 rotations that sum partial products can be replaced by log₂ N rotate-and-add steps. At 31×32 @ 32×33 this cuts 31 rotations to 5 and latency from 2.03 ms to 0.49 ms (4.2× faster), with identical output.
- **One ciphertext holds the whole product at the largest shape.** 31×32 @ 32×33 uses 31·32·33 = 32,736 of the 32,768 slots: 32,736 encrypted multiply-adds and a 1,023-entry result in about 0.5 ms.
- **Error stays tiny** (below 1e-12 for bicyclic). The diagonal method's larger error comes from encoding the plaintext matrix at a 28-bit scale.
- The two methods are **not a like-for-like comparison**: bicyclic multiplies two encrypted matrices, while the diagonal method multiplies a plaintext matrix by an encrypted vector. The diagonal method is included because it is the standard approach (and what Cerium's BERT benchmark builds on).

## Comparison with Cerium's own BERT attention matmul

Cerium's BERT benchmark has a hand-tuned encrypted × encrypted kernel, `matmul_ct128x128_ct128x128` in `benchmarks/bert_base/bert_cerium.py`. It computes the attention context P_h @ V_h for two heads at once (P_h is 128×128, V_h is 128×64), from two ciphertexts. I copied its body unchanged into a standalone program, [`bert_attention_matmul`](../cerium_examples/bert_attention_matmul/), with the input and mask scales the BERT runner uses. I built it with BERT's compiler settings for this kernel (3,096 registers, 128 base conversion units, block size 256). Its decrypted output matches NumPy's P_h @ V_h, with a max abs error of 1.2e-07 on values around 0.1. Raw data: [`latency_results_bert.jsonl`](logs/pod/latency/latency_results_bert.jsonl).

| Kernel | Work | GPU latency | Encrypted multiply-adds per ms |
| --- | --- | --- | --- |
| BERT attention matmul (Cerium's) | 2 × (128×128 @ 128×64) = 2,097,152 multiply-adds | **12.72 ms** (12.77 ms isolated call) | ≈165,000 |
| Bicyclic, doubling (ours) | 31×32 @ 32×33 = 32,736 | 0.49 ms | ≈67,000 |
| Bicyclic, linear (ours) | 31×32 @ 32×33 = 32,736 | 2.03 ms | ≈16,000 |
| Diagonal (ours, plaintext matrix) | 128×128 @ 128 = 16,384 | 13.14 ms | ≈1,200 |

- **BERT's kernel is about 2.5× more throughput than our best bicyclic run.** It also handles far larger matrices, since bicyclic packing is limited to M·N·P ≤ 32,768 slots in one ciphertext. Its speed comes from baby-step giant-step (BSGS) rotations, hoisted rotations that share one key-switch decomposition across many shifts, and batching 8 ciphertexts into vector operations.
- **For tiny products, bicyclic still has the lowest latency** (0.25–0.5 ms), because it needs only one ciphertext multiply and a handful of rotations.
- **Larger matrices with bicyclic would need block decomposition** across several ciphertexts. That is the natural next experiment if we want a like-for-like 128×128 comparison.
- One-time costs for the BERT kernel: compile 8.2 s, evaluation keys 0.5 s (about 0.5 GB on disk), load 1.2 s, encrypt about 80 ms, decrypt about 25 ms.

## Rotation strategies and batching (bicyclic, 31×32 @ 32×33)

A follow-up run on a second H200 pod (`1a7b2e145fjhab`) tested where BERT's advantage comes from. All 16 runs passed. Raw data: [`latency_results_extra.jsonl`](logs/pod/latency/latency_results_extra.jsonl).

| How the N partial products are summed | Rotations | 15×16 @ 16×17 | 31×32 @ 32×33 |
| --- | --- | --- | --- |
| Linear: one rotate-and-add at a time | N − 1 | 1.01 ms | 2.03 ms |
| Hoisted: `HoistedRotate`, sharing one key-switch decomposition | N − 1 | 0.77 ms | 1.52 ms |
| Fused: `RotateAccumulateMany`, one fused kernel | N − 1 | 0.46 ms | 0.82 ms |
| Doubling: log₂ N rotate-and-adds | log₂ N | 0.41 ms | **0.49 ms** |

- **Cerium's fused rotate-and-sum operation gives a 2.5× speedup with the same 31 rotations** (2.03 ms down to 0.82 ms). Hoisting alone gives 1.3×.
- **Doubling remains fastest for this shape,** because 5 rotations beat even well-optimized 31 rotations. When N is not a power of two, the fused operation is the best option.
- **`RotateAccumulate` (the single-operand form) crashes the compiler** with a segfault at this Cerium commit, with or without a zero shift. `RotateAccumulateMany` with the same ciphertext repeated works and is what BERT uses.

**Batching independent products in one program** (doubling schedule, 31×32 @ 32×33 each):

| Products per program | GPU latency | Per product | Encrypted multiply-adds per ms |
| --- | --- | --- | --- |
| 1 | 0.49 ms | 0.49 ms | ≈67,000 |
| 2 | 0.79 ms | 0.40 ms | ≈83,000 |
| 4 | 1.48 ms | 0.37 ms | ≈89,000 |
| 8 | 3.11 ms | 0.39 ms | ≈84,000 |
| 16 | 6.33 ms | 0.40 ms | ≈83,000 |

- **One product does not fill the H200.** Running 2–4 at once improves throughput by about 30%, and it plateaus by 4 at about 85,000–89,000 multiply-adds per ms.
- **BERT's kernel still does about 1.9× better (≈165,000 per ms).** So its advantage is mostly algorithmic (baby-step giant-step rotations, hoisting, and vectorized operations sharing work), not just better GPU utilization.
- **Client-side costs scale with the batch.** Encryption takes about 95 ms per pair of input ciphertexts, and decryption about 62 ms per output, both on the CPU. Evaluation keys are shared, so key generation stays at about 0.24 s.

## One-time and client-side costs (not GPU compute)

| Step | Time |
| --- | --- |
| Compile DSL to CUDA, then `nvcc` | 3.0–6.9 s |
| Generate evaluation keys | 0.3–2.4 s (grows with the number of distinct rotations) |
| Encode plaintexts | 0.26–0.31 s |
| Load compiled program and build CUDA graph | 0.9–3.8 s |
| Encrypt inputs (CPU) | ≈100 ms for 2 ciphertexts, ≈50 ms for 1 |
| Copy ciphertexts to GPU | ≈2–5 ms |
| Decrypt and decode the output (CPU) | 66–95 ms |

These steps run once per program or per input, and the encryption and decryption run on the CPU in Cerium's runtime. Compile time is from the sweep log (`latency_results.jsonl.compile`); the rest are from `bench_matmul.py`.

## Setup and caveats

- CKKS parameters are the examples' defaults: 32,768 slots, starting level 12, 28-bit RNS primes. Inputs are at scale 2^56; rotations run after the multiply, at 2^112.
- Compiled with 8,192 GPU registers and 256 base conversion units for every configuration. The tutorials' defaults (1,024 and 128) cannot compile the 31-rotation and 128×128 cases; Cerium's own benchmarks use 4,096–8,192 registers.
- Single GPU only; small integer matrix entries (0–4); one multiplication deep (no bootstrapping).
- Not yet compared with other libraries. Both `bicyclic_matmul` and `bert_attention_matmul` compile for 2 GPUs (one kernel library per GPU; see [`probe_2gpu.log`](logs/pod/latency/probe_2gpu.log)), but they have not been run on more than one GPU.

## Reproduce

On the pod (after `bash setup_pod.sh 90`):

```bash
cd /workspace/cerium_work
bash sweep_latency.sh          # all 12 configurations, about 4 minutes
SWEEP=bert bash sweep_latency.sh   # BERT attention matmul, about 1 minute
SWEEP=extra bash sweep_latency.sh  # rotation strategies and batching, about 4 minutes
```

The sweep builds into `/root/cerium_scratch` (container disk) because the BERT kernel's evaluation keys alone are about 0.5 GB and `/workspace` is near its quota.

To run one configuration by hand:

```bash
source env.sh && cd Cerium/examples
BICYCLIC_SHAPE=31,32,33 BICYCLIC_SCHEDULE=doubling ../venv_cerium/bin/python3 -m examples.bicyclic_matmul.program --vregs 8192 --bcus 256
bash make_example.sh bicyclic_matmul 1
BICYCLIC_SHAPE=31,32,33 BICYCLIC_SCHEDULE=doubling CERIUM_RUNTIME_NUM_ITERS=100 ../venv_cerium/bin/python3 ../../bench_matmul.py bicyclic_matmul --calls 5
```

Scripts: [`pod/bench_matmul.py`](pod/bench_matmul.py), [`pod/sweep_latency.sh`](pod/sweep_latency.sh). Examples: [`../cerium_examples/bicyclic_matmul/`](../cerium_examples/bicyclic_matmul/), [`../cerium_examples/diagonal_matvec/`](../cerium_examples/diagonal_matvec/), [`../cerium_examples/bert_attention_matmul/`](../cerium_examples/bert_attention_matmul/).
