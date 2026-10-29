# Cerium BMM, bootstrapping and Tricycle: report for Divy, Sohaib and Lawrence

All measurements: one NVIDIA H200 (RunPod, driver 595.91.07), Cerium commit
`6b48704`, CUDA 13.3 `nvcc` for the generated kernels, SEAL 4.1.1, NCCL 2.30.7,
Python 3.12.3, NumPy 2.2.6, Xeon Platinum 8568Y+. The only GPU process
recorded during each run was the benchmark itself.

The raw results are in `results/2026-10-29_h200/`.
`contended_do_not_use/` holds an earlier sweep run while another job shared the GPU. Its timings are about 2× slow and are not used here.
The `environment.nvcc` field in the raw records says "release 12.8". That is the pod's default `/usr/local/cuda`; the Makefiles build the kernels with `/usr/local/cuda-13.3/bin/nvcc`, and the field is now fixed in `env_info.py`.

| Request | Status |
| --- | --- |
| 1. BMM at realistic sizes | **Measured** |
| 2. Latency, op counts, attribution, environment | **Measured** |
| 3. Cerium bootstrapping | **Measured** (standalone 33-level variant) |
| 4. Tricycle SST-2 reference | **Inspected; not reproduced.** Blocker: proprietary HEaaN GPU library |
| 5. Hoisting / double hoisting | **Findings and plan;** baseline measured, nothing new implemented |

## 1. BMM-2 at realistic sizes

**Assumption:** "BMM-2" is our bicyclic ciphertext × ciphertext product
(`Cerium/examples/bicyclic_matmul`).

**Slot mapping.** Entry (i, j) of an r×c matrix with gcd(r, c) = 1 goes in every slot k with
k mod r = i and k mod c = j.
- For C = A·B with A M×N and B N×P, the program computes `relinearize(a*b)`, then the rotation sum over shifts t·M·P (t = 0..N−1).
- C[i, l] then lands in slot k < M·P with k mod M = i, k mod P = l.
- Constraints: M, N and P pairwise coprime, and M·N·P ≤ 32768 slots (no wrap-around).
- Our shapes near the limit: 31·32·33 = 32736, 15·128·17 = 32640, 17·64·29 = 31552, 127·16·15 = 30480.

**Useful slot occupancy** is (independent products in the ciphertext × M·N·P) / 32768.
- A single small product replicated across the ciphertext still counts as 1 product.
- Putting K independent small products in consecutive M·N·P blocks of one ciphertext is **slot batching**. It is not a larger matrix product; rows marked "batch K" below.

**Error metrics.** Computed on useful output entries only, against NumPy `A @ B` in float64:
- **max abs** = max |Ĉ − C|.
- **rel Frobenius** = ‖Ĉ − C‖_F / ‖C‖_F, over all products stacked.
- **max/max** = max |Ĉ − C| / max |C|.

Decrypted values are complex, and the imaginary part counts as error. Real inputs are uniform in [−1, 1]; the tiny case uses integers 0..4.

**Latency definitions.**
- **run median / p95:** one `program.run()` with one CUDA-graph launch. 200 timed calls after 10 warm-ups. The call ends after `cudaDeviceSynchronize`, so it includes host launch overhead.
- **steady/launch:** 100 graph launches per call, 20 timed calls after 2 warm-ups, divided by 100.

All inputs are 2^56-scale ciphertexts at 12 RNS limbs (about 331.8-bit Q). Outputs are at 12 limbs, scale 2^112, with no rescale.

| Shape (schedule) | Rotations | Occupancy | run median / p95 (ms) | steady/launch (ms) | max abs | rel Frobenius | max/max |
| --- | --: | --: | --: | --: | --: | --: | --: |
| 3×4 @ 4×5 (linear, tiny check, integers) | 3 | 0.0018 | 0.360 / 0.409 | 0.272 | 1.4e-13 | 3.8e-15 | 3.5e-15 |
| 31×32 @ 32×33 (doubling) | 5 | 0.9990 | 0.556 / 0.600 | 0.490 | 1.5e-13 | 2.9e-14 | 2.2e-14 |
| 31×32 @ 32×33 (linear) | 31 | 0.9990 | 2.095 / 2.126 | 2.059 | 1.5e-13 | 2.9e-14 | 2.2e-14 |
| 31×32 @ 32×33 (hoisted) | 31 | 0.9990 | 1.575 / 1.597 | 1.526 | 1.5e-13 | 2.9e-14 | 2.2e-14 |
| 31×32 @ 32×33 (rotacc) | 31 | 0.9990 | 0.882 / 0.919 | 0.824 | 1.5e-13 | 2.9e-14 | 2.2e-14 |
| 15×128 @ 128×17 (doubling) | 7 | 0.9961 | 0.715 / 0.771 | 0.651 | 2.7e-13 | 2.9e-14 | 1.8e-14 |
| 17×64 @ 64×29 (doubling) | 6 | 0.9629 | 0.629 / 0.660 | 0.570 | 1.9e-13 | 2.8e-14 | 2.1e-14 |
| 127×16 @ 16×15 (doubling) | 4 | 0.9302 | 0.472 / 0.526 | 0.410 | 1.1e-13 | 2.8e-14 | 2.0e-14 |
| 127×16 @ 16×15 (linear) | 15 | 0.9302 | 1.080 / 1.125 | 1.013 | 1.1e-13 | 2.8e-14 | 2.0e-14 |
| batch 546 × (3×4 @ 4×5) (doubling) | 2 | 0.9998 | 0.306 / 0.340 | 0.248 | 7.6e-14 | 2.9e-14 | 3.0e-14 |
| batch 65 × (7×8 @ 8×9) (doubling) | 3 | 0.9998 | 0.387 / 0.416 | 0.328 | 9.9e-14 | 2.9e-14 | 2.9e-14 |
| batch 8 × (15×16 @ 16×17) (doubling) | 4 | 0.9961 | 0.468 / 0.507 | 0.410 | 1.1e-13 | 2.9e-14 | 2.3e-14 |

All cases also match the NumPy model of the full slot vector (`passed`).

## 2. Timing phases, op counts, attribution, environment

**One-time and per-inference phases.** Example: 31×32 @ 32×33 linear; other shapes are similar. Host wall time, `time.perf_counter`:

| Phase | What it includes | ms |
| --- | --- | --: |
| DSL build | Python builds the operation graph | 0.4 |
| Cerium compile | graph passes and CUDA code generation | 88 (11–88 across BMM cases) |
| nvcc build | builds the generated kernels into `function_map_0.so` | 4199 (2240–4199) |
| keygen (secret) | context, secret key and ephemeral key, CPU | 527 |
| keygen (eval) | relinearization and rotation keys, serialized to disk, CPU | 402 |
| plaintext encode | none for BMM; the call itself only | 293 |
| load program | read keys, copy them to the GPU, capture the CUDA graph | 1068 |
| encrypt | encode and encrypt both inputs, CPU | 97 |
| host-to-device | input ciphertexts, blocking `cudaMemcpy` | 31 (usually 2) |
| compute | `program.run()`, see table above | 2.1 |
| device-to-host | output ciphertext, ends with device synchronize | 0.9 |
| decrypt | decrypt and decode, CPU | 67 |

**Operation counts per product.** These are identical across BMM shapes except for rotations.
- Source level (operations written in the DSL), per product: 1 ct-ct multiply, 0 ct-pt multiplies, R rotations (the table's "Rotations" column), 1 relinearization, 0 rescales, 0 bootstraps.
- Compiled level (lowered kernel-group graph, `CERIUM_PRINT_GRAPHS=1`): key switches = (key-switch mod-down instructions) / 2, and automorphisms = (rotation instructions) / 2.
- For every schedule except `rotacc`, compiled key switches equal source relinearizations plus rotations (for example 32 = 1 + 31). `rotacc` compiles to **2** mod-downs instead of 32: its rotations are summed in the extended basis and divided down once.
- Launched kernels per run range from 1 (`add` ablation) to 115 (`linear`).

**Time attribution, measured two ways.** Neither is inferred from op counts.

Ablation (steady/launch):
- `add` (a + b only): 0.005 ms.
- `none` (a·b + relinearize): 0.085 ms.
- Rotation stage = full − `none`:

  | Schedule | Rotation stage | Rotations | Per rotation incl. add |
  | --- | --: | --: | --: |
  | linear | 1.974 ms | 31 | 0.064 ms |
  | doubling | 0.405 ms | 5 sequential | 0.081 ms |
  | hoisted | 1.441 ms | 31 | 0.046 ms |
  | rotacc | 0.739 ms | 31 | 0.024 ms |

- Caveat: the compiler fuses across op boundaries, so the subtraction is an attribution of the program difference, not of isolated ops.

Nsight Systems (`--cuda-graph-trace=node`, GPU kernel time per launch, 31×32 @ 32×33):

| Schedule | Total | ModDown (`sud`) | NTT | INTT | base conv | fused element-wise |
| --- | --: | --: | --: | --: | --: | --: |
| none | 0.102 | 0.018 | 0.012 | 0.024 | 0.016 | 0.032 |
| linear | 2.143 | 0.407 | 0.198 | 0.627 | 0.253 | 0.658 |
| hoisted | 1.553 | 0.406 | **0.023** | 0.439 | 0.184 | 0.501 |
| rotacc | 0.888 | **0.035** | 0.211 | 0.237 | 0.107 | 0.296 |
| doubling | 0.589 | 0.104 | 0.074 | 0.148 | 0.090 | 0.173 |

- These classes are primitives. Multiplication and rotation share the key-switch kernels (NTT, base conversion, ModDown), and the fused kernels mix tensor products, key products and automorphisms. So the profile cannot split "multiply time" from "rotation time" directly; the ablation does that.
- The profile does show where hoisting saves time: `hoisted` removes NTT time (one shared ModUp), and `rotacc` removes ModDown time (one ModDown).

**CKKS parameters.**
- Ring dimension 65536, 32768 slots.
- 27.6-bit RNS primes from `examples/primes.py`, a 64-prime table.
- BMM inputs at 12 limbs. Key-switching extension primes are indices 52–63 (read from the compiled base conversions).
- Dense secret key (Hamming weight 32768); ephemeral key with Hamming weight 32.

## 3. Cerium bootstrapping

**Program.** `Cerium/benchmarks/bootstrap_32K/bootstrap.py`, function
`bootstrap_32Kslots_33levels_1gpus`, unchanged. It runs:
1. key switch to the ephemeral key;
2. ModRaise to 51 limbs;
3. 4-stage BSGS CoeffToSlot;
4. homomorphic mod (Chebyshev degree 30, then 3 double-angle squarings);
5. 3-stage BSGS SlotToCoeff;
6. rescale.

Relation to the requested `bert_base/bootstrap.py`:
- `bert_base/bootstrap.py` is the in-model library copy with the same algorithm. It differs in the vectorized homomorphic-mod variant and the multi-GPU split, and BERT itself calls its 35-level variants (`b35`, `b35_2`).
- The 33-level standalone benchmark is the runnable one Cerium ships, so we measured that. A standalone `b35` benchmark would need extraction work.

The run step reproduces `bootstrap_runner.py`: same seeds, keys and input (32768 unit-modulus complex numbers with random phase).

| Quantity | Value |
| --- | --- |
| Input | 2 limbs, encoded scale 2^50.23 (declared 2^56 in the DSL) |
| Output | 18 limbs, scale 2^56 (decoded at 2^50.23) |
| Latency, 1 launch per call (30 timed after 3 warm-ups; 2 runs) | median 17.39–17.44 ms, p95 17.41–17.47 ms |
| Steady, 10 launches per call | 17.32–17.33 ms per bootstrap |
| GPU kernel time (nsys) | 18.26 ms. Shares: fused 33%, ModDown 20%, base conv 18%, NTT 15%, INTT 13% |
| Error before bootstrapping vs plaintext | max abs 2.1e-12 (38.8 bits) |
| Error after bootstrapping vs plaintext (4 runs) | max abs 2.0e-6 to 3.3e-6 (18.2–18.9 bits); mean abs 5.5e-7; rel L2 6.2e-7 |
| Error added by bootstrapping (output − input) | about equal to the after-bootstrap error |
| Ops, source level | 1 ephemeral key switch, 1 ModRaise, 298 ct-pt multiplies, 24 ct-ct multiplies, 22 relinearizations, 73 rotations, 2 conjugations, 9 rescales, 28 double rescales |
| Ops, compiled level | 98 key switches and 73 automorphisms (match the source counts); 462 launched functions |
| One-time costs | constants 5.3 s, eval keygen 3.6–4.3 s, load and CUDA graph 4.0–5.0 s, compile 3.2 s plus nvcc 4.4 s |
| Key and plaintext storage | eval keys 5.48 GB, plaintexts 454 MB |

The after-bootstrap error varies by up to about 1.7× between runs with
identical seeds. The before-bootstrap error is identical across runs, so the
variation arises inside the GPU computation. Report the range, not one number.

## 4. Tricycle SST-2 reference

Details are in `TRICYCLE_REPRO.md`.
- **Procedure:**
  - Python venv: transformers 4.48.2, torch ≥ 2.6.
  - Export the checkpoint `textattack/bert-base-uncased-SST-2` at `95f0f6f8…`.
  - CMake build against HEaaN.
  - `benchmark_bert_base_128` with `TRICYCLE_BENCHMARK_SYNC=1`, 1 repetition, no warm-up. The 128 tokens are padded to 132 positions (4×33), with HEaaN `FGb` parameters.
- **Not reproduced:** we have no HEaaN GPU library or license, and the reference targets an A100 80 GB (ours is an H200).
- Lawrence's **~105 s for 128 tokens is a reference claim, not our measurement.** The repository README gives no time.
- Scope of the reference timer: the 12 encoder layers' wall time, with bootstrapping time recorded separately, plus a separately timed head. It excludes key generation, per-layer plaintext weight encoding, embedding and encryption.
- The README's accuracy figure (0.02545) is described as L1 (a sum of errors). The CSV column the code writes is the max of the two logit errors.

## 5. Hoisting and double hoisting

Details are in `HOISTING_PLAN.md`.
- **tricyclic-matmul (OpenFHE)** already has both halves: hoisted baby steps (`EvalFastRotationPrecompute` / `EvalFastRotation`), and giant steps accumulated in Q·P with one `KeySwitchDown`.
- **The HEaaN SST-2 port** has neither. It does use lazy rescale and lazy relinearization.
- **Cerium** has both halves only inside `BsgsMultiplyAccumulate` (ct×pt). For ct×ct it offers `HoistedRotate` (shared ModUp) and `RotateAccumulateMany` (single ModDown) separately. `RotateAccumulate`, the combined single-input version, segfaults the compiler at this commit. The compiler's automatic fusion passes are defined but never invoked.
- The measured baseline above already shows each half working on the GPU (NTT time drops with `hoisted`, ModDown time drops with `rotacc`). Double hoisting would combine both.
- Following the request, **nothing was implemented.** The plan (BSGS-with-ones schedule, `RotAcc2` fix, Tricycle ct-ct port) and its correctness and performance checks are in `HOISTING_PLAN.md`.

## Reproduce

On a 1-GPU Hopper machine, from this repo:

```sh
# one-time setup: CUDA 13.3, g++-12, NCCL, SEAL 4.1.1, Cerium clone + build, venv
export CERIUM_WORK=/workspace/cerium_work
git clone --recurse-submodules https://github.com/CMU-CAOS/Cerium.git $CERIUM_WORK/Cerium
git -C $CERIUM_WORK/Cerium checkout 6b48704      # the commit measured here
bash familiarization/pod/setup_pod.sh
cp cerium_examples/bicyclic_matmul/*.py $CERIUM_WORK/Cerium/examples/bicyclic_matmul/   # updated example
source $CERIUM_WORK/env.sh

bash benchmarks/cerium_bench/run_bmm.sh       results/   # BMM sweep        -> bmm_results.jsonl
bash benchmarks/cerium_bench/run_bootstrap.sh results/   # bootstrapping    -> bootstrap_results.jsonl
bash benchmarks/cerium_bench/profile_nsys.sh  results/   # kernel breakdown -> nsys/*_breakdown.json

# one BMM case by hand (from $CERIUM_REPO/examples):
BICYCLIC_SHAPE=31,32,33 BICYCLIC_SCHEDULE=doubling BICYCLIC_VALUES=real \
  python benchmarks/cerium_bench/bench_bmm.py compile
BICYCLIC_SHAPE=31,32,33 BICYCLIC_SCHEDULE=doubling BICYCLIC_VALUES=real CERIUM_RUNTIME_NUM_ITERS=1 \
  python benchmarks/cerium_bench/bench_bmm.py run --warmup 10 --reps 200
```

Scripts are in `cerium_bench/`:
- `bench_bmm.py` and `bench_bootstrap.py`: timed phases, errors, environment record.
- `opcount.py`: source-level and compiled-level operation counts.
- `kernel_breakdown.py`: nsys kernel-time classes.
- `env_info.py`.

## Email draft

> Hi Divy, Sohaib, Lawrence,
>
> Here is where things stand. All numbers are from one H200 running Cerium at commit 6b48704; raw results and scripts are in the repo.
>
> **BMM at realistic sizes.**
> - The bicyclic ct×ct product needs pairwise-coprime M, N and P with M·N·P ≤ 32768 slots.
> - Shapes that nearly fill the ciphertext, all with max abs error ≤ 2.7e-13 and relative Frobenius error about 2.9e-14 against NumPy:
>   - 31×32 @ 32×33 (99.9% useful slots): 0.49 ms per product in steady state (median of 100-launch runs; 0.56 ms median and 0.60 ms p95 for a single launch including host overhead);
>   - 15×128 @ 128×17: 0.65 ms;
>   - 17×64 @ 64×29: 0.57 ms;
>   - 127×16 @ 16×15: 0.41 ms.
> - Packing many small products into one ciphertext (for example 546 copies of 3×4 @ 4×5) is slot batching, which we report separately; it takes 0.25 ms.
>
> **Costs and op counts.**
> - Each product is 1 ct-ct multiply, 1 relinearization and log2 N rotations (doubling schedule), with no rescale or bootstrap.
> - The compiled graph confirms these counts.
> - By ablation, multiply plus relinearize takes 0.085 ms; the rest is the rotation sum.
> - One-time costs (nvcc build about 3 s, key generation, CUDA-graph setup about 1 s) are reported separately.
>
> **Bootstrapping.**
> - Cerium's single-GPU 32K-slot bootstrap (the standalone 33-level benchmark) takes 17.4 ms median (p95 17.5 ms).
> - It takes a 2-limb ciphertext to 18 limbs.
> - Error against the original plaintext is 2.1e-12 before and 2.0–3.3e-6 after (about 18–19 bits), so almost all of it comes from bootstrapping.
>
> **Tricycle SST-2.**
> - I documented the reproduction procedure, but could not run it: it needs the proprietary HEaaN GPU library, which we don't have, and it targets an A100 80 GB.
> - I am treating the ~105 s for 128 tokens as Lawrence's reference number, not something we measured.
>
> **Hoisting.**
> - The OpenFHE tricyclic matmul already double-hoists (hoisted baby steps, giant steps with one ModDown); the HEaaN port does not.
> - Cerium has both halves, and on the GPU each one measurably cuts its kernel class (NTT time, or ModDown time).
> - I have a concrete plan and validation approach for combining them, and will implement it after your feedback.
>
> Best,
> Sagnik
