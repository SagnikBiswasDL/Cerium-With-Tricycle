# Tricycle SST-2 reference: reproduction requirements

Source: https://bitbucket.org/tricyclepaper/tricycle-sst2-checkpoint/src/sst2-checkpoint/,
cloned at commit `86466cf` ("Add SST-2 BERT inference and calibrated nonlinear
approximations"). Everything below comes from reading that checkout; we have
**not** run it.

## Status

**Not reproduced.**
- **Blocker:** the encrypted runners link the proprietary CryptoLab HEaaN GPU
  library (`target_link_libraries(... HEaaN cudart)`, headers expected in
  `/usr/local/include`). We do not have a HEaaN installation or license, and it
  is not available on our RunPod image.
- **Hardware:** the README targets an A100 80 GB; our measurements use an H200.
  Even with HEaaN, a run on our hardware would not be a like-for-like reproduction.

Lawrence's figure of about 105 s for 128 tokens is **a reference claim, not our
measurement**. The README gives no end-to-end time; it reports only accuracy
(final two-logit error ≈ 0.02545).

## Reproduction procedure (from the README)

1. Environment: CUDA GPU, HEaaN C++ development environment, A100 80 GB.
2. Python, used for checkpoint export and the tokenizer, which the C++ code calls through `TRICYCLE_PYTHON`:
   ```sh
   python3 -m venv .venv && . .venv/bin/activate
   python -m pip install -r requirements-pytorch.txt   # transformers==4.48.2, torch>=2.6,<3, numpy>=1.24,<3, safetensors>=0.4
   export TRICYCLE_PYTHON="$PWD/.venv/bin/python"
   ```
3. Build (CMake ≥ 3.16, C++17, optional OpenMP, NVTX headers for `nvtxRangePush`):
   ```sh
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --target run_ptxt_bert_tiny_heaan run_bert_tiny_heaan \
     run_ptxt_bert_base_heaan run_bert_base_heaan benchmark_bert_base_128 -j4
   ```
4. Export the checkpoint. This needs Hugging Face Hub access the first time; the
   output directory must be new.
   ```sh
   python pytorch/bert_base.py --output model/bert-base-weights --text 'This movie was wonderful!'
   ```
   - The checkpoint is `textattack/bert-base-uncased-SST-2`, pinned at `95f0f6f859b35c8ff0863ae3cd4e2dbc702c0ae2`: 12 layers, hidden size 768, 12 heads.
   - The converter writes little-endian float64 tensors, the tokenizer, and a SHA-256 manifest.
   - The C++ code reads the weights from `$TRICYCLE_WEIGHTS` (default `/model/bert-base-weights`; `include/transformer_params.h`).
5. 128-token run:
   ```sh
   mkdir -p results/benchmark-128 && cd results/benchmark-128
   TRICYCLE_BENCHMARK_SYNC=1 TRICYCLE_BENCHMARK_REPEATS=1 TRICYCLE_BENCHMARK_NO_WARMUP=1 \
   TRICYCLE_SOFTMAX_CONFIG_DIR="$PWD/../../configs/bert_base_softmax_128" \
   TRICYCLE_GELU_FEATURE_GROUPS="$PWD/../../configs/bert_base_gelu_feature_groups" \
   ../../build/benchmark_bert_base_128 ../../examples/bert_base_benchmark_128_input.txt
   ```
   Outputs: `classification_results.csv`, `operation_timings.csv` (per layer and
   per operation), `head_timing.csv` (pooler + classifier) and `chain_results.csv`
   (per-layer error against the plaintext reference).

## Evaluation configuration (from `programs/benchmark_bert_base_128.cpp`)

**CKKS setup**
- HEaaN `ParameterPreset::FGb` on CUDA device 0, with full slots (`getLogFullSlots`).
- Input ciphertexts are encrypted, then `levelDown` to level 12.

**Packing**
- The input has 128 tokenizer tokens, padded to **132 positions**: 4 blocks of 33, one ciphertext per block.
- PAD tokens are **not** masked in attention (README, "Padding attention").

**Model and approximations**
- BERT-Base, 12 encoder layers, then the pooler and classifier.
- The nonlinear layers (softmax, GELU, LayerNorm) are recalibrated for this checkpoint. Details:
  - softmax exponential with k = 8;
  - degree-119 inverse square root;
  - degree-119 GELU with per-layer ranges.
- This makes them deeper than the original Tricycle, so latency differs from the original paper.

**Keys**
- Encryption, multiplication and conjugation keys.
- One left-rotation key per distinct rotation the layers use.
- Bootstrapping rotation keys.

**Timing method**
- Timestamps use `high_resolution_clock`. With `TRICYCLE_BENCHMARK_SYNC` set, each timestamp is preceded by `cudaDeviceSynchronize()`.
- The README command runs **one** repetition with **no warm-up**.
- Timed:
  - each encoder layer's `evaluate_layer_reusing_precomp` (`wall_ms` per layer), with bootstrapping time accumulated separately (`bootstrap_ms`);
  - the head, timed separately.
- Not timed:
  - key generation;
  - per-layer plaintext weight encoding (`setup_transformer_layer_specific_precomp`);
  - the embedding;
  - encryption;
  - decryption and the error checks.

**Accuracy**
- `classification_results.csv` holds `max_logit_error = max(|Δlogit0|, |Δlogit1|)`.
- The README describes its 0.02545 figure as L1, the *sum* of the two absolute logit errors. The code writes the max, so these differ unless one error is zero.

## What a comparison with Cerium needs

1. A reproduction, by Lawrence or anyone with HEaaN, on a stated GPU.
   Report the sum of the 12 per-layer `wall_ms` values and the head time
   separately, so the scope of the ~105 s figure is unambiguous.
2. Our side does not yet exist. Cerium ships BERT-Base inference
   (`Cerium/benchmarks/bert_base`), but it uses its own packing and
   approximations, not Tricycle's tricyclic encoding. A Tricycle-on-Cerium
   measurement needs the tricyclic matmuls ported to Cerium. The matmul part of
   that is planned in `HOISTING_PLAN.md`.
3. Differences that must be stated alongside any comparison:
   - GPU (A100 80 GB vs H200);
   - library (HEaaN FGb vs Cerium with 28-bit RNS primes and N = 2^16);
   - bootstrapping variant;
   - token count and padding (132 positions);
   - warm-up and repetition count (1 run, cold, in the reference command).
