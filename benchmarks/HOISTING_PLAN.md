# Hoisting and double hoisting: findings and plan

Status: **plan only; nothing in this file is implemented or measured yet.**
Request 5 asked for this after the core benchmarks.

## Terms

A rotation (or relinearization) is a key switch: **ModUp**, an automorphism, an
inner product with the key, then **ModDown**.
- ModUp: decompose the input into digits and base-convert them to the extended basis Q·P.
- Inner product with the key: done in Q·P.
- ModDown: divide by P and return to Q.

- **Single hoisting (ModUp hoisting).** Several rotations of the *same*
  ciphertext share one ModUp, because the automorphism commutes with the
  digit decomposition. Each rotation still pays its own inner product and ModDown.
- **ModDown hoisting.** When rotated ciphertexts are only *summed*, the sum can be
  formed in Q·P and ModDown applied once to the total.
- **Double hoisting.** Both together, usually in a BSGS linear transform:
  - baby steps share one ModUp;
  - giant-step rotations are accumulated in Q·P and finish with one ModDown per output.

## What exists today

| Code | ModUp hoisting | ModDown hoisting | Where |
| --- | --- | --- | --- |
| tricyclic-matmul (OpenFHE 1.2.3, CPU), commit 6f9fa57 | Yes, baby steps: `EvalFastRotationPrecompute` + `EvalFastRotation` | Yes, giant steps: `EvalFastRotationExt` (stays in Q·P), accumulate, then one `KeySwitchDown` per output block | `tricyclic_matmul.cpp`: ct×pt 586–693 and 739–845; ct-ct baby steps 1045–1053 and 1083–1100; ct-ct giant steps 1144–1209 |
| tricycle-sst2-checkpoint (HEaaN GPU), commit 86466cf | No. Baby steps are chained `leftRotate` calls, each rotating the previous baby step | No. Each giant step is relinearized and fully rotated, then summed | `src/tricyclic_matmul_heaan.cpp`: `left_rotate_with_signed` (line 21); baby steps 1328–1337 and 1804–1810; ct-ct giant steps 1494–1546 and 1893–1944 |
| Cerium `BsgsMultiplyAccumulate` (ct×pt only) | Yes: `hoisted_input_broadcast_keyswitching_internal_vec`, key-switch type `Broadcast` | Yes: `bsgs_giantstep_accumulate_keyswitch_iterations_vec`, type `Aggregation` | `cerium/compiler/backend/compiler.cc` 3001–3066. Used by bootstrapping (CoeffToSlot / SlotToCoeff) |
| Cerium `HoistedRotate` (DSL) → `Op::HoistInpBroadcast` | Yes | No | `frontend/function.h` 836–868 |
| Cerium `RotateAccumulateMany` (DSL) → `Op::RotAcc` | No (one operand per rotation) | Yes (`Aggregation` key switch, one ModDown) | `frontend/function.h` 781–833; `backend/compiler.cc` 2158ff |
| Cerium `RotateAccumulate` (DSL) → `Op::RotAcc2` (one operand, many rotations) | Intended | Intended | Segfaults the compiler at commit 6b48704 (observed earlier, with and without a 0 shift) |
| Cerium `FuseAggregationPass` / `HoistInputBroadcastPass` | n/a | n/a | Defined in `backend/keyswitch_pass.cc` but never invoked, so the compiler does not hoist automatically; the program must use the hoisted ops |

The HEaaN port does use lazy rescaling (`multWithoutRescale`, one `rescale` per
output) and lazy relinearization (`tensor`, summed, one `relinearize` per
giant step). Neither is hoisting.

## Where hoisting applies

1. **Our bicyclic BMM rotation sum** `z = Σ_t rot(p, t·M·P)`, with `t = 0..N-1`
   and every rotation applied to the same ciphertext `p`.
   - `linear`: N−1 full key switches.
   - `hoisted` (`HoistedRotate`): one ModUp and N−1 ModDowns.
   - `rotacc` (`RotateAccumulateMany`): N−1 ModUps and one ModDown.
   - `doubling`: log2 N sequential full key switches.
   - **Double hoisting**: one ModUp, N−1 inner products in Q·P and one ModDown.
     This is what `RotAcc2` is meant to do. A BSGS split `t = b + g·B` gives a
     cheaper variant: B−1 baby rotations sharing one ModUp, then N/B−1 giant
     rotations aggregated into one ModDown.
2. **A Tricycle ct-ct matmul port to Cerium.**
   - Baby steps of the right matrix: `HoistedRotate`.
   - Products: tensor without relinearization, accumulated per giant step.
   - Relinearize once per giant step.
   - Giant steps: `RotateAccumulateMany`, so one ModDown per output block.
   - Rescale once.

   That reproduces the OpenFHE structure with Cerium primitives. The left
   matrix's pre-rotations and the replication by doubling are rotations of one
   ciphertext, so they are also candidates for `HoistedRotate`.
3. **The HEaaN port.** Hoisting there would need a hoisted-rotation API in the
   HEaaN SDK. We have no SDK access to check; see `TRICYCLE_REPRO.md`.

## Plan, in order

1. **Rotation-sum variants on the BMM benchmark**, no compiler changes:
   - Measure `hoisted` and `rotacc` with the new harness (`run_bmm.sh`, `CASES` extended) on an idle GPU.
   - Add a `bsgs` schedule. It computes the rotation sum with
     `BsgsMultiplyAccumulate(p, ones, baby=[0..B-1]·MP, giant=[0, B, 2B, ...]·MP)`,
     using plaintext ones (scale 0 bits, so the product's scale is unchanged).
   - This gets double hoisting from Cerium's existing BSGS lowering, at the cost
     of N cheap ct×pt multiplies.
   - Risk: plaintext-scale handling with a 0-bit scale. Check `z.scale()` /
     `z.level()` at compile time, and compare decrypted values against NumPy.
2. **`RotAcc2` segfault.**
   - Get a backtrace (gdb on `cerium_compile`, debug build).
   - Check whether `rotate_accumulate(term, indices, 0)` assumes one operand per index
     (`RotAcc`) when given a single operand (`RotAcc2`).
   - If the fix is local, the `rotacc` schedule becomes true double hoisting
     with no extra multiplies.
   - Upstream the fix or report it to CMU-CAOS.
3. **Tricycle ct-ct BSGS port** (item 2 above), sized like BERT attention
   (128×128 @ 128×64, 128-token blocks). Compare against Cerium's own BERT matmul
   (`examples/bert_attention_matmul`).

## Validation

**Correctness, each variant:**
- Decrypt and compare with NumPy `A @ B` on useful output entries only. Report
  max abs error, relative Frobenius error and max abs / max |C| (same as `bench_bmm.py`).
- Compare with the non-hoisted variant: errors should match within about 2×.
  Hoisting changes the key-switch noise slightly; one ModDown should not increase it.
- Check `z.level()` / `z.scale()` match the non-hoisted program.

**Proof that hoisting actually happened.** Use compiled counts from
`*_cerium_graph_simple.dot` (`opcount.count_compiled`), not source counts:
- ModUp work = `int` + `bco` + `ntt` instructions on the input digits.
- ModDowns = `sud` instructions / 2.
- Expected, for R rotations of one ciphertext:

  | Variant | ModUps | ModDowns |
  | --- | --- | --- |
  | linear | R | R |
  | single hoisting | 1 | R |
  | ModDown hoisting | R | 1 |
  | double hoisting | 1 | 1 |

  A variant that does not show the expected drop is not hoisted, whatever the source says.

**Performance:**
- Same harness and definitions as the core benchmark, on an idle GPU:
  - `NUM_ITERS=1`, 10 warmups, 200 timed runs, median and p95;
  - `NUM_ITERS=100` for steady per-launch time.
- Attribution:
  - ablation (`BICYCLIC_SCHEDULE=none` gives the multiply + relinearize cost; subtract it from each variant to get its rotation-stage time);
  - nsys kernel time by primitive class (NTT, base conversion, ModDown, element-wise).
- Shapes: N ∈ {16, 32, 64, 128} with M·N·P near 32768, plus one slot-batched case.
- Report the crossover against `doubling` (log2 N sequential key switches).
- Acceptance: a hoisted variant is worth adopting where it beats `doubling` at
  equal error, or where N is not a power of two (doubling does not apply).
