"""Checks for the Tricycle layer products as backend programs.

Run from the repo root: ``python3 -m tricycle.test_kernels``
"""

import unittest
from fractions import Fraction

import numpy as np

from tricycle.encoding import decode, dense_matmul
from tricycle.kernels import (
    SHAPES,
    LayerShape,
    NumpyBackend,
    simulate,
    track_scales,
    tricycle_layer_matmuls,
)
from tricycle.matmul import (
    DEFAULT_SLOTS,
    direct_ct_ct,
    direct_ct_pt,
    direct_pt_ct,
    matmul_ct_ct,
    matmul_ct_pt,
    matmul_pt_ct,
    put_encoded,
)
from tricycle.nonlinear import gelu, softmax

# Miniature of the Tricycle packing: every tricyclic shape below is pairwise
# coprime, keys are padded by 2, two token row blocks, two feed-forward blocks.
MINI = LayerShape(tokens=5, d_model=7, heads=3, d_k=4, d_ff=11, row_blocks=2, ff_blocks=2)
# Stand-ins for 28-bit RNS primes; the bookkeeping only needs distinct values.
FAKE_PRIMES = [(1 << 28) - 57 - 30 * i for i in range(16)]


class BroadcastAndMaskTests(unittest.TestCase):
    def assert_close(self, got, expected):
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)

    def test_activations_broadcast_against_per_head_weights(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(1, 5, 7))
        w = rng.normal(size=(3, 7, 4))
        expected = dense_matmul(x, w)
        self.assertEqual(expected.shape, (3, 5, 4))
        self.assert_close(direct_ct_pt(x, w), expected)
        self.assert_close(matmul_ct_pt(x, w), expected)

    def test_plaintext_reshape_broadcasts_over_heads(self):
        rng = np.random.default_rng(3)
        pad = np.zeros((1, 7, 5))
        pad[0, :5, :5] = np.eye(5)
        keys = rng.normal(size=(3, 5, 4))
        expected = dense_matmul(pad, keys)
        self.assert_close(direct_pt_ct(pad, keys), expected)
        self.assert_close(matmul_pt_ct(pad, keys), expected)
        # The padded keys, read with the last two dims swapped, are K^T.
        slots = put_encoded(matmul_pt_ct(pad, keys), DEFAULT_SLOTS)
        key_t = decode(slots, (3, 4, 7))
        self.assert_close(key_t[:, :, :5], keys.transpose(0, 2, 1))
        self.assert_close(key_t[:, :, 5:], 0.0)

    def test_both_ct_ct_schedules_with_output_masks(self):
        rng = np.random.default_rng(4)
        q = rng.normal(size=(3, 5, 4))
        k_t = rng.normal(size=(3, 4, 7))
        expected = dense_matmul(q, k_t)
        for side in ("right", "left"):
            for mask in (None, 0.5):
                scale = 1.0 if mask is None else mask
                self.assert_close(matmul_ct_ct(q, k_t, baby_side=side, mask_value=mask),
                                  expected * scale)
                self.assert_close(direct_ct_ct(q, k_t, baby_side=side, mask_value=mask),
                                  expected * scale)

    def test_complex_packed_weights(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(1, 5, 7))
        w = rng.normal(size=(3, 7, 4)) + 1j * rng.normal(size=(3, 7, 4))
        self.assert_close(matmul_ct_pt(x, w), dense_matmul(x, w))


class LayerProductTests(unittest.TestCase):
    def test_each_product_matches_the_dense_blocks(self):
        rng = np.random.default_rng(6)
        for matmul in tricycle_layer_matmuls(MINI):
            with self.subTest(matmul.name):
                left, right = matmul.random_blocks(rng)
                decoded, slots, counts = simulate(matmul, left, right)
                expected = matmul.reference(left, right)
                self.assertEqual(set(decoded), set(expected))
                for name in expected:
                    np.testing.assert_allclose(decoded[name], expected[name],
                                               rtol=1e-9, atol=1e-9)
                    # Nothing leaks past the output, so the next product can replicate it.
                    tail = np.asarray(slots[name])[matmul.plan.output_elements:]
                    np.testing.assert_allclose(tail, 0.0, atol=1e-9)
                self.assertGreater(counts["multiplies"], 0)

    def test_bert_tiny_and_base_shapes(self):
        rng = np.random.default_rng(8)
        for shape_name in ("tiny", "base"):
            for matmul in tricycle_layer_matmuls(SHAPES[shape_name]):
                with self.subTest(f"{shape_name}/{matmul.name}"):
                    left, right = matmul.random_blocks(rng)
                    decoded, _, _ = simulate(matmul, left, right)
                    expected = matmul.reference(left, right)
                    for name in expected:
                        np.testing.assert_allclose(decoded[name], expected[name],
                                                   rtol=1e-8, atol=1e-8)

    def test_scales_and_levels_are_exact(self):
        level = 12
        cipher_scale = 1 << 56
        for matmul in tricycle_layer_matmuls(MINI):
            with self.subTest(matmul.name):
                outputs, plaintext_scales = track_scales(matmul, FAKE_PRIMES, cipher_scale, level)
                self.assertTrue(plaintext_scales)
                for value in outputs.values():
                    self.assertEqual(value.degree, 1)
                    if matmul.kind == "ct_ct":
                        self.assertEqual(value.level, level - 2)
                        self.assertEqual(value.scale,
                                         Fraction(cipher_scale ** 2, FAKE_PRIMES[level - 2]))
                    else:
                        self.assertEqual(value.level, level - 1)
                        self.assertEqual(value.scale, cipher_scale)


def _blocks(tensor, axis, size):
    return [np.take(tensor, range(i * size, (i + 1) * size), axis=axis)
            for i in range(tensor.shape[axis] // size)]


class ChainedLayerTest(unittest.TestCase):
    """Attention and feed-forward with slot outputs fed straight into the next product.

    Softmax and GELU are Divy's half; here they are applied to decoded values
    and re-encoded. Everything between them stays in slots.
    """

    def test_attention_then_feed_forward(self):
        s = MINI
        rng = np.random.default_rng(7)
        n = DEFAULT_SLOTS
        matmuls = {m.name: m for m in tricycle_layer_matmuls(s)}
        T, Tp, D, H, K, F, R = (s.tokens, s.padded_tokens, s.d_model, s.heads, s.d_k, s.d_ff,
                                s.row_blocks)

        x = rng.uniform(-1, 1, (R * T, D))
        wq, wk, wv = (rng.uniform(-0.5, 0.5, (H, D, K)) for _ in range(3))
        wo = rng.uniform(-0.5, 0.5, (H, K, D))
        w1 = rng.uniform(-0.5, 0.5, (D, s.ff_blocks * F))
        w2 = rng.uniform(-0.5, 0.5, (s.ff_blocks * F, D))

        def run(name, left, right):
            m = matmuls[name]
            inputs = m.inputs(left, right, n)
            out = m.emit(NumpyBackend(inputs, n), n)
            rows, cols = m.out_grid
            return [[out[m.output_name(i, k)] for k in range(cols)] for i in range(rows)]

        x_blocks = [[xb[None]] for xb in _blocks(x, 0, T)]
        qkv = run("qkv", x_blocks, [[wq, wk, wv]])

        pad = np.zeros((1, Tp, T))
        pad[0, :T, :T] = np.eye(T)
        kv = [[qkv[i][1] for i in range(R)] + [qkv[i][2] for i in range(R)]]
        reshaped = run("reshape", [[pad]], kv)

        # K^T is the reshape output's slots, unchanged.
        scores = run("qk", [[qkv[i][0]] for i in range(R)], [reshaped[0][:R]])
        sdims = matmuls["qk"].plan.result_dims

        probs = []
        for i in range(R):
            row = np.concatenate([decode(scores[i][j], sdims)[:, :, :T] for j in range(R)], -1)
            p = softmax(row, axis=-1)
            padded = []
            for j in range(R):
                block = np.zeros((H, T, Tp))
                block[:, :, :T] = p[:, :, j * T:(j + 1) * T]
                padded.append(block)
            probs.append(padded)
        context = run("pv", probs, [[reshaped[0][R + j]] for j in range(R)])
        attended = run("attn_out", context, [[wo]])
        w1_blocks = [[wb[None] for wb in _blocks(w1, 1, F)]]
        ff1 = run("ff1", attended, w1_blocks)
        fdims = matmuls["ff1"].plan.result_dims
        activated = [[gelu(decode(ff1[i][j], fdims)) for j in range(s.ff_blocks)]
                     for i in range(R)]
        w2_blocks = [[wb[None]] for wb in _blocks(w2, 0, F)]
        ff2 = run("ff2", activated, w2_blocks)
        odims = matmuls["ff2"].plan.result_dims
        got = np.concatenate([decode(ff2[i][0], odims)[0] for i in range(R)], 0)

        q = np.einsum("td,hdk->htk", x, wq)
        k = np.einsum("td,hdk->htk", x, wk)
        v = np.einsum("td,hdk->htk", x, wv)
        p = softmax(q @ k.transpose(0, 2, 1) / np.sqrt(K), axis=-1)
        attention = np.einsum("htk,hkd->td", p @ v, wo)
        expected = gelu(attention @ w1) @ w2
        np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
