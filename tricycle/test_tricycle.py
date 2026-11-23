"""Checks for the tricyclic matmul, the nonlinear approximations, and one encoder block.

Run from the repo root: ``python3 -m tricycle.test_tricycle``
"""

import unittest

import numpy as np

from tricycle.encoder import EncoderWeights, dense_encoder_layer, encoder_layer
from tricycle.encoding import (
    concatenated_matmul,
    decode,
    dense_matmul,
    encode,
    flatten_block_grid,
    split_blocks,
)
from tricycle.kernels import block_matmul
from tricycle.matmul import (
    DEFAULT_SLOTS,
    ct_ct_plan,
    ct_pt_plan,
    direct_ct_ct,
    direct_ct_pt,
    direct_pt_ct,
    matmul_ct_ct,
    matmul_ct_pt,
    matmul_pt_ct,
    schedule_summary,
)
from tricycle.nonlinear import exp_power, gelu, gelu_chebyshev, inv_sqrt_chebyshev, layer_norm, softmax
from tricycle.slots import left_rotate, replicate


def _sequence(shape):
    return np.arange(1, int(np.prod(shape)) + 1, dtype=float).reshape(shape)


class EncodingTests(unittest.TestCase):
    def test_roundtrip_and_slot_formula(self):
        tensor = _sequence((1, 3, 5))
        slots = encode(tensor)
        self.assertEqual(slots.shape, (15,))
        self.assertEqual(slots[7], tensor[7 % 1, 7 % 3, 7 % 5])
        np.testing.assert_array_equal(decode(slots, (1, 3, 5)), tensor)

    def test_rejects_a_shared_divisor(self):
        with self.assertRaises(ValueError):
            encode(np.zeros((2, 4, 6)))


class ReplicationTests(unittest.TestCase):
    def test_doubling_tiles_the_first_block(self):
        slots = np.zeros(64)
        slots[:3] = (1, 2, 3)
        tiled = replicate(slots, 3, 4)
        for copy in range(4):
            np.testing.assert_array_equal(tiled[copy * 3:(copy + 1) * 3], (1, 2, 3))
        self.assertTrue(np.all(tiled[12:] == 0))

    def test_one_repetition_is_a_copy(self):
        slots = np.arange(8, dtype=float)
        np.testing.assert_array_equal(replicate(slots, 3, 1), slots)

    def test_left_rotate_matches_openfhe_direction(self):
        np.testing.assert_array_equal(left_rotate(np.array([1.0, 2, 3, 4]), 1), [2, 3, 4, 1])
        np.testing.assert_array_equal(left_rotate(np.array([1.0, 2, 3, 4]), -1), [4, 1, 2, 3])


class MatmulTests(unittest.TestCase):
    def assert_close(self, got, expected):
        np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)

    def test_small_ct_pt_matches_dense_and_the_direct_sum(self):
        # Output (1, 2, 5) is pairwise coprime, so every entry has its own slot.
        left = _sequence((1, 2, 3))
        right = _sequence((1, 3, 5))
        expected = dense_matmul(left, right)
        self.assert_close(direct_ct_pt(left, right), expected)
        self.assert_close(matmul_ct_pt(left, right), expected)
        summary = schedule_summary(ct_pt_plan(left.shape, right.shape))
        self.assertGreaterEqual(summary["diagonal_multiplies"], left.shape[2])
        self.assertEqual(summary["result_dims"], (1, 2, 5))

    def test_small_pt_ct_and_ct_ct(self):
        left = _sequence((1, 2, 3))
        right = _sequence((1, 3, 5))
        expected = dense_matmul(left, right)
        self.assert_close(direct_pt_ct(left, right), expected)
        self.assert_close(matmul_pt_ct(left, right), expected)
        self.assert_close(direct_ct_ct(left, right), expected)
        self.assert_close(matmul_ct_ct(left, right), expected)

    def test_lawrence_single_block_shapes(self):
        # ctxt_ctxt_matmul_test.cpp: blocks (1, 3, 5) @ (1, 5, 7) = (1, 3, 7)
        left = _sequence((1, 3, 5))
        right = _sequence((1, 5, 7))
        expected = dense_matmul(left, right)
        self.assert_close(matmul_ct_pt(left, right), expected)
        self.assert_close(matmul_pt_ct(left, right), expected)
        self.assert_close(matmul_ct_ct(left, right), expected)
        self.assert_close(direct_ct_ct(left, right), matmul_ct_ct(left, right))

    def test_concatenated_sums_the_batch(self):
        # ctxt_ptxt_concatenated_matmul_test.cpp: (2, 3, 5) @ (2, 5, 7) = (1, 3, 7)
        left = _sequence((2, 3, 5))
        right = _sequence((2, 5, 7))
        expected = concatenated_matmul(left, right)
        self.assertEqual(expected.shape, (1, 3, 7))
        self.assert_close(direct_ct_pt(left, right, concatenated=True), expected)
        self.assert_close(matmul_ct_pt(left, right, concatenated=True), expected)
        self.assert_close(direct_pt_ct(left, right, concatenated=True), expected)
        self.assert_close(matmul_pt_ct(left, right, concatenated=True), expected)

    def test_block_product_from_the_openfhe_test(self):
        # (1, 6, 10) @ (1, 10, 21), blocks (1, 3, 5) and (1, 5, 7)
        left = _sequence((1, 6, 10))
        right = _sequence((1, 10, 21))
        expected = dense_matmul(left, right)
        left_blocks = flatten_block_grid(split_blocks(left, (1, 3, 5)))
        right_blocks = flatten_block_grid(split_blocks(right, (1, 5, 7)))
        self.assertEqual((len(left_blocks), len(left_blocks[0])), (2, 2))
        self.assertEqual((len(right_blocks), len(right_blocks[0])), (2, 3))

        expected_blocks = flatten_block_grid(split_blocks(expected, (1, 3, 7)))
        products = [
            block_matmul("ct_pt", left_blocks, right_blocks),
            block_matmul("pt_ct", left_blocks, right_blocks),
            block_matmul("ct_ct", left_blocks, right_blocks, baby_side="right"),
            block_matmul("ct_ct", left_blocks, right_blocks, baby_side="left"),
        ]
        for product in products:
            for i in range(2):
                for k in range(3):
                    self.assert_close(product[i][k], expected_blocks[i][k])

    def test_random_rectangular_product(self):
        rng = np.random.default_rng(0)
        left = rng.normal(size=(1, 3, 4))
        right = rng.normal(size=(1, 4, 5))
        expected = dense_matmul(left, right)
        self.assert_close(matmul_ct_pt(left, right), expected)
        self.assert_close(matmul_ct_ct(left, right), expected)

    def test_ct_ct_key_schedule_is_nonempty(self):
        plan = ct_ct_plan((1, 3, 5), (1, 5, 7), DEFAULT_SLOTS)
        summary = schedule_summary(plan)
        self.assertGreater(summary["giant_steps"], 0)
        self.assertGreater(summary["rotations_in_keygen"], 0)
        self.assertEqual(summary["result_dims"], (1, 3, 7))


class NonlinearTests(unittest.TestCase):
    def test_softmax_rows_sum_to_one(self):
        logits = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        weights = softmax(logits, axis=-1)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0)
        self.assertGreater(weights[0, 2], weights[0, 0])

    def test_layernorm_zero_mean_unit_variance(self):
        values = np.array([[1.0, 2.0, 3.0, 4.0]])
        normed = layer_norm(values, np.ones(4), np.zeros(4))
        np.testing.assert_allclose(normed.mean(), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.mean(normed ** 2), 1.0, atol=1e-12)

    def test_exp_power_tracks_exp_near_zero(self):
        xs = np.linspace(-1.0, 0.0, 21)
        approx = exp_power(xs, 7)
        self.assertLess(np.max(np.abs(approx - np.exp(xs))), 0.02)
        # k squarings, so the base is multiplied, not exponentiated by a library exp.
        self.assertAlmostEqual(exp_power(np.array([0.0]), 7)[0], 1.0)

    def test_gelu_polynomial_on_the_tiny_interval(self):
        # Tiny GELU: degree 59, mult_factor 1/15, input of the series on [-1, 1].
        xs = np.linspace(-1.0, 1.0, 201)
        approx, _ = gelu_chebyshev(xs, degree=59, mult_factor=1.0 / 15.0)
        self.assertLess(np.max(np.abs(approx - gelu(xs * 15.0))), 1e-4)

    def test_inverse_sqrt_polynomial_on_a_positive_interval(self):
        # Tiny LayerNorm builds a degree-27 series. The interval here is the
        # image of a variance after the postscale of 10 used in transformer.h.
        lower, upper = 10.0, 1000.0
        xs = np.linspace(lower, upper, 200)
        approx, _ = inv_sqrt_chebyshev(xs, degree=27, initial_lower=lower, initial_upper=upper)
        # Degree 27 on a 100x interval, which is the Tiny LayerNorm degree.
        # The absolute error sits near 7e-4; a broken fit is off by O(1).
        self.assertLess(np.max(np.abs(approx - xs ** -0.5)), 1e-3)


class EncoderTests(unittest.TestCase):
    def test_block_matches_dense_bert_math(self):
        rng = np.random.default_rng(1)
        tokens, d_model, d_k, d_ff = 3, 5, 4, 7
        weights = EncoderWeights(
            w_q=rng.normal(size=(d_model, d_k)),
            w_k=rng.normal(size=(d_model, d_k)),
            w_v=rng.normal(size=(d_model, d_k)),
            w_o=rng.normal(size=(d_k, d_model)),
            w_ff1=rng.normal(size=(d_model, d_ff)),
            w_ff2=rng.normal(size=(d_ff, d_model)),
            norm1_weight=rng.normal(size=(d_model,)) + 1.0,
            norm1_bias=rng.normal(size=(d_model,)),
            norm2_weight=rng.normal(size=(d_model,)) + 1.0,
            norm2_bias=rng.normal(size=(d_model,)),
        )
        hidden = rng.normal(size=(tokens, d_model))
        got = encoder_layer(hidden, weights)
        expected = dense_encoder_layer(hidden, weights)
        np.testing.assert_allclose(got, expected, rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
