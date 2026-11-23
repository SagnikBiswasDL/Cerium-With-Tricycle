"""One encoder block, with every matrix product going through the tricyclic matmul.

The dimensions are a miniature of the packing in ``transformer.cpp``: the
attention scores are rectangular because the key axis is padded by
``attention_pad`` (Tricycle's ``self_attention_reshape``, which is 2 on
BERT-Tiny and BERT-Base). That pad is what keeps the three tricyclic
dimensions coprime. Softmax ignores the pad columns. GELU and both LayerNorms
are the exact functions; swap in ``gelu_chebyshev`` or ``exp_power`` from
``tricycle.nonlinear`` when measuring the polynomial error on this block.

Shapes, with the defaults ``tokens=3``, ``d_model=5``, ``d_k=4``, ``d_ff=7``,
``attention_pad=2``:

- ``X @ Wq`` is ciphertext-plaintext, ``(1, 3, 5) @ (1, 5, 4)``
- ``Q @ K^T`` is ciphertext-ciphertext, ``(1, 3, 4) @ (1, 4, 5)``
- ``probs @ V`` is ciphertext-ciphertext, ``(1, 3, 5) @ (1, 5, 4)``
- the output projection and both feed-forward multiplies are ciphertext-plaintext

The homomorphic reshape that Tricycle uses to transpose ``K`` inside a
ciphertext is not required here: ``K`` is transposed after ``matmul_ct_pt``
decodes it. The product that consumes ``K^T`` is still a ciphertext-ciphertext
matmul.
"""

from dataclasses import dataclass

import numpy as np

from tricycle.matmul import matmul_ct_ct, matmul_ct_pt
from tricycle.nonlinear import gelu, layer_norm, softmax


@dataclass
class EncoderWeights:
    w_q: np.ndarray
    w_k: np.ndarray
    w_v: np.ndarray
    w_o: np.ndarray
    w_ff1: np.ndarray
    w_ff2: np.ndarray
    norm1_weight: np.ndarray
    norm1_bias: np.ndarray
    norm2_weight: np.ndarray
    norm2_bias: np.ndarray
    attention_pad: int = 2
    eps: float = 1e-12

    @property
    def d_model(self):
        return int(self.w_q.shape[0])

    @property
    def d_k(self):
        return int(self.w_q.shape[1])

    @property
    def d_ff(self):
        return int(self.w_ff1.shape[1])


def _as_batch(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {matrix.shape}")
    return matrix.reshape(1, matrix.shape[0], matrix.shape[1])


def _matmul_pt(left, right, num_slots):
    return matmul_ct_pt(_as_batch(left), _as_batch(right), num_slots=num_slots)[0]


def _matmul_ct(left, right, num_slots):
    return matmul_ct_ct(_as_batch(left), _as_batch(right), num_slots=num_slots)[0]


def encoder_layer(hidden, weights, num_slots=1 << 15):
    """Run one block. ``hidden`` is ``(tokens, d_model)``."""
    hidden = np.asarray(hidden, dtype=float)
    tokens, d_model = hidden.shape
    if d_model != weights.d_model:
        raise ValueError(f"hidden width {d_model} does not match weights {weights.d_model}")

    query = _matmul_pt(hidden, weights.w_q, num_slots)
    key = _matmul_pt(hidden, weights.w_k, num_slots)
    value = _matmul_pt(hidden, weights.w_v, num_slots)

    pad = weights.attention_pad
    key_t = np.zeros((weights.d_k, tokens + pad), dtype=float)
    key_t[:, :tokens] = key.T
    scores = _matmul_ct(query, key_t, num_slots)
    scores = scores[:, :tokens] / np.sqrt(weights.d_k)
    probs = softmax(scores, axis=-1)
    probs_padded = np.zeros((tokens, tokens + pad), dtype=float)
    probs_padded[:, :tokens] = probs

    value_padded = np.zeros((tokens + pad, weights.d_k), dtype=float)
    value_padded[:tokens] = value
    context = _matmul_ct(probs_padded, value_padded, num_slots)
    attention = _matmul_pt(context, weights.w_o, num_slots)

    residual = layer_norm(
        hidden + attention, weights.norm1_weight, weights.norm1_bias, weights.eps)
    ff = _matmul_pt(residual, weights.w_ff1, num_slots)
    ff = gelu(ff)
    ff = _matmul_pt(ff, weights.w_ff2, num_slots)
    return layer_norm(residual + ff, weights.norm2_weight, weights.norm2_bias, weights.eps)


def dense_encoder_layer(hidden, weights):
    """The same block with ``@``. Used to check the tricyclic path."""
    hidden = np.asarray(hidden, dtype=float)
    tokens = hidden.shape[0]
    query = hidden @ weights.w_q
    key = hidden @ weights.w_k
    value = hidden @ weights.w_v
    scores = (query @ key.T) / np.sqrt(weights.d_k)
    probs = softmax(scores, axis=-1)
    attention = (probs @ value) @ weights.w_o
    residual = layer_norm(
        hidden + attention, weights.norm1_weight, weights.norm1_bias, weights.eps)
    ff = gelu(residual @ weights.w_ff1) @ weights.w_ff2
    return layer_norm(residual + ff, weights.norm2_weight, weights.norm2_bias, weights.eps)
