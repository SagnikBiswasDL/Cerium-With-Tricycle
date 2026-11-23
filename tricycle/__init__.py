"""Plaintext port of Lawrence's Tricycle pieces that can run without HEaaN or OpenFHE.

Two halves, matching the split of the work:

- ``tricycle.matmul`` is the OpenFHE tricyclic matmul
  (bitbucket.org/tricyclepaper/tricyclic-matmul, commit 6f9fa57):
  tricyclic slot encoding, ciphertext-plaintext, plaintext-ciphertext, and
  ciphertext-ciphertext products, including the BSGS rotation schedule.
- ``tricycle.nonlinear`` is the nonlinear half of
  bitbucket.org/tricyclepaper/tricycle (commit 83c4294): exact softmax, GELU
  and LayerNorm, plus the polynomial approximations those layers evaluate
  under encryption (power-of-two exponential, Chebyshev GELU, Chebyshev
  inverse square root).
- ``tricycle.encoder`` runs one encoder block by calling both.
- ``tricycle.kernels`` lowers the seven products of a Tricycle layer (HEaaN
  schedules from ``transformer.cpp``) over block grids to a backend: NumPy,
  exact scale/level tracking, or the Cerium DSL (``tricycle.cerium_backend``,
  which needs Cerium's Python package).

The slot routines are the encrypted algorithms with ciphertexts replaced by
their slot vectors, so a result that matches NumPy here is the same linear
combination of rotations the CKKS circuit computes. Bootstrapping and the
HEaaN runtime are not in this package.
"""

from tricycle.encoder import EncoderWeights, encoder_layer
from tricycle.encoding import decode, encode
from tricycle.kernels import SHAPES, LayerShape, tricycle_layer_matmuls
from tricycle.matmul import matmul_ct_ct, matmul_ct_pt, matmul_pt_ct
from tricycle.nonlinear import gelu, layer_norm, softmax

__all__ = [
    "SHAPES",
    "EncoderWeights",
    "LayerShape",
    "decode",
    "encode",
    "encoder_layer",
    "gelu",
    "layer_norm",
    "matmul_ct_ct",
    "matmul_ct_pt",
    "matmul_pt_ct",
    "softmax",
    "tricycle_layer_matmuls",
]
