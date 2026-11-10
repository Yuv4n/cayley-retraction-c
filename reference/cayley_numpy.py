"""
The Cayley retraction in NumPy: the reference that the C code is checked against.

Four problems, in order. Each has tests in tests/test_reference.py, which fail
until the problem is done. Run them from the repository root with

    python3 tests/test_reference.py

NumPy is the only dependency (python3 -m pip install numpy). Matrices are
float64 NumPy arrays, n x n unless stated. Clarity matters more than speed
here: this is the version the C code must agree with.

The maths is in docs/derivation.md.
"""

import numpy as np


def form_a(W, G):
    """
    Problem 1: the skew-symmetric matrix A.

    Return A = G W^T - W G^T as a new array.

    Rules:
      - Do not modify W or G.
      - The result must be skew-symmetric (A^T = -A) for any W and G, not only
        for orthogonal W.

    Think about: why does that hold with no assumption on W or G?
    """
    raise NotImplementedError("problem 1: form_a")


def orthogonality_error(W):
    """
    Problem 2: how far W is from orthogonal.

    Return the Frobenius norm of W^T W - I as a Python float.

    Think about: which NumPy function gives the Frobenius norm of a matrix, and
    what is its default for a 2-D array?
    """
    raise NotImplementedError("problem 2: orthogonality_error")


def cayley_step(W, G, eta):
    """
    Problem 3: one Cayley update.

    Return W+ = (I + K)^-1 (I - K) W, where K = (eta / 2) * A and A = form_a(W, G).

    Rules:
      - Build M = I + K and B = (I - K) W, then solve M X = B with
        np.linalg.solve. Do not call np.linalg.inv, or invert in any other way.
      - Return a new array. Do not modify W or G.
      - Any n >= 1 must work, including n = 1 (where A is zero).

    Think about: what does np.linalg.solve do with the n columns of B?
    """
    raise NotImplementedError("problem 3: cayley_step")


def first_order_step(W, G, eta):
    """
    Problem 4: the first-order approximation, used to sanity-check the update.

    Return W - eta * A @ W, with A = form_a(W, G). For small eta, cayley_step
    agrees with this to first order, so their difference should shrink like eta
    squared.

    Do not modify W or G.
    """
    raise NotImplementedError("problem 4: first_order_step")
