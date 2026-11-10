"""
Data for the orthogonal Procrustes demonstration.

Given X and Y of size n x m with m >= n, minimise f(W) = ||W X - Y||_F^2 over
orthogonal W. The C driver runs the iterations. This file makes the problem and
its exact answer.

Seven problems plus a main function. Tests are in tests/test_examples.py. Run
them from the repository root with

    python3 tests/test_examples.py

The update preserves the determinant, so iterating from a rotation can only
reach rotations. The target is therefore the best rotation, not the best
orthogonal matrix. That is the point of problem 5.
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"


def random_rotation(n, rng):
    """
    Problem 1: a random rotation.

    Return an n x n orthogonal matrix with determinant +1, from the QR
    factorisation of a matrix of standard normal entries drawn from rng.

    Rules:
      - Fix the column signs using the diagonal of R, as for any QR-based
        orthogonal matrix.
      - If the determinant is then -1, flip the sign of one column.
      - The same rng state gives the same matrix.
    """
    raise NotImplementedError("problem 1: random_rotation")


def make_problem(n, m, sigma, seed):
    """
    Problem 2: synthetic data.

    Using np.random.default_rng(seed), in this order: draw X (n x m, standard
    normal), draw W_true with random_rotation, draw E (n x m, standard normal).
    Return (X, Y, W_true) with Y = W_true @ X + sigma * E.

    Raise ValueError if m < n.
    """
    raise NotImplementedError("problem 2: make_problem")


def objective(W, X, Y):
    """
    Problem 3: the loss f(W) = ||W X - Y||_F^2, as a Python float.
    """
    raise NotImplementedError("problem 3: objective")


def gradient(W, X, Y):
    """
    Problem 4: the Euclidean gradient of f with respect to W.

    Return G = 2 (W X - Y) X^T, an n x n array. This is the G that the C update
    takes as input.
    """
    raise NotImplementedError("problem 4: gradient")


def kabsch(X, Y):
    """
    Problem 5: the exact answer over rotations.

    Take the SVD  Y X^T = U S V^T. Over all orthogonal matrices the minimiser
    is U V^T. If det(U V^T) is -1 that is a reflection, which the update can
    never reach, so return the best rotation instead:

        W* = U diag(1, ..., 1, det(U V^T)) V^T.

    Use np.linalg.svd (note that it returns V^T, not V). Return an n x n array
    with determinant +1.

    Think about: why does flipping the sign for the smallest singular value
    cost the least?
    """
    raise NotImplementedError("problem 5: kabsch")


def write_problem(directory, n, m, sigma, seed):
    """
    Problem 6: write a problem and its answer to disk.

    Make the problem with make_problem, compute W* with kabsch, create
    directory if needed, and write these as raw float64 (ndarray.tofile,
    row-major):

        X_n{n}_m{m}.bin       n x m
        Y_n{n}_m{m}.bin       n x m
        W0_n{n}.bin           n x n, the identity (the starting point)
        Wstar_n{n}_m{m}.bin   n x n

    Return a dict mapping "X", "Y", "W0" and "Wstar" to the Path of each file.
    """
    raise NotImplementedError("problem 6: write_problem")


def read_matrix(path, rows, cols):
    """
    Problem 7: read a matrix back.

    Return a rows x cols float64 array read from path with np.fromfile. If the
    file does not hold exactly rows * cols doubles, raise ValueError.
    """
    raise NotImplementedError("problem 7: read_matrix")


def main():
    """
    Write one problem into DATA_DIR (suggestion: n = 16, m = 64, sigma = 0.1,
    seed = 0) and print each path written. Not tested.
    """
    raise NotImplementedError("main")


if __name__ == "__main__":
    main()
