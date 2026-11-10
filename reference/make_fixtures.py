"""
Fixtures for the C tests: seeded inputs and the reference output, written as
raw float64 files that the C code reads with fread.

Five problems plus a main function. Tests are in tests/test_reference.py. Run
them from the repository root with

    python3 tests/test_reference.py

Each fixture is a flat file of n * n doubles in row-major order (NumPy's default
C order), with the size in the file name. For n = 3 the files are
W_n3.bin, G_n3.bin and Wplus_n3.bin.

The step size is fixed so that the C tests can use the same value.
"""

from pathlib import Path

import numpy as np

import cayley_numpy as ref

FIXTURE_ETA = 0.1
SIZES = (3, 128)
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def random_orthogonal(n, rng):
    """
    Problem 1: a random orthogonal matrix.

    Return an n x n orthogonal matrix, built from the QR factorisation of an
    n x n matrix of standard normal entries drawn from rng (a NumPy Generator).

    Rules:
      - The same rng state must give the same matrix.
      - QR routines may return Q with arbitrary column signs. Multiply each
        column of Q by the sign of the matching diagonal entry of R, so that the
        result does not depend on that convention.
      - Return a C-contiguous float64 array.

    Think about: what is the product Q * diag(sign(diag(R))), and why is it
    still orthogonal?
    """
    raise NotImplementedError("problem 1: random_orthogonal")


def make_case(n, seed):
    """
    Problem 2: one test case.

    Using np.random.default_rng(seed): draw W with random_orthogonal, then G as
    an n x n matrix of standard normal entries. Return the tuple
    (W, G, W_plus), where W_plus = ref.cayley_step(W, G, FIXTURE_ETA).

    The same n and seed must always give the same three arrays.
    """
    raise NotImplementedError("problem 2: make_case")


def fixture_paths(directory, n):
    """
    Problem 3: where the files for size n live.

    Return a dict mapping "W", "G" and "Wplus" to Path objects inside
    directory, named W_n{n}.bin, G_n{n}.bin and Wplus_n{n}.bin. Do not touch
    the file system.
    """
    raise NotImplementedError("problem 3: fixture_paths")


def write_case(directory, n, seed):
    """
    Problem 4: write one case to disk.

    Make the case with make_case, create directory if it does not exist, and
    write W, G and W_plus as raw float64 with ndarray.tofile, in row-major
    order. Return the dict from fixture_paths.
    """
    raise NotImplementedError("problem 4: write_case")


def read_matrix(path, n):
    """
    Problem 5: read a fixture back.

    Return an n x n float64 array read from path with np.fromfile. If the file
    does not hold exactly n * n doubles, raise ValueError. This is what the C
    reader must check too when it calls fread.
    """
    raise NotImplementedError("problem 5: read_matrix")


def main():
    """
    Write the fixtures for every size in SIZES into FIXTURE_DIR, using the size
    itself as the seed, and print each path written. Not tested.
    """
    raise NotImplementedError("main")


if __name__ == "__main__":
    main()
