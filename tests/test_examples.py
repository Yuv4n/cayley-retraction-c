"""
Tests for examples/procrustes_data.py.

Run from the repository root:

    python3 tests/test_examples.py

pytest also works if you have it, since every test is a function named test_*.
Every test starts out failing, because the functions are stubs. Work through
the problems in order.

Tolerances are provisional.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

import procrustes_data as pd  # noqa: E402


# ---- helpers (the tests' own checks, independent of the code under test) ----

def loss(W, X, Y):
    return float(np.sum((W @ X - Y) ** 2))


def departure(W):
    return float(np.linalg.norm(W.T @ W - np.eye(W.shape[0])))


def reflection_flip(n):
    D = np.eye(n)
    D[-1, -1] = -1.0
    return D


# ---- problem 1: random_rotation ----

def test_random_rotation_is_a_rotation():
    rng = np.random.default_rng(0)
    for _ in range(20):  # enough draws to hit both signs of the raw determinant
        R = pd.random_rotation(5, rng)
        assert departure(R) < 1e-13, "not orthogonal"
        assert abs(np.linalg.det(R) - 1.0) < 1e-12, f"determinant is {np.linalg.det(R)}"


def test_random_rotation_is_deterministic():
    a = pd.random_rotation(4, np.random.default_rng(7))
    b = pd.random_rotation(4, np.random.default_rng(7))
    assert np.array_equal(a, b), "same rng state should give the same rotation"


# ---- problem 2: make_problem ----

def test_make_problem_shapes_and_noise_free_case():
    X, Y, W_true = pd.make_problem(4, 10, 0.0, seed=1)
    assert X.shape == (4, 10) and Y.shape == (4, 10) and W_true.shape == (4, 4), "wrong shapes"
    assert np.allclose(Y, W_true @ X, atol=1e-13), "with sigma = 0, Y should equal W_true X"
    assert abs(np.linalg.det(W_true) - 1.0) < 1e-12, "W_true should be a rotation"


def test_make_problem_noise_and_determinism():
    a = pd.make_problem(3, 8, 0.5, seed=2)
    b = pd.make_problem(3, 8, 0.5, seed=2)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)), "same arguments should repeat"
    X, Y, W_true = a
    assert not np.allclose(Y, W_true @ X), "with sigma > 0, Y should differ from W_true X"


def test_make_problem_rejects_m_less_than_n():
    try:
        pd.make_problem(5, 3, 0.1, seed=0)
    except ValueError:
        return
    raise AssertionError("m < n should raise ValueError")


# ---- problem 3: objective ----

def test_objective_hand_checked():
    X = np.eye(2)
    Y = np.array([[1.0, 1.0], [0.0, 0.0]])
    # W X - Y = [[0, -1], [0, 1]], whose squared Frobenius norm is 2.
    got = pd.objective(np.eye(2), X, Y)
    assert isinstance(got, float), "return a Python float"
    assert abs(got - 2.0) < 1e-14, f"got {got}, wanted 2"
    assert pd.objective(np.eye(2), np.eye(2), np.eye(2)) < 1e-15, "a perfect fit should give 0"


# ---- problem 4: gradient ----

def test_gradient_hand_checked():
    X = np.eye(2)
    Y = np.array([[1.0, 1.0], [0.0, 0.0]])
    want = np.array([[0.0, -2.0], [0.0, 2.0]])  # 2 (W X - Y) X^T with W = I
    got = pd.gradient(np.eye(2), X, Y)
    assert np.allclose(got, want, atol=1e-14), f"got\n{got}"


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    n, m = 3, 5
    W = rng.standard_normal((n, n))
    X = rng.standard_normal((n, m))
    Y = rng.standard_normal((n, m))
    G = pd.gradient(W, X, Y)
    eps = 1e-6
    for i in range(n):
        for j in range(n):
            bump = np.zeros((n, n))
            bump[i, j] = eps
            numeric = (pd.objective(W + bump, X, Y) - pd.objective(W - bump, X, Y)) / (2 * eps)
            assert abs(G[i, j] - numeric) < 1e-6, f"entry ({i}, {j}): {G[i, j]} vs {numeric}"


# ---- problem 5: kabsch ----

def test_kabsch_recovers_the_rotation_without_noise():
    X, Y, W_true = pd.make_problem(4, 10, 0.0, seed=4)
    got = pd.kabsch(X, Y)
    assert np.allclose(got, W_true, atol=1e-10), "should recover W_true when there is no noise"


def test_kabsch_is_optimal_among_rotations():
    X, Y, W_true = pd.make_problem(4, 12, 0.3, seed=5)
    best = pd.kabsch(X, Y)
    assert departure(best) < 1e-12 and abs(np.linalg.det(best) - 1.0) < 1e-10, "not a rotation"
    rng = np.random.default_rng(55)
    for _ in range(500):
        R = pd.random_rotation(4, rng)
        assert loss(best, X, Y) <= loss(R, X, Y) + 1e-12, "a random rotation beat W*"
    assert loss(best, X, Y) <= loss(W_true, X, Y) + 1e-12, "W_true beat W*"


def test_kabsch_returns_a_rotation_when_the_best_orthogonal_matrix_is_a_reflection():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((3, 8))
    Y = reflection_flip(3) @ X  # the best orthogonal fit is a reflection
    best = pd.kabsch(X, Y)
    assert abs(np.linalg.det(best) - 1.0) < 1e-10, f"determinant is {np.linalg.det(best)}, not +1"
    assert departure(best) < 1e-12, "not orthogonal"
    assert loss(best, X, Y) > 1e-3, "a rotation cannot fit reflected data exactly"
    for _ in range(500):
        R = pd.random_rotation(3, rng)
        assert loss(best, X, Y) <= loss(R, X, Y) + 1e-12, "a random rotation beat W*"


def test_kabsch_solution_is_stationary():
    # At a minimum over rotations the skew part of W^T G vanishes.
    for seed, sigma in ((8, 0.3), (9, 0.0)):
        X, Y, _ = pd.make_problem(4, 12, sigma, seed=seed)
        best = pd.kabsch(X, Y)
        S = best.T @ pd.gradient(best, X, Y)
        assert np.allclose(S, S.T, atol=1e-8), "W^T G should be symmetric at the optimum"


# ---- problem 6: write_problem ----

def test_write_problem_file_names_and_contents():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "new" / "dir"  # does not exist yet
        paths = pd.write_problem(target, 3, 5, 0.1, seed=2)
        assert set(paths) == {"X", "Y", "W0", "Wstar"}, f"keys are {set(paths)}"
        assert paths["X"].name == "X_n3_m5.bin"
        assert paths["Y"].name == "Y_n3_m5.bin"
        assert paths["W0"].name == "W0_n3.bin"
        assert paths["Wstar"].name == "Wstar_n3_m5.bin"
        X, Y, _ = pd.make_problem(3, 5, 0.1, seed=2)
        assert paths["X"].stat().st_size == 3 * 5 * 8, "X should hold 15 doubles"
        assert np.array_equal(np.fromfile(paths["X"], dtype=np.float64), X.ravel()), "X contents"
        assert np.array_equal(np.fromfile(paths["Y"], dtype=np.float64), Y.ravel()), "Y contents"
        assert np.array_equal(np.fromfile(paths["W0"], dtype=np.float64), np.eye(3).ravel()), "W0 is I"
        assert np.array_equal(np.fromfile(paths["Wstar"], dtype=np.float64), pd.kabsch(X, Y).ravel()), "W*"


# ---- problem 7: read_matrix ----

def test_read_matrix_round_trip_and_size_check():
    with tempfile.TemporaryDirectory() as tmp:
        paths = pd.write_problem(Path(tmp), 3, 5, 0.1, seed=3)
        X, _, _ = pd.make_problem(3, 5, 0.1, seed=3)
        got = pd.read_matrix(paths["X"], 3, 5)
        assert got.shape == (3, 5) and got.dtype == np.float64, "wrong shape or dtype"
        assert np.array_equal(got, X), "X did not survive the round trip"
        try:
            pd.read_matrix(paths["X"], 4, 5)
        except ValueError:
            return
        raise AssertionError("reading 15 doubles as 4 x 5 should raise ValueError")


# ---- runner ----

def run_all():
    failed = 0
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        print(name)
        try:
            fn()
            print("  ok")
        except NotImplementedError as exc:
            print(f"  not implemented yet ({exc})")
            failed += 1
        except AssertionError as exc:
            print(f"  FAIL: {exc}")
            failed += 1
        except Exception as exc:  # a crash is a failure too
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{len(tests) - failed} of {len(tests)} tests passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)
