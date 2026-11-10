"""
Tests for reference/cayley_numpy.py and reference/make_fixtures.py.

Run from the repository root:

    python3 tests/test_reference.py

pytest also works if you have it, since every test is a function named test_*.
Every test starts out failing, because the functions are stubs. Work through
the problems in each file in order.

Tolerances are provisional. Tighten them once you know how the code behaves and
record the reasoning in docs/validation.md.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference"))

import cayley_numpy as ref  # noqa: E402
import make_fixtures as fx  # noqa: E402


# ---- helpers (the tests' own checks, independent of the code under test) ----

def orthogonal(n, seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def departure(W):
    return float(np.linalg.norm(W.T @ W - np.eye(W.shape[0])))


# ---- cayley_numpy.py, problem 1: form_a ----

def test_form_a_hand_checked():
    W = np.eye(3)
    G = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 10]])
    want = np.array([[0.0, -2, -4], [2, 0, -2], [4, 2, 0]])
    A = ref.form_a(W, G)
    assert np.allclose(A, want, atol=1e-14, rtol=0), f"A is\n{A}\nwanted\n{want}"


def test_form_a_is_skew_for_any_w():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((5, 5))  # deliberately not orthogonal
    G = rng.standard_normal((5, 5))
    A = ref.form_a(W, G)
    assert A.shape == (5, 5), "A has the wrong shape"
    assert np.allclose(A + A.T, 0, atol=1e-14), "A is not skew-symmetric"


def test_form_a_does_not_modify_inputs():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((4, 4))
    G = rng.standard_normal((4, 4))
    W0, G0 = W.copy(), G.copy()
    ref.form_a(W, G)
    assert np.array_equal(W, W0) and np.array_equal(G, G0), "an input was modified"


# ---- cayley_numpy.py, problem 2: orthogonality_error ----

def test_orthogonality_error_zero_for_orthogonal():
    assert ref.orthogonality_error(np.eye(4)) < 1e-15, "identity should give 0"
    assert ref.orthogonality_error(orthogonal(6, 3)) < 1e-13, "orthogonal W should give ~0"


def test_orthogonality_error_hand_checked():
    # W = 2I gives W^T W - I = 3I, whose Frobenius norm is sqrt(3 * 9).
    got = ref.orthogonality_error(2 * np.eye(3))
    assert isinstance(got, float), "return a Python float"
    assert abs(got - np.sqrt(27.0)) < 1e-12, f"got {got}, wanted {np.sqrt(27.0)}"


# ---- cayley_numpy.py, problem 3: cayley_step ----

def test_cayley_step_2x2_closed_form():
    # For W = I, A = s J with s = G01 - G10. With t = eta * s / 2 the answer is a
    # rotation. Here s = 1 and eta = 1, so t = 0.5 and W+ = [[0.6, -0.8], [0.8, 0.6]].
    W = np.eye(2)
    G = np.array([[1.0, 1], [0, 3]])
    want = np.array([[0.6, -0.8], [0.8, 0.6]])
    got = ref.cayley_step(W, G, 1.0)
    assert np.allclose(got, want, atol=1e-14, rtol=0), f"got\n{got}"


def test_cayley_step_zero_gradient_leaves_w_unchanged():
    W = np.random.default_rng(4).standard_normal((4, 4))
    got = ref.cayley_step(W, np.zeros((4, 4)), 0.3)
    assert np.allclose(got, W, atol=1e-14, rtol=0), "W changed although G = 0"


def test_cayley_step_n_equals_one():
    got = ref.cayley_step(np.array([[1.0]]), np.array([[5.0]]), 0.2)
    assert np.allclose(got, [[1.0]]), "for n = 1, A = 0 and W must not change"


def test_cayley_step_stays_orthogonal():
    n = 16
    rng = np.random.default_rng(5)
    W = orthogonal(n, 5)
    for _ in range(10):
        W = ref.cayley_step(W, rng.standard_normal((n, n)), 0.5)
        assert departure(W) < 1e-12, f"orthogonality lost: {departure(W)}"


def test_cayley_step_preserves_determinant():
    W = orthogonal(6, 6)
    G = np.random.default_rng(6).standard_normal((6, 6))
    got = ref.cayley_step(W, G, 0.4)
    assert np.isclose(np.linalg.det(got), np.linalg.det(W), rtol=1e-10), "determinant changed"


def test_cayley_step_does_not_modify_inputs():
    rng = np.random.default_rng(7)
    W = orthogonal(4, 7)
    G = rng.standard_normal((4, 4))
    W0, G0 = W.copy(), G.copy()
    got = ref.cayley_step(W, G, 0.2)
    assert np.array_equal(W, W0) and np.array_equal(G, G0), "an input was modified"
    assert got is not W, "return a new array"


def test_cayley_step_does_not_call_inv():
    original = np.linalg.inv

    def forbidden(*args, **kwargs):
        raise AssertionError("np.linalg.inv was called")

    np.linalg.inv = forbidden
    try:
        rng = np.random.default_rng(8)
        ref.cayley_step(orthogonal(4, 8), rng.standard_normal((4, 4)), 0.2)
    finally:
        np.linalg.inv = original


# ---- cayley_numpy.py, problem 4: first_order_step ----

def test_first_order_step_hand_checked():
    # For orthogonal W, A W = G - W G^T W.
    W = orthogonal(5, 9)
    G = np.random.default_rng(9).standard_normal((5, 5))
    eta = 0.01
    want = W - eta * (G - W @ G.T @ W)
    got = ref.first_order_step(W, G, eta)
    assert np.allclose(got, want, atol=1e-13), "does not match W - eta * (G - W G^T W)"


def test_cayley_step_agrees_with_first_order_to_second_order():
    W = orthogonal(5, 10)
    G = np.random.default_rng(10).standard_normal((5, 5))

    def gap(eta):
        return np.linalg.norm(ref.cayley_step(W, G, eta) - ref.first_order_step(W, G, eta))

    ratio = gap(0.01) / gap(0.005)
    assert 3.5 < ratio < 4.5, f"halving eta should cut the gap by about 4, got {ratio}"


# ---- make_fixtures.py, problem 1: random_orthogonal ----

def test_random_orthogonal_is_orthogonal():
    Q = fx.random_orthogonal(10, np.random.default_rng(0))
    assert Q.shape == (10, 10), "wrong shape"
    assert Q.dtype == np.float64, "use float64"
    assert Q.flags["C_CONTIGUOUS"], "return a C-contiguous array"
    assert departure(Q) < 1e-13, f"not orthogonal: {departure(Q)}"


def test_random_orthogonal_is_deterministic():
    a = fx.random_orthogonal(6, np.random.default_rng(3))
    b = fx.random_orthogonal(6, np.random.default_rng(3))
    c = fx.random_orthogonal(6, np.random.default_rng(4))
    assert np.array_equal(a, b), "same rng state should give the same matrix"
    assert not np.array_equal(a, c), "different seeds should give different matrices"


# ---- make_fixtures.py, problem 2: make_case ----

def test_make_case_consistent_with_reference():
    W, G, Wp = fx.make_case(5, seed=0)
    assert W.shape == G.shape == Wp.shape == (5, 5), "wrong shapes"
    assert departure(W) < 1e-13, "W should be orthogonal"
    assert np.array_equal(Wp, ref.cayley_step(W, G, fx.FIXTURE_ETA)), "W_plus is not the reference output"


def test_make_case_is_deterministic():
    a = fx.make_case(4, seed=2)
    b = fx.make_case(4, seed=2)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)), "same n and seed should repeat"


# ---- make_fixtures.py, problem 3: fixture_paths ----

def test_fixture_paths_names():
    paths = fx.fixture_paths(Path("somewhere"), 3)
    assert set(paths) == {"W", "G", "Wplus"}, f"keys are {set(paths)}"
    assert paths["W"] == Path("somewhere") / "W_n3.bin"
    assert paths["G"] == Path("somewhere") / "G_n3.bin"
    assert paths["Wplus"] == Path("somewhere") / "Wplus_n3.bin"
    assert not Path("somewhere").exists(), "fixture_paths must not touch the file system"


# ---- make_fixtures.py, problem 4: write_case ----

def test_write_case_file_contents():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "new" / "dir"  # does not exist yet
        paths = fx.write_case(target, 3, seed=0)
        W, G, Wp = fx.make_case(3, seed=0)
        for key, want in (("W", W), ("G", G), ("Wplus", Wp)):
            assert paths[key].stat().st_size == 3 * 3 * 8, f"{key}: file should hold 9 doubles"
            raw = np.fromfile(paths[key], dtype=np.float64)
            assert np.array_equal(raw, want.ravel()), f"{key}: contents are not the row-major array"


# ---- make_fixtures.py, problem 5: read_matrix ----

def test_read_matrix_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        paths = fx.write_case(Path(tmp), 4, seed=1)
        W, G, Wp = fx.make_case(4, seed=1)
        got = fx.read_matrix(paths["W"], 4)
        assert got.shape == (4, 4) and got.dtype == np.float64, "wrong shape or dtype"
        assert np.array_equal(got, W), "W did not survive the round trip"
        assert np.array_equal(fx.read_matrix(paths["Wplus"], 4), Wp), "W_plus did not survive"


def test_read_matrix_rejects_wrong_size():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "short.bin"
        np.arange(5, dtype=np.float64).tofile(path)
        try:
            fx.read_matrix(path, 3)
        except ValueError:
            return
        raise AssertionError("a file with 5 doubles should not read as 3 x 3")


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
