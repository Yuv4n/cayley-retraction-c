# Validation

Nothing in this file has been run. It fixes the method before any result exists, so that the results cannot shape it.

## Reference and fixtures

`reference/cayley_numpy.py` is a NumPy version of the update written in this repository from the formulas in the README. It uses a linear solver and not an explicit inverse, and the benchmark states which it is.

`reference/make_fixtures.py` builds W from the QR factorisation of a seeded Gaussian matrix, draws G from the same seed, runs the reference and writes W, G and W⁺. The proposed format is raw float64 arrays written with NumPy's `tofile`, with the size in each filename, read in C with `fread`. The C reader checks that `fread` returns the expected count. Test sizes are n = 3, small enough to print and check by hand, and n = 128.

Before any C is compared with it, the reference is checked in NumPy: W⁺ is orthogonal, and for small η it matches the first-order expansion W − ηAW.

## Correctness tests

1. Form M. M − I is exactly skew, entry for entry, and M agrees with NumPy.
2. Build B. B agrees with the reference's right-hand side.
3. Solve. The norm of M W⁺ − B is at rounding level. This needs no reference, so it catches an error the C and NumPy code might share. One test uses a system that forces row swaps.
4. Full step. W⁺ agrees with the reference, and ‖W⁺ᵀW⁺ − I‖_F is at rounding level.
5. Failure handling. NaN or infinity in G or W produces an error return.
6. Allocation. A step makes no allocations once the workspace exists. The way to test this is undecided.

Tests run under AddressSanitizer and UndefinedBehaviorSanitizer. No tolerance is fixed yet. Each is set from the observed run-to-run behaviour of the reference and recorded here with its reasoning before any claim of agreement is made.

## Demonstration: orthogonal Procrustes

Given X and Y of size n × m with m ≥ n, minimise f(W) = ‖WX − Y‖²_F over orthogonal W. The Euclidean gradient is G = 2(WX − Y)Xᵀ.

`examples/procrustes_data.py` draws X with Gaussian entries, takes a rotation W_true with determinant +1 from the QR factorisation of a Gaussian matrix with the signs fixed, and sets Y = W_true X + σE with Gaussian noise E. It then computes the SVD YXᵀ = UΣVᵀ. Over all orthogonal matrices the minimiser is UVᵀ. The update preserves the determinant, so from a rotation it can only reach rotations, and the target is

W* = U diag(1, …, 1, det(UVᵀ)) Vᵀ.

The script writes X, Y, the starting W₀ (the identity is acceptable, as its determinant is +1) and W*. The SVD is computed in NumPy only.

`examples/procrustes.c` runs the iterations. Every few iterations it logs f(W), ‖W − W*‖_F and ‖WᵀW − I‖_F. The run succeeds if f falls to f(W*), ‖W − W*‖_F becomes small, and the orthogonality error stays at rounding level throughout. The step size η is chosen by trying a few values, and the values tried and the one chosen are recorded here.

## Benchmark

Both implementations time one full step, from W and G to W⁺, on the same fixture inputs. Sizes of 64, 128, 256, 512 and 1024 are proposed and may change once early timings show where the interesting range lies.

NumPy's solver calls LAPACK and its products call BLAS. The first C version calls neither, so the result is a comparison of hand-written code with an optimised library and is reported that way. NumPy is timed twice, with its default thread count and with one thread (`VECLIB_MAXIMUM_THREADS=1` for Accelerate on macOS). The BLAS build, thread settings, compiler, flags and machine are recorded with every run.

For each size:

- A few untimed warm-up steps come first.
- The number of timed repetitions is fixed in advance and large enough for the smallest size to time reliably.
- The median and the spread are reported, never the best run.
- Workspace creation is timed separately from the step.

Both drivers write plain text to `bench/results/`. Results are committed only after they have been measured, together with the command that produced them.
