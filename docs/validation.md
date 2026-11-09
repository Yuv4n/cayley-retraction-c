# Validation

Nothing in this file has been run. It fixes the method before any result exists, so that the results cannot shape it.

## Correctness

`reference/cayley_numpy.py` will hold the update exactly as it appears in the RNN repository (LINK-TO-RNN-REPO), with the commit noted. `reference/make_fixtures.py` will build W from the QR factorisation of a seeded random matrix, draw G from the same seed, run the reference and write inputs and outputs to files that the C tests read. The file format is undecided. Where the reference differs from the documentation on the transpose convention or the sign of η, the reference wins and the documentation is corrected.

The tests will check:

1. Agreement with the reference. The largest absolute difference between the C and NumPy W⁺ at each size.
2. The residual of the solve. The norm of M W⁺ − B, which needs no reference and so catches an error that the C code and the NumPy code might share.
3. Orthogonality after one step. The Frobenius norm of W⁺ᵀW⁺ − I.
4. Drift. The same norm after many steps with fresh gradients, to see how quickly rounding error accumulates.
5. Failure handling. A gradient containing NaN or infinity produces an error return.
6. Allocation. A step makes no allocations once the workspace exists. The way to test this is undecided.

No tolerance is fixed yet. Each will be set from the observed run-to-run behaviour of the reference and recorded here with its reasoning before any claim of agreement is made.

## Benchmark

Both implementations time one full step, from W and G to W⁺, on the same fixture inputs. Hidden sizes of 64, 128, 256, 512 and 1024 are proposed and may change once early timings show where the interesting range lies.

NumPy's solver calls LAPACK and its products call BLAS. The first C version will not, so the result is a comparison of hand-written code with an optimised library and will be reported that way. NumPy will be timed with its default thread count and again restricted to one thread. The BLAS build, thread settings, compiler, flags and machine are recorded with every run. The reference may use an explicit inverse or a solver, and the benchmark notes which.

For each size:

- A few untimed warm-up steps come first.
- The number of timed repetitions is fixed in advance and large enough for the smallest size to time reliably.
- The median and the spread are reported, not the best run.
- Workspace creation is timed separately from the step.

Both drivers write plain text to `bench/results/`. Results are committed only after they have been measured, together with the command that produced them.
