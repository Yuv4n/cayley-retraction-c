# Design

This is the plan for code that does not exist yet. Where a choice is still open, the text says so.

## Solving the system

Let M = I + (η/2)A. Each step factorises M as PM = LU by Gaussian elimination with partial pivoting, then solves M W⁺ = B one column at a time by forward and back substitution, where B = (I − (η/2)A)W. The cost is cubic in n, the same order as an explicit inverse, with a smaller constant and better rounding behaviour.

The symmetric part of M is I, so every leading principal submatrix of M is nonsingular and elimination would never meet a zero pivot even without pivoting. Partial pivoting stays in for its effect on rounding. Whether it is worth its cost is something the benchmark can answer, so the choice is provisional.

## Memory

Two observations keep the workspace small.

1. A can be formed entry by entry, A[i][j] = Σₖ (G[i][k] W[j][k] − W[i][k] G[j][k]), straight into the buffer that will hold M. No product matrix is needed.
2. Since I − (η/2)A = 2I − M, the right-hand side is B = 2W − MW, so M is enough to build it.

That gives two n × n buffers of doubles (M and B) and an array of n pivot indices. The solution is written over B, then copied into W or swapped with it. This is the expected layout and will be confirmed when the step is written.

The workspace is created once for a given n, reused for every step and freed once at the end. A step never calls the allocator.

## Interface

Three operations: create a workspace for a given n, apply one step to W in place given G and η, and free the workspace. The step returns a status, so a zero pivot or a non-finite input is reported to the caller and not silently propagated. A zero pivot cannot occur in exact arithmetic, but a gradient containing NaN or infinity can still produce one. The shape is settled and the signatures are not.

## Numerical choices

- Precision is double. NumPy defaults to float64, but the dtype used in the original RNN code has to be checked before this is final.
- Storage is row-major and contiguous, which is NumPy's default layout, so fixtures can be read without transposing.
- The first version is single-threaded C with no BLAS. The comparison with NumPy is therefore between hand-written elimination and a tuned library, a point [validation.md](validation.md) returns to.
