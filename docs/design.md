# Design

This is the plan for code that does not exist yet. The decisions in the first four sections are requirements of the project. The last section lists what is still open.

## Memory access

Storage is row-major and contiguous, which is also NumPy's default, so fixtures can be read without transposing. Every loop in the design is arranged so that its inner loop walks along a row. A column walk on a row-major matrix jumps n doubles at a time and makes poor use of the cache, and avoiding that is the main hardware-aware decision in the project.

## Forming M

Let K = (η/2)A and M = I + K. A is skew-symmetric, so only half of it needs computing. For i < j,

A[i][j] = Σₖ (G[i][k] W[j][k] − W[i][k] G[j][k]),

which reads two rows of G and two rows of W, all contiguous. A[j][i] is the negative of A[i][j] and the diagonal is zero. The scaled values, with 1 added on the diagonal, are written straight into the buffer for M, so no separate A and no product matrix is stored. This halves the arithmetic for A and leaves M − I exactly skew, entry for entry, because each pair is computed once and negated.

## Solving the system

The update needs M⁻¹B with B = (I − K)W. Since I − K = 2I − M, B = 2W − MW, so M is enough to build it. The product MW is formed with a loop order that keeps the inner loop along rows.

The inverse is never formed. The system M W⁺ = B is solved by Gaussian elimination with partial pivoting. B is n × n and row-major, and its columns are the n right-hand sides. All of them are processed together, along rows:

- A row swap exchanges two rows of M and the same two rows of B.
- Eliminating below a pivot subtracts a multiple of the pivot row from a lower row, across the remaining columns of M and across the whole row of B.
- Back substitution takes each row of B in turn from the bottom, subtracts the multiples of the rows already solved, and divides by the diagonal entry.

Every access is along a row. Solving one column at a time would read B with stride n and is ruled out.

The cost is cubic in n, the same order as forming an explicit inverse, with a smaller constant and better rounding behaviour.

## Allocation and failure

A workspace is created for a given n before the first step, reused for every step and freed at the end. Expected contents: two n × n buffers of doubles (M and B) and n pivot indices. The solution overwrites B and is then copied into W or swapped with it. A step never calls the allocator. The Procrustes driver allocates its own buffers, for WX − Y and for G, once outside its loop, under the same rule.

The step returns a status. A zero pivot, or a non-finite value in G or W, is reported to the caller and not passed on silently. A zero pivot cannot occur in exact arithmetic for the reasons in [derivation.md](derivation.md), but non-finite input can still produce one.

## Numerical choices

- Precision is double, matching NumPy's float64 default.
- The first version is single-threaded C with no BLAS or LAPACK. The comparison with NumPy is therefore between hand-written code and a tuned library, a point [validation.md](validation.md) returns to.

## Interface

Three operations: create a workspace for a given n, apply one update to W in place given G and η, and free the workspace. The shape is settled and the signatures are open.

## Open questions

- Whether partial pivoting is worth its cost, given that M needs no pivoting in exact arithmetic. Measure with and without.
- How to test that a step makes no allocation.
- The exact function signatures.
- The step size and iteration count for Procrustes.
