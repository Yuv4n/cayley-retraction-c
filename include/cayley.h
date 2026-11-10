/*
 * cayley.h : the Cayley retraction step, set out as five problems.
 *
 * Work through them in order. Each one has a test in tests/test_cayley.c, and
 * each test fails until you have done that problem. The signatures below are a
 * proposal. Change them if you prefer, and update the tests to match.
 *
 * All matrices are n x n, double precision, row-major: element (i, j) of a
 * matrix X lives at X[i * n + j].
 *
 * The maths is in docs/derivation.md and the reasoning behind the memory
 * decisions is in docs/design.md.
 */

#ifndef CAYLEY_H
#define CAYLEY_H

#include <stddef.h>

typedef enum {
    CAYLEY_OK = 0,
    CAYLEY_ERR_SINGULAR = 1,  /* a pivot was exactly zero */
    CAYLEY_ERR_NONFINITE = 2, /* NaN or infinity in the input */
    CAYLEY_TODO = 99          /* returned by the unfinished stubs; delete once unused */
} CayleyStatus;

typedef struct CayleyWorkspace CayleyWorkspace;

/*
 * Problem 1: the workspace
 *
 * Allocate everything one step needs, once. That is two n x n buffers of
 * doubles (call them M and B) and n pivot indices (size_t). Define the struct
 * in cayley.c.
 *
 * cayley_create returns NULL if n is 0, if the buffer size would overflow
 * size_t, or if any allocation fails. If it fails part-way, free what it has
 * already allocated before returning.
 *
 * cayley_free releases everything. Passing NULL does nothing.
 *
 * Think about: what happens if the second allocation fails? Is n * n * 8
 * computed before or after you check it fits?
 */
CayleyWorkspace *cayley_create(size_t n);
void cayley_free(CayleyWorkspace *ws);

/*
 * Problem 2: form M
 *
 * Write M = I + (eta / 2) * A into M, where A = G W^T - W G^T.
 *
 * A is skew-symmetric, so compute only the entries with i < j, using
 *
 *     A[i][j] = sum over k of (G[i][k] * W[j][k] - W[i][k] * G[j][k])
 *
 * Both reads walk along a row. Set the mirrored entry by negation, and set the
 * diagonal of M to exactly 1. Every entry of M must be written; do not assume
 * it starts at zero. M does not overlap W or G.
 *
 * Think about: why does computing each pair once make M - I exactly skew?
 */
void cayley_form_m(size_t n, const double *W, const double *G, double eta,
                   double *M);

/*
 * Problem 3: build the right-hand side
 *
 * Write B = 2W - M W into B, where M is the matrix from problem 2.
 *
 * (Why does this equal (I - (eta / 2) A) W? See docs/design.md.)
 *
 * The inner loop must walk along a row of W and a row of B. A loop that walks
 * down a column of a row-major matrix is the thing to avoid. Write every entry
 * of B. B does not overlap M or W.
 *
 * Think about: of the six orderings of the i, j and k loops, which keep the
 * innermost access contiguous for all three matrices?
 */
void cayley_build_b(size_t n, const double *M, const double *W, double *B);

/*
 * Problem 4: solve M X = B
 *
 * Gaussian elimination with partial pivoting, in place. On return B holds X
 * and M has been destroyed.
 *
 * Rules:
 *   - No inverse, and no allocation.
 *   - All n right-hand sides are handled together. Every row swap, every
 *     elimination update and every back-substitution step acts on whole rows
 *     of M and B. Never solve one column of B at a time.
 *   - At step k, choose the row p >= k whose entry in column k has the largest
 *     absolute value (lowest index on a tie), swap rows p and k of both M and
 *     B, and record p in piv[k]. piv[k] == k means no swap.
 *   - If the chosen pivot is exactly zero, return CAYLEY_ERR_SINGULAR. The
 *     contents of M, B and piv are then unspecified.
 *   - Otherwise return CAYLEY_OK.
 *
 * Think about: how do you apply the multiplier to B? What does back
 * substitution look like when the unknowns are whole rows?
 */
CayleyStatus cayley_solve(size_t n, double *M, double *B, size_t *piv);

/*
 * Problem 5: the full step
 *
 * Replace W by W+ = (I + (eta/2) A)^-1 (I - (eta/2) A) W, using the workspace
 * for all scratch memory. G is the gradient. W and G are n x n, where n is the
 * size the workspace was created with.
 *
 * Rules:
 *   - If any entry of W or G, or eta itself, is NaN or infinite, return
 *     CAYLEY_ERR_NONFINITE. Check before doing any work.
 *   - On any error return, W is left exactly as it was. Only copy the result
 *     into W once the solve has succeeded.
 *   - A step never calls malloc or free.
 *   - A workspace can be used for any number of steps.
 *
 * This is where problems 2 to 4 are put together. It should be short.
 */
CayleyStatus cayley_step(CayleyWorkspace *ws, double *W, const double *G,
                         double eta);

#endif
