/*
 * Tests for the five problems in include/cayley.h.
 *
 * Build and run from the repository root:
 *
 *     mkdir -p build
 *     clang -Wall -Wextra -g -fsanitize=address,undefined -I include \
 *         src/cayley.c tests/test_cayley.c -o build/test_cayley -lm
 *     ./build/test_cayley
 *
 * Every test starts out failing, because the functions in src/cayley.c are
 * stubs. Work through the problems in order and watch the tests turn to ok.
 * The program exits with a non-zero status if any test fails.
 *
 * Tolerances are provisional. When you know how the code really behaves,
 * tighten them and record the reasoning in docs/validation.md.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cayley.h"

#define TOL 1e-12

static int g_failed;

static void expect(int ok, const char *what)
{
    if (!ok) {
        printf("    FAIL: %s\n", what);
        g_failed++;
    }
}

/* ---- helpers ---------------------------------------------------------- */

static uint64_t rng_state;

/* Deterministic uniform values in [-1, 1). */
static double rng_next(void)
{
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(rng_state >> 11) * (2.0 / 9007199254740992.0) - 1.0;
}

static double *new_matrix(size_t n)
{
    double *p = calloc(n * n, sizeof *p);
    if (p == NULL) {
        fputs("test harness: out of memory\n", stderr);
        exit(2);
    }
    return p;
}

static void fill_random(double *a, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        a[i] = rng_next();
    }
}

static void fill_nan(double *a, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        a[i] = NAN;
    }
}

/* Largest |a - b|, or infinity if any difference is NaN. */
static double max_abs_diff(const double *a, const double *b, size_t count)
{
    double worst = 0.0;
    for (size_t i = 0; i < count; i++) {
        double d = fabs(a[i] - b[i]);
        if (isnan(d)) {
            return INFINITY;
        }
        if (d > worst) {
            worst = d;
        }
    }
    return worst;
}

/* C = A * B. Slow and obvious; this is the test's own check, not the library. */
static void matmul(size_t n, const double *A, const double *B, double *C)
{
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/* Frobenius norm of W^T W - I. */
static double orthogonality_error(size_t n, const double *W)
{
    double total = 0.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double dot = 0.0;
            for (size_t k = 0; k < n; k++) {
                dot += W[k * n + i] * W[k * n + j];
            }
            double d = dot - (i == j ? 1.0 : 0.0);
            total += d * d;
        }
    }
    return sqrt(total);
}

/* A random matrix with a large diagonal, so it is safely nonsingular. */
static void fill_dominant(size_t n, double *M)
{
    fill_random(M, n * n);
    for (size_t i = 0; i < n; i++) {
        M[i * n + i] += (double)n;
    }
}

/* ---- problem 1: workspace --------------------------------------------- */

static void test_create_free(void)
{
    CayleyWorkspace *ws = cayley_create(3);
    expect(ws != NULL, "cayley_create(3) returns a workspace");
    cayley_free(ws);

    ws = cayley_create(1);
    expect(ws != NULL, "cayley_create(1) returns a workspace");
    cayley_free(ws);

    cayley_free(NULL); /* must not crash */

    expect(cayley_create(0) == NULL, "cayley_create(0) returns NULL");

    for (int i = 0; i < 1000; i++) { /* run under the sanitiser to catch misuse */
        cayley_free(cayley_create(16));
    }
}

/* Kept last in the list: a missing overflow check may crash the program. */
static void test_create_overflow(void)
{
    CayleyWorkspace *small = cayley_create(2);
    expect(small != NULL, "a small workspace can be created (problem 1 first)");
    cayley_free(small);

    CayleyWorkspace *ws = cayley_create(SIZE_MAX / 2);
    expect(ws == NULL, "an n whose buffers overflow size_t returns NULL");
    cayley_free(ws);
}

/* ---- problem 2: form M ------------------------------------------------ */

static void test_form_m_hand_checked(void)
{
    /* W = I, so A = G - G^T. With eta = 0.5, M = I + 0.25 * (G - G^T). */
    const double W[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double G[9] = {1, 2, 3, 4, 5, 6, 7, 8, 10};
    const double want[9] = {1,   -0.5, -1,
                            0.5, 1,    -0.5,
                            1,   0.5,  1};
    double M[9];
    fill_nan(M, 9);

    cayley_form_m(3, W, G, 0.5, M);

    expect(max_abs_diff(M, want, 9) == 0.0, "M matches the hand-worked 3 x 3 case");
}

static void test_form_m_skew_exact(void)
{
    const size_t n = 5;
    const double eta = 0.37;
    double *W = new_matrix(n);
    double *G = new_matrix(n);
    double *M = new_matrix(n);
    double *want = new_matrix(n);

    rng_state = 11;
    fill_random(W, n * n); /* W need not be orthogonal for this check */
    fill_random(G, n * n);
    fill_nan(M, n * n);

    cayley_form_m(n, W, G, eta, M);

    for (size_t i = 0; i < n; i++) {
        expect(M[i * n + i] == 1.0, "diagonal of M is exactly 1");
        for (size_t j = i + 1; j < n; j++) {
            expect(M[i * n + j] == -M[j * n + i],
                   "M - I is exactly skew, entry for entry");
        }
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double a = 0.0;
            for (size_t k = 0; k < n; k++) {
                a += G[i * n + k] * W[j * n + k] - W[i * n + k] * G[j * n + k];
            }
            want[i * n + j] = (i == j ? 1.0 : 0.0) + 0.5 * eta * a;
        }
    }
    expect(max_abs_diff(M, want, n * n) < TOL, "M agrees with a direct computation");

    free(W);
    free(G);
    free(M);
    free(want);
}

/* ---- problem 3: right-hand side --------------------------------------- */

static void test_build_b_hand_checked(void)
{
    const double M[9] = {1,   -0.5, -1,
                         0.5, 1,    -0.5,
                         1,   0.5,  1};
    const double I3[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double want_i[9] = {1,  0.5,  1,
                              -0.5, 1,  0.5,
                              -1, -0.5, 1}; /* 2I - M */
    const double W[9] = {1, 2, 0, 0, 1, 3, 4, 0, 1};
    const double want_w[9] = {5,   2.5,  2.5,
                              1.5, 0,    3.5,
                              3,   -2.5, -0.5};
    double B[9];

    fill_nan(B, 9);
    cayley_build_b(3, M, I3, B);
    expect(max_abs_diff(B, want_i, 9) == 0.0, "B = 2I - M when W = I");

    fill_nan(B, 9);
    cayley_build_b(3, M, W, B);
    expect(max_abs_diff(B, want_w, 9) == 0.0, "B matches the hand-worked 3 x 3 case");
}

static void test_build_b_identity_m(void)
{
    const size_t n = 4;
    double *M = new_matrix(n);
    double *W = new_matrix(n);
    double *B = new_matrix(n);

    rng_state = 21;
    for (size_t i = 0; i < n; i++) {
        M[i * n + i] = 1.0;
    }
    fill_random(W, n * n);
    fill_nan(B, n * n);

    cayley_build_b(n, M, W, B);

    expect(max_abs_diff(B, W, n * n) < TOL, "B = W when M = I");

    free(M);
    free(W);
    free(B);
}

static void test_build_b_random(void)
{
    const size_t n = 6;
    double *M = new_matrix(n);
    double *W = new_matrix(n);
    double *B = new_matrix(n);
    double *MW = new_matrix(n);
    double *want = new_matrix(n);

    rng_state = 31;
    fill_random(M, n * n);
    fill_random(W, n * n);
    fill_nan(B, n * n);
    matmul(n, M, W, MW);
    for (size_t i = 0; i < n * n; i++) {
        want[i] = 2.0 * W[i] - MW[i];
    }

    cayley_build_b(n, M, W, B);

    expect(max_abs_diff(B, want, n * n) < TOL, "B = 2W - MW for random 6 x 6 input");

    free(M);
    free(W);
    free(B);
    free(MW);
    free(want);
}

/* ---- problem 4: solve ------------------------------------------------- */

static void test_solve_known_solution(void)
{
    const size_t n = 6;
    double *M = new_matrix(n);
    double *X = new_matrix(n);
    double *B = new_matrix(n);
    size_t piv[6];

    rng_state = 41;
    fill_dominant(n, M);
    fill_random(X, n * n);
    matmul(n, M, X, B); /* so the true solution of M X = B is X */

    CayleyStatus status = cayley_solve(n, M, B, piv);

    expect(status == CAYLEY_OK, "solve reports CAYLEY_OK");
    expect(max_abs_diff(B, X, n * n) < 1e-10, "B is overwritten with the solution");

    free(M);
    free(X);
    free(B);
}

static void test_solve_forces_row_swaps(void)
{
    /* M[0][0] is zero, so elimination without a row swap divides by zero. */
    double M[9] = {0, 2, 1, 1, 1, 1, 3, 0, 2};
    const double X[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double B[9];
    size_t piv[3] = {99, 99, 99};
    const size_t want_piv[3] = {2, 2, 2};

    const double M_copy[9] = {0, 2, 1, 1, 1, 1, 3, 0, 2};
    matmul(3, M_copy, X, B);

    CayleyStatus status = cayley_solve(3, M, B, piv);

    expect(status == CAYLEY_OK, "solve reports CAYLEY_OK");
    expect(max_abs_diff(B, X, 9) < TOL, "solution is correct when swaps are needed");
    expect(memcmp(piv, want_piv, sizeof piv) == 0,
           "piv records the swaps {2, 2, 2}");
}

static void test_solve_singular(void)
{
    /* Row 1 is twice row 0, so the matrix is singular. In this case the
     * arithmetic is exact, so a pivot of exactly zero appears. */
    double M[9] = {1, 2, 3, 2, 4, 6, 1, 1, 1};
    double B[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    size_t piv[3];

    CayleyStatus status = cayley_solve(3, M, B, piv);

    expect(status == CAYLEY_ERR_SINGULAR, "a singular matrix returns CAYLEY_ERR_SINGULAR");
}

/* ---- problem 5: the full step ----------------------------------------- */

static void test_step_zero_gradient(void)
{
    const size_t n = 4;
    double *W = new_matrix(n);
    double *G = new_matrix(n); /* all zeros */
    double *before = new_matrix(n);
    CayleyWorkspace *ws = cayley_create(n);

    if (ws == NULL) {
        expect(0, "a workspace is needed for this test (problem 1)");
        goto done;
    }
    rng_state = 51;
    fill_random(W, n * n);
    memcpy(before, W, n * n * sizeof *W);

    CayleyStatus status = cayley_step(ws, W, G, 0.3);

    expect(status == CAYLEY_OK, "step reports CAYLEY_OK");
    expect(max_abs_diff(W, before, n * n) == 0.0, "W is unchanged when G = 0");

done:
    cayley_free(ws);
    free(W);
    free(G);
    free(before);
}

static void test_step_2x2_rotation(void)
{
    /* For n = 2 and W = I, A = s J with J = [[0, 1], [-1, 0]] and s = G01 - G10.
     * With t = eta * s / 2 the answer is a rotation:
     *     W+ = 1 / (1 + t^2) * [[1 - t^2, -2t], [2t, 1 - t^2]].
     * Here s = 1 and eta = 1, so t = 0.5 and W+ = [[0.6, -0.8], [0.8, 0.6]]. */
    double W[4] = {1, 0, 0, 1};
    const double G[4] = {1, 1, 0, 3};
    const double want[4] = {0.6, -0.8, 0.8, 0.6};
    CayleyWorkspace *ws = cayley_create(2);

    if (ws == NULL) {
        expect(0, "a workspace is needed for this test (problem 1)");
        return;
    }

    CayleyStatus status = cayley_step(ws, W, G, 1.0);

    expect(status == CAYLEY_OK, "step reports CAYLEY_OK");
    expect(max_abs_diff(W, want, 4) < 1e-14, "W+ is the closed-form rotation");
    cayley_free(ws);
}

static void test_step_stays_orthogonal(void)
{
    const size_t n = 8;
    double *W = new_matrix(n);
    double *G = new_matrix(n);
    CayleyWorkspace *ws = cayley_create(n);

    if (ws == NULL) {
        expect(0, "a workspace is needed for this test (problem 1)");
        goto done;
    }
    rng_state = 61;
    for (size_t i = 0; i < n; i++) {
        W[i * n + i] = 1.0;
    }

    /* The same workspace is reused for every step. */
    for (int step = 0; step < 20; step++) {
        fill_random(G, n * n);
        CayleyStatus status = cayley_step(ws, W, G, 0.1);
        expect(status == CAYLEY_OK, "step reports CAYLEY_OK");
        if (status != CAYLEY_OK) {
            break;
        }
        expect(orthogonality_error(n, W) < TOL,
               "||W^T W - I||_F stays at rounding level");
    }

done:
    cayley_free(ws);
    free(W);
    free(G);
}

static void test_step_rejects_nonfinite(void)
{
    const size_t n = 3;
    double *W = new_matrix(n);
    double *G = new_matrix(n);
    double *before = new_matrix(n);
    CayleyWorkspace *ws = cayley_create(n);

    if (ws == NULL) {
        expect(0, "a workspace is needed for this test (problem 1)");
        goto done;
    }
    rng_state = 71;
    fill_random(W, n * n);
    fill_random(G, n * n);
    memcpy(before, W, n * n * sizeof *W);

    G[4] = NAN;
    expect(cayley_step(ws, W, G, 0.1) == CAYLEY_ERR_NONFINITE, "NaN in G is reported");
    expect(memcmp(W, before, n * n * sizeof *W) == 0, "W is untouched after a NaN in G");

    G[4] = 0.25;
    W[2] = INFINITY;
    memcpy(before, W, n * n * sizeof *W);
    expect(cayley_step(ws, W, G, 0.1) == CAYLEY_ERR_NONFINITE, "infinity in W is reported");
    expect(memcmp(W, before, n * n * sizeof *W) == 0, "W is untouched after infinity in W");

    W[2] = 0.5;
    memcpy(before, W, n * n * sizeof *W);
    expect(cayley_step(ws, W, G, NAN) == CAYLEY_ERR_NONFINITE, "NaN eta is reported");
    expect(memcmp(W, before, n * n * sizeof *W) == 0, "W is untouched after a NaN eta");

    /* The workspace must still work after errors. */
    memset(W, 0, n * n * sizeof *W);
    for (size_t i = 0; i < n; i++) {
        W[i * n + i] = 1.0;
    }
    expect(cayley_step(ws, W, G, 0.1) == CAYLEY_OK, "the workspace is usable after errors");

done:
    cayley_free(ws);
    free(W);
    free(G);
    free(before);
}

/* ---- runner ----------------------------------------------------------- */

typedef struct {
    const char *name;
    void (*run)(void);
} Test;

static const Test tests[] = {
    {"problem 1: create and free", test_create_free},
    {"problem 2: form M, hand-checked", test_form_m_hand_checked},
    {"problem 2: form M, exactly skew", test_form_m_skew_exact},
    {"problem 3: build B, hand-checked", test_build_b_hand_checked},
    {"problem 3: build B, M = I", test_build_b_identity_m},
    {"problem 3: build B, random", test_build_b_random},
    {"problem 4: solve, known solution", test_solve_known_solution},
    {"problem 4: solve, forced row swaps", test_solve_forces_row_swaps},
    {"problem 4: solve, singular matrix", test_solve_singular},
    {"problem 5: step, zero gradient", test_step_zero_gradient},
    {"problem 5: step, 2 x 2 closed form", test_step_2x2_rotation},
    {"problem 5: step, stays orthogonal", test_step_stays_orthogonal},
    {"problem 5: step, non-finite input", test_step_rejects_nonfinite},
    {"problem 1: size overflow is rejected", test_create_overflow},
};

int main(void)
{
    const size_t count = sizeof tests / sizeof tests[0];
    size_t passed = 0;

    for (size_t i = 0; i < count; i++) {
        g_failed = 0;
        printf("%s\n", tests[i].name);
        tests[i].run();
        if (g_failed == 0) {
            printf("  ok\n");
            passed++;
        }
    }

    printf("\n%zu of %zu tests passed\n", passed, count);
    return passed == count ? 0 : 1;
}
