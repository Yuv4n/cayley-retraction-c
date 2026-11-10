/*
 * Implementation. Each function below is a stub: it compiles, does nothing
 * and reports that it is unfinished. The problem statements are in
 * include/cayley.h. Replace the bodies one at a time and run the tests after
 * each. The (void) lines only silence unused-parameter warnings; delete them
 * as you start using the parameters.
 */

#include "cayley.h"

/* TODO (problem 1): define struct CayleyWorkspace here. */

CayleyWorkspace *cayley_create(size_t n)
{
    (void)n;
    return NULL;
}

void cayley_free(CayleyWorkspace *ws)
{
    (void)ws;
}

void cayley_form_m(size_t n, const double *W, const double *G, double eta,
                   double *M)
{
    (void)n;
    (void)W;
    (void)G;
    (void)eta;
    (void)M;
}

void cayley_build_b(size_t n, const double *M, const double *W, double *B)
{
    (void)n;
    (void)M;
    (void)W;
    (void)B;
}

CayleyStatus cayley_solve(size_t n, double *M, double *B, size_t *piv)
{
    (void)n;
    (void)M;
    (void)B;
    (void)piv;
    return CAYLEY_TODO;
}

CayleyStatus cayley_step(CayleyWorkspace *ws, double *W, const double *G,
                         double eta)
{
    (void)ws;
    (void)W;
    (void)G;
    (void)eta;
    return CAYLEY_TODO;
}
