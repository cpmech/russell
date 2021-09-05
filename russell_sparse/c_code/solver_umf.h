#ifndef SOLVER_UMF_H
#define SOLVER_UMF_H

#include <inttypes.h>
#include <stdlib.h>

#include "constants.h"
#include "umfpack.h"

struct SolverUMF {
    double control[UMFPACK_CONTROL];
    double info[UMFPACK_INFO];
    int n;
    int nnz;
    int *ap;
    int *ai;
    double *ax;
    void *symbolic;
    void *numeric;
};

static inline void set_umf_verbose(struct SolverUMF *solver, int32_t verbose) {
    if (verbose > C_TRUE) {
        solver->control[UMFPACK_PRL] = UMF_PRINT_LEVEL_VERBOSE;
    } else {
        solver->control[UMFPACK_PRL] = UMF_PRINT_LEVEL_SILENT;
    }
}

struct SolverUMF *new_solver_umf(int32_t symmetry) {
    struct SolverUMF *solver = (struct SolverUMF *)malloc(sizeof(struct SolverUMF));

    if (solver == NULL) {
        return NULL;
    }

    umfpack_di_defaults(solver->control);

    solver->control[UMFPACK_STRATEGY] = UMF_SYMMETRY[symmetry];

    solver->n = 0;
    solver->nnz = 0;
    solver->ap = NULL;
    solver->ai = NULL;
    solver->ax = NULL;

    solver->symbolic = NULL;
    solver->numeric = NULL;

    return solver;
}

void drop_solver_umf(struct SolverUMF *solver) {
    if (solver == NULL) {
        return;
    }

    if (solver->ap != NULL) {
        free(solver->ap);
    }
    if (solver->ai != NULL) {
        free(solver->ai);
    }
    if (solver->ax != NULL) {
        free(solver->ax);
    }

    if (solver->symbolic != NULL) {
        umfpack_di_free_symbolic(&solver->symbolic);
        free(solver->symbolic);
    }
    if (solver->numeric != NULL) {
        umfpack_di_free_numeric(&solver->numeric);
        free(solver->numeric);
    }

    free(solver);
}

int32_t solver_umf_initialize(struct SolverUMF *solver,
                              int32_t n,
                              int32_t nnz,
                              int32_t const *indices_i,
                              int32_t const *indices_j,
                              double const *values_a,
                              int32_t ordering,
                              int32_t scaling) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    solver->ap = (int *)malloc((n + 1) * sizeof(int));
    if (solver->ap == NULL) {
        return MALLOC_ERROR;
    }

    solver->ai = (int *)malloc(nnz * sizeof(int));
    if (solver->ai == NULL) {
        free(solver->ap);
        return MALLOC_ERROR;
    }

    solver->ax = (double *)malloc(nnz * sizeof(double));
    if (solver->ax == NULL) {
        free(solver->ai);
        free(solver->ap);
        return MALLOC_ERROR;
    }

    solver->n = n;
    solver->nnz = nnz;

    int code = umfpack_di_triplet_to_col(n, n, nnz, indices_i, indices_j, values_a,
                                         solver->ap, solver->ai, solver->ax, NULL);
    if (code != UMFPACK_OK) {
        free(solver->ap);
        free(solver->ai);
        free(solver->ax);
        return code;
    }

    solver->control[UMFPACK_ORDERING] = UMF_ORDERING[ordering];
    solver->control[UMFPACK_SCALE] = UMF_SCALING[scaling];

    return UMFPACK_OK;
}

int32_t solver_umf_factorize(struct SolverUMF *solver, int32_t verbose) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    set_umf_verbose(solver, verbose);

    int code = umfpack_di_symbolic(solver->n, solver->n, solver->ap, solver->ai, solver->ax,
                                   &solver->symbolic, solver->control, solver->info);
    if (code != UMFPACK_OK) {
        return code;
    }

    code = umfpack_di_numeric(solver->ap, solver->ai, solver->ax,
                              solver->symbolic, &solver->numeric, solver->control, solver->info);

    return code;
}

int32_t solver_umf_solve(struct SolverUMF *solver, double *x, const double *rhs, int32_t verbose) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    set_umf_verbose(solver, verbose);

    int code = umfpack_di_solve(UMFPACK_A, solver->ap, solver->ai, solver->ax,
                                x, rhs, solver->numeric, solver->control, solver->info);

    return code;
}

#endif