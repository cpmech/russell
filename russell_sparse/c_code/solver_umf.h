#ifndef SOLVER_UMF_H
#define SOLVER_UMF_H

#include <inttypes.h>
#include <stdlib.h>

#include "constants.h"
#include "umfpack.h"

struct SolverUMF {
    double control[UMFPACK_CONTROL];
    double info[UMFPACK_INFO];
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

struct SolverUMF *new_solver_umf(int32_t symmetric) {
    struct SolverUMF *solver = (struct SolverUMF *)malloc(sizeof(struct SolverUMF));

    if (solver == NULL) {
        return NULL;
    }

    umfpack_di_defaults(solver->control);

    if (symmetric == C_TRUE) {
        solver->control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;
    }

    solver->symbolic = NULL;
    solver->numeric = NULL;

    return solver;
}

void drop_solver_umf(struct SolverUMF *solver) {
    if (solver == NULL) {
        return;
    }

    if (solver->symbolic != NULL) {
        free(solver->symbolic);
    }
    if (solver->numeric != NULL) {
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
                              int32_t scaling,
                              int32_t verbose) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    return 1;  // error
}

int32_t solver_umf_factorize(struct SolverMMP *solver, int32_t verbose) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    return 1;  // error
}

int32_t solver_umf_solve(struct SolverMMP *solver, double *rhs, int32_t verbose) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    return 1;  // error
}

#endif