#ifndef SOLVER_MUMPS_H
#define SOLVER_MUMPS_H

#include <inttypes.h>
#include <stdlib.h>

#include "constants.h"
#include "dmumps_c.h"

#define ICNTL(i) icntl[(i)-1]  // macro to make indices match documentation
#define INFOG(i) infog[(i)-1]  // macro to make indices match documentation

static inline void set_verbose(DMUMPS_STRUC_C *data, int32_t verbose) {
    if (verbose > 0) {
        data->ICNTL(1) = 6;  // standard output stream
        data->ICNTL(2) = 0;  // output stream
        data->ICNTL(3) = 6;  // standard output stream
        data->ICNTL(4) = 3;  // errors, warnings, and main statistics printed
    } else {
        data->ICNTL(1) = -1;  // no output messages
        data->ICNTL(2) = -1;  // no warnings
        data->ICNTL(3) = -1;  // no global information
        data->ICNTL(4) = -1;  // message level
    }
}

struct SolverMumps {
    DMUMPS_STRUC_C dmumps;  // MUMPS data structure for C-code
    int32_t initialized;    // initialize has been called
    int32_t factorized;     // factorize has been called
    int32_t n;              // a*x=rhs system dimension
    int32_t nnz;            // number of non-zero values
    MUMPS_INT *indices_i;   // [nnz] zero- or one-based indices stored here
    MUMPS_INT *indices_j;   // [nnz] zero- or one-based indices stored here
    double *values_a;       // [nnz] the non-zero entries in the matrix
    double *rhs;            // [n] the right-hand-side vector of a*x=rhs
};

struct SolverMumps *new_solver_mumps(int32_t symmetry, int32_t verbose) {
    struct SolverMumps *solver = (struct SolverMumps *)malloc(sizeof(struct SolverMumps));

    solver->dmumps.comm_fortran = MUMPS_IGNORED;
    solver->dmumps.par = MUMPS_PAR_HOST_ALSO_WORKS;
    solver->dmumps.sym = symmetry;

    set_verbose(&solver->dmumps, verbose);
    solver->dmumps.job = MUMPS_JOB_INITIALIZE;
    dmumps_c(&solver->dmumps);

    if (solver->dmumps.INFOG(1) != 0) {
        free(solver);
        return NULL;
    }

    solver->initialized = C_FALSE;
    solver->factorized = C_FALSE;
    solver->n = 0;
    solver->nnz = 0;
    solver->indices_i = NULL;
    solver->indices_j = NULL;
    solver->values_a = NULL;
    solver->rhs = NULL;

    return solver;
}

static inline int32_t solver_mumps_new_data(struct SolverMumps *solver, int32_t n, int32_t nnz) {
    if (solver->n != 0 ||
        solver->nnz != 0 ||
        solver->indices_i != NULL ||
        solver->indices_j != NULL ||
        solver->values_a != NULL) {
        return C_HAS_ERROR;
    }
    solver->indices_i = (MUMPS_INT *)malloc(nnz * sizeof(MUMPS_INT));
    if (solver->indices_i == NULL) {
        return C_HAS_ERROR;
    }
    solver->indices_j = (MUMPS_INT *)malloc(nnz * sizeof(MUMPS_INT));
    if (solver->indices_j == NULL) {
        free(solver->indices_i);
        return C_HAS_ERROR;
    }
    solver->values_a = (double *)malloc(nnz * sizeof(double));
    if (solver->values_a == NULL) {
        free(solver->indices_j);
        free(solver->indices_i);
        return C_HAS_ERROR;
    }
    solver->rhs = (double *)malloc(n * sizeof(double));
    if (solver->rhs == NULL) {
        free(solver->indices_j);
        free(solver->indices_i);
        free(solver->values_a);
        return C_HAS_ERROR;
    }
    solver->n = n;
    solver->nnz = nnz;
    return C_NO_ERROR;
}

static inline void solver_mumps_free_data(struct SolverMumps *solver) {
    if (solver->rhs != NULL) {
        free(solver->rhs);
    }
    if (solver->values_a != NULL) {
        free(solver->values_a);
    }
    if (solver->indices_j != NULL) {
        free(solver->indices_j);
    }
    if (solver->indices_i != NULL) {
        free(solver->indices_i);
    }
    solver->initialized = C_FALSE;
    solver->factorized = C_FALSE;
    solver->n = 0;
    solver->nnz = 0;
    solver->indices_i = NULL;
    solver->indices_j = NULL;
    solver->values_a = NULL;
}

static inline int32_t solver_mumps_set_ija(struct SolverMumps *solver,
                                           int32_t nnz,
                                           int32_t const *indices_i,
                                           int32_t const *indices_j,
                                           double const *values_a) {
    if (nnz != solver->nnz ||
        solver->nnz < 1 ||
        solver->indices_i == NULL ||
        solver->indices_j == NULL ||
        solver->values_a == NULL) {
        return C_HAS_ERROR;
    }
    int32_t m = solver->nnz % 4;
    int32_t p;
    for (p = 0; p < m; p++) {
        solver->indices_i[p] = indices_i[p] + 1;
        solver->indices_j[p] = indices_j[p] + 1;
        solver->values_a[p] = values_a[p];
    }
    for (p = m; p < solver->nnz; p += 4) {
        solver->indices_i[p] = indices_i[p] + 1;
        solver->indices_j[p] = indices_j[p] + 1;
        solver->values_a[p] = values_a[p];
    }
    return C_NO_ERROR;
}

static inline int32_t solver_mumps_set_rhs(struct SolverMumps *solver,
                                           int32_t n,
                                           double const *rhs) {
    if (n != solver->n ||
        solver->n < 1 ||
        solver->rhs == NULL) {
        return C_HAS_ERROR;
    }
    int32_t m = solver->n % 4;
    int32_t p;
    for (p = 0; p < m; p++) {
        solver->rhs[p] = rhs[p];
    }
    for (p = m; p < solver->n; p += 4) {
        solver->rhs[p] = rhs[p];
    }
    return C_NO_ERROR;
}

void drop_solver_mumps(struct SolverMumps *solver) {
    if (solver == NULL) {
        return;
    }

    solver_mumps_free_data(solver);

    set_verbose(&solver->dmumps, 0);
    solver->dmumps.job = MUMPS_JOB_TERMINATE;
    dmumps_c(&solver->dmumps);

    free(solver);
}

int32_t solver_mumps_initialize(struct SolverMumps *solver,
                                int32_t n,
                                int32_t nnz,
                                int32_t const *indices_i,
                                int32_t const *indices_j,
                                double const *values_a,
                                int32_t ordering,
                                int32_t scaling,
                                int32_t pct_inc_workspace,
                                int32_t max_work_memory,
                                int32_t openmp_num_threads,
                                int32_t verbose) {
    if (solver == NULL ||
        n < 1 ||
        nnz < 1 ||
        indices_i == NULL ||
        indices_j == NULL ||
        values_a == NULL) {
        return C_HAS_ERROR;
    }

    if (solver->initialized) {
        return C_HAS_ERROR;
    }

    int32_t res = solver_mumps_new_data(solver, n, nnz);
    if (res != C_NO_ERROR) {
        return C_HAS_ERROR;
    }

    res = solver_mumps_set_ija(solver, nnz, indices_i, indices_j, values_a);
    if (res != C_NO_ERROR) {
        return C_HAS_ERROR;
    }

    solver->dmumps.n = n;
    solver->dmumps.nz = nnz,
    solver->dmumps.irn = solver->indices_i;
    solver->dmumps.jcn = solver->indices_j;
    solver->dmumps.a = solver->values_a;

    solver->dmumps.ICNTL(5) = MUMPS_ICNTL5_ASSEMBLED_MATRIX;
    solver->dmumps.ICNTL(6) = MUMPS_ICNTL6_PERMUT_AUTO;
    solver->dmumps.ICNTL(7) = ordering;
    solver->dmumps.ICNTL(8) = scaling;
    solver->dmumps.ICNTL(14) = pct_inc_workspace;
    solver->dmumps.ICNTL(16) = openmp_num_threads;
    solver->dmumps.ICNTL(18) = MUMPS_ICNTL18_CENTRALIZED;
    solver->dmumps.ICNTL(23) = max_work_memory;
    solver->dmumps.ICNTL(28) = MUMPS_ICNTL28_SEQUENTIAL;
    solver->dmumps.ICNTL(29) = MUMPS_IGNORED;

    set_verbose(&solver->dmumps, verbose);
    solver->dmumps.job = MUMPS_JOB_ANALYZE;
    dmumps_c(&solver->dmumps);

    if (solver->dmumps.INFOG(1) != 0) {
        return solver->dmumps.INFOG(1);
    }

    solver->initialized = C_TRUE;

    return C_NO_ERROR;
}

int32_t solver_mumps_factorize(struct SolverMumps *solver, int32_t verbose) {
    if (solver == NULL) {
        return C_HAS_ERROR;
    }

    if (solver->initialized == C_FALSE) {
        return C_HAS_ERROR;
    }

    set_verbose(&solver->dmumps, verbose);

    solver->dmumps.job = MUMPS_JOB_FACTORIZE;
    dmumps_c(&solver->dmumps);

    if (solver->dmumps.INFOG(1) != 0) {
        return solver->dmumps.INFOG(1);
    }

    solver->factorized = C_TRUE;

    return C_NO_ERROR;
}

int32_t solver_mumps_solve(struct SolverMumps *solver,
                           int32_t n,
                           double *x,
                           double const *rhs,
                           int32_t verbose) {
    if (solver == NULL ||
        x == NULL ||
        rhs == NULL) {
        return C_HAS_ERROR;
    }

    if (solver->factorized == C_FALSE) {
        return C_HAS_ERROR;
    }

    set_verbose(&solver->dmumps, verbose);

    solver_mumps_set_rhs(solver, n, rhs);
    solver->dmumps.rhs = solver->rhs;

    solver->dmumps.job = MUMPS_JOB_SOLVE;
    dmumps_c(&solver->dmumps);

    if (solver->dmumps.INFOG(1) != 0) {
        return solver->dmumps.INFOG(1);
    }

    int32_t m = solver->n % 4;
    int32_t p;
    for (p = 0; p < m; p++) {
        x[p] = solver->rhs[p];
    }
    for (p = m; p < solver->n; p += 4) {
        x[p] = solver->rhs[p];
    }

    return C_NO_ERROR;
}

#undef INFOG
#undef ICNTL

#endif
