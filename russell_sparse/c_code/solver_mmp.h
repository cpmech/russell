#ifndef SOLVER_MMP_H
#define SOLVER_MMP_H

#include <inttypes.h>
#include <stdlib.h>

#include "constants.h"
#include "dmumps_c.h"

#define ICNTL(i) icntl[(i)-1]  // macro to make indices match documentation
#define INFOG(i) infog[(i)-1]  // macro to make indices match documentation

static inline void set_mmp_verbose(DMUMPS_STRUC_C *data, int32_t verbose) {
    if (verbose == C_TRUE) {
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

struct SolverMMP {
    DMUMPS_STRUC_C data;  // data structure
};

struct SolverMMP *new_solver_mmp(int32_t symmetry, int32_t verbose) {
    struct SolverMMP *solver = (struct SolverMMP *)malloc(sizeof(struct SolverMMP));

    if (solver == NULL) {
        return NULL;
    }

    solver->data.comm_fortran = MUMPS_IGNORED;
    solver->data.par = MUMPS_PAR_HOST_ALSO_WORKS;
    solver->data.sym = symmetry;

    set_mmp_verbose(&solver->data, verbose);
    solver->data.job = MUMPS_JOB_INITIALIZE;
    dmumps_c(&solver->data);

    if (solver->data.INFOG(1) != 0) {
        free(solver);
        return NULL;
    }

    return solver;
}

void drop_solver_mmp(struct SolverMMP *solver) {
    if (solver == NULL) {
        return;
    }

    set_mmp_verbose(&solver->data, 0);
    solver->data.job = MUMPS_JOB_TERMINATE;
    dmumps_c(&solver->data);

    if (solver->data.irn != NULL) {
        free(solver->data.irn);
    }
    if (solver->data.jcn != NULL) {
        free(solver->data.jcn);
    }
    if (solver->data.a != NULL) {
        free(solver->data.a);
    }

    free(solver);
}

int32_t solver_mmp_initialize(struct SolverMMP *solver,
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
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    solver->data.irn = (MUMPS_INT *)malloc(nnz * sizeof(MUMPS_INT));
    if (solver->data.irn == NULL) {
        return MALLOC_ERROR;
    }

    solver->data.jcn = (MUMPS_INT *)malloc(nnz * sizeof(MUMPS_INT));
    if (solver->data.jcn == NULL) {
        free(solver->data.irn);
        return MALLOC_ERROR;
    }

    solver->data.a = (double *)malloc(nnz * sizeof(double));
    if (solver->data.a == NULL) {
        free(solver->data.jcn);
        free(solver->data.irn);
        return MALLOC_ERROR;
    }

    int32_t p;
    for (p = 0; p < nnz; p++) {
        solver->data.irn[p] = indices_i[p] + 1;
        solver->data.jcn[p] = indices_j[p] + 1;
        solver->data.a[p] = values_a[p];
    }

    solver->data.n = n;
    solver->data.nz = nnz;

    solver->data.ICNTL(5) = MUMPS_ICNTL5_ASSEMBLED_MATRIX;
    solver->data.ICNTL(6) = MUMPS_ICNTL6_PERMUT_AUTO;
    solver->data.ICNTL(7) = ordering;
    solver->data.ICNTL(8) = scaling;
    solver->data.ICNTL(14) = pct_inc_workspace;
    solver->data.ICNTL(16) = openmp_num_threads;
    solver->data.ICNTL(18) = MUMPS_ICNTL18_CENTRALIZED;
    solver->data.ICNTL(23) = max_work_memory;
    solver->data.ICNTL(28) = MUMPS_ICNTL28_SEQUENTIAL;
    solver->data.ICNTL(29) = MUMPS_IGNORED;

    set_mmp_verbose(&solver->data, verbose);
    solver->data.job = MUMPS_JOB_ANALYZE;
    dmumps_c(&solver->data);

    return solver->data.INFOG(1);
}

int32_t solver_mmp_factorize(struct SolverMMP *solver, int32_t verbose) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    set_mmp_verbose(&solver->data, verbose);
    solver->data.job = MUMPS_JOB_FACTORIZE;
    dmumps_c(&solver->data);

    return solver->data.INFOG(1);
}

int32_t solver_mmp_solve(struct SolverMMP *solver, double *rhs, int32_t verbose) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    solver->data.rhs = rhs;

    set_mmp_verbose(&solver->data, verbose);
    solver->data.job = MUMPS_JOB_SOLVE;
    dmumps_c(&solver->data);

    return solver->data.INFOG(1);
}

#undef INFOG
#undef ICNTL

#endif
