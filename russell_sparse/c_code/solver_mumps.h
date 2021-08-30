#ifndef SOLVER_MUMPS_H
#define SOLVER_MUMPS_H

#include <stdlib.h>

#include "constants.h"
#include "dmumps_c.h"
#include "sparse_triplet.h"

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

    return solver;
}

void drop_solver_mumps(struct SolverMumps *solver) {
    if (solver == NULL) {
        return;
    }
    set_verbose(&solver->dmumps, 0);
    solver->dmumps.job = MUMPS_JOB_TERMINATE;
    dmumps_c(&solver->dmumps);
    free(solver);
}

int32_t solver_mumps_get_last_error(struct SolverMumps *solver) {
    return solver->dmumps.INFOG(1);
}

int32_t solver_mumps_analyze(struct SolverMumps *solver,
                             struct SparseTriplet *trip,
                             int32_t ndim,
                             int32_t nnz,
                             int32_t ordering,
                             int32_t scaling,
                             int32_t pct_inc_workspace,
                             int32_t max_work_memory,
                             int32_t openmp_num_threads,
                             int32_t verbose) {
    if (solver == NULL) {
        return C_HAS_ERROR;
    }

    set_verbose(&solver->dmumps, verbose);

    solver->dmumps.ICNTL(5) = MUMPS_ICNTL5_ASSEMBLED_MATRIX;
    solver->dmumps.ICNTL(7) = ordering;
    solver->dmumps.ICNTL(8) = scaling;
    solver->dmumps.ICNTL(14) = pct_inc_workspace;
    solver->dmumps.ICNTL(23) = max_work_memory;
    solver->dmumps.ICNTL(16) = openmp_num_threads;
    solver->dmumps.n = ndim;

    solver->dmumps.ICNTL(18) = MUMPS_ICNTL18_CENTRALIZED;
    solver->dmumps.ICNTL(6) = MUMPS_ICNTL6_PERMUT_AUTO;
    solver->dmumps.nz = nnz,
    solver->dmumps.irn = trip->indices_i;
    solver->dmumps.jcn = trip->indices_j;
    solver->dmumps.a = trip->values_x;

    solver->dmumps.ICNTL(28) = MUMPS_ICNTL28_SEQUENTIAL;
    solver->dmumps.ICNTL(29) = MUMPS_IGNORED;

    solver->dmumps.job = MUMPS_JOB_ANALYZE;
    dmumps_c(&solver->dmumps);

    return solver->dmumps.INFOG(1);
}

#undef INFOG
#undef ICNTL

#endif
