#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "constants.h"
#include "interface_mumps.h"

#define ICNTL(i) icntl[(i)-1]   // macro to make indices match documentation
#define RINFOG(i) rinfog[(i)-1] // macro to make indices match documentation
#define INFOG(i) infog[(i)-1]   // macro to make indices match documentation
#define INFO(i) info[(i)-1]     // macro to make indices match documentation

static inline void set_mmp_verbose(DMUMPS_STRUC_C *data, int32_t verbose) {
    if (verbose == C_TRUE) {
        data->ICNTL(1) = 6; // standard output stream
        data->ICNTL(2) = 0; // output stream
        data->ICNTL(3) = 6; // standard output stream
        data->ICNTL(4) = 3; // errors, warnings, and main statistics printed
    } else {
        data->ICNTL(1) = -1; // no output messages
        data->ICNTL(2) = -1; // no warnings
        data->ICNTL(3) = -1; // no global information
        data->ICNTL(4) = -1; // message level
    }
}

struct SolverMMP {
    DMUMPS_STRUC_C data;   // data structure
    int32_t done_job_init; // job init successfully
};

struct SolverMMP *new_solver_mmp() {
    struct SolverMMP *solver = (struct SolverMMP *)malloc(sizeof(struct SolverMMP));

    if (solver == NULL) {
        return NULL;
    }

    solver->data.irn = NULL;
    solver->data.jcn = NULL;
    solver->data.a = NULL;
    solver->done_job_init = C_FALSE;

    return solver;
}

void drop_solver_mmp(struct SolverMMP *solver) {
    if (solver == NULL) {
        return;
    }

    if (solver->data.irn != NULL) {
        free(solver->data.irn);
        solver->data.irn = NULL;
    }
    if (solver->data.jcn != NULL) {
        free(solver->data.jcn);
        solver->data.jcn = NULL;
    }
    if (solver->data.a != NULL) {
        free(solver->data.a);
        solver->data.a = NULL;
    }

    if (solver->done_job_init == C_TRUE) {
        set_mmp_verbose(&solver->data, C_FALSE);
        solver->data.job = MUMPS_JOB_TERMINATE;
        dmumps_c(&solver->data);
    }

    free(solver);
}

int32_t solver_mmp_initialize(struct SolverMMP *solver,
                              int32_t n,
                              int32_t nnz,
                              int32_t symmetry,
                              int32_t ordering,
                              int32_t scaling,
                              int32_t pct_inc_workspace,
                              int32_t max_work_memory,
                              int32_t openmp_num_threads,
                              int32_t compute_determinant) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    solver->data.comm_fortran = MUMPS_IGNORED;
    solver->data.par = MUMPS_PAR_HOST_ALSO_WORKS;
    solver->data.sym = MMP_SYMMETRY[symmetry];

    set_mmp_verbose(&solver->data, C_FALSE);
    solver->data.job = MUMPS_JOB_INITIALIZE;
    dmumps_c(&solver->data);
    if (solver->data.INFOG(1) != 0) {
        return solver->data.INFOG(1);
    }
    solver->done_job_init = C_TRUE;

    if (strcmp(solver->data.version_number, MUMPS_VERSION) != 0) {
        printf("\n\n\nERROR: MUMPS LIBRARY VERSION = ");
        int i;
        for (i = 0; i < MUMPS_VERSION_MAX_LEN; i++) {
            printf("%c", solver->data.version_number[i]);
        }
        printf(" != INCLUDE VERSION = %s \n\n\n", MUMPS_VERSION);
        return VERSION_ERROR;
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

    solver->data.n = n;
    solver->data.nz = nnz;

    solver->data.ICNTL(5) = MUMPS_ICNTL5_ASSEMBLED_MATRIX;
    solver->data.ICNTL(6) = MUMPS_ICNTL6_PERMUT_AUTO;
    solver->data.ICNTL(7) = MMP_ORDERING[ordering];
    solver->data.ICNTL(8) = MMP_SCALING[scaling];
    solver->data.ICNTL(14) = pct_inc_workspace;
    solver->data.ICNTL(16) = openmp_num_threads;
    solver->data.ICNTL(18) = MUMPS_ICNTL18_CENTRALIZED;
    solver->data.ICNTL(23) = max_work_memory;
    solver->data.ICNTL(28) = MUMPS_ICNTL28_SEQUENTIAL;
    solver->data.ICNTL(29) = MUMPS_IGNORED;

    if (compute_determinant == C_TRUE) {
        // The determinant is obtained by computing
        // (a + ib) * 2^c where a = RINFOG(12), b = RINFOG(13) and c = INFOG(34).
        // In real arithmetic b = RINFOG(13) is equal to 0.
        solver->data.ICNTL(33) = 1;
        solver->data.ICNTL(8) = 0; // it's recommended to disable scaling when computing the determinant
    } else {
        solver->data.ICNTL(33) = 0;
    }

    return 0; // success
}

int32_t solver_mmp_factorize(struct SolverMMP *solver,
                             int32_t const *indices_i,
                             int32_t const *indices_j,
                             double const *values_aij,
                             double *determinant_coefficient_a,
                             double *determinant_exponent_c,
                             int32_t verbose) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    // set matrix components and perform analysis (must be done for each factorization)

    int32_t p;
    for (p = 0; p < solver->data.nz; p++) {
        solver->data.irn[p] = indices_i[p] + 1;
        solver->data.jcn[p] = indices_j[p] + 1;
        solver->data.a[p] = values_aij[p];
    }

    set_mmp_verbose(&solver->data, verbose);
    solver->data.job = MUMPS_JOB_ANALYZE;
    dmumps_c(&solver->data);

    if (solver->data.INFO(1) != 0) {
        // error
        return solver->data.INFOG(1);
    }

    // perform factorization

    set_mmp_verbose(&solver->data, verbose);
    solver->data.job = MUMPS_JOB_FACTORIZE;
    dmumps_c(&solver->data);

    // read determinant

    if (solver->data.ICNTL(33) == 1) {
        *determinant_coefficient_a = solver->data.RINFOG(12);
        *determinant_exponent_c = solver->data.INFOG(34);
    } else {
        *determinant_coefficient_a = 0.0;
        *determinant_exponent_c = 0.0;
    }

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

int32_t solver_mmp_used_ordering(struct SolverMMP *solver) {
    return solver->data.INFOG(7);
}

int32_t solver_mmp_used_scaling(struct SolverMMP *solver) {
    return solver->data.INFOG(33);
}
