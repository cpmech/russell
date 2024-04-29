#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi.h"

#include "dmumps_c.h"

#include "constants.h"

#define ICNTL(i) icntl[(i)-1]   // macro to make indices match documentation
#define RINFOG(i) rinfog[(i)-1] // macro to make indices match documentation
#define INFOG(i) infog[(i)-1]   // macro to make indices match documentation
#define INFO(i) info[(i)-1]     // macro to make indices match documentation

/// @brief Holds the data for MUMPS
struct InterfaceMUMPS {
    /// @brief Holds the MUMPS data structure
    DMUMPS_STRUC_C data;

    /// @brief job init completed successfully
    int32_t done_job_init;

    /// @brief indicates that the initialization has been completed
    C_BOOL initialization_completed;

    /// @brief indicates that the factorization (at least once) has been completed
    C_BOOL factorization_completed;
};

/// @brief Sets verbose mode
static inline void set_mumps_verbose(DMUMPS_STRUC_C *data, int32_t verbose) {
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

/// @brief Allocates a new MUMPS interface
struct InterfaceMUMPS *solver_mumps_new() {
    struct InterfaceMUMPS *solver = (struct InterfaceMUMPS *)malloc(sizeof(struct InterfaceMUMPS));

    if (solver == NULL) {
        return NULL;
    }

    solver->data.irn = NULL;
    solver->data.jcn = NULL;
    solver->data.a = NULL;
    solver->done_job_init = C_FALSE;
    solver->initialization_completed = C_FALSE;
    solver->factorization_completed = C_FALSE;

    return solver;
}

/// @brief Deallocates the MUMPS interface
void solver_mumps_drop(struct InterfaceMUMPS *solver) {
    if (solver == NULL) {
        return;
    }

    // prevent MUMPS freeing these
    solver->data.irn = NULL;
    solver->data.jcn = NULL;
    solver->data.a = NULL;

    if (solver->done_job_init == C_TRUE) {
        set_mumps_verbose(&solver->data, C_FALSE);
        solver->data.job = MUMPS_JOB_TERMINATE;
        dmumps_c(&solver->data);
    }

    free(solver);
}

/// @brief Perform analysis just once (considering that the matrix structure remains constant)
int32_t solver_mumps_initialize(struct InterfaceMUMPS *solver,
                                int32_t ordering,
                                int32_t scaling,
                                int32_t pct_inc_workspace,
                                int32_t max_work_memory,
                                int32_t openmp_num_threads,
                                C_BOOL verbose,
                                C_BOOL general_symmetric,
                                C_BOOL positive_definite,
                                int32_t ndim,
                                int32_t nnz,
                                int32_t const *indices_i,
                                int32_t const *indices_j,
                                double const *values_aij) {
    if (solver == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (solver->initialization_completed == C_TRUE) {
        return ERROR_ALREADY_INITIALIZED;
    }

    solver->data.comm_fortran = MUMPS_FORTRAN_COMM;
    solver->data.par = MUMPS_PAR_HOST_ALSO_WORKS;
    solver->data.sym = 0; // unsymmetric (page 27)
    if (general_symmetric == C_TRUE) {
        solver->data.sym = 2; // general symmetric (page 27)
    } else if (positive_definite == C_TRUE) {
        solver->data.sym = 1; // symmetric positive-definite (page 27)
    }

    int flag = 0;
    MPI_Initialized(&flag);
    if (flag != 1) {
        int status = MPI_Init(NULL, NULL); // initializes the MPI execution environment
        if (status != MPI_SUCCESS) {
            return ERROR_MPI_INIT_FAILED;
        }
    }

    set_mumps_verbose(&solver->data, C_FALSE);
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
        return ERROR_VERSION;
    }

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

    solver->data.n = ndim;
    solver->data.nz = nnz;
    solver->data.irn = (int *)indices_i;
    solver->data.jcn = (int *)indices_j;
    solver->data.a = (double *)values_aij;

    set_mumps_verbose(&solver->data, verbose);
    solver->data.job = MUMPS_JOB_ANALYZE;
    dmumps_c(&solver->data);

    if (solver->data.INFO(1) != 0) {
        return solver->data.INFOG(1); // error
    }

    solver->initialization_completed = C_TRUE;

    return SUCCESSFUL_EXIT;
}

/// @brief Performs the factorization
int32_t solver_mumps_factorize(struct InterfaceMUMPS *solver,
                               int32_t *effective_ordering,
                               int32_t *effective_scaling,
                               double *determinant_coefficient,
                               double *determinant_exponent,
                               C_BOOL compute_determinant,
                               C_BOOL verbose) {
    if (solver == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (solver->initialization_completed == C_FALSE) {
        return ERROR_NEED_INITIALIZATION;
    }

    // handle requests

    if (compute_determinant == C_TRUE) {
        solver->data.ICNTL(33) = 1;
        solver->data.ICNTL(8) = 0; // it's recommended to disable scaling when computing the determinant
    } else {
        solver->data.ICNTL(33) = 0;
    }

    // perform factorization

    set_mumps_verbose(&solver->data, verbose);
    solver->data.job = MUMPS_JOB_FACTORIZE;
    dmumps_c(&solver->data);

    // save the output params

    *effective_ordering = solver->data.INFOG(7);
    *effective_scaling = solver->data.INFOG(33);

    // read the determinant

    if (compute_determinant == C_TRUE && solver->data.ICNTL(33) == 1) {
        *determinant_coefficient = solver->data.RINFOG(12);
        *determinant_exponent = solver->data.INFOG(34);
    } else {
        *determinant_coefficient = 0.0;
        *determinant_exponent = 0.0;
    }

    solver->factorization_completed = C_TRUE;

    return solver->data.INFOG(1);
}

/// @brief Computes the solution of the linear system
/// @param error_analysis_array_len_8 array of size 8 to hold the results from the error analysis
/// @param error_analysis_option ICNTL(11): 0 (nothing), 1 (all; slow), 2 (just errors)
int32_t solver_mumps_solve(struct InterfaceMUMPS *solver,
                           double *rhs,
                           double *error_analysis_array_len_8,
                           int32_t error_analysis_option,
                           C_BOOL verbose) {
    if (solver == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (solver->factorization_completed == C_FALSE) {
        return ERROR_NEED_FACTORIZATION;
    }

    solver->data.ICNTL(11) = error_analysis_option;

    solver->data.rhs = rhs;

    set_mumps_verbose(&solver->data, verbose);
    solver->data.job = MUMPS_JOB_SOLVE;
    dmumps_c(&solver->data);

    if (error_analysis_option > 0) {
        error_analysis_array_len_8[0] = solver->data.RINFOG(4); // norm_a
        error_analysis_array_len_8[1] = solver->data.RINFOG(5); // norm_x
        error_analysis_array_len_8[2] = solver->data.RINFOG(6); // resid
        error_analysis_array_len_8[3] = solver->data.RINFOG(7); // omega1
        error_analysis_array_len_8[4] = solver->data.RINFOG(8); // omega2
        if (error_analysis_option == 1) {
            error_analysis_array_len_8[5] = solver->data.RINFOG(9);  // delta
            error_analysis_array_len_8[6] = solver->data.RINFOG(10); // cond1
            error_analysis_array_len_8[7] = solver->data.RINFOG(11); // cond2
        }
    }

    return solver->data.INFOG(1);
}
