#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dmumps_c.h"

#include "constants.h"

#define ICNTL(i) icntl[(i)-1]   // macro to make indices match documentation
#define RINFOG(i) rinfog[(i)-1] // macro to make indices match documentation
#define INFOG(i) infog[(i)-1]   // macro to make indices match documentation
#define INFO(i) info[(i)-1]     // macro to make indices match documentation

const MUMPS_INT MUMPS_IGNORED = 0; // to ignore the Fortran communicator since we're not using MPI

const MUMPS_INT MUMPS_JOB_INITIALIZE = -1; // section 5.1.1, page 24
const MUMPS_INT MUMPS_JOB_TERMINATE = -2;  // section 5.1.1, page 24
const MUMPS_INT MUMPS_JOB_ANALYZE = 1;     // section 5.1.1, page 24
const MUMPS_INT MUMPS_JOB_FACTORIZE = 2;   // section 5.1.1, page 25
const MUMPS_INT MUMPS_JOB_SOLVE = 3;       // section 5.1.1, page 25

const MUMPS_INT MUMPS_PAR_HOST_ALSO_WORKS = 1;     // section 5.1.4, page 26
const MUMPS_INT MUMPS_ICNTL5_ASSEMBLED_MATRIX = 0; // section 5.2.2, page 27
const MUMPS_INT MUMPS_ICNTL18_CENTRALIZED = 0;     // section 5.2.2, page 27
const MUMPS_INT MUMPS_ICNTL6_PERMUT_AUTO = 7;      // section 5.3, page 32
const MUMPS_INT MUMPS_ICNTL28_SEQUENTIAL = 1;      // section 5.4, page 33

/// @brief Holds the data for MUMPS
struct InterfaceMUMPS {
    DMUMPS_STRUC_C data;              // data structure
    int32_t done_job_init;            // job init successfully
    double determinant_coefficient_a; // if asked, stores the determinant coefficient a
    double determinant_exponent_c;    // if asked, stores the determinant exponent c
};

/// @brief Sets verbose mode
/// @param data Is the MUMPS data structure
/// @param verbose Shows messages or not
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

/// @brief Performs the initialization
/// @param solver Is a pointer to the solver interface
/// @param n Is the number of rows and columns of the coefficient matrix
/// @param nnz Is the number of non-zero values in the coefficient matrix
/// @param symmetry Is the MUMPS code for the kind of symmetry, if any
/// @param ordering Is the MUMPS ordering code
/// @param scaling Is the MUMPS scaling code
/// @param pct_inc_workspace Is the allowed percentage increase of the workspace
/// @param max_work_memory Is the allowed maximum memory
/// @param openmp_num_threads Is the number of threads allowed for OpenMP
/// @param compute_determinant Requests that determinant be computed
/// @return A success or fail code
int32_t solver_mumps_initialize(struct InterfaceMUMPS *solver,
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
    solver->data.sym = symmetry;

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
        return VERSION_ERROR;
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

/// @brief Performs the factorization
/// @param solver Is a pointer to the solver interface
/// @param indices_i Are the CooMatrix row indices
/// @param indices_j Are the CooMatrix column indices
/// @param values_aij Are the CooMatrix values
/// @param verbose Shows messages
/// @return A success or fail code
int32_t solver_mumps_factorize(struct InterfaceMUMPS *solver,
                               int32_t n,
                               int32_t nnz,
                               int32_t const *indices_i,
                               int32_t const *indices_j,
                               double const *values_aij,
                               int32_t verbose) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    // set matrix components and perform analysis (must be done for each factorization)

    solver->data.n = n;
    solver->data.nz = nnz;
    solver->data.irn = (int *)indices_i;
    solver->data.jcn = (int *)indices_j;
    solver->data.a = (double *)values_aij;

    set_mumps_verbose(&solver->data, verbose);
    solver->data.job = MUMPS_JOB_ANALYZE;
    dmumps_c(&solver->data);

    if (solver->data.INFO(1) != 0) {
        // error
        return solver->data.INFOG(1);
    }

    // perform factorization

    set_mumps_verbose(&solver->data, verbose);
    solver->data.job = MUMPS_JOB_FACTORIZE;
    dmumps_c(&solver->data);

    // read determinant

    if (solver->data.ICNTL(33) == 1) {
        solver->determinant_coefficient_a = solver->data.RINFOG(12);
        solver->determinant_exponent_c = solver->data.INFOG(34);
    } else {
        solver->determinant_coefficient_a = 0.0;
        solver->determinant_exponent_c = 0.0;
    }

    return solver->data.INFOG(1);
}

/// @brief Computes the solution of the linear system
/// @param solver Is a pointer to the solver interface
/// @param rhs Is the right-hand side on the input and the vector of unknow values x on the output
/// @param verbose Shows messages
/// @return A success or fail code
int32_t solver_mumps_solve(struct InterfaceMUMPS *solver, double *rhs, int32_t verbose) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    solver->data.rhs = rhs;

    set_mumps_verbose(&solver->data, verbose);
    solver->data.job = MUMPS_JOB_SOLVE;
    dmumps_c(&solver->data);

    return solver->data.INFOG(1);
}

/// @brief Returns the effective ordering using during the computations
/// @param solver Is a pointer to the solver
/// @return The used MUMPS ordering code
int32_t solver_mumps_get_ordering(const struct InterfaceMUMPS *solver) {
    if (solver == NULL) {
        return -1;
    }
    return solver->data.INFOG(7);
}

/// @brief Returns the effective scaling using during the computations
/// @param solver Is a pointer to the solver
/// @return The used MUMPS scaling code
int32_t solver_mumps_get_scaling(const struct InterfaceMUMPS *solver) {
    if (solver == NULL) {
        return -1;
    }
    return solver->data.INFOG(33);
}

/// @brief Returns the coefficient needed to compute the determinant, if requested
/// @param solver Is a pointer to the solver
/// @return The coefficient a of a * 2^c
double solver_mumps_get_det_coef_a(const struct InterfaceMUMPS *solver) {
    if (solver == NULL) {
        return 0.0;
    }
    return solver->determinant_coefficient_a;
}

/// @brief Returns the exponent needed to compute the determinant, if requested
/// @param solver Is a pointer to the solver
/// @return The exponent c of a * 2^c
double solver_mumps_get_det_exp_c(const struct InterfaceMUMPS *solver) {
    if (solver == NULL) {
        return 0.0;
    }
    return solver->determinant_exponent_c;
}
