#pragma once

#include "dmumps_c.h"
#include "stdlib.h"

#define ICNTL(i) icntl[(i)-1]   // macro to make indices match documentation
#define RINFOG(i) rinfog[(i)-1] // macro to make indices match documentation
#define INFOG(i) infog[(i)-1]   // macro to make indices match documentation
#define INFO(i) info[(i)-1]     // macro to make indices match documentation

const MUMPS_INT MUMPS_IGNORED = 0;

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

/// @brief Allocates a new MUMPS interface
struct InterfaceMUMPS *solver_mumps_new();

/// @brief Deallocates the MUMPS interface
void solver_mumps_drop(struct InterfaceMUMPS *solver);

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
                                int32_t n,
                                int32_t nnz,
                                int32_t symmetry,
                                int32_t ordering,
                                int32_t scaling,
                                int32_t pct_inc_workspace,
                                int32_t max_work_memory,
                                int32_t openmp_num_threads,
                                int32_t compute_determinant);

/// @brief Performs the factorization
/// @param solver Is a pointer to the solver interface
/// @param indices_i Are the CooMatrix row indices
/// @param indices_j Are the CooMatrix column indices
/// @param values_aij Are the CooMatrix values
/// @param verbose Shows messages
/// @return A success or fail code
int32_t solver_mumps_factorize(struct InterfaceMUMPS *solver,
                               int32_t const *indices_i,
                               int32_t const *indices_j,
                               double const *values_aij,
                               int32_t verbose);

/// @brief Computes the solution of the linear system
/// @param solver Is a pointer to the solver interface
/// @param rhs Is the right-hand side on the input and the vector of unknow values x on the output
/// @param verbose Shows messages
/// @return A success or fail code
int32_t solver_mumps_solve(struct InterfaceMUMPS *solver, double *rhs, int32_t verbose);

/// @brief Returns the effective ordering using during the computations
/// @param solver Is a pointer to the solver
/// @return The used MUMPS ordering code
int32_t solver_mumps_get_ordering(const struct InterfaceMUMPS *solver);

/// @brief Returns the effective scaling using during the computations
/// @param solver Is a pointer to the solver
/// @return The used MUMPS scaling code
int32_t solver_mumps_get_scaling(const struct InterfaceMUMPS *solver);

/// @brief Returns the coefficient needed to compute the determinant, if requested
/// @param solver Is a pointer to the solver
/// @return The coefficient a of a * 2^c
double solver_mumps_get_det_coef_a(const struct InterfaceMUMPS *solver);

/// @brief Returns the exponent needed to compute the determinant, if requested
/// @param solver Is a pointer to the solver
/// @return The exponent c of a * 2^c
double solver_mumps_get_det_exp_c(const struct InterfaceMUMPS *solver);
