#pragma once

#include <stdlib.h>

#ifdef WITH_INTEL_DSS
#include "mkl_dss.h"
#else
#define MKL_INT int
#define _MKL_DSS_HANDLE_t void *
#define UNUSED(expr)  \
    do {              \
        (void)(expr); \
    } while (0)
#endif

#include "constants.h"

/// @brief Wraps the Intel DSS solver
/// @note The DSS uses a row-major UPPER triangular storage format.
/// @note The matrix is compressed row-by-row.
/// @note For symmetric matrices only non-zero elements in the UPPER triangular half of the matrix are stored.
/// @note For symmetric matrices, the zero diagonal entries must be stored as well.
struct InterfaceIntelDSS {
    /// @brief Holds the Intel DSS options
    MKL_INT dss_opt;

    /// @brief Holds the Intel DSS sym option
    MKL_INT dss_sym;

    /// @brief Holds the Intel DSS type option
    MKL_INT dss_type;

    /// @brief Holds the Intel DSS handle (allocated in analyze)
    _MKL_DSS_HANDLE_t handle;
};

/// @brief Allocates a new Intel DSS interface
/// @return A pointer to the solver interface
struct InterfaceIntelDSS *solver_intel_dss_new() {
#ifdef WITH_INTEL_DSS
    struct InterfaceIntelDSS *solver = (struct InterfaceIntelDSS *)malloc(sizeof(struct InterfaceIntelDSS));

    if (solver == NULL) {
        return NULL;
    }

    solver->handle = NULL;

    return solver;
#else
    return NULL;
#endif
}

/// @brief Deallocates the Intel DSS interface
/// @param solver Is a pointer to the solver interface
void solver_intel_dss_drop(struct InterfaceIntelDSS *solver) {
#ifdef WITH_INTEL_DSS
    if (solver == NULL) {
        return;
    }

    if (solver->handle != NULL) {
        dss_delete(solver->handle, solver->dss_opt);
    }

    free(solver);
#else
    UNUSED(solver);
#endif
}

/// @brief Performs the initialization
/// @param solver Is a pointer to the solver interface
/// @param symmetric Whether the matrix is symmetric or not
/// @param positive_definite Whether the matrix is positive-definite or not
/// @return A success or fail code
int32_t solver_intel_dss_initialize(struct InterfaceIntelDSS *solver,
                                    int32_t symmetric,
                                    int32_t positive_definite) {
#ifdef WITH_INTEL_DSS
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    solver->dss_opt = MKL_DSS_MSG_LVL_WARNING + MKL_DSS_TERM_LVL_ERROR + MKL_DSS_ZERO_BASED_INDEXING;

    solver->dss_sym = MKL_DSS_NON_SYMMETRIC;
    if (symmetric == C_TRUE) {
        solver->dss_sym = MKL_DSS_SYMMETRIC;
    }

    solver->dss_type = MKL_DSS_INDEFINITE;
    if (positive_definite == C_TRUE) {
        solver->dss_type = MKL_DSS_POSITIVE_DEFINITE;
    }

    MKL_INT status = dss_create(solver->handle, solver->dss_opt);
    return status;
#else
    UNUSED(solver);
    UNUSED(symmetric);
    UNUSED(positive_definite);
    return NOT_AVAILABLE;
#endif
}

/// @brief Performs the factorization
/// @param solver Is a pointer to the solver interface
/// @param ndim Is the number of rows and columns of the coefficient matrix
/// @param row_pointers The row pointers array with size = nrow + 1
/// @param col_indices The columns indices array with size = nnz (number of non-zeros)
/// @param values The values array with size = nnz (number of non-zeros)
/// @return A success or fail code
int32_t solver_intel_dss_factorize(struct InterfaceIntelDSS *solver,
                                   int32_t ndim,
                                   const int32_t *row_pointers,
                                   const int32_t *col_indices,
                                   const double *values) {
#ifdef WITH_INTEL_DSS
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }
    if (solver->handle == NULL) {
        return NULL_POINTER_ERROR;
    }

    // define the non-zero structure of the matrix
    MKL_INT nnz = row_pointers[ndim];
    MKL_INT status = dss_define_structure(solver->handle,
                                          solver->dss_sym,
                                          row_pointers,
                                          ndim,
                                          ndim,
                                          col_indices,
                                          nnz);
    if (status != MKL_DSS_SUCCESS) {
        return status;
    }

    // reorder the matrix
    status = dss_reorder(solver->handle, solver->dss_opt, 0);
    if (status != MKL_DSS_SUCCESS) {
        return status;
    }

    // factor the matrix
    status = dss_factor_real(solver->handle, solver->dss_type, values);
    return status;
#else
    UNUSED(solver);
    UNUSED(ndim);
    UNUSED(row_pointers);
    UNUSED(col_indices);
    UNUSED(values);
    return NOT_AVAILABLE;
#endif
}

/// @brief Computes the solution of the linear system
/// @param solver Is a pointer to the solver interface
/// @param x Is the left-hand side vector (unknowns)
/// @param rhs Is the right-hand side vector
/// @param row_pointers The row pointers array with size = nrow + 1
/// @param col_indices The column indices array with size = nnz (number of non-zeros)
/// @param values The values array with size = nnz (number of non-zeros)
/// @return A success or fail code
int32_t solver_intel_dss_solve(struct InterfaceIntelDSS *solver,
                               double *x,
                               const double *rhs,
                               const int32_t *col_pointers,
                               const int32_t *row_indices,
                               const double *values) {
#ifdef WITH_INTEL_DSS
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }
    if (solver->handle == NULL) {
        return NULL_POINTER_ERROR;
    }

    // get the solution vector
    MKL_INT n_rhs = 1;
    MKL_INT status = dss_solve_real(solver->handle,
                                    solver->dss_opt,
                                    rhs,
                                    n_rhs,
                                    x);
    return status;
#else
    UNUSED(solver);
    UNUSED(x);
    UNUSED(rhs);
    UNUSED(col_pointers);
    UNUSED(row_indices);
    UNUSED(values);
    return NOT_AVAILABLE;
#endif
}
