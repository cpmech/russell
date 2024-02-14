#include <stdio.h>
#include <stdlib.h>

#ifdef WITH_INTEL_DSS
#include "mkl_dss.h"
#else
#define MKL_INT int
#define _MKL_DSS_HANDLE_t void *
#define UNUSED(var) (void)var
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

    /// @brief indicates that the initialization has been completed
    C_BOOL initialization_completed;

    /// @brief Indicates that the factorization (at least once) has been completed
    C_BOOL factorization_completed;
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
    solver->initialization_completed = C_FALSE;
    solver->factorization_completed = C_FALSE;

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

/// @brief Defines the structure and reorder analysis
/// @return A success or fail code
int32_t solver_intel_dss_initialize(struct InterfaceIntelDSS *solver,
                                    C_BOOL general_symmetric,
                                    C_BOOL positive_definite,
                                    int32_t ndim,
                                    const int32_t *row_pointers,
                                    const int32_t *col_indices) {
#ifdef WITH_INTEL_DSS

    if (solver == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (solver->initialization_completed == C_TRUE) {
        return ERROR_ALREADY_INITIALIZED;
    }

    solver->dss_opt = MKL_DSS_MSG_LVL_WARNING + MKL_DSS_TERM_LVL_ERROR + MKL_DSS_ZERO_BASED_INDEXING;

    solver->dss_sym = MKL_DSS_NON_SYMMETRIC;
    if (general_symmetric == C_TRUE) {
        solver->dss_sym = MKL_DSS_SYMMETRIC;
    }

    solver->dss_type = MKL_DSS_INDEFINITE;
    if (positive_definite == C_TRUE) {
        solver->dss_type = MKL_DSS_POSITIVE_DEFINITE;
    }

    MKL_INT status = dss_create(solver->handle, solver->dss_opt);

    if (status != MKL_DSS_SUCCESS) {
        return status;
    }

    if (solver->handle == NULL) {
        return ERROR_NULL_POINTER;
    }

    // define the non-zero structure of the matrix
    MKL_INT nnz = row_pointers[ndim];
    status = dss_define_structure(solver->handle,
                                  solver->dss_sym,
                                  row_pointers,
                                  ndim,
                                  ndim,
                                  col_indices,
                                  nnz);
    if (status != MKL_DSS_SUCCESS) {
        return status;
    }

    // reorder the matrix (NOTE: we cannot call reorder again)
    status = dss_reorder(solver->handle, solver->dss_opt, 0);
    if (status != MKL_DSS_SUCCESS) {
        return status;
    }

    solver->initialization_completed = C_TRUE;

    return SUCCESSFUL_EXIT;

#else
    UNUSED(solver);
    UNUSED(general_symmetric);
    UNUSED(positive_definite);
    UNUSED(ndim);
    UNUSED(row_pointers);
    UNUSED(col_indices);
    return ERROR_NOT_AVAILABLE;
#endif
}

/// @brief Performs the factorization
/// @return A success or fail code
int32_t solver_intel_dss_factorize(struct InterfaceIntelDSS *solver,
                                   double *determinant_coefficient,
                                   double *determinant_exponent,
                                   C_BOOL compute_determinant,
                                   const double *values) {
#ifdef WITH_INTEL_DSS

    if (solver == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (solver->initialization_completed == C_FALSE) {
        return ERROR_NEED_INITIALIZATION;
    }

    // factor the matrix
    MKL_INT status = dss_factor_real(solver->handle, solver->dss_type, values);

    // compute determinant
    *determinant_coefficient = 0.0;
    *determinant_exponent = 0.0;
    if (compute_determinant == C_TRUE) {
        _CHARACTER_t stat_in[] = "determinant";
        _DOUBLE_PRECISION_t stat_out[5];
        status = dss_statistics(solver->handle, solver->dss_opt, stat_in, stat_out);
        if (status != MKL_DSS_SUCCESS) {
            return status;
        }
        *determinant_exponent = stat_out[0];
        *determinant_coefficient = stat_out[1];
    } else {
        *determinant_coefficient = 0.0;
        *determinant_exponent = 0.0;
    }

    solver->factorization_completed = C_TRUE;

    // done
    return status;

#else
    UNUSED(solver);
    UNUSED(determinant_coefficient);
    UNUSED(determinant_exponent);
    UNUSED(compute_determinant);
    UNUSED(values);
    return ERROR_NOT_AVAILABLE;
#endif
}

/// @brief Computes the solution of the linear system
/// @param solver Is a pointer to the solver interface
/// @param x Is the left-hand side vector (unknowns)
/// @param rhs Is the right-hand side vector
/// @return A success or fail code
int32_t solver_intel_dss_solve(struct InterfaceIntelDSS *solver,
                               double *x,
                               const double *rhs) {
#ifdef WITH_INTEL_DSS

    if (solver == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (solver->handle == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (solver->factorization_completed == C_FALSE) {
        return ERROR_NEED_FACTORIZATION;
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
    return ERROR_NOT_AVAILABLE;
#endif
}
