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

void print_csr(int32_t ndim,
               const int32_t *row_pointers,
               const int32_t *col_indices,
               const double *values) {
    for (int32_t i = 0; i < ndim; i++) {
        for (int32_t p = row_pointers[i]; p < row_pointers[i + 1]; p++) {
            int32_t j = col_indices[p];
            printf("%d %d => %g\n", i, j, values[p]);
        }
    }
}

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
    int32_t initialization_completed;

    /// @brief Indicates that the factorization (at least once) has been completed
    int32_t factorization_completed;
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

/// @brief Performs the factorization
/// @param solver Is a pointer to the solver interface
/// @note Output
/// @param determinant_coefficient determinant coefficient: det = coefficient * pow(2, exponent)
/// @param determinant_exponent determinant exponent: det = coefficient * pow(2, exponent)
/// @note Requests
/// @param compute_determinant Requests that determinant be computed
/// @note Matrix config
/// @param general_symmetric Whether the matrix is general symmetric (not necessarily positive-definite) or not
/// @param positive_definite Whether the matrix is symmetric and positive-definite or not
/// @param ndim Is the number of rows and columns of the coefficient matrix
/// @note Matrix
/// @param row_pointers The row pointers array with size = nrow + 1
/// @param col_indices The columns indices array with size = nnz (number of non-zeros)
/// @param values The values array with size = nnz (number of non-zeros)
/// @return A success or fail code
int32_t solver_intel_dss_factorize(struct InterfaceIntelDSS *solver,
                                   // output
                                   double *determinant_coefficient,
                                   double *determinant_exponent,
                                   // requests
                                   int32_t compute_determinant,
                                   // matrix config
                                   int32_t general_symmetric,
                                   int32_t positive_definite,
                                   int32_t ndim,
                                   // matrix
                                   const int32_t *row_pointers,
                                   const int32_t *col_indices,
                                   const double *values) {
#ifdef WITH_INTEL_DSS
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    // perform initialization
    if (solver->initialization_completed == C_FALSE) {
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
            return NULL_POINTER_ERROR;
        }

        solver->initialization_completed = C_TRUE;

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
    }

    // print_csr(ndim, row_pointers, col_indices, values);

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
    }

    // done
    return status;
#else
    UNUSED(solver);
    UNUSED(determinant_coefficient);
    UNUSED(determinant_exponent);
    UNUSED(compute_determinant);
    UNUSED(general_symmetric);
    UNUSED(positive_definite);
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
/// @return A success or fail code
int32_t solver_intel_dss_solve(struct InterfaceIntelDSS *solver,
                               double *x,
                               const double *rhs) {
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
    return NOT_AVAILABLE;
#endif
}
