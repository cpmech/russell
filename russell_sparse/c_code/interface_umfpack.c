#include <inttypes.h>
#include <stdlib.h>

#include "umfpack.h"

#include "constants.h"

const double UMFPACK_PRINT_LEVEL_SILENT = 0.0;  // page 116
const double UMFPACK_PRINT_LEVEL_VERBOSE = 2.0; // page 116

/// @brief Holds the data for UMFPACK
struct InterfaceUMFPACK {
    /// @brief Holds control flags
    double control[UMFPACK_CONTROL];

    /// @brief Holds information data
    double info[UMFPACK_INFO];

    /// @brief Is a handle to symbolic factorization results
    void *symbolic;

    /// @brief Is a handle to numeric factorization results
    void *numeric;

    /// @brief indicates that the initialization has been completed
    C_BOOL initialization_completed;

    /// @brief Indicates that the factorization (at least once) has been completed
    C_BOOL factorization_completed;
};

/// @brief Sets verbose mode
/// @param solver Is a pointer to the solver interface
/// @param verbose Shows messages or not
static inline void set_umfpack_verbose(struct InterfaceUMFPACK *solver, int32_t verbose) {
    if (verbose == C_TRUE) {
        solver->control[UMFPACK_PRL] = UMFPACK_PRINT_LEVEL_VERBOSE;
    } else {
        solver->control[UMFPACK_PRL] = UMFPACK_PRINT_LEVEL_SILENT;
    }
}

/// @brief Allocates a new UMFPACK interface
struct InterfaceUMFPACK *solver_umfpack_new() {
    struct InterfaceUMFPACK *solver = (struct InterfaceUMFPACK *)malloc(sizeof(struct InterfaceUMFPACK));

    if (solver == NULL) {
        return NULL;
    }

    umfpack_di_defaults(solver->control);

    solver->symbolic = NULL;
    solver->numeric = NULL;
    solver->initialization_completed = C_FALSE;
    solver->factorization_completed = C_FALSE;

    return solver;
}

/// @brief Deallocates the UMFPACK interface
void solver_umfpack_drop(struct InterfaceUMFPACK *solver) {
    if (solver == NULL) {
        return;
    }

    if (solver->symbolic != NULL) {
        umfpack_di_free_symbolic(&solver->symbolic);
        free(solver->symbolic);
    }
    if (solver->numeric != NULL) {
        umfpack_di_free_numeric(&solver->numeric);
        free(solver->numeric);
    }

    free(solver);
}

/// @brief Performs the factorization
/// @param solver Is a pointer to the solver interface
/// @note Output
/// @param effective_strategy used strategy regarding symmetry (after factorize)
/// @param effective_ordering used ordering (after factorize)
/// @param effective_scaling used scaling (after factorize)
/// @param reciprocal_condition_number_estimate reciprocal condition number estimate (if requested)
/// @param determinant_coefficient determinant coefficient: det = coefficient * pow(base, exponent)
/// @param determinant_exponent determinant exponent: det = coefficient * pow(base, exponent)
/// @note Input
/// @param ordering Is the ordering code
/// @param scaling Is the scaling code
/// @note Requests
/// @param compute_condition_number_estimate estimates the reciprocal condition number (rcond)
/// @param compute_determinant Requests that determinant be computed
/// @param verbose Shows messages
/// @note Matrix config
/// @param enforce_unsymmetric_strategy Indicates to enforce unsymmetric strategy (not recommended)
/// @param ndim Is the number of rows and columns of the coefficient matrix
/// @note Matrix
/// @param col_pointers The column pointers array with size = ncol + 1
/// @param row_indices The row indices array with size = nnz (number of non-zeros)
/// @param values The values array with size = nnz (number of non-zeros)
/// @return A success or fail code
int32_t solver_umfpack_factorize(struct InterfaceUMFPACK *solver,
                                 // output
                                 int32_t *effective_strategy,
                                 int32_t *effective_ordering,
                                 int32_t *effective_scaling,
                                 double *reciprocal_condition_number_estimate,
                                 double *determinant_coefficient,
                                 double *determinant_exponent,
                                 // input
                                 int32_t ordering,
                                 int32_t scaling,
                                 // requests
                                 C_BOOL compute_condition_number_estimate,
                                 C_BOOL compute_determinant,
                                 C_BOOL verbose,
                                 // matrix config
                                 C_BOOL enforce_unsymmetric_strategy,
                                 int32_t ndim,
                                 // matrix
                                 const int32_t *col_pointers,
                                 const int32_t *row_indices,
                                 const double *values) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    if (solver->initialization_completed == C_FALSE) {
        // perform initialization

        solver->control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_AUTO;
        if (enforce_unsymmetric_strategy == C_TRUE) {
            solver->control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_UNSYMMETRIC;
        }

        solver->control[UMFPACK_ORDERING] = ordering;
        solver->control[UMFPACK_SCALE] = scaling;

        set_umfpack_verbose(solver, verbose);

        // perform symbolic factorization
        int code = umfpack_di_symbolic(ndim,
                                       ndim,
                                       col_pointers,
                                       row_indices,
                                       values,
                                       &solver->symbolic,
                                       solver->control,
                                       solver->info);
        if (code != UMFPACK_OK) {
            return code;
        }

        solver->initialization_completed = C_TRUE;

    } else {
        // free the previous numeric to avoid memory leak
        umfpack_di_free_numeric(&solver->numeric);
    }

    // perform numeric factorization
    int code = umfpack_di_numeric(col_pointers,
                                  row_indices,
                                  values,
                                  solver->symbolic,
                                  &solver->numeric,
                                  solver->control,
                                  solver->info);
    if (verbose == C_TRUE) {
        umfpack_di_report_info(solver->control, solver->info);
    }

    // save strategy, ordering, and scaling
    *effective_strategy = solver->info[UMFPACK_STRATEGY_USED];
    *effective_ordering = solver->info[UMFPACK_ORDERING_USED];
    *effective_scaling = solver->control[UMFPACK_SCALE];

    // condition number
    if (compute_condition_number_estimate == C_TRUE) {
        *reciprocal_condition_number_estimate = solver->info[UMFPACK_RCOND];
    }

    // compute determinant
    if (compute_determinant == C_TRUE) {
        code = umfpack_di_get_determinant(determinant_coefficient,
                                          determinant_exponent,
                                          solver->numeric,
                                          solver->info);
    }

    solver->factorization_completed = C_TRUE;

    return code;
}

/// @brief Computes the solution of the linear system
/// @param solver Is a pointer to the solver interface
/// @param x Is the left-hand side vector (unknowns)
/// @param rhs Is the right-hand side vector
/// @param col_pointers The column pointers array with size = ncol + 1
/// @param row_indices The row indices array with size = nnz (number of non-zeros)
/// @param values The values array with size = nnz (number of non-zeros)
/// @param verbose Shows messages
/// @return A success or fail code
int32_t solver_umfpack_solve(struct InterfaceUMFPACK *solver,
                             double *x,
                             const double *rhs,
                             const int32_t *col_pointers,
                             const int32_t *row_indices,
                             const double *values,
                             C_BOOL verbose) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    if (solver->factorization_completed == C_FALSE) {
        return NEED_FACTORIZATION;
    }

    set_umfpack_verbose(solver, verbose);

    int code = umfpack_di_solve(UMFPACK_A,
                                col_pointers,
                                row_indices,
                                values,
                                x,
                                rhs,
                                solver->numeric,
                                solver->control,
                                solver->info);
    if (verbose == C_TRUE) {
        umfpack_di_report_info(solver->control, solver->info);
    }

    return code;
}

/// @brief Converts COO matrix (with possible duplicates) to CSC matrix
/// @param nrow Is the number of rows
/// @param ncol Is the number of columns
/// @param nnz Is the number of non-zero values, including duplicates
/// @param indices_i Are the CooMatrix row indices
/// @param indices_j Are the CooMatrix column indices
/// @param values_aij Are the CooMatrix values, including duplicates
/// @return A success or fail code
int32_t umfpack_coo_to_csc(int32_t *col_pointers,
                           int32_t *row_indices,
                           double *values,
                           int32_t nrow,
                           int32_t ncol,
                           int32_t nnz,
                           int32_t const *indices_i,
                           int32_t const *indices_j,
                           double const *values_aij) {
    int code = umfpack_di_triplet_to_col(nrow, ncol, nnz,
                                         indices_i, indices_j, values_aij,
                                         col_pointers, row_indices, values, NULL);
    return code;
}
