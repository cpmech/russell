#include <inttypes.h>
#include <stdlib.h>

#include "umfpack.h"

#include "constants.h"

/// @brief Holds the data for UMFPACK
struct InterfaceComplexUMFPACK {
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
static inline void set_complex_umfpack_verbose(struct InterfaceComplexUMFPACK *solver, int32_t verbose) {
    if (verbose == C_TRUE) {
        solver->control[UMFPACK_PRL] = UMFPACK_PRINT_LEVEL_VERBOSE;
    } else {
        solver->control[UMFPACK_PRL] = UMFPACK_PRINT_LEVEL_SILENT;
    }
}

/// @brief Allocates a new UMFPACK interface
struct InterfaceComplexUMFPACK *complex_solver_umfpack_new() {
    struct InterfaceComplexUMFPACK *solver = (struct InterfaceComplexUMFPACK *)malloc(sizeof(struct InterfaceComplexUMFPACK));

    if (solver == NULL) {
        return NULL;
    }

    umfpack_zi_defaults(solver->control);

    solver->symbolic = NULL;
    solver->numeric = NULL;
    solver->initialization_completed = C_FALSE;
    solver->factorization_completed = C_FALSE;

    return solver;
}

/// @brief Deallocates the UMFPACK interface
void complex_solver_umfpack_drop(struct InterfaceComplexUMFPACK *solver) {
    if (solver == NULL) {
        return;
    }

    if (solver->symbolic != NULL) {
        umfpack_zi_free_symbolic(&solver->symbolic);
        free(solver->symbolic);
    }
    if (solver->numeric != NULL) {
        umfpack_zi_free_numeric(&solver->numeric);
        free(solver->numeric);
    }

    free(solver);
}

/// @brief Performs the symbolic factorization
/// @return A success or fail code
int32_t complex_solver_umfpack_initialize(struct InterfaceComplexUMFPACK *solver,
                                          int32_t ordering,
                                          int32_t scaling,
                                          C_BOOL verbose,
                                          C_BOOL enforce_unsymmetric_strategy,
                                          int32_t ndim,
                                          const int32_t *col_pointers,
                                          const int32_t *row_indices,
                                          const COMPLEX64 *values) {
    if (solver == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (solver->initialization_completed == C_TRUE) {
        return ERROR_ALREADY_INITIALIZED;
    }

    solver->control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_AUTO;
    if (enforce_unsymmetric_strategy == C_TRUE) {
        solver->control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_UNSYMMETRIC;
    }

    solver->control[UMFPACK_ORDERING] = ordering;
    solver->control[UMFPACK_SCALE] = scaling;

    set_complex_umfpack_verbose(solver, verbose);

    int code = umfpack_zi_symbolic(ndim,
                                   ndim,
                                   col_pointers,
                                   row_indices,
                                   values,
                                   NULL,
                                   &solver->symbolic,
                                   solver->control,
                                   solver->info);
    if (code != UMFPACK_OK) {
        return code;
    }

    solver->initialization_completed = C_TRUE;

    return SUCCESSFUL_EXIT;
}

/// @brief Performs the numeric factorization
int32_t complex_solver_umfpack_factorize(struct InterfaceComplexUMFPACK *solver,
                                         int32_t *effective_strategy,
                                         int32_t *effective_ordering,
                                         int32_t *effective_scaling,
                                         double *rcond_estimate,
                                         double *determinant_coefficient_real,
                                         double *determinant_coefficient_imag,
                                         double *determinant_exponent,
                                         C_BOOL compute_determinant,
                                         C_BOOL verbose,
                                         const int32_t *col_pointers,
                                         const int32_t *row_indices,
                                         const COMPLEX64 *values) {
    if (solver == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (solver->initialization_completed == C_FALSE) {
        return ERROR_NEED_INITIALIZATION;
    }

    if (solver->factorization_completed == C_TRUE) {
        // free the previous numeric to avoid memory leak
        umfpack_zi_free_numeric(&solver->numeric);
    }

    // perform numeric factorization
    int code = umfpack_zi_numeric(col_pointers,
                                  row_indices,
                                  values,
                                  NULL,
                                  solver->symbolic,
                                  &solver->numeric,
                                  solver->control,
                                  solver->info);
    if (verbose == C_TRUE) {
        umfpack_zi_report_info(solver->control, solver->info);
    }

    // save strategy, ordering, and scaling
    *effective_strategy = solver->info[UMFPACK_STRATEGY_USED];
    *effective_ordering = solver->info[UMFPACK_ORDERING_USED];
    *effective_scaling = solver->control[UMFPACK_SCALE];

    // reciprocal condition number estimate
    *rcond_estimate = solver->info[UMFPACK_RCOND];

    // compute determinant
    if (compute_determinant == C_TRUE) {
        code = umfpack_zi_get_determinant(determinant_coefficient_real,
                                          determinant_coefficient_imag,
                                          determinant_exponent,
                                          solver->numeric,
                                          solver->info);
    } else {
        *determinant_coefficient_real = 0.0;
        *determinant_coefficient_imag = 0.0;
        *determinant_exponent = 0.0;
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
int32_t complex_solver_umfpack_solve(struct InterfaceComplexUMFPACK *solver,
                                     COMPLEX64 *x,
                                     const COMPLEX64 *rhs,
                                     const int32_t *col_pointers,
                                     const int32_t *row_indices,
                                     const COMPLEX64 *values,
                                     C_BOOL verbose) {
    if (solver == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (solver->factorization_completed == C_FALSE) {
        return ERROR_NEED_FACTORIZATION;
    }

    set_complex_umfpack_verbose(solver, verbose);

    int code = umfpack_zi_solve(UMFPACK_A,
                                col_pointers,
                                row_indices,
                                values,
                                NULL,
                                x,
                                NULL,
                                rhs,
                                NULL,
                                solver->numeric,
                                solver->control,
                                solver->info);
    if (verbose == C_TRUE) {
        umfpack_zi_report_info(solver->control, solver->info);
    }

    return code;
}
