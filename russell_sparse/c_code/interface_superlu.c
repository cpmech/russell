#include <inttypes.h>
#include <stdlib.h>

#include "slu_ddefs.h"

#include "constants.h"

#define UNUSED(var) (void)var

/// @brief Holds the data for SuperLU
struct InterfaceSuperLU {

    /// @brief indicates that the initialization has been completed
    int32_t initialization_completed;

    /// @brief Indicates that the factorization (at least once) has been completed
    int32_t factorization_completed;
};

/// @brief Sets verbose mode
/// @param solver Is a pointer to the solver interface
/// @param verbose Shows messages or not
static inline void set_superlu_verbose(struct InterfaceSuperLU *solver, int32_t verbose) {

    UNUSED(solver);

    if (verbose == C_TRUE) {
        // todo
    } else {
        // todo
    }
}

/// @brief Allocates a new SuperLU interface
struct InterfaceSuperLU *solver_superlu_new() {
    struct InterfaceSuperLU *solver = (struct InterfaceSuperLU *)malloc(sizeof(struct InterfaceSuperLU));

    if (solver == NULL) {
        return NULL;
    }

    solver->initialization_completed = C_FALSE;
    solver->factorization_completed = C_FALSE;

    return solver;
}

/// @brief Deallocates the SuperLU interface
void solver_superlu_drop(struct InterfaceSuperLU *solver) {
    if (solver == NULL) {
        return;
    }

    free(solver);
}

/// @brief Performs the factorization
/// @param solver Is a pointer to the solver interface
/// @note Output
/// @param effective_strategy used strategy regarding symmetry (after factorize)
/// @param effective_ordering used ordering (after factorize)
/// @param effective_scaling used scaling (after factorize)
/// @param determinant_coefficient determinant coefficient: det = coefficient * pow(base, exponent)
/// @param determinant_exponent determinant exponent: det = coefficient * pow(base, exponent)
/// @note Input
/// @param ordering Is the ordering code
/// @param scaling Is the scaling code
/// @note Requests
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
int32_t solver_superlu_factorize(struct InterfaceSuperLU *solver,
                                 // output
                                 int32_t *effective_strategy,
                                 int32_t *effective_ordering,
                                 int32_t *effective_scaling,
                                 double *determinant_coefficient,
                                 double *determinant_exponent,
                                 // input
                                 int32_t ordering,
                                 int32_t scaling,
                                 // requests
                                 int32_t compute_determinant,
                                 int32_t verbose,
                                 // matrix config
                                 int32_t enforce_unsymmetric_strategy,
                                 int32_t ndim,
                                 // matrix
                                 const int32_t *col_pointers,
                                 const int32_t *row_indices,
                                 const double *values) {

    UNUSED(solver);
    UNUSED(effective_strategy);
    UNUSED(effective_ordering);
    UNUSED(effective_scaling);
    UNUSED(determinant_coefficient);
    UNUSED(determinant_exponent);
    UNUSED(ordering);
    UNUSED(scaling);
    UNUSED(compute_determinant);
    UNUSED(verbose);
    UNUSED(enforce_unsymmetric_strategy);
    UNUSED(ndim);
    UNUSED(col_pointers);
    UNUSED(row_indices);
    UNUSED(values);

    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    if (solver->initialization_completed == C_FALSE) {
        // perform initialization

        solver->initialization_completed = C_TRUE;

    } else {
    }

    // perform numeric factorization

    // save strategy, ordering, and scaling

    // compute determinant
    if (compute_determinant == C_TRUE) {
    }

    solver->factorization_completed = C_TRUE;

    return 0;
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
int32_t solver_superlu_solve(struct InterfaceSuperLU *solver,
                             double *x,
                             const double *rhs,
                             const int32_t *col_pointers,
                             const int32_t *row_indices,
                             const double *values,
                             int32_t verbose) {

    UNUSED(x);
    UNUSED(rhs);
    UNUSED(col_pointers);
    UNUSED(row_indices);
    UNUSED(values);

    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    if (solver->factorization_completed == C_FALSE) {
        return NEED_FACTORIZATION;
    }

    set_superlu_verbose(solver, verbose);

    if (verbose == C_TRUE) {
    }

    return 0;
}
