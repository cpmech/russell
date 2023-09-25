#include <inttypes.h>
#include <stdlib.h>

#include "umfpack.h"

#include "constants.h"

const double UMFPACK_PRINT_LEVEL_SILENT = 0.0;  // page 116
const double UMFPACK_PRINT_LEVEL_VERBOSE = 2.0; // page 116

/// @brief Holds the data for UMFPACK
struct InterfaceUMFPACK {
    double control[UMFPACK_CONTROL]; // control flags
    double info[UMFPACK_INFO];       // information data
    void *symbolic;                  // handle to symbolic factorization results
    void *numeric;                   // handle to numeric factorization results
    int effective_strategy;          // used strategy regarding symmetry (after factorize)
    int effective_ordering;          // used ordering (after factorize)
    int effective_scaling;           // used scaling (after factorize)
    double mx[1];                    // mx[0] is the coefficient mx of the determinant = mx * 10 ^ ex (after factorize)
    double ex[1];                    // ex[0] is the exponent ex of the determinant = mx * 10 ^ ex (after factorize)
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
    solver->effective_strategy = 0;
    solver->effective_ordering = 0;
    solver->effective_scaling = 0;
    solver->mx[0] = 0.0;
    solver->ex[0] = 0.0;

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
/// @param ndim Is the number of rows and columns of the coefficient matrix
/// @param symmetry Is the UMFPACK code for the kind of symmetry, if any
/// @param ordering Is the UMFPACK ordering code
/// @param scaling Is the UMFPACK scaling code
/// @param col_pointers The column pointers array with size = ncol + 1
/// @param row_indices The row indices array with size = nnz (number of non-zeros)
/// @param values The values array with size = nnz (number of non-zeros)
/// @param compute_determinant Whether the determinant should be computed or not
/// @param verbose Shows messages
/// @return A success or fail code
int32_t solver_umfpack_factorize(struct InterfaceUMFPACK *solver,
                                 int32_t ndim,
                                 int32_t symmetry,
                                 int32_t ordering,
                                 int32_t scaling,
                                 const int32_t *col_pointers,
                                 const int32_t *row_indices,
                                 const double *values,
                                 int32_t compute_determinant,
                                 int32_t verbose) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    solver->control[UMFPACK_STRATEGY] = symmetry;
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

    // perform numeric factorization
    code = umfpack_di_numeric(col_pointers,
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
    solver->effective_strategy = solver->info[UMFPACK_STRATEGY_USED];
    solver->effective_ordering = solver->info[UMFPACK_ORDERING_USED];
    solver->effective_scaling = solver->control[UMFPACK_SCALE];

    // compute determinant
    if (compute_determinant == C_TRUE) {
        code = umfpack_di_get_determinant(solver->mx, solver->ex, solver->numeric, solver->info);
    }

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
                             int32_t verbose) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
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

/// @brief Returns the effective ordering using during the computations
/// @param solver Is a pointer to the solver
/// @return The used UMFPACK ordering code
int32_t solver_umfpack_get_ordering(const struct InterfaceUMFPACK *solver) {
    if (solver == NULL) {
        return -1;
    }
    return solver->effective_ordering;
}

/// @brief Returns the effective scaling using during the computations
/// @param solver Is a pointer to the solver
/// @return The used UMFPACK scaling code
int32_t solver_umfpack_get_scaling(const struct InterfaceUMFPACK *solver) {
    if (solver == NULL) {
        return -1;
    }
    return solver->effective_scaling;
}

/// @brief Returns the effective symmetry/non-symmetry strategy using during the computations
/// @param solver Is a pointer to the solver
/// @return The strategy (concerning symmetry or the lack of it) taken by UMFPACK
int32_t solver_umfpack_get_strategy(const struct InterfaceUMFPACK *solver) {
    if (solver == NULL) {
        return -1;
    }
    return solver->effective_strategy;
}

/// @brief Returns the coefficient needed to compute the determinant, if requested
/// @param solver Is a pointer to the solver
/// @return The coefficient mx of the determinant = mx * 10 ^ ex
double solver_umfpack_get_det_mx(const struct InterfaceUMFPACK *solver) {
    if (solver == NULL) {
        return 0.0;
    }
    return solver->mx[0];
}

/// @brief Returns the exponent needed to compute the determinant, if requested
/// @param solver Is a pointer to the solver
/// @return The exponent ex of the determinant = mx * 10 ^ ex
double solver_umfpack_get_det_ex(const struct InterfaceUMFPACK *solver) {
    if (solver == NULL) {
        return 0.0;
    }
    return solver->ex[0];
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
