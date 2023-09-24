#include <inttypes.h>
#include <stdlib.h>

#include "constants.h"
#include "interface_umfpack.h"

static inline void set_umfpack_verbose(struct InterfaceUMFPACK *solver, int32_t verbose) {
    if (verbose == C_TRUE) {
        solver->control[UMFPACK_PRL] = UMFPACK_PRINT_LEVEL_VERBOSE;
    } else {
        solver->control[UMFPACK_PRL] = UMFPACK_PRINT_LEVEL_SILENT;
    }
}

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

int32_t solver_umfpack_get_ordering(const struct InterfaceUMFPACK *solver) {
    return solver->effective_ordering;
}

int32_t solver_umfpack_get_scaling(const struct InterfaceUMFPACK *solver) {
    return solver->effective_scaling;
}

int32_t solver_umfpack_get_strategy(const struct InterfaceUMFPACK *solver) {
    return solver->effective_strategy;
}

double solver_umfpack_get_det_mx(const struct InterfaceUMFPACK *solver) {
    if (solver == NULL) {
        return 0.0;
    }
    return solver->mx[0];
}

double solver_umfpack_get_det_ex(const struct InterfaceUMFPACK *solver) {
    if (solver == NULL) {
        return 0.0;
    }
    return solver->ex[0];
}

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
