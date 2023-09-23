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

    solver->n = 0;
    solver->nnz = 0;
    solver->ap = NULL;
    solver->ai = NULL;
    solver->ax = NULL;

    solver->symbolic = NULL;
    solver->numeric = NULL;

    return solver;
}

void solver_umfpack_drop(struct InterfaceUMFPACK *solver) {
    if (solver == NULL) {
        return;
    }

    if (solver->ap != NULL) {
        free(solver->ap);
    }
    if (solver->ai != NULL) {
        free(solver->ai);
    }
    if (solver->ax != NULL) {
        free(solver->ax);
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

int32_t solver_umfpack_initialize(struct InterfaceUMFPACK *solver,
                                  int32_t n,
                                  int32_t nnz,
                                  int32_t symmetry,
                                  int32_t ordering,
                                  int32_t scaling,
                                  int32_t compute_determinant) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    umfpack_di_defaults(solver->control);
    solver->control[UMFPACK_STRATEGY] = symmetry;
    solver->effective_strategy = symmetry;

    solver->ap = (int *)malloc((n + 1) * sizeof(int));
    if (solver->ap == NULL) {
        return MALLOC_ERROR;
    }

    solver->ai = (int *)malloc(nnz * sizeof(int));
    if (solver->ai == NULL) {
        free(solver->ap);
        return MALLOC_ERROR;
    }

    solver->ax = (double *)malloc(nnz * sizeof(double));
    if (solver->ax == NULL) {
        free(solver->ai);
        free(solver->ap);
        return MALLOC_ERROR;
    }

    solver->n = n;
    solver->nnz = nnz;

    solver->control[UMFPACK_ORDERING] = ordering;
    solver->control[UMFPACK_SCALE] = scaling;
    solver->effective_ordering = ordering;
    solver->effective_scaling = scaling;

    solver->compute_determinant = compute_determinant;

    return UMFPACK_OK;
}

int32_t solver_umfpack_factorize(struct InterfaceUMFPACK *solver,
                                 int32_t const *indices_i,
                                 int32_t const *indices_j,
                                 double const *values_aij,
                                 int32_t verbose) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    set_umfpack_verbose(solver, verbose);

    // convert triplet to compressed column (must be done for every factorization)

    int code = umfpack_di_triplet_to_col(solver->n, solver->n, solver->nnz,
                                         indices_i, indices_j, values_aij,
                                         solver->ap, solver->ai, solver->ax, NULL);
    if (code != UMFPACK_OK) {
        return code;
    }
    if (verbose == C_TRUE) {
        umfpack_di_report_status(solver->control, code);
    }

    // perform factorization

    code = umfpack_di_symbolic(solver->n, solver->n, solver->ap, solver->ai, solver->ax,
                               &solver->symbolic, solver->control, solver->info);
    if (code != UMFPACK_OK) {
        return code;
    }

    code = umfpack_di_numeric(solver->ap, solver->ai, solver->ax,
                              solver->symbolic, &solver->numeric, solver->control, solver->info);

    if (verbose == C_TRUE) {
        umfpack_di_report_info(solver->control, solver->info);
    }

    // save strategy, ordering, and scaling

    solver->effective_strategy = solver->info[UMFPACK_STRATEGY_USED];
    solver->effective_ordering = solver->info[UMFPACK_ORDERING_USED];
    solver->effective_scaling = solver->control[UMFPACK_SCALE];

    // compute determinant

    if (solver->compute_determinant == C_TRUE) {
        code = umfpack_di_get_determinant(solver->mx, solver->ex, solver->numeric, solver->info);
    }

    return code;
}

int32_t solver_umfpack_solve(struct InterfaceUMFPACK *solver, double *x, const double *rhs, int32_t verbose) {
    if (solver == NULL) {
        return NULL_POINTER_ERROR;
    }

    set_umfpack_verbose(solver, verbose);

    int code = umfpack_di_solve(UMFPACK_A, solver->ap, solver->ai, solver->ax,
                                x, rhs, solver->numeric, solver->control, solver->info);

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
