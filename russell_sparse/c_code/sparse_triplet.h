#ifndef SPARSE_TRIPLET_H
#define SPARSE_TRIPLET_H

#include <stdlib.h>

#include "constants.h"
#include "dmumps_c.h"

struct SparseTriplet {
    MUMPS_INT *indices_i;  // zero- or one-based indices stored here
    MUMPS_INT *indices_j;  // zero- or one-based indices stored here
    double *values_x;      // the non-zero entries in the matrix
};

struct SparseTriplet *new_sparse_triplet(int32_t max) {
    struct SparseTriplet *trip = (struct SparseTriplet *)malloc(sizeof(struct SparseTriplet));

    if (trip == NULL) {
        return NULL;
    }

    MUMPS_INT *indices_i = (MUMPS_INT *)malloc(max * sizeof(MUMPS_INT));

    if (indices_i == NULL) {
        free(trip);
        return NULL;
    }

    MUMPS_INT *indices_j = (MUMPS_INT *)malloc(max * sizeof(MUMPS_INT));

    if (indices_j == NULL) {
        free(trip);
        return NULL;
    }

    double *values_x = (double *)malloc(max * sizeof(double));

    if (values_x == NULL) {
        free(trip);
        return NULL;
    }

    trip->indices_i = indices_i;
    trip->indices_j = indices_j;
    trip->values_x = values_x;

    return trip;
}

void drop_sparse_triplet(struct SparseTriplet *trip) {
    if (trip == NULL) {
        return;
    }
    free(trip->indices_i);
    free(trip->indices_j);
    free(trip->values_x);
    free(trip);
}

int32_t sparse_triplet_set(struct SparseTriplet *trip, int32_t pos, int32_t i, int32_t j, double x) {
    if (trip == NULL) {
        return C_HAS_ERROR;
    }
    trip->indices_i[pos] = i + 1;
    trip->indices_j[pos] = j + 1;
    trip->values_x[pos] = x;
    return C_NO_ERROR;
}

int32_t sparse_triplet_get(struct SparseTriplet *trip, int32_t pos, int32_t *i, int32_t *j, double *x) {
    if (trip == NULL) {
        return C_HAS_ERROR;
    }
    *i = trip->indices_i[pos] - 1;
    *j = trip->indices_j[pos] - 1;
    *x = trip->values_x[pos];
    return C_NO_ERROR;
}

#endif
