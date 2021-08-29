#ifndef SPARSE_TRIPLET_H
#define SPARSE_TRIPLET_H

#include <stdio.h>
#include <stdlib.h>

#include "constants.h"
#include "dmumps_c.h"

struct SparseTriplet {
    int32_t m;    // number of rows
    int32_t n;    // number of columns
    int32_t pos;  // current index => nnz in the end
    int32_t max;  // max allowed number of entries
    // bool one_based;        // indices (i; j) start with 1 instead of 0 (e.g. for MUMPS)
    // bool symmetric;        // symmetric matrix?, but WITHOUT both sides of the diagonal
    MUMPS_INT *indices_i;  // zero- or one-based indices stored here
    MUMPS_INT *indices_j;  // zero- or one-based indices stored here
    double *values_x;      // the non-zero entries in the matrix
};

//  int32_t is_one_based,
//  int32_t is_symmetric

struct SparseTriplet *new_sparse_triplet(int32_t m,
                                         int32_t n,
                                         int32_t max) {
    if (m == 0 || n == 0 || max == 0) {
        return NULL;
    }

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

    trip->m = m;
    trip->n = n;
    trip->pos = 0;
    trip->max = max;
    // trip->one_based = is_one_based == TRUE;
    // trip->symmetric = is_symmetric == TRUE;
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

int32_t sparse_triplet_start(struct SparseTriplet *trip) {
    if (trip == NULL) {
        return HAS_ERROR;
    }
    trip->pos = 0;
    return NO_ERROR;
}

int32_t sparse_triplet_put(struct SparseTriplet *trip, int32_t i, int32_t j, double x) {
    if (trip == NULL) {
        return HAS_ERROR;
    }
    trip->indices_i[trip->pos] = i;
    trip->indices_j[trip->pos] = j;
    trip->values_x[trip->pos] = x;
    trip->pos += 1;
    return NO_ERROR;
}

#endif
