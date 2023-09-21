#pragma once

#include "umfpack.h"

const double UMF_SYMMETRY[2] = {
    UMFPACK_STRATEGY_UNSYMMETRIC, // Unsymmetric
    UMFPACK_STRATEGY_SYMMETRIC,   // General symmetric
};

const double UMF_ORDERING[10] = {
    UMFPACK_ORDERING_AMD,     // Amd
    UMFPACK_DEFAULT_ORDERING, // Amf => Auto
    UMFPACK_DEFAULT_ORDERING, // Auto
    UMFPACK_ORDERING_BEST,    // Best
    UMFPACK_ORDERING_CHOLMOD, // Cholmod
    UMFPACK_ORDERING_METIS,   // Metis
    UMFPACK_ORDERING_NONE,    // No
    UMFPACK_DEFAULT_ORDERING, // Pord => Auto
    UMFPACK_DEFAULT_ORDERING, // Qamd => Auto
    UMFPACK_DEFAULT_ORDERING, // Scotch => Auto
};

const double UMF_SCALING[9] = {
    UMFPACK_DEFAULT_SCALE, // Auto
    UMFPACK_DEFAULT_SCALE, // Column => Auto
    UMFPACK_DEFAULT_SCALE, // Diagonal => Auto
    UMFPACK_SCALE_MAX,     // Max
    UMFPACK_SCALE_NONE,    // No
    UMFPACK_DEFAULT_SCALE, // RowCol => Auto
    UMFPACK_DEFAULT_SCALE, // RowColIter => Auto
    UMFPACK_DEFAULT_SCALE, // RowColRig => Auto
    UMFPACK_SCALE_SUM,     // Sum
};

const double UMF_PRINT_LEVEL_SILENT = 0.0;  // page 116
const double UMF_PRINT_LEVEL_VERBOSE = 2.0; // page 116
