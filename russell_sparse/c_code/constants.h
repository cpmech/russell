#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <inttypes.h>

#include "dmumps_c.h"
#include "umfpack.h"

const int32_t NULL_POINTER_ERROR = 100000;
const int32_t MALLOC_ERROR = 200000;
const int32_t VERSION_ERROR = 300000;
const int32_t C_TRUE = 1;
const int32_t C_FALSE = 0;

const MUMPS_INT MUMPS_IGNORED = 0;

const MUMPS_INT MUMPS_JOB_INITIALIZE = -1;
const MUMPS_INT MUMPS_JOB_TERMINATE = -2;
const MUMPS_INT MUMPS_JOB_ANALYZE = 1;
const MUMPS_INT MUMPS_JOB_FACTORIZE = 2;
const MUMPS_INT MUMPS_JOB_SOLVE = 3;

const MUMPS_INT MUMPS_PAR_HOST_ALSO_WORKS = 1;     // section 5.1.4, page 26
const MUMPS_INT MUMPS_ICNTL5_ASSEMBLED_MATRIX = 0; // section 5.2.2, page 27
const MUMPS_INT MUMPS_ICNTL18_CENTRALIZED = 0;     // section 5.2.2, page 27
const MUMPS_INT MUMPS_ICNTL6_PERMUT_AUTO = 7;      // section 5.3, page 32
const MUMPS_INT MUMPS_ICNTL28_SEQUENTIAL = 1;      // section 5.4, page 33

const double UMF_PRINT_LEVEL_SILENT = 0.0;  // page 116
const double UMF_PRINT_LEVEL_VERBOSE = 2.0; // page 116

const MUMPS_INT MMP_SYMMETRY[3] = {
    0, // Unsymmetric
    1, // Positive-definite symmetric
    2, // General symmetric
};

const MUMPS_INT MMP_ORDERING[10] = {
    0, // 0: Amd
    2, // 1: Amf
    7, // 2: Auto
    7, // 3: Best => Auto
    7, // 4: Cholmod => Auto
    5, // 5: Metis
    7, // 6: No => Auto
    4, // 7: Pord
    6, // 8: Qamd
    3, // 9: Scotch
};

const MUMPS_INT MMP_SCALING[9] = {
    77, // 0: Auto
    3,  // 1: Column
    1,  // 2: Diagonal
    77, // 3: Max => Auto
    0,  // 4: No
    4,  // 5: RowCol
    7,  // 6: RowColIter
    8,  // 7: RowColRig
    77, // 8: Sum => Auto
};

inline int32_t get_russell_ordering(MUMPS_INT mmp_ordering) {
    if (mmp_ordering == 0) {
        return 0; // Amd
    } else if (mmp_ordering == 2) {
        return 1; // Amf
    } else if (mmp_ordering == 7) {
        return 2; // Auto
    } else if (mmp_ordering == 5) {
        return 5; // Metis
    } else if (mmp_ordering == 4) {
        return 7; // Pord
    } else if (mmp_ordering == 6) {
        return 8; // Qamd
    } else if (mmp_ordering == 3) {
        return 9; // Scotch
    } else {
        return 2; // Auto
    }
}

inline int32_t get_russell_scaling(MUMPS_INT mmp_scaling) {
    if (mmp_scaling == 77) {
        return 0; // Auto
    } else if (mmp_scaling == 3) {
        return 1; // Column
    } else if (mmp_scaling == 1) {
        return 2; // Diagonal
    } else if (mmp_scaling == 0) {
        return 4; // No
    } else if (mmp_scaling == 4) {
        return 5; // RowCol
    } else if (mmp_scaling == 7) {
        return 6; // RowColIter
    } else if (mmp_scaling == 8) {
        return 7; // RowColRig
    } else {
        return 0; // Auto
    }
}

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

#endif
