#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <inttypes.h>

#include "dmumps_c.h"
#include "umfpack.h"

const int32_t NULL_POINTER_ERROR = 100000;
const int32_t MALLOC_ERROR = 200000;
const int32_t C_TRUE = 1;
const int32_t C_FALSE = 0;

const MUMPS_INT MUMPS_IGNORED = 0;

const MUMPS_INT MUMPS_JOB_INITIALIZE = -1;
const MUMPS_INT MUMPS_JOB_TERMINATE = -2;
const MUMPS_INT MUMPS_JOB_ANALYZE = 1;
const MUMPS_INT MUMPS_JOB_FACTORIZE = 2;
const MUMPS_INT MUMPS_JOB_SOLVE = 3;

const MUMPS_INT MUMPS_PAR_HOST_ALSO_WORKS = 1;      // section 5.1.4, page 26
const MUMPS_INT MUMPS_ICNTL5_ASSEMBLED_MATRIX = 0;  // section 5.2.2, page 27
const MUMPS_INT MUMPS_ICNTL18_CENTRALIZED = 0;      // section 5.2.2, page 27
const MUMPS_INT MUMPS_ICNTL6_PERMUT_AUTO = 7;       // section 5.3, page 32
const MUMPS_INT MUMPS_ICNTL28_SEQUENTIAL = 1;       // section 5.4, page 33

const double UMF_PRINT_LEVEL_SILENT = 0.0;   // page 116
const double UMF_PRINT_LEVEL_VERBOSE = 2.0;  // page 116

const MUMPS_INT MMP_SYMMETRY[4] = {
    0,  // Auto => Unsymmetric
    2,  // General symmetric
    0,  // Unsymmetric
    1,  // Positive-definite symmetric
};

const MUMPS_INT MMP_ORDERING[10] = {
    0,  // Amd
    2,  // Amf
    7,  // Auto
    7,  // Best => Auto
    7,  // Cholmod => Auto
    5,  // Metis
    7,  // No => Auto
    4,  // Pord
    6,  // Qamd
    3,  // Scotch
};

const MUMPS_INT MMP_SCALING[9] = {
    77,  // Auto
    3,   // Column
    1,   // Diagonal
    77,  // Max => Auto
    0,   // No
    4,   // RowCol
    7,   // RowColIter
    8,   // RowColRig
    77,  // Sum => Auto
};

const double UMF_SYMMETRY[4] = {
    UMFPACK_STRATEGY_AUTO,         // Auto
    UMFPACK_STRATEGY_SYMMETRIC,    // General symmetric
    UMFPACK_STRATEGY_UNSYMMETRIC,  // Unsymmetric
    UMFPACK_STRATEGY_SYMMETRIC,    // Positive-definite symmetric => General symmetric
};

const double UMF_ORDERING[10] = {
    UMFPACK_ORDERING_AMD,      // Amd
    UMFPACK_DEFAULT_ORDERING,  // Amf => Auto
    UMFPACK_DEFAULT_ORDERING,  // Auto
    UMFPACK_ORDERING_BEST,     // Best
    UMFPACK_ORDERING_CHOLMOD,  // Cholmod
    UMFPACK_ORDERING_METIS,    // Metis
    UMFPACK_ORDERING_NONE,     // No
    UMFPACK_DEFAULT_ORDERING,  // Pord => Auto
    UMFPACK_DEFAULT_ORDERING,  // Qamd => Auto
    UMFPACK_DEFAULT_ORDERING,  // Scotch => Auto
};

const double UMF_SCALING[9] = {
    UMFPACK_DEFAULT_SCALE,  // Auto
    UMFPACK_DEFAULT_SCALE,  // Column => Auto
    UMFPACK_DEFAULT_SCALE,  // Diagonal => Auto
    UMFPACK_SCALE_MAX,      // Max
    UMFPACK_SCALE_NONE,     // No
    UMFPACK_DEFAULT_SCALE,  // RowCol => Auto
    UMFPACK_DEFAULT_SCALE,  // RowColIter => Auto
    UMFPACK_DEFAULT_SCALE,  // RowColRig => Auto
    UMFPACK_SCALE_SUM,      // Sum
};

#endif
