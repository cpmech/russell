#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <inttypes.h>

#include "dmumps_c.h"
#include "umfpack.h"

const int32_t NULL_POINTER_ERROR = 100000;
const int32_t C_TRUE = 1;

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

const double UMF_ORDERING[6] = {UMFPACK_ORDERING_AMD,
                                UMFPACK_ORDERING_BEST,
                                UMFPACK_ORDERING_CHOLMOD,
                                UMFPACK_DEFAULT_ORDERING,
                                UMFPACK_ORDERING_METIS,
                                UMFPACK_ORDERING_NONE};

const double UMF_SCALING[4] = {UMFPACK_DEFAULT_SCALE,
                               UMFPACK_SCALE_MAX,
                               UMFPACK_SCALE_NONE,
                               UMFPACK_SCALE_SUM};

#endif
