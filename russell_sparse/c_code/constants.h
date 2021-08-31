#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <stdlib.h>

#include "dmumps_c.h"

const int32_t C_HAS_ERROR = 1;
const int32_t C_NO_ERROR = 0;

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

#endif
