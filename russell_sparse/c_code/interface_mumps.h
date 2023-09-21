#pragma once

#include "dmumps_c.h"

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


const MUMPS_INT MUMPS_SYMMETRY[3] = {
    0, // Unsymmetric
    1, // Positive-definite symmetric
    2, // General symmetric
};

const MUMPS_INT MUMPS_ORDERING[10] = {
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

const MUMPS_INT MUMPS_SCALING[9] = {
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
