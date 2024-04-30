#pragma once

#include <inttypes.h>

#define SUCCESSFUL_EXIT 0
#define ERROR_NULL_POINTER 100000
#define ERROR_MALLOC 200000
#define ERROR_VERSION 300000
#define ERROR_NOT_AVAILABLE 400000
#define ERROR_NEED_INITIALIZATION 500000
#define ERROR_NEED_FACTORIZATION 600000
#define ERROR_ALREADY_INITIALIZED 700000
#define C_TRUE 1
#define C_FALSE 0

#define C_BOOL int32_t

#define COMPLEX64 double

// UMFPACK -------------------------------------------------------------------------------------------

#define UMFPACK_PRINT_LEVEL_SILENT 0.0  // page 116
#define UMFPACK_PRINT_LEVEL_VERBOSE 2.0 // page 116

// KLU -----------------------------------------------------------------------------------------------

#define KLU_ERROR_ANALYZE -9  // defined here
#define KLU_ERROR_FACTOR -8   // defined here
#define KLU_ERROR_COND_EST -7 // defined here

// MUMPS ---------------------------------------------------------------------------------------------

#define MUMPS_IGNORED 0 // to ignore the Fortran communicator since we're not using MPI

#define MUMPS_JOB_INITIALIZE -1 // section 5.1.1, page 24
#define MUMPS_JOB_TERMINATE -2  // section 5.1.1, page 24
#define MUMPS_JOB_ANALYZE 1     // section 5.1.1, page 24
#define MUMPS_JOB_FACTORIZE 2   // section 5.1.1, page 25
#define MUMPS_JOB_SOLVE 3       // section 5.1.1, page 25

#define MUMPS_PAR_HOST_ALSO_WORKS 1     // section 5.1.4, page 26
#define MUMPS_ICNTL5_ASSEMBLED_MATRIX 0 // section 5.2.2, page 27
#define MUMPS_ICNTL18_CENTRALIZED 0     // section 5.2.2, page 27
#define MUMPS_ICNTL6_PERMUT_AUTO 7      // section 5.3, page 32
#define MUMPS_ICNTL28_SEQUENTIAL 1      // section 5.4, page 33
