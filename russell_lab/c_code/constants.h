#pragma once

#include <inttypes.h>

// Represents the type of boolean flags interchanged with the C-code
#define C_BOOL int32_t

// Boolean flags
#define C_TRUE 1
#define C_FALSE 0

// Norm codes
#define NORM_EUC 0
#define NORM_FRO 1
#define NORM_INF 2
#define NORM_MAX 3
#define NORM_ONE 4

// SVD codes
#define SVD_CODE_A 0
#define SVD_CODE_S 1
#define SVD_CODE_O 2
#define SVD_CODE_N 3

// Error codes
#define SUCCESSFUL_EXIT 0
#define UNKNOWN_ERROR 1
