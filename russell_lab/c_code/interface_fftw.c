#include <inttypes.h>
#include <stdlib.h>

#include "fftw3.h"

// WARNING: FFTW is not thread-safe
//
// The default FFTW interface uses double precision for all floating-point numbers,
// and defines a fftw_complex type to hold complex numbers as:
//
// typedef double fftw_complex[2];
//
// Here, the [0] element holds the real part and the [1] element holds the imaginary part.
//
// Alternatively, if you have a C compiler (such as gcc) that supports the C99 revision
// of the ANSI C standard, you can use C's new native complex type
// (WHICH IS BINARY-COMPATIBLE WITH THE TYPEDEF ABOVE).
//
// Reference: <https://www.fftw.org/fftw3_doc/Complex-numbers.html>
//
// FFTW says "so you should initialize your input data after creating the plan."
// Therefore, the plan can be created and reused several times.
// <http://www.fftw.org/fftw3_doc/Planner-Flags.html>
//
// Also: "The plan can be reused as many times as needed. In typical high-performance
// applications, many transforms of the same size are computed"
// <http://www.fftw.org/fftw3_doc/Introduction.html>
//
// WARNING: "[...] transforms operate on contiguous arrays in the C-standard ROW-MAJOR order."

#include "constants.h"

int32_t interface_fftw_dft_1d(int32_t n0,
                              fftw_complex *data,
                              C_BOOL inverse) {
    int sign = inverse == C_TRUE ? FFTW_BACKWARD : FFTW_FORWARD;
    int flag = FFTW_ESTIMATE;

    fftw_plan plan = fftw_plan_dft_1d(n0, data, data, sign, flag);

    if (plan == NULL) {
        return UNKNOWN_ERROR;
    }

    fftw_execute(plan);
    fftw_destroy_plan(plan);

    return SUCCESSFUL_EXIT;
}

int32_t interface_fftw_dft_2d(int32_t n0,
                              int32_t n1,
                              fftw_complex *data_row_major,
                              C_BOOL inverse) {
    int sign = inverse == C_TRUE ? FFTW_BACKWARD : FFTW_FORWARD;
    int flag = FFTW_ESTIMATE;

    fftw_plan plan = fftw_plan_dft_2d(n0, n1, data_row_major, data_row_major, sign, flag);

    if (plan == NULL) {
        return UNKNOWN_ERROR;
    }

    fftw_execute(plan);
    fftw_destroy_plan(plan);

    return SUCCESSFUL_EXIT;
}

int32_t interface_fftw_dft_3d(int32_t n0,
                              int32_t n1,
                              int32_t n2,
                              fftw_complex *data_row_major,
                              C_BOOL inverse) {
    int sign = inverse == C_TRUE ? FFTW_BACKWARD : FFTW_FORWARD;
    int flag = FFTW_ESTIMATE;

    fftw_plan plan = fftw_plan_dft_3d(n0, n1, n2, data_row_major, data_row_major, sign, flag);

    if (plan == NULL) {
        return UNKNOWN_ERROR;
    }

    fftw_execute(plan);
    fftw_destroy_plan(plan);

    return SUCCESSFUL_EXIT;
}
