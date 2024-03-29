#include <inttypes.h>
#include <stdlib.h>

#include "fftw3.h"

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

#include "constants.h"

/// @brief Wraps FFTW data structures
struct InterfaceFFTW {
    /// @brief Holds the main FFTW data structure
    fftw_plan plan;

    /// @brief indicates that the initialization has been completed
    C_BOOL initialization_completed;
};

/// @brief Allocates a new FFTW interface object
struct InterfaceFFTW *interface_fftw_new() {
    struct InterfaceFFTW *interface = (struct InterfaceFFTW *)malloc(sizeof(struct InterfaceFFTW));

    if (interface == NULL) {
        return NULL;
    }

    interface->initialization_completed = C_FALSE;

    return interface;
}

/// @brief Deallocates the FFTW interface object
void interface_fftw_drop(struct InterfaceFFTW *interface) {
    if (interface == NULL) {
        return;
    }

    if (interface->plan != NULL) {
        fftw_destroy_plan(interface->plan);
    }

    free(interface);
}

/// @brief Performs the initialization
int32_t interface_fftw_initialize(struct InterfaceFFTW *interface,
                                  int32_t num_point,
                                  fftw_complex *data,
                                  C_BOOL inverse,
                                  C_BOOL measure) {
    if (interface == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (interface->initialization_completed == C_TRUE) {
        return ERROR_ALREADY_INITIALIZED;
    }

    int sign = inverse ? FFTW_BACKWARD : FFTW_FORWARD;
    int flag = measure ? FFTW_MEASURE : FFTW_ESTIMATE;

    interface->plan = fftw_plan_dft_1d(num_point, data, data, sign, flag);

    interface->initialization_completed = C_TRUE;

    return SUCCESSFUL_EXIT;
}

/// @brief Computes the Fourier transform
int32_t interface_fftw_execute(struct InterfaceFFTW *interface) {
    if (interface == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (interface->plan == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (interface->initialization_completed == C_FALSE) {
        return ERROR_NEED_INITIALIZATION;
    }

    fftw_execute(interface->plan);

    return SUCCESSFUL_EXIT;
}
