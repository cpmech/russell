#include <inttypes.h>
#include <stdlib.h>

#include "fftw3.h"

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
                                  double /*Complex64*/ *data,
                                  C_BOOL inverse,
                                  C_BOOL measure) {
    if (interface == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (interface->initialization_completed == C_TRUE) {
        return ERROR_ALREADY_INITIALIZED;
    }

    // TODO

    interface->initialization_completed = C_TRUE;

    return SUCCESSFUL_EXIT;
}

/// @brief Computes the Fourier transform
int32_t interface_fftw_execute(struct InterfaceFFTW *interface) {
    if (interface == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (interface->initialization_completed == C_FALSE) {
        return ERROR_NEED_INITIALIZATION;
    }

    // TODO

    return SUCCESSFUL_EXIT;
}
