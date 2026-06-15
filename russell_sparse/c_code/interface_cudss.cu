// kernel.cu
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudss.h"

#include "constants.h"

/// @brief Holds the data for cuDSS
struct InterfaceCUDSS {
    /// @brief indicates that the initialization has been completed
    C_BOOL initialization_completed;

    /// @brief Indicates that the factorization (at least once) has been completed
    C_BOOL factorization_completed;

    /// @brief CUDA stream
    cudaStream_t stream;

    /// @brief cuDSS library handle
    cudssHandle_t handle;

    /// @brief cuDSS solver configuration
    cudssConfig_t config;

    /// @brief cuDSS data object
    cudssData_t data;

    /// @brief Solution vector
    cudssMatrix_t x;

    /// @brief Right-hand side vector
    cudssMatrix_t b;

    /// @brief Matrix object (we use repeated characters instead of capital letters here; e.g. we use 'bb' for 'B')
    cudssMatrix_t aa;
};

/// @brief Allocates a new cuDSS interface
extern "C" struct InterfaceCUDSS *solver_cudss_new() {
    /* Create a CUDA stream */
    cudaStream_t stream = NULL;
    cudaError_t cuda_error = cudaStreamCreate(&stream);
    if (cuda_error != cudaSuccess) {
        return NULL;
    }

    /* Create the cuDSS library handle */
    cudssHandle_t handle;
    cudssStatus_t status = cudssCreate(&handle);
    if (status != CUDSS_STATUS_SUCCESS) {
        cudaStreamDestroy(stream);
        return NULL;
    }

    /* Set the custom stream for the library handle */
    status = cudssSetStream(handle, stream);
    if (status != CUDSS_STATUS_SUCCESS) {
        cudssDestroy(handle);
        cudaStreamDestroy(stream);
        return NULL;
    }

    /* Create cuDSS solver configuration*/
    cudssConfig_t config;
    status = cudssConfigCreate(&config);
    if (status != CUDSS_STATUS_SUCCESS) {
        cudssDestroy(handle);
        cudaStreamDestroy(stream);
        return NULL;
    }

    /* Create data object */
    cudssData_t data;
    status = cudssDataCreate(handle, &data);
    if (status != CUDSS_STATUS_SUCCESS) {
        cudssConfigDestroy(config);
        cudssDestroy(handle);
        cudaStreamDestroy(stream);
        return NULL;
    }

    /* Create pivot strategy */
    cudssPivotType_t pivot = CUDSS_PIVOT_AUTO;
    status = cudssConfigSet(config, CUDSS_CONFIG_PIVOT_TYPE, &pivot, sizeof(pivot));
    if (status != CUDSS_STATUS_SUCCESS) {
        cudssDataDestroy(handle, data);
        cudssConfigDestroy(config);
        cudssDestroy(handle);
        cudaStreamDestroy(stream);
        return NULL;
    }

    /* Allocate the solver object */
    struct InterfaceCUDSS *solver = (struct InterfaceCUDSS *)malloc(sizeof(struct InterfaceCUDSS));
    if (solver == NULL) {
        cudssDataDestroy(handle, data);
        cudssConfigDestroy(config);
        cudssDestroy(handle);
        cudaStreamDestroy(stream);
        return NULL;
    }

    /* Set just allocated members in the solver object */
    solver->stream = stream;
    solver->handle = handle;
    solver->config = config;
    solver->data = data;

    /* Success: return the pointer to the solver object */
    return solver;
}

/// @brief De-allocates the cuDSS interface
extern "C" void solver_cudss_drop(struct InterfaceCUDSS *solver) {
    if (solver == NULL) {
        return;
    }

    if (solver->handle != NULL & solver->data != NULL) {
        cudssDataDestroy(solver->handle, solver->data);
    }
    if (solver->config != NULL) {
        cudssConfigDestroy(solver->config);
    }
    if (solver->handle != NULL) {
        cudssDestroy(solver->handle);
    }
    if (solver->stream != NULL) {
        cudaStreamDestroy(solver->stream);
    }

    free(solver);
}

/// @brief Performs the symbolic factorization
extern "C" int32_t solver_cudss_initialize(struct InterfaceCUDSS *solver,
                                           C_BOOL verbose,
                                           C_BOOL general_symmetric,
                                           C_BOOL positive_definite,
                                           int32_t ndim,
                                           const int32_t *row_pointers,
                                           const int32_t *col_indices,
                                           const double *values) {
    if (solver == NULL) {
        return ERROR_NULL_POINTER;
    }

    /* TODO: implement this later */

    solver->initialization_completed = C_TRUE;
    return SUCCESSFUL_EXIT;
}

/// @brief Performs the numeric factorization
extern "C" int32_t solver_cudss_factorize(struct InterfaceCUDSS *solver,
                                          C_BOOL verbose,
                                          const int32_t *row_pointers,
                                          const int32_t *col_indices,
                                          const double *values) {
    if (solver == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (solver->initialization_completed == C_FALSE) {
        return ERROR_NEED_INITIALIZATION;
    }

    /* TODO: implement this later */

    solver->factorization_completed = C_TRUE;
    return SUCCESSFUL_EXIT;
}

/// @brief Computes the solution of the linear system
extern "C" int32_t solver_cudss_solve(struct InterfaceCUDSS *solver,
                                      double *x,
                                      const double *rhs,
                                      const int32_t *row_pointers,
                                      const int32_t *col_indices,
                                      const double *values,
                                      C_BOOL verbose) {
    if (solver == NULL) {
        return ERROR_NULL_POINTER;
    }

    if (solver->factorization_completed == C_FALSE) {
        return ERROR_NEED_FACTORIZATION;
    }

    /* TODO: implement this later */

    return SUCCESSFUL_EXIT;
}