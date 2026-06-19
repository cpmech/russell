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

    /// @brief cuDSS coefficient matrix (A)
    cudssMatrix_t aa_mat;

    /// @brief cuDSS right-hand side vector (b)
    cudssMatrix_t b_vec;

    /// @brief cuDSS solution vector (x)
    cudssMatrix_t x_vec;

    /// @brief GPU (device) row pointers
    int *gpu_row_pointers;

    /// @brief GPU (device) col indices
    int *gpu_col_indices;

    /// @brief GPU (device) values
    double *gpu_values;

    /// @brief GPU (device) right-hand side
    double *gpu_b;

    /// @brief GPU (device) solution vector
    double *gpu_x;

    /// @brief System dimension
    int ndim;

    /// @brief Number of non-zeros
    int nnz;
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

    /* Allocate the solver object */
    struct InterfaceCUDSS *solver = (struct InterfaceCUDSS *)calloc(1, sizeof(struct InterfaceCUDSS));
    if (solver == NULL) {
        cudssDataDestroy(handle, data);
        cudssConfigDestroy(config);
        cudssDestroy(handle);
        cudaStreamDestroy(stream);
        return NULL;
    }

    /* Set the non-zero members (all other members are zero-initialized by calloc) */
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

    if (solver->aa_mat != NULL) {
        cudssMatrixDestroy(solver->aa_mat);
    }
    if (solver->b_vec != NULL) {
        cudssMatrixDestroy(solver->b_vec);
    }
    if (solver->x_vec != NULL) {
        cudssMatrixDestroy(solver->x_vec);
    }

    if (solver->handle != NULL && solver->data != NULL) {
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

    if (solver->gpu_row_pointers != NULL) {
        cudaFree(solver->gpu_row_pointers);
    }
    if (solver->gpu_col_indices != NULL) {
        cudaFree(solver->gpu_col_indices);
    }
    if (solver->gpu_values != NULL) {
        cudaFree(solver->gpu_values);
    }
    if (solver->gpu_b != NULL) {
        cudaFree(solver->gpu_b);
    }
    if (solver->gpu_x != NULL) {
        cudaFree(solver->gpu_x);
    }

    free(solver);
}

/// @brief Performs the symbolic factorization
extern "C" int32_t solver_cudss_initialize(struct InterfaceCUDSS *solver,
                                           int32_t ordering,
                                           int32_t matching,
                                           int32_t pivoting,
                                           double pivot_epsilon,
                                           int32_t refinement_nstep,
                                           int32_t hybrid_memory,
                                           C_BOOL verbose,
                                           C_BOOL general_symmetric,
                                           C_BOOL positive_definite,
                                           int32_t ndim,
                                           const int32_t *row_pointers,
                                           const int32_t *col_indices,
                                           const double *values) {
    /* Check */
    if (solver == NULL) {
        return ERROR_NULL_POINTER;
    }

    /* Save the system dimension for later */
    solver->ndim = ndim;

    /* Save the number of non-zeros for later */
    int32_t nnz = row_pointers[ndim];
    solver->nnz = nnz;

    /* Allocate device memory for row pointers */
    cudaError_t cuda_error = cudaMalloc(&solver->gpu_row_pointers, (ndim + 1) * sizeof(int));
    if (cuda_error != cudaSuccess) {
        return ERROR_CUDA_MALLOC;
    }

    /* Allocate device memory for col indices */
    cuda_error = cudaMalloc(&solver->gpu_col_indices, nnz * sizeof(int));
    if (cuda_error != cudaSuccess) {
        return ERROR_CUDA_MALLOC;
    }

    /* Allocate device memory for values */
    cuda_error = cudaMalloc(&solver->gpu_values, nnz * sizeof(double));
    if (cuda_error != cudaSuccess) {
        return ERROR_CUDA_MALLOC;
    }

    /* Allocate device memory for b */
    cuda_error = cudaMalloc(&solver->gpu_b, ndim * sizeof(double));
    if (cuda_error != cudaSuccess) {
        return ERROR_CUDA_MALLOC;
    }

    /* Allocate device memory for x */
    cuda_error = cudaMalloc(&solver->gpu_x, ndim * sizeof(double));
    if (cuda_error != cudaSuccess) {
        return ERROR_CUDA_MALLOC;
    }

    /* Show message */
    if (verbose == C_TRUE) {
        printf("solver_cudss_initialize: Memory allocated\n");
    }

    /* Copy host memory to device for row_pointers */
    cuda_error = cudaMemcpy(solver->gpu_row_pointers, row_pointers, (ndim + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        return ERROR_CUDA_MEMCPY;
    }

    /* Copy host memory to device for col_indices */
    cuda_error = cudaMemcpy(solver->gpu_col_indices, col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        return ERROR_CUDA_MEMCPY;
    }

    /* Copy host memory to device for values */
    cuda_error = cudaMemcpy(solver->gpu_values, values, nnz * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        return ERROR_CUDA_MEMCPY;
    }

    /* Create object for the right-hand side b (as dense matrix) */
    cudssStatus_t status = cudssMatrixCreateDn(&solver->b_vec, ndim, 1, ndim, solver->gpu_b, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    if (status != CUDSS_STATUS_SUCCESS) {
        return ERROR_CUDSS_MATRIX_CREATE_DN;
    }

    /* Create object for the solution x (as dense matrix) */
    status = cudssMatrixCreateDn(&solver->x_vec, ndim, 1, ndim, solver->gpu_x, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    if (status != CUDSS_STATUS_SUCCESS) {
        return ERROR_CUDSS_MATRIX_CREATE_DN;
    }

    /* Set the pivot type */
    cudssPivotType_t pivot = (cudssPivotType_t)pivoting;
    status = cudssConfigSet(solver->config, CUDSS_CONFIG_PIVOT_TYPE, &pivot, sizeof(cudssPivotType_t));
    if (status != CUDSS_STATUS_SUCCESS) {
        return ERROR_CUDSS_CONFIG_SET;
    }

    /* Set the ordering algorithm */
    cudssReorderingAlg_t reorder_alg = (cudssReorderingAlg_t)ordering;
    status = cudssConfigSet(solver->config, CUDSS_CONFIG_REORDERING_ALG, &reorder_alg, sizeof(cudssReorderingAlg_t));
    if (status != CUDSS_STATUS_SUCCESS) {
        return ERROR_CUDSS_CONFIG_SET;
    }

    /* Set the matching algorithm */
    cudssMatchingAlg_t matching_alg = (cudssMatchingAlg_t)matching;
    status = cudssConfigSet(solver->config, CUDSS_CONFIG_MATCHING_ALG, &matching_alg, sizeof(cudssMatchingAlg_t));
    if (status != CUDSS_STATUS_SUCCESS) {
        return ERROR_CUDSS_CONFIG_SET;
    }

    /* Set the pivot epsilon */
    if (pivot_epsilon > 0.0) {
        status = cudssConfigSet(solver->config, CUDSS_CONFIG_PIVOT_EPSILON, &pivot_epsilon, sizeof(double));
        if (status != CUDSS_STATUS_SUCCESS) {
            return ERROR_CUDSS_CONFIG_SET;
        }
    }

    /* Set iterative refinement number of steps */
    if (refinement_nstep > 0) {
        status = cudssConfigSet(solver->config, CUDSS_CONFIG_IR_N_STEPS, &refinement_nstep, sizeof(int32_t));
        if (status != CUDSS_STATUS_SUCCESS) {
            return ERROR_CUDSS_CONFIG_SET;
        }
    }

    /* Create a matrix object for the sparse input matrix */
    cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;  /* default to general matrix (possibly unsymmetric) */
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL; /* default to full view (not just one triangle) */
    if (general_symmetric) {
        mtype = CUDSS_MTYPE_SYMMETRIC;
        mview = CUDSS_MVIEW_LOWER;
    }
    if (positive_definite) {
        mtype = CUDSS_MTYPE_SPD;
        mview = CUDSS_MVIEW_LOWER;
    }
    status = cudssMatrixCreateCsr(&solver->aa_mat, ndim, ndim, nnz, solver->gpu_row_pointers, NULL,
                                  solver->gpu_col_indices, solver->gpu_values, CUDSS_R_32I,
                                  CUDSS_R_32I, CUDSS_R_64F, mtype, mview,
                                  CUDSS_BASE_ZERO);
    if (status != CUDSS_STATUS_SUCCESS) {
        return ERROR_CUDSS_MATRIX_CREATE_CSR;
    }

    /* Show message */
    if (verbose == C_TRUE) {
        printf("solver_cudss_initialize: Coefficient matrix allocated and initialized\n");
    }

    /* Enable hybrid mode where factors are stored in host memory
       Note: It must be set before the first call to ANALYSIS step.*/
    if (hybrid_memory == C_TRUE) {
        int hybrid_mode = 1;
        status = cudssConfigSet(solver->config, CUDSS_CONFIG_HYBRID_MEMORY_MODE, &hybrid_mode, sizeof(int));
        if (status != CUDSS_STATUS_SUCCESS) {
            return ERROR_CUDSS_CONFIG_SET;
        }
        if (verbose == C_TRUE) {
            printf("solver_cudss_initialize: Hybrid memory mode enabled\n");
        }
    }

    /* Symbolic factorization */
    status = cudssExecute(solver->handle, CUDSS_PHASE_ANALYSIS, solver->config, solver->data, solver->aa_mat, NULL, NULL);
    if (status != CUDSS_STATUS_SUCCESS) {
        return ERROR_CUDSS_SYM_FACTORIZATION + (int)status;
    }

    /* Synchronize */
    cuda_error = cudaStreamSynchronize(solver->stream);
    if (cuda_error != cudaSuccess) {
        return ERROR_CUDA_SYNCHRONIZE;
    }

    /* Show message */
    if (verbose == C_TRUE) {
        printf("solver_cudss_initialize: Symbolic factorization completed\n");
    }

    /* Done */
    solver->initialization_completed = C_TRUE;
    return SUCCESSFUL_EXIT;
}

/// @brief Performs the numeric factorization
extern "C" int32_t solver_cudss_factorize(struct InterfaceCUDSS *solver,
                                          int32_t *effective_matching,
                                          int32_t *effective_pivoting,
                                          C_BOOL verbose,
                                          const double *values) {

    /* Check */
    if (solver == NULL) {
        return ERROR_NULL_POINTER;
    }
    if (solver->initialization_completed == C_FALSE) {
        return ERROR_NEED_INITIALIZATION;
    }

    /* Number of non-zeros */
    int32_t nnz = solver->nnz;

    /* Copy the updated coefficient matrix values to the device (GPU) */
    cudaError_t cuda_error = cudaMemcpy(solver->gpu_values, values, nnz * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        return ERROR_CUDA_MEMCPY;
    }

    /* Notify cuDSS that the matrix values have been updated */
    cudssStatus_t status = cudssMatrixSetValues(solver->aa_mat, solver->gpu_values);
    if (status != CUDSS_STATUS_SUCCESS) {
        return ERROR_CUDSS_MATRIX_SET_VALUES;
    }

    /* Numeric factorization */
    status = cudssExecute(solver->handle, CUDSS_PHASE_FACTORIZATION, solver->config,
                          solver->data, solver->aa_mat, NULL, NULL);
    if (status != CUDSS_STATUS_SUCCESS) {
        return ERROR_CUDSS_NUM_FACTORIZATION + (int)status;
    }

    /* Synchronize */
    cuda_error = cudaStreamSynchronize(solver->stream);
    if (cuda_error != cudaSuccess) {
        return ERROR_CUDA_SYNCHRONIZE;
    }

    /* Check for device-side errors */
    int device_info = 0;
    size_t size_written = 0;
    status = cudssDataGet(solver->handle, solver->data, CUDSS_DATA_INFO, &device_info, sizeof(device_info), &size_written);
    if (status != CUDSS_STATUS_SUCCESS) {
        return ERROR_CUDSS_CONFIG_GET;
    }
    if (device_info != 0) {
        return ERROR_CUDSS_DEVICE;
    }

    /* Print number of pivots (indicates (near) singularity) */
    if (verbose == C_TRUE) {
        int npivots = 0;
        status = cudssDataGet(solver->handle, solver->data, CUDSS_DATA_NPIVOTS, &npivots, sizeof(npivots), &size_written);
        if (status != CUDSS_STATUS_SUCCESS) {
            return ERROR_CUDSS_CONFIG_GET;
        }
        if (npivots > 0) {
            printf("solver_cudss_factorize: WARNING: %d pivot(s) perturbed (matrix may be (nearly) singular)\n", npivots);
        }
    }

    /* Retrieve the effective (used) matching algorithm */
    cudssMatchingAlg_t used_matching_alg = (cudssMatchingAlg_t)-1;
    status = cudssConfigGet(solver->config, CUDSS_CONFIG_MATCHING_ALG, &used_matching_alg, sizeof(cudssMatchingAlg_t), &size_written);
    if (status != CUDSS_STATUS_SUCCESS) {
        return ERROR_CUDSS_CONFIG_GET;
    }
    *effective_matching = used_matching_alg;

    /* Retrieve the effective (used) pivoting strategy */
    cudssPivotType_t used_pivot = (cudssPivotType_t)-1;
    status = cudssConfigGet(solver->config, CUDSS_CONFIG_PIVOT_TYPE, &used_pivot, sizeof(cudssPivotType_t), &size_written);
    if (status != CUDSS_STATUS_SUCCESS) {
        return ERROR_CUDSS_CONFIG_GET;
    }
    *effective_pivoting = used_pivot;

    /* Show message */
    if (verbose == C_TRUE) {
        printf("solver_cudss_factorize: Numeric factorization completed\n");
    }

    /* Done */
    solver->factorization_completed = C_TRUE;
    return SUCCESSFUL_EXIT;
}

/// @brief Computes the solution of the linear system
extern "C" int32_t solver_cudss_solve(struct InterfaceCUDSS *solver,
                                      double *x,
                                      const double *rhs,
                                      C_BOOL verbose) {
    /* Check */
    if (solver == NULL) {
        return ERROR_NULL_POINTER;
    }
    if (solver->factorization_completed == C_FALSE) {
        return ERROR_NEED_FACTORIZATION;
    }

    /* Set the right-hand side vector */
    int ndim = solver->ndim;
    cudaError_t cuda_error = cudaMemcpy(solver->gpu_b, rhs, ndim * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        return ERROR_CUDA_MEMCPY;
    }

    /* Call solve */
    cudssStatus_t status = cudssExecute(solver->handle, CUDSS_PHASE_SOLVE, solver->config, solver->data, solver->aa_mat, solver->x_vec, solver->b_vec);
    if (status != CUDSS_STATUS_SUCCESS) {
        return ERROR_CUDSS_SOLVE + (int)status;
    }

    /* Synchronize */
    cuda_error = cudaStreamSynchronize(solver->stream);
    if (cuda_error != cudaSuccess) {
        return ERROR_CUDA_SYNCHRONIZE;
    }

    /* Check for device-side errors */
    int device_info = 0;
    size_t size_written = 0;
    status = cudssDataGet(solver->handle, solver->data, CUDSS_DATA_INFO, &device_info, sizeof(device_info), &size_written);
    if (status != CUDSS_STATUS_SUCCESS) {
        return ERROR_CUDSS_CONFIG_GET;
    }
    if (device_info != 0) {
        return ERROR_CUDSS_DEVICE;
    }

    /* Copy solution to output vector */
    cuda_error = cudaMemcpy(x, solver->gpu_x, ndim * sizeof(double),
                            cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) {
        return ERROR_CUDA_MEMCPY;
    }

    /* Show message */
    if (verbose == C_TRUE) {
        printf("solver_cudss_solve: Solution completed\n");
    }

    /* Done */
    return SUCCESSFUL_EXIT;
}
