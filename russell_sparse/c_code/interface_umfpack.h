#pragma once

#include "umfpack.h"

const double UMFPACK_OPTION_SYMMETRY[2] = {
    UMFPACK_STRATEGY_UNSYMMETRIC, // Unsymmetric
    UMFPACK_STRATEGY_SYMMETRIC,   // General symmetric
};

const double UMFPACK_OPTION_ORDERING[10] = {
    UMFPACK_ORDERING_AMD,     // Amd
    UMFPACK_DEFAULT_ORDERING, // Amf => Auto
    UMFPACK_DEFAULT_ORDERING, // Auto
    UMFPACK_ORDERING_BEST,    // Best
    UMFPACK_ORDERING_CHOLMOD, // Cholmod
    UMFPACK_ORDERING_METIS,   // Metis
    UMFPACK_ORDERING_NONE,    // No
    UMFPACK_DEFAULT_ORDERING, // Pord => Auto
    UMFPACK_DEFAULT_ORDERING, // Qamd => Auto
    UMFPACK_DEFAULT_ORDERING, // Scotch => Auto
};

const double UMFPACK_OPTION_SCALING[9] = {
    UMFPACK_DEFAULT_SCALE, // Auto
    UMFPACK_DEFAULT_SCALE, // Column => Auto
    UMFPACK_DEFAULT_SCALE, // Diagonal => Auto
    UMFPACK_SCALE_MAX,     // Max
    UMFPACK_SCALE_NONE,    // No
    UMFPACK_DEFAULT_SCALE, // RowCol => Auto
    UMFPACK_DEFAULT_SCALE, // RowColIter => Auto
    UMFPACK_DEFAULT_SCALE, // RowColRig => Auto
    UMFPACK_SCALE_SUM,     // Sum
};

const double UMFPACK_PRINT_LEVEL_SILENT = 0.0;  // page 116
const double UMFPACK_PRINT_LEVEL_VERBOSE = 2.0; // page 116

/// @brief Holds the data for UMFPACK
struct InterfaceUMFPACK {
    double control[UMFPACK_CONTROL]; // control flags
    double info[UMFPACK_INFO];       // information data
    int n;                           // number of rows = number of columns
    int nnz;                         // number of non-zero values
    int *ap;                         // pointers in the CscMatrix
    int *ai;                         // indices in the CscMatrix
    double *ax;                      // values in the CscMatrix
    void *symbolic;                  // handle to symbolic factorization results
    void *numeric;                   // handle to numeric factorization results
    int32_t compute_determinant;     // whether the determinant should be computed or not (after factorize)
    double mx[1];                    // mx[0] is the coefficient mx of the determinant = mx * 10 ^ ex
    double ex[1];                    // ex[0] is the exponent ex of the determinant = mx * 10 ^ ex
};

/// @brief Allocates a new UMFPACK interface
struct InterfaceUMFPACK *solver_umfpack_new();

/// @brief Deallocates the UMFPACK interface
void solver_umfpack_drop(struct InterfaceUMFPACK *solver);

/// @brief Performs the initialization
/// @param solver Is a pointer to the solver interface
/// @param n Is the number of rows and columns of the coefficient matrix
/// @param nnz Is the number of non-zero values in the coefficient matrix
/// @param symmetry Is the UMFPACK code for the kind of symmetry, if any
/// @param ordering Is the UMFPACK ordering code
/// @param scaling Is the UMFPACK scaling code
/// @param verbose Shows messages
/// @return A success or fail code
int32_t solver_umfpack_initialize(struct InterfaceUMFPACK *solver,
                                  int32_t n,
                                  int32_t nnz,
                                  int32_t symmetry,
                                  int32_t ordering,
                                  int32_t scaling,
                                  int32_t compute_determinant);

/// @brief Performs the factorization
/// @param solver Is a pointer to the solver interface
/// @param indices_i Are the CooMatrix row indices
/// @param indices_j Are the CooMatrix column indices
/// @param values_aij Are the CooMatrix values
/// @param verbose Shows messages
/// @return A success or fail code
int32_t solver_umfpack_factorize(struct InterfaceUMFPACK *solver,
                                 int32_t const *indices_i,
                                 int32_t const *indices_j,
                                 double const *values_aij,
                                 int32_t verbose);

/// @brief Computes the solution of the linear system
/// @param solver Is a pointer to the solver interface
/// @param x Is the left-hand side vector (unknowns)
/// @param rhs Is the right-hand side vector
/// @param verbose Shows messages
/// @return A success or fail code
int32_t solver_umfpack_solve(struct InterfaceUMFPACK *solver, double *x, const double *rhs, int32_t verbose);

/// @brief Returns the effective ordering using during the computations
/// @param solver Is a pointer to the solver
/// @return The used UMFPACK ordering code
int32_t solver_umfpack_get_ordering(const struct InterfaceUMFPACK *solver);

/// @brief Returns the effective scaling using during the computations
/// @param solver Is a pointer to the solver
/// @return The used UMFPACK scaling code
int32_t solver_umfpack_get_scaling(const struct InterfaceUMFPACK *solver);

/// @brief Returns the coefficient needed to compute the determinant, if requested
/// @param solver Is a pointer to the solver
/// @return The coefficient mx of the determinant = mx * 10 ^ ex
double solver_umfpack_get_det_mx(const struct InterfaceUMFPACK *solver);

/// @brief Returns the exponent needed to compute the determinant, if requested
/// @param solver Is a pointer to the solver
/// @return The exponent ex of the determinant = mx * 10 ^ ex
double solver_umfpack_get_det_ex(const struct InterfaceUMFPACK *solver);
