//! Russell - Rust Scientific Library
//!
//! **sparse**: Sparse matrix tools and solvers
//!
//! # Introduction
//!
//! We have three storage formats for sparse matrices:
//!
//! * [CooMatrix] (COO) -- COOrdinates matrix, also known as a sparse triplet.
//! * [CscMatrix] (CSC) -- Compressed Sparse Column matrix
//! * [CsrMatrix] (CSR) -- Compressed Sparse Row matrix
//!
//! Additionally, to unify the handling of the above sparse matrix data structures, we have:
//!
//! * [SparseMatrix] -- Either a COO, CSC, or CSR matrix
//!
//! The COO matrix is the best when we need to update the values of the matrix because it has easy access to the triples (i, j, aij). For instance, the repetitive access is the primary use case for codes based on the finite element method (FEM) for approximating partial differential equations. Moreover, the COO matrix allows storing duplicate entries; for example, the triple `(0, 0, 123.0)` can be stored as two triples `(0, 0, 100.0)` and `(0, 0, 23.0)`. Again, this is the primary need for FEM codes because of the so-called assembly process where elements add to the same positions in the "global stiffness" matrix. Nonetheless, the duplicate entries must be summed up at some stage for the linear solver (e.g., MUMPS, UMFPACK, and Intel DSS). These linear solvers also use the more memory-efficient storage formats CSC and CSR. The following is the default input for these solvers:
//!
//! * [MUMPS](https://mumps-solver.org) -- requires a COO matrix as input internally
//! * [UMFPACK](https://github.com/DrTimothyAldenDavis/SuiteSparse) -- requires a CSC matrix as input internally
//! * [Intel DSS](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/direct-sparse-solver-dss-interface-routines.html) -- requires a CSR matrix as input internally
//!
//! Nonetheless, the implemented interface to the above linear solvers takes a [SparseMatrix] as input, which will automatically be converted from COO to CSC or COO to CSR, as appropriate.
//!
//! The best way to use a COO matrix is to initialize it with the maximum possible number of non-zero values and repetitively call the [CooMatrix::put()] function to insert triples (i, j, aij) into the data structure. This procedure is computationally efficient. Later, we can create a Compressed Sparse Column (CSC) or a Compressed Sparse Row (CSC) matrix from the COO matrix. The CSC and CSR will sum up any duplicates in the COO matrix during the conversion process. To reinitialize the counter for "putting" entries into the triplet structure, we can call the [CooMatrix::reset()] function (e.g., to recreate the global stiffness matrix in FEM simulations).
//!
//! The three individual sparse matrix structures ([CooMatrix], [CscMatrix], and [CsrMatrix]) and the wrapping (unifying) structure SparseMatrix have functions to calculate the (sparse) matrix-vector product, which, albeit not computer optimized, are convenient for checking the solution to the linear problem A * x = b (see also the VerifyLinSys structure).
//!
//! We recommend using the [SparseMatrix] directly unless your computations need a more specialized interaction with the CSC or CSR formats. Also, the [SparseMatrix] returns "pointers" to the CSC and CSR structures (constant access and mutable access).
//!
//! We call the actual linear system solver implementation [Genie] because they work like "magic" after being "wrapped" via a C-interface. Note that these fantastic solvers are implemented in Fortran and C. You may easily access the linear solvers directly via the following structures:
//!
//! * [SolverMUMPS] -- thin wrapper to the MUMPS solver
//! * [SolverUMFPACK] -- thin wrapper to the UMFPACK solver
//! * [SolverIntelDSS] -- thin wrapper to the Intel DSS solver
//!
//! This library also provides a unifying Trait called [LinSolTrait], which the above structures implement. In addition, the [LinSolver] structure holds a "pointer" to one of the above structures and is a more convenient way to use the linear solvers in generic codes when we need to switch from solver to solver (e.g., for benchmarking). After allocating a [LinSolver], if needed, we can access the actual implementations (interfaces/thin wrappers) via the [LinSolver::actual] data member.
//!
//! The [LinSolTrait] has two main functions (that should be called in this order):
//!
//! * [LinSolTrait::factorize()] -- performs the initialization of the linear solver, if needed, analysis, and symbolic and numerical factorization of the coefficient matrix A from A * x = b
//! * [LinSolTrait::solve()] -- find x from the linear system A * x = b after completing the factorization.
//!
//! If the **structure** of the coefficient matrix remains constant, but its values change during a simulation, we can repeatedly call `factorize` again to perform the factorization. However, if the structure changes, the solver must be **dropped** and another solver allocated to force the **initialization** process.
//!
//! We call the **structure** of the coefficient matrix A from A * x = b the set with the following characteristics:
//!
//! * `nrow` -- number of rows
//! * `ncol` -- number of columns
//! * `nnz` -- number of non-zero values
//! * `symmetry` -- symmetry type
//! * the locations of the non-zero values
//!
//! If neither the structure nor the values of the coefficient matrix change, we can call `solve` repeatedly if needed (e.g., in a simulation of linear dynamics using the FEM).
//!
//! The linear solvers have numerous configuration parameters; however, we can use the default parameters initially. The configuration parameters are collected in the [LinSolParams] structures, which is an input to the [LinSolTrait::factorize()]. The parameters include options such as [Ordering] and [Scaling].
//!
//! Some solvers (e.g., MUMPS) may return error analysis variables that are not "unified" by the linear solver trait. In this case, we can simply access the `actual` implementation (which will return a SolverMUMPS) to access the statistics/error analysis via the [SolverMUMPS::get_stats_after_solve()] function. Analogously, UMFPACK provides the [SolverUMFPACK::get_rcond_after_factorize()] function to access the reciprocal condition number estimate.
//!
//! This library also provides functions to read and write Matrix Market files containing (huge) sparse matrices that can be used in performance benchmarking or other studies. The [read_matrix_market()] function reads a Matrix Market file and returns a [CooMatrix]. To write a Matrix Market file, we can use the function [write_matrix_market()], which takes a [SparseMatrix] and, thus, automatically convert COO to CSC or COO to CSR, also performing the sum of duplicates. The `write_matrix_market` also writes an SMAT file (almost like the Matrix Market format) without the header and with zero-based indices. The SMAT file can be given to the fantastic [Vismatrix](https://github.com/cpmech/vismatrix) tool to visualize the sparse matrix structure and values interactively; see the example below.
//!
//! ![doc-example-vismatrix](https://raw.githubusercontent.com/cpmech/russell/main/russell_sparse/data/figures/doc-example-vismatrix.png)
//!
//! # Examples
//!
//! ## Solving a sparse linear system using UMFPACK
//!
//! TODO
//!
//! ## Using the common solver interface
//!
//! TODO
//!
//! ## Using the common solver interface (single-use)
//!
//! TODO

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod auxiliary_and_constants;
mod coo_matrix;
mod csc_matrix;
mod csr_matrix;
mod enums;
mod lin_solver;
pub mod prelude;
mod read_matrix_market;
mod samples;
mod solver_intel_dss;
mod solver_mumps;
mod solver_umfpack;
mod sparse_matrix;
mod stats_lin_sol;
mod verify_lin_sys;
mod write_matrix_market;
use crate::auxiliary_and_constants::*;
pub use crate::coo_matrix::*;
pub use crate::csc_matrix::*;
pub use crate::csr_matrix::*;
pub use crate::enums::*;
pub use crate::lin_solver::*;
pub use crate::read_matrix_market::*;
pub use crate::samples::*;
pub use crate::solver_intel_dss::*;
pub use crate::solver_mumps::*;
pub use crate::solver_umfpack::*;
pub use crate::sparse_matrix::*;
pub use crate::stats_lin_sol::*;
pub use crate::verify_lin_sys::*;
pub use crate::write_matrix_market::*;

// run code from README file
#[cfg(doctest)]
mod test_readme {
    macro_rules! external_doc_test {
        ($x:expr) => {
            #[doc = $x]
            extern "C" {}
        };
    }
    external_doc_test!(include_str!("../README.md"));
    external_doc_test!(include_str!("../../README.md"));
}
