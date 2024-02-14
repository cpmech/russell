//! Russell - Rust Scientific Library
//!
//! `russell_sparse`: Sparse matrix tools and solvers
//!
//! **Important:** This crate depends on external libraries (non-Rust). Thus, please check the [Installation Instructions on the GitHub Repository](https://github.com/cpmech/russell).
//!
//! # Introduction
//!
//! This crate implements three storage formats for sparse matrices:
//!
//! * [NumCooMatrix] (COO) -- COOrdinates matrix, also known as a sparse triplet.
//! * [NumCscMatrix] (CSC) -- Compressed Sparse Column matrix
//! * [NumCsrMatrix] (CSR) -- Compressed Sparse Row matrix
//!
//! Additionally, to unify the handling of the above data structures, this implements:
//!
//! * [NumSparseMatrix] -- Either a COO, CSC, or CSR matrix. We recommend using `NumSparseMatrix` solely, if possible.
//!
//! For convenience, this crate defines the following type aliases for Real and Complex matrices (with double precision):
//!
//! * [CooMatrix], [CscMatrix], [CsrMatrix], [SparseMatrix] -- For real numbers represented by `f64`
//! * [ComplexCooMatrix], [ComplexCscMatrix], [ComplexCsrMatrix], [ComplexSparseMatrix] -- For complex numbers represented by [num_complex::Complex64]
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
//! This library also provides functions to read and write Matrix Market files containing (huge) sparse matrices that can be used in performance benchmarking or other studies. The [read_matrix_market()] function reads a Matrix Market file and returns a [CooMatrix]. To write a Matrix Market file, we can use the function [csc_write_matrix_market()] (and similar), which automatically converts COO to CSC or COO to CSR, also performing the sum of duplicates. The `write_matrix_market` can also writs an SMAT file (almost like the Matrix Market format) without the header and with zero-based indices. The SMAT file can be given to the fantastic [Vismatrix](https://github.com/cpmech/vismatrix) tool to visualize the sparse matrix structure and values interactively; see the example below.
//!
//! ![doc-example-vismatrix](https://raw.githubusercontent.com/cpmech/russell/main/russell_sparse/data/figures/doc-example-vismatrix.png)
//!
//! # Examples
//!
//! ## Create CSR matrix from COO
//!
//! ```
//! use russell_sparse::prelude::*;
//! use russell_sparse::StrError;
//!
//! fn main() -> Result<(), StrError> {
//!     // allocate a square matrix and store as COO matrix
//!     // ┌          ┐
//!     // │  1  0  2 │
//!     // │  0  0  3 │ << the diagonal 0 entry is optional,
//!     // │  4  5  6 │    but should be saved for Intel DSS
//!     // └          ┘
//!     let (nrow, ncol, nnz) = (3, 3, 6);
//!     let mut coo = CooMatrix::new(nrow, ncol, nnz, None, false)?;
//!     coo.put(0, 0, 1.0)?;
//!     coo.put(0, 2, 2.0)?;
//!     coo.put(1, 2, 3.0)?;
//!     coo.put(2, 0, 4.0)?;
//!     coo.put(2, 1, 5.0)?;
//!     coo.put(2, 2, 6.0)?;
//!
//!     // convert to CCR matrix
//!     let csc = CscMatrix::from_coo(&coo)?;
//!     let correct_v = &[
//!         //                               p
//!         1.0, 4.0, //      j = 0, count = 0, 1
//!         5.0, //           j = 1, count = 2
//!         2.0, 3.0, 6.0, // j = 2, count = 3, 4, 5
//!              //                  count = 6
//!     ];
//!     let correct_i = &[
//!         //                         p
//!         0, 2, //    j = 0, count = 0, 1
//!         2, //       j = 1, count = 2
//!         0, 1, 2, // j = 2, count = 3, 4, 5
//!            //              count = 6
//!     ];
//!     let correct_p = &[0, 2, 3, 6];
//!
//!     // check
//!     assert_eq!(csc.get_col_pointers(), correct_p);
//!     assert_eq!(csc.get_row_indices(), correct_i);
//!     assert_eq!(csc.get_values(), correct_v);
//!     Ok(())
//! }
//! ```
//!
//! ## Solving a tiny sparse linear system using LinSolver (Umfpack)
//!
//! ```
//! use russell_lab::{vec_approx_eq, Vector};
//! use russell_sparse::prelude::*;
//! use russell_sparse::StrError;
//!
//! fn main() -> Result<(), StrError> {
//!     // constants
//!     let ndim = 3; // number of rows = number of columns
//!     let nnz = 5; // number of non-zero values
//!
//!     // allocate the linear solver
//!     let mut solver = LinSolver::new(Genie::Umfpack)?;
//!
//!     // allocate the coefficient matrix
//!     let mut coo = SparseMatrix::new_coo(ndim, ndim, nnz, None, false)?;
//!     coo.put(0, 0, 0.2)?;
//!     coo.put(0, 1, 0.2)?;
//!     coo.put(1, 0, 0.5)?;
//!     coo.put(1, 1, -0.25)?;
//!     coo.put(2, 2, 0.25)?;
//!
//!     // print matrix
//!     let mut a = coo.as_dense();
//!     let correct = "┌                   ┐\n\
//!                    │   0.2   0.2     0 │\n\
//!                    │   0.5 -0.25     0 │\n\
//!                    │     0     0  0.25 │\n\
//!                    └                   ┘";
//!     assert_eq!(format!("{}", a), correct);
//!
//!     // call factorize
//!     solver.actual.factorize(&mut coo, None)?;
//!
//!     // allocate two right-hand side vectors
//!     let rhs1 = Vector::from(&[1.0, 1.0, 1.0]);
//!     let rhs2 = Vector::from(&[2.0, 2.0, 2.0]);
//!
//!     // calculate the solution
//!     let mut x1 = Vector::new(ndim);
//!     solver.actual.solve(&mut x1, &coo, &rhs1, false)?;
//!     let correct = vec![3.0, 2.0, 4.0];
//!     vec_approx_eq(x1.as_data(), &correct, 1e-14);
//!
//!     // calculate the solution again
//!     let mut x2 = Vector::new(ndim);
//!     solver.actual.solve(&mut x2, &coo, &rhs2, false)?;
//!     let correct = vec![6.0, 4.0, 8.0];
//!     vec_approx_eq(x2.as_data(), &correct, 1e-14);
//!     Ok(())
//! }
//! ```
//!
//! ## Solving a tiny sparse linear system using SolverUMFPACK
//!
//! ```
//! use russell_lab::{vec_approx_eq, Vector};
//! use russell_sparse::prelude::*;
//! use russell_sparse::StrError;
//!
//! fn main() -> Result<(), StrError> {
//!     // constants
//!     let ndim = 5; // number of rows = number of columns
//!     let nnz = 13; // number of non-zero values, including duplicates
//!
//!     // allocate solver
//!     let mut umfpack = SolverUMFPACK::new()?;
//!
//!     // allocate the coefficient matrix
//!     //  2  3  .  .  .
//!     //  3  .  4  .  6
//!     //  . -1 -3  2  .
//!     //  .  .  1  .  .
//!     //  .  4  2  .  1
//!     let mut coo = SparseMatrix::new_coo(ndim, ndim, nnz, None, false)?;
//!     coo.put(0, 0, 1.0)?; // << (0, 0, a00/2) duplicate
//!     coo.put(0, 0, 1.0)?; // << (0, 0, a00/2) duplicate
//!     coo.put(1, 0, 3.0)?;
//!     coo.put(0, 1, 3.0)?;
//!     coo.put(2, 1, -1.0)?;
//!     coo.put(4, 1, 4.0)?;
//!     coo.put(1, 2, 4.0)?;
//!     coo.put(2, 2, -3.0)?;
//!     coo.put(3, 2, 1.0)?;
//!     coo.put(4, 2, 2.0)?;
//!     coo.put(2, 3, 2.0)?;
//!     coo.put(1, 4, 6.0)?;
//!     coo.put(4, 4, 1.0)?;
//!
//!     // parameters
//!     let mut params = LinSolParams::new();
//!     params.verbose = false;
//!     params.compute_determinant = true;
//!
//!     // call factorize
//!     umfpack.factorize(&mut coo, Some(params))?;
//!
//!     // allocate x and rhs
//!     let mut x = Vector::new(ndim);
//!     let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
//!
//!     // calculate the solution
//!     umfpack.solve(&mut x, &coo, &rhs, false)?;
//!     println!("x =\n{}", x);
//!
//!     // check the results
//!     let correct = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//!     vec_approx_eq(x.as_data(), &correct, 1e-14);
//!
//!     // analysis
//!     let mut stats = StatsLinSol::new();
//!     umfpack.update_stats(&mut stats);
//!     let (mx, ex) = (stats.determinant.mantissa_real, stats.determinant.exponent);
//!     println!("det(a) = {:?}", mx * f64::powf(10.0, ex));
//!     println!("rcond  = {:?}", stats.output.umfpack_rcond_estimate);
//!     Ok(())
//! }
//! ```

/// Defines the error output as a static string
pub type StrError = &'static str;

mod aliases;
mod auxiliary_and_constants;
mod complex_lin_solver;
mod complex_solver_mumps;
mod complex_solver_umfpack;
mod coo_matrix;
mod csc_matrix;
mod csr_matrix;
mod enums;
mod lin_sol_params;
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
pub use crate::aliases::*;
use crate::auxiliary_and_constants::*;
pub use crate::complex_lin_solver::*;
pub use crate::complex_solver_mumps::*;
pub use crate::complex_solver_umfpack::*;
pub use crate::coo_matrix::*;
pub use crate::csc_matrix::*;
pub use crate::csr_matrix::*;
pub use crate::enums::*;
pub use crate::lin_sol_params::*;
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
