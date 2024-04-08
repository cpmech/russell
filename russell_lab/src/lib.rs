//! Russell - Rust Scientific Library
//!
//! `russell_lab`: Matrix-vector laboratory including linear algebra tools
//!
//! **Important:** This crate depends on external libraries (non-Rust). Thus, please check the [Installation Instructions on the GitHub Repository](https://github.com/cpmech/russell).
//!
//! # Introduction
//!
//! This crate implements specialized mathematical functions (e.g., Bessel, Chebyshev, Erf, Gamma, Heaviside, Logistic, ...) and functions to perform linear algebra computations (e.g., Matrix, Vector, Matrix-Vector, Eigen-decomposition, SVD, Inverse, ...). This crate also implements a set of helpful function for comparing floating-point numbers, measuring computer time, reading table-formatted data, and more.
//!
//! The code shall be implemented in *native Rust* code as much as possible. However, thin interfaces ("wrappers") are implemented for some of the best tools available in numerical mathematics, including [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) and [Intel MKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/overview.html).
//!
//! The code is organized in modules:
//!
//! * [check] -- implements functions to assist in unit and integration testing
//! * [base] -- implements a "base" functionality to help other modules
//! * [math] -- implements mathematical (specialized) functions and constants
//! * [vector] -- implements the [NumVector] struct and associated functions
//! * [matrix] -- implements the [NumMatrix] struct and associated functions
//! * [matvec] -- implements functions operating on matrices and vectors
//! * [fftw] -- implements a think wrap to a few [FFTW](https://www.fftw.org/) routines. Warning: these routines are thread-unsafe
//! * [algo] -- implements algorithms that depend on the other modules (e.g, Lagrange interpolation)
//!
//! For linear algebra, the main structures are [NumVector] and [NumMatrix], that are generic Vector and Matrix structures. The Matrix data is stored as **column-major**. The [Vector] and [Matrix] are `f64` and `Complex64` aliases of `NumVector` and `NumMatrix`, respectively.
//!
//! The linear algebra functions currently handle only `(f64, i32)` pairs, i.e., accessing the `(double, int)` C functions. We also consider `(Complex64, i32)` pairs.
//!
//! There are many functions for linear algebra, such as (for Real and Complex types):
//!
//! * Vector addition ([vec_add()]), copy ([vec_copy()]), inner ([vec_inner()]) and outer ([vec_outer()]) products, norms ([vec_norm()]), and more
//! * Matrix addition ([mat_add()]), multiplication ([mat_mat_mul()], [mat_t_mat_mul()]), copy ([mat_copy()]), singular-value decomposition ([mat_svd()]), eigenvalues ([mat_eigen()], [mat_eigen_sym()]), pseudo-inverse ([mat_pseudo_inverse()]), inverse ([mat_inverse()]), norms ([mat_norm()]), and more
//! * Matrix-vector multiplication ([mat_vec_mul()])
//! * Solution of dense linear systems with symmetric ([mat_cholesky()]) or non-symmetric ([solve_lin_sys()]) coefficient matrices
//! * Reading writing files ([read_table()]) , linspace ([NumVector::linspace()]), grid generators ([generate2d()]), [generate3d()]), [linear_fitting()], [Stopwatch] and more
//! * Checking results, comparing float point numbers, and verifying the correctness of derivatives; see [crate::check]
//!
//! **Note:** For the functions dealing with complex numbers, the following line must be added to all derived code:
//!
//! ```
//! use num_complex::Complex64;
//! ```
//!
//! This line will bring [num_complex::Complex64] to the scope. For convenience the (russell_lab) macro [crate::cpx!] may be used to allocate complex numbers.
//!
//! # Example - Cholesky factorization
//!
//! ```
//! use russell_lab::{mat_cholesky, Matrix, StrError};
//!
//! fn main() -> Result<(), StrError> {
//!     // set matrix
//!     let sym = 0.0;
//!     #[rustfmt::skip]
//!     let mut a = Matrix::from(&[
//!         [  4.0,   sym,   sym],
//!         [ 12.0,  37.0,   sym],
//!         [-16.0, -43.0,  98.0],
//!     ]);
//!
//!     // perform factorization
//!     mat_cholesky(&mut a, false)?;
//!
//!     // define alias (for convenience)
//!     let l = &a;
//!
//!     // compare with solution
//!     let l_correct = "┌          ┐\n\
//!                      │  2  0  0 │\n\
//!                      │  6  1  0 │\n\
//!                      │ -8  5  3 │\n\
//!                      └          ┘";
//!     assert_eq!(format!("{}", l), l_correct);
//!
//!     // check:  l ⋅ lᵀ = a
//!     let m = a.nrow();
//!     let mut l_lt = Matrix::new(m, m);
//!     for i in 0..m {
//!         for j in 0..m {
//!             for k in 0..m {
//!                 l_lt.add(i, j, l.get(i, k) * l.get(j, k));
//!             }
//!         }
//!     }
//!     let l_lt_correct = "┌             ┐\n\
//!                         │   4  12 -16 │\n\
//!                         │  12  37 -43 │\n\
//!                         │ -16 -43  98 │\n\
//!                         └             ┘";
//!     assert_eq!(format!("{}", l_lt), l_lt_correct);
//!     Ok(())
//! }
//! ```

/// Defines the error output as a static string
pub type StrError = &'static str;

pub mod algo;
pub mod base;
pub mod check;
pub mod fftw;
mod internal;
pub mod math;
pub mod matrix;
pub mod matvec;
pub mod vector;
// pub use crate::algo::*;
pub use crate::base::*;
pub use crate::check::*;
pub use crate::fftw::*;
use crate::internal::*;
pub use crate::matrix::*;
pub use crate::matvec::*;
pub use crate::vector::*;
