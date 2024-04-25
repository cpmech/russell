//! Russell - Rust Scientific Library
//!
//! `russell_lab`: Scientific laboratory for linear algebra and numerical mathematics
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
//! * [algo] -- Structs and algorithms that roughly depend on the other modules
//! * [base] -- "Base" functionality to help other modules
//! * [check] -- Functions to assist in unit and integration testing
//! * [math] (*not re-exported*) -- Mathematical "special" functions and constants
//! * [matrix] -- Matrix struct and associated functions
//! * [matvec] -- Functions operating on matrices and vectors
//! * [vector] -- Vector struct and associated functions
//!
//! ## Linear algebra
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
//! * Reading writing files ([read_table()]) , linspace ([NumVector::linspace()]), grid generators ([generate2d()]), [generate3d()]), [Stopwatch] and more
//! * Checking results, comparing float point numbers, and verifying the correctness of derivatives; see [crate::check]
//!
//! ## Complex numbers
//!
//! **Note:** For the functions dealing with complex numbers, the following line must be added to all derived code:
//!
//! ```
//! use num_complex::Complex64;
//! ```
//!
//! This line will bring [num_complex::Complex64] to the scope. For convenience the (russell_lab) macro [crate::cpx!] may be used to allocate complex numbers.
//!
//! # Examples
//!
//! ## (matrix) Eigen-decomposition of a small matrix
//!
//! ```
//! use russell_lab::{mat_eigen, Matrix, Vector};
//! use russell_lab::StrError;
//!
//! fn main() -> Result<(), StrError> {
//!     let data = [
//!         [2.0, 0.0, 0.0],
//!         [0.0, 3.0, 4.0],
//!         [0.0, 4.0, 9.0],
//!     ];
//!     let mut a = Matrix::from(&data);
//!     let m = a.nrow();
//!     let mut l_real = Vector::new(m);
//!     let mut l_imag = Vector::new(m);
//!     let mut v_real = Matrix::new(m, m);
//!     let mut v_imag = Matrix::new(m, m);
//!     mat_eigen(&mut l_real, &mut l_imag, &mut v_real, &mut v_imag, &mut a)?;
//!     println!("eigenvalues =\n{}", l_real);
//!     println!("eigenvectors =\n{}", v_real);
//!     Ok(())
//! }
//! ```
//!
//! ## (math) Bessel functions
//!
//! ```
//! use russell_lab::math;
//! use russell_lab::{Vector, StrError};
//!
//! fn main() -> Result<(), StrError> {
//!     let xx = Vector::linspace(0.0, 15.0, 101)?;
//!     let j0 = xx.get_mapped(|x| math::bessel_j0(x));
//!     let j1 = xx.get_mapped(|x| math::bessel_j1(x));
//!     let j2 = xx.get_mapped(|x| math::bessel_jn(2, x));
//!     Ok(())
//! }
//! ```

/// Defines the error output as a static string
pub type StrError = &'static str;

pub mod algo;
pub mod base;
pub mod check;
mod internal;
pub mod math;
pub mod matrix;
pub mod matvec;
pub mod vector;
pub use crate::algo::*;
pub use crate::base::*;
pub use crate::check::*;
use crate::internal::*;
pub use crate::matrix::*;
pub use crate::matvec::*;
pub use crate::vector::*;

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
}
