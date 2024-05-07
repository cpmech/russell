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
//! The figure below illustrates the base structures (struct) for linear algebra computations. The vector and matrix structures are generic but require the type parameter to implement a set of traits. The essential trait is the `Num` trait provided by the `num-traits` crate. This trait allows vectors and matrices in Russell to hold any numerical type. The `Copy` trait enables the possibility of cloning vectors and matrices directly; however, for efficiency, the BLAS functions can be called instead. `DeserializedOwned` and `Serialize` allow vectors and matrices to be serialized/deserialized, a feature that propagates for all other structures depending on `NumVector` and `NumMatrix`. This feature allows, for instance, writing and reading JSON files with simulation results. `NumCast` is required by `NumVector` because the `linspace` member function needs to cast the number of grid points to the final type (e.g., `f64`). `NumMatrix` requires the `AddAssign` and `MulAssign` traits to implement the convenience member functions named `add` and `mul`.
//!
//! ![Vector-Matrix](https://raw.githubusercontent.com/cpmech/russell/main/russell_lab/data/figures/vector-matrix.svg)
//!
//! `NumVector` has a single data member named `data` (a `Vec<T>`) holding all elements. Thus, `NumVector` wraps the native Rust vector struct. `NumVector` implements various methods, including functions for constructing and transforming vectors. One helpful method is `linspace` to generate a sequence of equally spaced numbers. `NumVector` implements the `Index` trait, allowing the access of elements using the `[i]` notation where `i` is the index.
//!
//! `NumMatrix` stores all entries in the *column-major* representation. The main reason for using the column-major representation is to make the code work directly with Fortran routines, such as the BLAS/LAPACK functions. It is worth noting that BLAS/LAPACK have functions that accept row-major matrices. Nonetheless, these functions add an overhead due to temporary memory allocation and copies, including transposing matrices.
//!
//! Thus, column-major data storage is essential. However, this requirement yields an undesirable side effect. Unfortunately, it is no longer possible to use the Index trait to access matrix elements using a notation such as `a[i][j]` as in other languages. Alternatively, `russell_lab` implements member functions `get(i, j)` and `set(i, j, value)` to access and update matrix elements.
//!
//! `NumMatrix` also implements the `AsArray2D` trait, allowing matrices to be constructed from fixed-size nested lists of elements (stack-allocated), growable nested arrays of type `Vec<Vec<T>>` (heap-allocated) and slices that are views into an array. For example, matrices can be created as follows:
//!
//! ```
//! use russell_lab::{NumMatrix, StrError};
//!
//! fn main() -> Result<(), StrError> {
//!     let a = NumMatrix::<i32>::from(&[
//!         [1, 2, 3], //
//!         [4, 5, 6], //
//!         [7, 8, 9], //
//!     ]);
//!     println!("{}", a);
//!     assert_eq!(
//!         format!("{}", a),
//!         "┌       ┐\n\
//!          │ 1 2 3 │\n\
//!          │ 4 5 6 │\n\
//!          │ 7 8 9 │\n\
//!          └       ┘"
//!     );
//!     assert_eq!(a.as_data(), &[1, 4, 7, 2, 5, 8, 3, 6, 9]);
//!     Ok(())
//! }
//! ```
//!
//! Finally, `NumVector` and `NumMatrix` implement the `Display` trait, allowing them to be pretty printed.
//!
//! ### Functionality
//!
//! The linear algebra functions currently handle only `(f64, i32)` pairs, i.e., accessing the `(double, int)` C functions. We also consider `(Complex64, i32)` pairs.
//!
//! There are many functions for linear algebra, such as (for Real and Complex types):
//!
//! * Vector addition ([vec_add()]), copy ([vec_copy()]), inner ([vec_inner()]) and outer ([vec_outer()]) products, norms ([vec_norm()]), and more
//! * Matrix addition ([mat_add()]), multiplication ([mat_mat_mul()], [mat_t_mat_mul()]), copy ([mat_copy()]), singular-value decomposition ([mat_svd()]), eigenvalues ([mat_eigen()], [mat_eigen_sym()]), pseudo-inverse ([mat_pseudo_inverse()]), inverse ([mat_inverse()]), norms ([mat_norm()]), and more
//! * Matrix-vector multiplication ([mat_vec_mul()])
//! * Solution of dense linear systems with symmetric ([mat_cholesky()]) or non-symmetric ([solve_lin_sys()]) coefficient matrices
//!
//! The `russell_lab` functions are higher-level than the BLAS/LAPACK counterparts, thus losing some of the generality of BLAS/LAPACK. Each BLAS/LAPACK function wrapped by `russell_lab` is carefully documented and thoroughly tested.
//!
//! `russell_lab` implements functions organized in the `vector`, `matvec`, and `matrix` directories. These directories correspond to the BLAS terminology as Level 1, Level 2, and Level 3.
//!
//! All vector functions are prefixed with `vec_` and `complex_vec_`, whereas all matrix functions are prefixed with `mat_` and `complex_mat_`. The `matvec` functions have varied names, albeit descriptive.
//!
//! `russell_lab` implements several other interfaces to BLAS/LAPACK. However, some functions are complemented with native Rust code to provide a higher-level interface or improve the functionality. For instance, after calling the DGEEV function, `mat_eigen` post-processes the results via the `dgeev_data` function to extract the results from LAPACK's compact representation, generating a more convenient interface to the user.
//!
//! An example of added functionality is the `mat_pseudo_inverse` function, which computes the pseudo-inverse of a rectangular matrix using the singular-value decomposition. This function is based on the singular-value decomposition routine provided by LAPACK.
//!
//! ## Complex numbers
//!
//! For convenience, this library re-exports:
//!
//! * `num_complex::Complex64` -- Needed for ComplexMatrix, ComplexVector and complex functions
//! * `num_complex::ComplexFloat` -- Needed for the intrinsic Complex64 operators such as `abs()` an others
//!
//! When using the [crate::cpx!] macro, the `Complex64` type must be imported as well. For example:
//!
//! ```
//! use russell_lab::{cpx, Complex64};
//!
//! println!("{}", cpx!(1.0, 2.0));
//! ```
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

/// Defines complex numbers with f64-real and f64-imaginary parts
pub use num_complex::Complex64;
pub use num_complex::ComplexFloat;

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
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctest;
