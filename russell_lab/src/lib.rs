//! Russell - Rust Scientific Library
//!
//! **lab**: Matrix-vector laboratory including linear algebra tools
//!
//! This crate depends on external libraries (not RUST; e.g., `liblapacke-dev` and `libopenblas-dev`). Thus, please check the [Installation Instructions on our GitHub Repository](https://github.com/cpmech/russell/tree/main/russell_lab).
//!
//! This crate implements several functions to perform linear algebra computations--it is a **mat**rix-vector **lab**oratory 😉. We implement some functions in native Rust code as much as possible but also wrap the best tools available, such as [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) and [Intel MKL](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-2/overview.html).
//!
//! The main structures are [NumVector] and [NumMatrix], which are generic Vector and Matrix structures. The Matrix data is stored as **column-major**. The [Vector] and [Matrix] are `f64` and `Complex64` aliases of `NumVector` and `NumMatrix`, respectively.
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
//! # Example - Cholesky factorization
//!
//! ```
//! use russell_lab::{mat_cholesky, Matrix, StrError};
//!
//! fn main() -> Result<(), StrError> {
//!     // TODO
//!     Ok(())
//! }
//! ```

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

pub mod base;
pub mod check;
mod internal;
pub mod math;
pub mod matrix;
pub mod matvec;
pub mod vector;
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
