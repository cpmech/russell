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
//!
//! # Example - Cholesky factorization
//!
//! ```
//! use russell_lab::{mat_cholesky, Matrix, StrError};
//!
//! fn main() -> Result<(), StrError> {
//!     // set matrix
//!     let a = Matrix::from(&[
//!         [  4.0,  12.0, -16.0],
//!         [ 12.0,  37.0, -43.0],
//!         [-16.0, -43.0,  98.0],
//!     ]);
//!
//!     // perform factorization
//!     let m = a.nrow();
//!     let mut l = Matrix::new(m, m);
//!     mat_cholesky(&mut l, &a)?;
//!
//!     // compare with solution
//!     let l_correct = "┌          ┐\n\
//!                      │  2  0  0 │\n\
//!                      │  6  1  0 │\n\
//!                      │ -8  5  3 │\n\
//!                      └          ┘";
//!     assert_eq!(format!("{}", l), l_correct);
//!     Ok(())
//! }
//! ```

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod add_arrays;
mod as_array;
mod auxiliary_and_constants;
mod enums;
mod formatters;
mod generators;
mod linear_fitting;
pub mod math;
mod matrix;
mod matvec;
pub mod prelude;
mod read_table;
mod sort;
mod sort_vec_mat;
mod stopwatch;
mod testing;
mod vector;
use crate::add_arrays::*;
pub use crate::as_array::*;
pub use crate::auxiliary_and_constants::*;
pub use crate::enums::*;
pub use crate::formatters::*;
pub use crate::generators::*;
pub use crate::linear_fitting::*;
pub use crate::matrix::*;
pub use crate::matvec::*;
pub use crate::read_table::*;
pub use crate::sort::*;
pub use crate::sort_vec_mat::*;
pub use crate::stopwatch::*;
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
