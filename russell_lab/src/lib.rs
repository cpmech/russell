//! Russell - Rust Scientific Library
//!
//! **lab**: Matrix-vector laboratory including linear algebra tools
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

mod as_array;
mod constants;
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
pub use crate::as_array::*;
use crate::constants::*;
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
