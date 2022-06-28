//! Russell - Rust Scientific Library
//!
//! **lab**: Matrix-vector laboratory including linear algebra tools
//!
//! # Example - Cholesky factorization
//!
//! ```
//! use russell_lab::{cholesky_factor, Matrix, StrError};
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
//!     cholesky_factor(&mut l, &a)?;
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

/// Returns package description
pub fn desc() -> String {
    "Matrix-vector laboratory including linear algebra tools".to_string()
}

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod as_array;
mod constants;
mod enums;
mod formatters;
mod generators;
mod math;
mod matrix;
mod matvec;
mod sort;
mod stopwatch;
mod vector;
pub use crate::as_array::*;
use crate::constants::*;
pub use crate::enums::*;
pub use crate::formatters::*;
pub use crate::generators::*;
pub use crate::math::*;
pub use crate::matrix::*;
pub use crate::matvec::*;
pub use crate::sort::*;
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
