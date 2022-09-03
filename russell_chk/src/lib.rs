//! Russell - Rust Scientific Library
//!
//! **chk**: Functions to check vectors and other data in tests
//!
//! # Example
//!
//! ```
//! ```

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod approx_eq;
mod assert_deriv_approx_eq;
mod complex_approx_eq;
mod complex_vec_approx_eq;
mod num_deriv;
mod vec_approx_eq;
pub use crate::approx_eq::*;
pub use crate::assert_deriv_approx_eq::*;
pub use crate::complex_approx_eq::*;
pub use crate::complex_vec_approx_eq::*;
pub use crate::num_deriv::*;
pub use crate::vec_approx_eq::*;

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
