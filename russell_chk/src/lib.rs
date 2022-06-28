//! Russell - Rust Scientific Library
//!
//! **chk**: Functions to check vectors and other data in tests
//!
//! # Example
//!
//! ```
//! use russell_chk::{assert_complex_approx_eq, assert_complex_vec_approx_eq,
//!     assert_approx_eq, assert_vec_approx_eq, assert_deriv_approx_eq};
//! use num_complex::Complex64;
//!
//! // check float point number
//! assert_approx_eq!(0.0000123, 0.000012, 1e-6);
//!
//! // check vector of float point numbers
//! assert_vec_approx_eq!(&[0.01, 0.012], &[0.012, 0.01], 1e-2);
//!
//! // check derivative using central differences
//! struct Arguments {}
//! let f = |x: f64, _: &mut Arguments| -x;
//! let args = &mut Arguments {};
//! let at_x = 8.0;
//! let dfdx = -1.01;
//! assert_deriv_approx_eq!(dfdx, at_x, f, args, 1e-2);
//!
//! // check complex numbers
//! assert_complex_approx_eq!(Complex64::new(1.0,8.0), Complex64::new(1.001,8.0), 1e-2);
//!
//! let a = [
//!     Complex64::new(0.123456789, 5.01),
//!     Complex64::new(0.123456789, 5.01),
//!     Complex64::new(0.123456789, 5.01)];
//! let b = [
//!     Complex64::new(0.12345678, 5.01),
//!     Complex64::new(0.1234567, 5.01),
//!     Complex64::new(0.123456, 5.01)];
//! assert_complex_vec_approx_eq!(&a, &b, 1e-6);
//! ```

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod assert_approx_eq;
mod assert_complex_approx_eq;
mod assert_complex_vec_approx_eq;
mod assert_deriv_approx_eq;
mod assert_vec_approx_eq;
mod num_deriv;
pub use crate::assert_approx_eq::*;
pub use crate::assert_complex_approx_eq::*;
pub use crate::assert_complex_vec_approx_eq::*;
pub use crate::assert_deriv_approx_eq::*;
pub use crate::assert_vec_approx_eq::*;
pub use crate::num_deriv::*;

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
