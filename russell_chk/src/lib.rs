//! Russell - Rust Scientific Library
//!
//! **chk**: Functions to check vectors and other data in tests
//!
//! # Example
//!
//! ```
//! use num_complex::Complex64;
//! use russell_chk::{approx_eq, complex_approx_eq, deriv_approx_eq, vec_approx_eq};
//!
//! fn main() {
//!     // check float point number
//!     approx_eq(0.0000123, 0.000012, 1e-6);
//!
//!     // check vector of float point numbers
//!     vec_approx_eq(&[0.01, 0.012], &[0.012, 0.01], 1e-2);
//!
//!     // check derivative using central differences
//!     struct Arguments {}
//!     let f = |x: f64, _: &mut Arguments| -x;
//!     let args = &mut Arguments {};
//!     let at_x = 8.0;
//!     let dfdx = -1.01;
//!     deriv_approx_eq(dfdx, at_x, args, 1e-2, f);
//!
//!     // check complex numbers
//!     complex_approx_eq(Complex64::new(1.0, 8.0), Complex64::new(1.001, 8.0), 1e-2);
//! }
//! ```

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod approx_eq;
mod complex_approx_eq;
mod complex_vec_approx_eq;
mod deriv_approx_eq;
mod num_deriv;
mod vec_approx_eq;
pub use crate::approx_eq::*;
pub use crate::complex_approx_eq::*;
pub use crate::complex_vec_approx_eq::*;
pub use crate::deriv_approx_eq::*;
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
