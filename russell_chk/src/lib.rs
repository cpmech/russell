//! Russell - Rust Scientific Library
//!
//! **chk**: Functions to check vectors and other data in tests
//!
//! # Example
//!
//! ```
//! # use russell_chk::*;
//! assert_approx_eq!(0.0000123, 0.000012, 1e-6);
//! assert_vec_approx_eq!(&[0.01, 0.012], &[0.012, 0.01], 1e-2);
//! ```

/// Returns package description
pub fn desc() -> String {
    "Functions to check vectors and other data in tests".to_string()
}

mod assert_approx_eq;
mod assert_vec_approx_eq;
pub use crate::assert_approx_eq::*;
pub use crate::assert_vec_approx_eq::*;

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
