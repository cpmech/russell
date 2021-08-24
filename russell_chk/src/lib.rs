//! Russell - Rust Scientific Library
//!
//! **chk**: Functions to check vectors and other data in tests

/// Returns package description
pub fn desc() -> String {
    "Functions to check vectors and other data in tests".to_string()
}

mod assert_approx_eq;
mod assert_vec_approx_eq;
pub use crate::assert_approx_eq::*;
pub use crate::assert_vec_approx_eq::*;
