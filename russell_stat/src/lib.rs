//! Russell - Rust Scientific Library
//!
//! **stat**: Statistics calculations, probability distributions, and pseudo random numbers

/// Returns package description
pub fn desc() -> String {
    "Statistics calculations, probability distributions, and pseudo random numbers".to_string()
}

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod histogram;
mod stat;
pub use crate::histogram::*;
pub use crate::stat::*;
