//! Russell - Rust Scientific Library
//!
//! **stat**: Statistics calculations, probability distributions, and pseudo random numbers

/// Returns package description
pub fn desc() -> String {
    "Statistics calculations, probability distributions, and pseudo random numbers".to_string()
}

mod histogram;
mod stat;
pub use crate::histogram::*;
pub use crate::stat::*;
