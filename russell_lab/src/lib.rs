//! Russell - Rust Scientific Library
//!
//! **lab**: Matrix-vector laboratory including linear algebra tools

/// Returns package description
pub fn desc() -> String {
    "Matrix-vector laboratory including linear algebra tools".to_string()
}

mod as_array;
mod enums;
mod formatters;
mod generators;
mod matrix;
mod matvec;
mod stopwatch;
mod vector;
pub use crate::as_array::*;
pub use crate::enums::*;
pub use crate::formatters::*;
pub use crate::generators::*;
pub use crate::matrix::*;
pub use crate::matvec::*;
pub use crate::stopwatch::*;
pub use crate::vector::*;
