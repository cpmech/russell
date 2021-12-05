//! Russell - Rust Scientific Library
//!
//! **tensor**: Tensor analysis structures and functions for continuum mechanics

/// Returns package description
pub fn desc() -> String {
    "Tensor analysis structures and functions for continuum mechanics".to_string()
}

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod constants;
mod operations;
mod samples;
mod tensor2;
mod tensor4;
mod util;
pub use crate::constants::*;
pub use crate::operations::*;
pub use crate::samples::*;
pub use crate::tensor2::*;
pub use crate::tensor4::*;
pub use crate::util::*;
