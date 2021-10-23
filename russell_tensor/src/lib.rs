//! Russell - Rust Scientific Library
//!
//! **tensor**: Tensor analysis structures and functions for continuum mechanics

/// Returns package description
pub fn desc() -> String {
    "Tensor analysis structures and functions for continuum mechanics".to_string()
}

mod constants;
mod operations;
mod samples;
mod tensor2;
mod tensor4;
pub use crate::constants::*;
pub use crate::operations::*;
pub use crate::samples::*;
pub use crate::tensor2::*;
pub use crate::tensor4::*;
