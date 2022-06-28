//! Russell - Rust Scientific Library
//!
//! **tensor**: Tensor analysis structures and functions for continuum mechanics

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod constants;
mod lin_elasticity;
mod operations;
mod samples;
mod tensor2;
mod tensor4;
mod util;
pub use crate::constants::*;
pub use crate::lin_elasticity::*;
pub use crate::operations::*;
pub use crate::samples::*;
pub use crate::tensor2::*;
pub use crate::tensor4::*;
pub use crate::util::*;
