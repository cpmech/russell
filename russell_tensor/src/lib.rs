//! Russell - Rust Scientific Library
//!
//! **tensor**: Tensor analysis structures and functions for continuum mechanics

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod as_matrix_3x3;
mod constants;
mod derivatives_t2;
mod derivatives_t4;
mod enums;
mod lin_elasticity;
mod operations;
mod samples_tensor2;
mod samples_tensor4;
mod spectral2;
mod tensor2;
mod tensor4;
pub use crate::as_matrix_3x3::*;
pub use crate::constants::*;
pub use crate::derivatives_t2::*;
pub use crate::derivatives_t4::*;
pub use crate::enums::*;
pub use crate::lin_elasticity::*;
pub use crate::operations::*;
pub use crate::samples_tensor2::*;
pub use crate::samples_tensor4::*;
pub use crate::spectral2::*;
pub use crate::tensor2::*;
pub use crate::tensor4::*;
