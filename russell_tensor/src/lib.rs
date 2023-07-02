//! Russell - Rust Scientific Library
//!
//! **tensor**: Tensor analysis structures and functions for continuum mechanics

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod as_matrix_3x3;
mod constants;
mod enums;
mod lin_elasticity;
mod operations;
mod order_high_derivatives;
mod order_low_derivatives;
mod samples_tensor2;
mod samples_tensor4;
mod spectral2;
mod tensor2;
mod tensor4;
pub use crate::as_matrix_3x3::*;
pub use crate::constants::*;
pub use crate::enums::*;
pub use crate::lin_elasticity::*;
pub use crate::operations::*;
pub use crate::order_high_derivatives::*;
pub use crate::order_low_derivatives::*;
pub use crate::samples_tensor2::*;
pub use crate::samples_tensor4::*;
pub use crate::spectral2::*;
pub use crate::tensor2::*;
pub use crate::tensor4::*;
