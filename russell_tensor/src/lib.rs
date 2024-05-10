//! Russell - Rust Scientific Library
//!
//! `russell_tensor`: Tensor analysis, calculus, and functions for continuum mechanics
//!
//! **Important:** This crate depends on external libraries (non-Rust). Thus, please check the [Installation Instructions on the GitHub Repository](https://github.com/cpmech/russell).
//!
//! This library implements structures and functions for tensor analysis and calculus. The library focuses on applications in engineering and [Continuum Mechanics](Continuum Mechanics). The essential functionality for the targeted applications includes second-order and fourth-order tensors, scalar "invariants," and derivatives.
//!
//! This library implements derivatives for scalar functions with respect to tensors, tensor functions with respect to tensors, and others. A convenient basis representation known as Mandel basis (similar to Voigt notation) is considered by this library internally. The user may also use the Mandel basis to perform simpler matrix-vector operations directly.

/// Defines the error output as a static string
pub type StrError = &'static str;

mod as_matrix_3x3;
mod as_matrix_9x9;
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
pub use crate::as_matrix_9x9::*;
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

// run code from README file
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctest;
