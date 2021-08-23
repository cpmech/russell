//! Russell - Rust Scientific Library
//!
//! **lab**: matrix-vector "laboratory" tools

#![feature(portable_simd)]

mod linspace;
mod matrix;
mod operations;
mod simd_operations;
mod vector;
pub use crate::linspace::*;
pub use crate::matrix::*;
pub use crate::operations::*;
// use crate::simd_operations::*;
pub use crate::vector::*;
