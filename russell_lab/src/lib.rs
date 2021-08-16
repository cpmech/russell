//! Russell - Rust Scientific Library
//!
//! **lab**: matrix-vector "laboratory" tools

// modules //////////////////////////////////////
mod linspace;
mod matrix;
mod operations;
mod vector;
pub use crate::linspace::*;
pub use crate::matrix::*;
pub use crate::operations::*;
pub use crate::vector::*;
