//! Russell - Rust Scientific Library
//!
//! **lab**: matrix-vector "laboratory" tools
//!
//! # Examples
//!
//! ```
//! use russell_lab::*;
//! let u = Vector::from(&[10.0, 20.0, 30.0, 40.0]);
//! let v = Vector::from(&[2.0, 1.5, 1.0, 0.5]);
//! let mut w = Vector::new(4);
//! add_vectors(&mut w, 0.1, &u, 2.0, &v);
//! let correct = "┌   ┐\n\
//!                │ 5 │\n\
//!                │ 5 │\n\
//!                │ 5 │\n\
//!                │ 5 │\n\
//!                └   ┘";
//! assert_eq!(format!("{}", w), correct);
//! ```
//!

#![feature(portable_simd)]

mod linspace;
mod matrix;
mod operations;
mod vector;
pub use crate::linspace::*;
pub use crate::matrix::*;
pub use crate::operations::*;
pub use crate::vector::*;
