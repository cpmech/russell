//! Russell - Rust Scientific Library
//!
//! **lab**: matrix-vector "laboratory" tools
//!
//! # Examples
//!
//! ## Vector addition
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
//! ## Matrix-Matrix multiplication
//!
//! ```
//! use russell_lab::*;
//! let a = Matrix::from(&[
//!     &[1.0, 2.0],
//!     &[3.0, 4.0],
//!     &[5.0, 6.0],
//! ]);
//! let b = Matrix::from(&[
//!     &[-1.0, -2.0, -3.0],
//!     &[-4.0, -5.0, -6.0],
//! ]);
//! let mut c = Matrix::new(3, 3);
//! mat_mat_mul(&mut c, 1.0, &a, &b);
//! let correct = "┌             ┐\n\
//!                │  -9 -12 -15 │\n\
//!                │ -19 -26 -33 │\n\
//!                │ -29 -40 -51 │\n\
//!                └             ┘";
//! assert_eq!(format!("{}", c), correct);
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
