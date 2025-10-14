//! Russell - Rust Scientific Library
//!
//! `russell_pde`: Solvers for ordinary differential equations and differential algebraic equations
//!
//! **Important:** This crate depends on external libraries (non-Rust). Thus, please check the [Installation Instructions on the GitHub Repository](https://github.com/cpmech/russell).

/// Defines the error output as a static string
pub type StrError = &'static str;

mod enums;
mod essential_bcs_2d;
mod fdm_laplacian_1d;
mod fdm_laplacian_2d_new;
mod grid_2d;

pub use enums::*;
pub use essential_bcs_2d::*;
pub use fdm_laplacian_1d::*;
pub use fdm_laplacian_2d_new::*;
pub use grid_2d::*;

// run code from README file
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctest;
