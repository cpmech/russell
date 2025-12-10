//! Russell - Rust Scientific Library
//!
//! `russell_pde`: Solvers for ordinary differential equations and differential algebraic equations
//!
//! **Important:** This crate depends on external libraries (non-Rust). Thus, please check the [Installation Instructions on the GitHub Repository](https://github.com/cpmech/russell).

/// Defines the error output as a static string
pub type StrError = &'static str;

mod constants;
mod enums;
mod equation_handler;
mod essential_bcs_1d;
mod essential_bcs_2d;
mod fdm_laplacian_1d;
mod fdm_laplacian_2d;
mod grid_1d;
mod grid_2d;
mod metrics;
mod metrics_2d;
mod spectral_laplacian_2d;
mod transfinite_2d;
mod transfinite_3d;
mod transfinite_samples;

pub use constants::*;
pub use enums::*;
pub use equation_handler::*;
pub use essential_bcs_1d::*;
pub use essential_bcs_2d::*;
pub use fdm_laplacian_1d::*;
pub use fdm_laplacian_2d::*;
pub use grid_1d::*;
pub use grid_2d::*;
pub use metrics::*;
pub use metrics_2d::*;
pub use spectral_laplacian_2d::*;
pub use transfinite_2d::*;
pub use transfinite_3d::*;
pub use transfinite_samples::*;

// run code from README file
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctest;
