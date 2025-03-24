//! Numerical Continuation methods to solve nonlinear systems of equations

/// Defines the error output as a static string
pub type StrError = &'static str;

mod arclength_solver;
mod nonlin_solver;
mod parametric_solver;

pub use arclength_solver::*;
pub use nonlin_solver::*;
pub use parametric_solver::*;

// run code from README file
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctest;
