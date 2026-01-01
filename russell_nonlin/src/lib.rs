//! Numerical Continuation methods to solve nonlinear systems of equations

/// Defines the error output as a static string
pub type StrError = &'static str;

mod config;
mod enums;
mod iteration_error;
mod logger;
mod output;
mod samples;
mod solver;
mod solver_arclength;
mod solver_natural;
mod solver_trait;
mod stats;
mod system;
mod workspace;

pub use config::*;
pub use enums::*;
use iteration_error::*;
use logger::*;
pub use output::*;
pub use samples::*;
pub use solver::*;
pub use solver_arclength::*;
pub use solver_natural::*;
use solver_trait::*;
pub use stats::*;
pub use system::*;
use workspace::*;

// run code from README file
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctest;
