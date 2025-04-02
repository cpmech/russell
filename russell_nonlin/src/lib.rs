//! Numerical Continuation methods to solve nonlinear systems of equations

/// Defines the error output as a static string
pub type StrError = &'static str;

mod enums;
mod logger;
mod nl_config;
mod nl_solver;
mod nl_solver_trait;
mod nl_state;
mod nl_system;
mod num_error;
mod output;
pub mod prelude;
mod samples;
mod solver_arclength;
mod solver_natural;
mod stats;
mod workspace;

pub use enums::*;
use logger::*;
pub use nl_config::*;
pub use nl_solver::*;
use nl_solver_trait::*;
pub use nl_state::*;
pub use nl_system::*;
use num_error::*;
pub use output::*;
pub use samples::*;
pub use solver_arclength::*;
pub use solver_natural::*;
pub use stats::*;
use workspace::*;

// run code from README file
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctest;
