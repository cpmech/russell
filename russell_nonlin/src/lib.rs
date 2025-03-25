//! Numerical Continuation methods to solve nonlinear systems of equations

/// Defines the error output as a static string
pub type StrError = &'static str;

mod enums;
mod logger;
mod nl_params;
mod nl_solver;
mod nl_solver_trait;
mod nl_state;
mod nl_system;
mod num_error;
mod output;
pub mod prelude;
mod samples;
mod solver_arclength;
mod solver_parametric;
mod solver_simple;
mod stats;
mod workspace;

pub use enums::*;
use logger::*;
pub use nl_params::*;
pub use nl_solver::*;
use nl_solver_trait::*;
pub use nl_state::*;
pub use nl_system::*;
use num_error::*;
pub use output::*;
pub use samples::*;
pub use solver_arclength::*;
pub use solver_parametric::*;
pub use solver_simple::*;
pub use stats::*;
use workspace::*;

// run code from README file
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctest;
