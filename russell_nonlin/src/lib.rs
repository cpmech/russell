//! Numerical Continuation methods to solve nonlinear systems of equations
//!
//! This crate implements Predictor-Corrector solvers based on the Euler-Newton method for
//! tracing solution branches of parameterized nonlinear systems of the form:
//!
//! ```text
//! G(u, λ) = 0
//! ```
//!
//! where `u` is the state vector and `λ` is a scalar load/control parameter.
//!
//! # Methods
//!
//! Two continuation strategies are available via [`Method`]:
//!
//! * [`Method::Natural`] — Natural parameter continuation. Increments λ step-by-step and solves
//!   `G(u, λ) = 0` at each step. Simple and efficient on regular branches, but **cannot pass
//!   folds** (limit points) where `∂λ/∂s = 0`.
//!
//! * [`Method::Arclength`] — Pseudo-arclength continuation. Parametrizes the solution curve by
//!   an arclength-like variable `s` and solves `G(u(s), λ(s)) = 0`. Can navigate **folds,
//!   turning points, and other singularities**.
//!
//! # Main types
//!
//! | Type | Role |
//! |------|------|
//! | [`System`] | Defines `G(u, λ)` and its Jacobian `∂G/∂u` via callbacks |
//! | [`Config`] | Solver settings: method, tolerances, step control, linear solver |
//! | [`Solver`] | Main entry point; created from a `Config` and a `System` |
//! | [`Stop`] | Stopping criterion (target λ, target u-component, or number of steps) |
//! | [`DeltaLambda`] | Step strategy: automatic, constant, or list-based Δλ |
//! | [`Output`] | Collects accepted-step results and/or runs a user callback |
//! | [`Stats`] | Benchmarking counters returned after solving |
//!
//! # Example
//!
//! ```
//! use russell_nonlin::{Config, DeltaLambda, IniDir, Output, Samples, Solver, Stop};
//! use russell_sparse::Sym;
//!
//! // G(u, λ) = u - λ  →  exact solution: u = λ
//! let (system, mut u, mut l, mut args) = Samples::simple_linear_problem(Sym::No);
//!
//! // configure the solver (Natural parameter continuation is the default)
//! let config = Config::new();
//! let mut solver = Solver::new(&config, system).unwrap();
//!
//! // record results at each accepted step
//! let out = &mut Output::new();
//! out.set_recording(true, &[0], &[]);
//!
//! // trace from λ = 0 to λ = 1 with constant Δλ = 0.1
//! solver
//!     .solve(
//!         &mut args,
//!         &mut u,
//!         &mut l,
//!         IniDir::Pos,
//!         Stop::MaxLambda(1.0),
//!         &DeltaLambda::constant(0.1),
//!         Some(out),
//!     )
//!     .unwrap();
//!
//! assert_eq!(out.get_l_values().len(), 11); // initial point + 10 steps
//! assert!((u[0] - 1.0).abs() < 1e-14);
//! assert!((l - 1.0).abs() < 1e-14);
//! ```

/// Defines the error output as a static string
pub type StrError = &'static str;

mod config;
mod delta_lambda;
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
pub use delta_lambda::*;
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
