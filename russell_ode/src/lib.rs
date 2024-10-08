//! Russell - Rust Scientific Library
//!
//! `russell_ode`: Solvers for ordinary differential equations and differential algebraic equations
//!
//! **Important:** This crate depends on external libraries (non-Rust). Thus, please check the [Installation Instructions on the GitHub Repository](https://github.com/cpmech/russell).
//!
//! ![arenstorf_dopri8](https://raw.githubusercontent.com/cpmech/russell/main/russell_ode/data/figures/arenstorf_dopri8.svg)
//!
//! The figure above is the solution of the Arenstorf Orbit obtained with [russell_ode](https://github.com/cpmech/russell/blob/main/russell_ode/examples/arenstorf_dopri8.rs).
//!
//! # Introduction
//!
//! This library implements (**natively**) numerical solvers for systems of ordinary equations (ODEs) and differential-algebraic equation systems (DAEs) of Index-1. One advantage of a native implementation is the "safety aspects" enforced by Rust. Moreover, we implement thread-safe code. For example, the performance is improved when the real-based linear and complex-based linear systems are factorized concurrently, as in our Radau5.
//!
//! The principal structs are (see the figure below):
//!
//! * [System] defines the ODE or DAE system
//! * [OdeSolver] implements the "time-stepping" loop and calls the *actual* numerical solver
//! * [Params] holds numeric parameters needed by all methods
//! * (optional) [Output] holds the results from accepted steps (all methods) or the *dense output* (DoPri5, DoPri8, and Radau5 only)
//! * (optional) [Stats] holds statistics and benchmarking data
//!
//! ![ODE principal structs](https://raw.githubusercontent.com/cpmech/russell/main/russell_ode/data/figures/ode-principal-structs.svg)
//!
//! The [System] struct holds the number of equations (system dimension), the right-hand side function `f(x, y)`, an optional function to compute the Jacobian matrix, and, also optionally, the mass matrix.
//!
//! The [OdeSolver] approximates the solution of the ODE/DAE using either fixed or variable steps. Some methods can only be run with fixed steps (this is automatically detected). In addition to the system struct, the solver takes [Params] as input.
//!
//! A set of default (~optimal) parameters are allocated by [Params::new()]. If needed, the user may *tweak* the parameters by accessing each parameter subgroup:
//!
//! * [ParamsNewton] parameters for Newton's iterations' for the methods that use iterations such asBwEuler and Radau5
//! * [ParamsStep] parameters for the variable-step control
//! * [ParamsStiffness] parameters to control and enable the stiffness detection (DoPri5 and DoPri8 only)
//! * [ParamsBwEuler] parameters for the BwEuler solver
//! * [ParamsRadau5] parameters for the Radau5 solver
//! * [ParamsERK] parameters for all explicit Runge-Kutta methods (e.g., DoPri5, DoPri8)
//!
//! The ODE and DAE systems are represented as follows:
//!
//! ```text
//!     d{y}
//! [M] ———— = {f}(x, {y})
//!      dx
//! ```
//!
//! where `x` is the independent scalar variable (e.g., time), `{y}` is the solution vector,
//! `{f}` is the right-hand side vector, and `[M]` is the so-called "mass matrix".
//!
//! **Note:** The mass matrix is optional and need not be specified (unless the DAE under study requires it).
//!
//! The Jacobian is defined by:
//!
//! ```text
//!               ∂{f}
//! [J](x, {y}) = ————
//!               ∂{y}
//! ```
//!
//! where `[J]` is the Jacobian matrix.
//!
//! **Note:** The Jacobian function is not required for explicit Runge-Kutta methods (see [Method] and [Information]).
//!
//! The flag [ParamsNewton::use_numerical_jacobian] may be set to true to compute the Jacobian matrix numerically. This option works with or without specifying the analytical Jacobian function.
//!
//! ## Recommended methods
//!
//! * [Method::DoPri5] for ODE systems and non-stiff problems using moderate tolerances
//! * [Method::DoPri8] for ODE systems and non-stiff problems using strict tolerances
//! * [Method::Radau5] for ODE and DAE systems, possibly stiff, with moderate to strict tolerances
//!
//! **Note:** A *Stiff problem* arises due to a combination of conditions, such as
//! the ODE system equations, the initial values, the stepsize, and the numerical method.
//!
//! ## Limitations
//!
//! * Currently, the only method that can solve DAE systems is [Method::Radau5]
//! * Currently, *dense output* is only available for [Method::DoPri5], [Method::DoPri8], and [Method::Radau5]
//!
//! ## References
//!
//! 1. E. Hairer, S. P. Nørsett, G. Wanner (2008) Solving Ordinary Differential Equations I.
//!    Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
//!    in Computational Mathematics, 528p
//! 2. E. Hairer, G. Wanner (2002) Solving Ordinary Differential Equations II.
//!    Stiff and Differential-Algebraic Problems. Second Revised Edition.
//!    Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p
//!
//! # Examples
//!
//! See also [the examples on GitHub page](https://github.com/cpmech/russell/tree/main/russell_ode)
//!
//! ### Simple system with mass matrix
//!
//! Solve with Radau5:
//!
//! ```text
//! y0' + y1'     = -y0 + y1
//! y0' - y1'     =  y0 + y1
//!           y2' = 1/(1 + x)
//!
//! y0(0) = 1,  y1(0) = 0,  y2(0) = 0
//! ```
//!
//! Thus:
//!
//! ```text
//! M y' = f(x, y)
//! ```
//!
//! with:
//!
//! ```text
//!     ┌          ┐       ┌           ┐
//!     │  1  1  0 │       │ -y0 + y1  │
//! M = │  1 -1  0 │   f = │  y0 + y1  │
//!     │  0  0  1 │       │ 1/(1 + x) │
//!     └          ┘       └           ┘
//! ```
//!
//! The Jacobian matrix is:
//!
//! ```text
//!          ┌          ┐
//!     df   │ -1  1  0 │
//! J = —— = │  1  1  0 │
//!     dy   │  0  0  0 │
//!          └          ┘
//! ```
//!
//! Code:
//!
//! ```
//! use russell_lab::{StrError, Vector};
//! use russell_ode::prelude::*;
//! use russell_sparse::{CooMatrix, Sym};
//!
//! fn main() -> Result<(), StrError> {
//!     // DAE system
//!     let ndim = 3;
//!     let jac_nnz = 4; // number of non-zero values in the Jacobian
//!     let mut system = System::new(ndim, |f: &mut Vector, x: f64, y: &Vector, _args: &mut NoArgs| {
//!         f[0] = -y[0] + y[1];
//!         f[1] = y[0] + y[1];
//!         f[2] = 1.0 / (1.0 + x);
//!         Ok(())
//!     });
//!
//!     // function to compute the Jacobian matrix
//!     let symmetric = Sym::No;
//!     system.set_jacobian(
//!         Some(jac_nnz),
//!         symmetric,
//!         move |jj: &mut CooMatrix, alpha: f64, _x: f64, _y: &Vector, _args: &mut NoArgs| {
//!             jj.reset();
//!             jj.put(0, 0, alpha * (-1.0))?;
//!             jj.put(0, 1, alpha * (1.0))?;
//!             jj.put(1, 0, alpha * (1.0))?;
//!             jj.put(1, 1, alpha * (1.0))?;
//!             Ok(())
//!         },
//!     )?;
//!
//!     // mass matrix
//!     let mass_nnz = 5; // number of non-zero values in the mass matrix
//!     system.set_mass(Some(mass_nnz), symmetric, |mm: &mut CooMatrix| {
//!         mm.put(0, 0, 1.0).unwrap();
//!         mm.put(0, 1, 1.0).unwrap();
//!         mm.put(1, 0, 1.0).unwrap();
//!         mm.put(1, 1, -1.0).unwrap();
//!         mm.put(2, 2, 1.0).unwrap();
//!     })?;
//!
//!     // solver
//!     let params = Params::new(Method::Radau5);
//!     let mut solver = OdeSolver::new(params, system)?;
//!
//!     // initial values
//!     let x = 0.0;
//!     let mut y = Vector::from(&[1.0, 0.0, 0.0]);
//!
//!     // solve from x = 0 to x = 2
//!     let x1 = 2.0;
//!     let mut args = 0;
//!     solver.solve(&mut y, x, x1, None, &mut args)?;
//!     println!("y =\n{}", y);
//!     Ok(())
//! }
//! ```

/// Defines the error output as a static string
pub type StrError = &'static str;

mod constants;
mod detect_stiffness;
mod enums;
mod erk_dense_out;
mod euler_backward;
mod euler_forward;
mod explicit_runge_kutta;
mod ode_solver;
mod ode_solver_trait;
mod output;
mod params;
mod pde_discrete_laplacian_2d;
pub mod prelude;
mod radau5;
mod samples;
mod stats;
mod system;
mod workspace;

pub use constants::*;
use detect_stiffness::*;
pub use enums::*;
use erk_dense_out::*;
use euler_backward::*;
use euler_forward::*;
use explicit_runge_kutta::*;
pub use ode_solver::*;
use ode_solver_trait::*;
pub use output::*;
pub use params::*;
pub use pde_discrete_laplacian_2d::*;
use radau5::*;
pub use samples::*;
pub use stats::*;
pub use system::*;
use workspace::*;

// run code from README file
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctest;
