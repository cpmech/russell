//! Russell - Rust Scientific Library
//!
//! `russell_ode`: Numerical solvers for systems of ordinary differential equations (ODEs) and differential algebraic equations (DAE)
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
//! **Note:** The mass matrix is optional and need not be specified.
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
//! # Recommended methods
//!
//! * [Method::DoPri5] for ODE systems and non-stiff problems using moderate tolerances
//! * [Method::DoPri8] for ODE systems and non-stiff problems using strict tolerances
//! * [Method::Radau5] for ODE and DAE systems, possibly stiff, with moderate to strict tolerances
//!
//! **Note:** A *Stiff problem* arises due to a combination of conditions, such as
//! the ODE system equations, the initial values, the stepsize, and the numerical method.
//!
//! # Limitations
//!
//! * Currently, the only method that can solve DAE systems is [Method::Radau5]
//! * Currently, *dense output* is only available for [Method::DoPri5], [Method::DoPri8], and [Method::Radau5]
//!
//! # References
//!
//! 1. E. Hairer, S. P. Nørsett, G. Wanner (2008) Solving Ordinary Differential Equations I.
//!    Non-stiff Problems. Second Revised Edition. Corrected 3rd printing 2008. Springer Series
//!    in Computational Mathematics, 528p
//! 2. E. Hairer, G. Wanner (2002) Solving Ordinary Differential Equations II.
//!    Stiff and Differential-Algebraic Problems. Second Revised Edition.
//!    Corrected 2nd printing 2002. Springer Series in Computational Mathematics, 614p

/// Defines the error output as a static string
pub type StrError = &'static str;

mod benchmark;
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
pub mod prelude;
mod radau5;
mod samples;
mod system;
mod workspace;
pub use crate::benchmark::*;
pub use crate::constants::*;
use crate::detect_stiffness::*;
pub use crate::enums::*;
use crate::erk_dense_out::*;
use crate::euler_backward::*;
use crate::euler_forward::*;
use crate::explicit_runge_kutta::*;
pub use crate::ode_solver::*;
use crate::ode_solver_trait::*;
pub use crate::output::*;
pub use crate::params::*;
use crate::radau5::*;
pub use crate::samples::*;
pub use crate::system::*;
use crate::workspace::*;

// run code from README file
#[cfg(doctest)]
mod test_readme {
    macro_rules! external_doc_test {
        ($x:expr) => {
            #[doc = $x]
            extern "C" {}
        };
    }
    external_doc_test!(include_str!("../README.md"));
}
