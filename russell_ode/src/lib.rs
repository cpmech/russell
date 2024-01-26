//! Russell - Rust Scientific Library
//!
//! `russell_ode`: Solvers for Ordinary Differential Equations

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod constants;
mod enums;
mod euler_forward;
mod explicit_runge_kutta;
mod num_solver;
mod ode_params;
mod ode_solver;
mod samples;
pub use crate::constants::*;
pub use crate::enums::*;
use crate::euler_forward::*;
use crate::explicit_runge_kutta::*;
use crate::num_solver::*;
pub use crate::ode_params::*;
pub use crate::ode_solver::*;
pub use crate::samples::*;

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
