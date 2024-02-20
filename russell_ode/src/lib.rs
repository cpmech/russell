//! Russell - Rust Scientific Library
//!
//! `russell_ode`: Solvers for Ordinary Differential Equations

/// Defines the error output as a static string
pub type StrError = &'static str;

mod auxiliary;
mod benchmark;
mod constants;
mod enums;
mod euler_backward;
mod euler_forward;
mod explicit_runge_kutta;
mod ode_solver;
mod ode_solver_trait;
mod output;
mod params;
mod radau5;
mod samples;
mod system;
mod workspace;
pub use crate::auxiliary::*;
pub use crate::benchmark::*;
pub use crate::constants::*;
pub use crate::enums::*;
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
