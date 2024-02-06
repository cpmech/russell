//! Russell - Rust Scientific Library
//!
//! `russell_ode`: Solvers for Ordinary Differential Equations

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod auxiliary;
mod benchmark;
mod constants;
mod enums;
mod euler_backward;
mod euler_forward;
mod explicit_runge_kutta;
mod num_solver;
mod params;
mod radau5;
mod samples;
mod solver;
mod system;
mod workspace;
pub use crate::auxiliary::*;
pub use crate::benchmark::*;
pub use crate::constants::*;
pub use crate::enums::*;
use crate::euler_backward::*;
use crate::euler_forward::*;
use crate::explicit_runge_kutta::*;
use crate::num_solver::*;
pub use crate::params::*;
use crate::radau5::*;
pub use crate::samples::*;
pub use crate::solver::*;
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
