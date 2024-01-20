#![allow(unused)]

//! Russell - Rust Scientific Library
//!
//! `russell_ode`: Solvers for Ordinary Differential Equations

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod constants;
mod definitions;
mod enums;
mod explicit_runge_kutta;
mod ode_output;
mod ode_sol_params;
mod ode_solver;
mod rk_work;
mod runge_kutta_trait;
use crate::constants::*;
use crate::definitions::*;
use crate::enums::*;
use crate::explicit_runge_kutta::*;
use crate::ode_output::*;
use crate::ode_sol_params::*;
use crate::ode_solver::*;
use crate::rk_work::*;
use crate::runge_kutta_trait::*;

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
