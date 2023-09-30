//! Russell - Rust Scientific Library
//!
//! **sparse**: Sparse matrix tools and solvers
//!
//! # Examples
//!
//! ## Solving a sparse linear system using UMFPACK
//!
//! TODO
//!
//! ## Using the common solver interface
//!
//! TODO
//!
//! ## Using the common solver interface (single-use)
//!
//! TODO

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod auxiliary_and_constants;
mod coo_matrix;
mod csc_matrix;
mod csr_matrix;
mod enums;
mod lin_sol_stats;
mod lin_solver;
pub mod prelude;
mod read_matrix_market;
mod samples;
mod solver_intel_dss;
mod solver_mumps;
mod solver_umfpack;
mod sparse_matrix;
mod verify_lin_sys;
mod write_matrix_market;
use crate::auxiliary_and_constants::*;
pub use crate::coo_matrix::*;
pub use crate::csc_matrix::*;
pub use crate::csr_matrix::*;
pub use crate::enums::*;
pub use crate::lin_sol_stats::*;
pub use crate::lin_solver::*;
pub use crate::read_matrix_market::*;
pub use crate::samples::*;
pub use crate::solver_intel_dss::*;
pub use crate::solver_mumps::*;
pub use crate::solver_umfpack::*;
pub use crate::sparse_matrix::*;
pub use crate::verify_lin_sys::*;
pub use crate::write_matrix_market::*;

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
    external_doc_test!(include_str!("../../README.md"));
}
