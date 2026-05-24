//! Makes available common structures and functions to perform computations
//!
//! You may write `use russell_sparse::prelude::*` in your code and obtain
//! access to commonly used functionality.

pub use crate::aliases::*;
pub use crate::complex_lin_solver::*;
pub use crate::complex_solver_umfpack::ComplexSolverUMFPACK;
pub use crate::coo_matrix::NumCooMatrix;
pub use crate::csc_matrix::NumCscMatrix;
pub use crate::csr_matrix::NumCsrMatrix;
pub use crate::enums::*;
pub use crate::lin_sol_params::LinSolParams;
pub use crate::lin_solver::*;
pub use crate::numerical_jacobian::numerical_jacobian;
pub use crate::read_matrix_market;
pub use crate::solver_umfpack::SolverUMFPACK;
pub use crate::stats_lin_sol::StatsLinSol;
pub use crate::verify_lin_sys::VerifyLinSys;

#[cfg(feature = "with_mumps")]
pub use crate::complex_solver_mumps::ComplexSolverMUMPS;

#[cfg(feature = "with_mumps")]
pub use crate::solver_mumps::SolverMUMPS;
