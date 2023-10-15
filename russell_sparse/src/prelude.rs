//! Makes available common structures and functions to perform computations
//!
//! You may write `use russell_sparse::prelude::*` in your code and obtain
//! access to commonly used functionality.

pub use crate::coo_matrix::CooMatrix;
pub use crate::csc_matrix::CscMatrix;
pub use crate::csr_matrix::CsrMatrix;
pub use crate::enums::*;
pub use crate::lin_solver::*;
pub use crate::read_matrix_market;
pub use crate::solver_intel_dss::SolverIntelDSS;
pub use crate::solver_mumps::SolverMUMPS;
pub use crate::solver_umfpack::SolverUMFPACK;
pub use crate::sparse_matrix::SparseMatrix;
pub use crate::stats_lin_sol::StatsLinSol;
pub use crate::verify_lin_sys::VerifyLinSys;
