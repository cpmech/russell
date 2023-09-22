//! Makes available common structures and functions to perform computations
//!
//! You may write `use russell_sparse::prelude::*` in your code and obtain
//! access to commonly used functionality.

pub use crate::config_solver::ConfigSolver;
pub use crate::coo_matrix::CooMatrix;
pub use crate::csr_matrix::CsrMatrix;
pub use crate::enums::*;
pub use crate::read_matrix_market;
pub use crate::solver::Solver;
pub use crate::solver_mumps::SolverMUMPS;
pub use crate::solver_umfpack::SolverUMFPACK;
pub use crate::verify_lin_sys::VerifyLinSys;
