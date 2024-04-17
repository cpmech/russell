//! This module implements algorithms built from base, math, and vector-matrix routines

mod common;
mod interp_lagrange;
mod min_bracketing;
mod min_solver;
mod num_jacobian;
mod quadrature;
mod root_solver_brent;
mod testing;
pub use crate::algo::common::*;
pub use crate::algo::interp_lagrange::*;
pub use crate::algo::min_bracketing::*;
pub use crate::algo::min_solver::*;
pub use crate::algo::num_jacobian::*;
pub use crate::algo::quadrature::*;
pub use crate::algo::root_solver_brent::*;
pub use crate::algo::testing::*;
