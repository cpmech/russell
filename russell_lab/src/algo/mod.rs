//! This module implements algorithms built from base, math, and vector-matrix routines

mod bracket_min;
mod common;
mod interp_lagrange;
mod num_jacobian;
mod root_solver_brent;
mod testing;
pub use crate::algo::bracket_min::*;
pub use crate::algo::common::*;
pub use crate::algo::interp_lagrange::*;
pub use crate::algo::num_jacobian::*;
pub use crate::algo::root_solver_brent::*;
