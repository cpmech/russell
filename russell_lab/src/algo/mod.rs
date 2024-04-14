//! This module implements algorithms built from base, math, and vector-matrix routines

mod interp_lagrange;
mod num_jacobian;
mod root_solver_bracket;
mod testing;
pub use crate::algo::interp_lagrange::*;
pub use crate::algo::num_jacobian::*;
pub use crate::algo::root_solver_bracket::*;
