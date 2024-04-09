//! This module implements algorithms built from base, math, and vector-matrix routines

mod fdm5_jacobian;
mod interp_lagrange;
pub use crate::algo::fdm5_jacobian::*;
pub use crate::algo::interp_lagrange::*;
