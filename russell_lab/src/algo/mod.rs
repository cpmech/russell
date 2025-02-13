//! This module implements algorithms built from base, math, and vector-matrix routines

mod common;
mod graph_dir;
mod interp_chebyshev;
mod interp_lagrange;
mod linear_fitting;
mod min_bracketing;
mod min_solver;
mod num_jacobian;
mod quadrature;
mod root_finder;
mod root_finder_brent;
mod testing;

pub use common::*;
pub use graph_dir::*;
pub use interp_chebyshev::*;
pub use interp_lagrange::*;
pub use linear_fitting::*;
pub use min_bracketing::*;
pub use min_solver::*;
pub use num_jacobian::*;
pub use quadrature::*;
pub use root_finder::*;
pub use testing::*;
