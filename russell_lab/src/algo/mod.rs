//! This module implements algorithms built from base, math, and vector-matrix routines

mod bracket;
mod bspline;
mod constants;
mod cubic;
mod interp_chebyshev;
mod interp_lagrange;
mod line_search;
mod linear_fitting;
mod min_bracketing;
mod min_solver;
mod newton_solver;
mod num_jacobian;
mod quadrature;
mod root_finder;
mod root_finder_brent;
mod stats;
mod testing;

pub use bracket::*;
pub use bspline::*;
pub use constants::*;
pub use cubic::*;
pub use interp_chebyshev::*;
pub use interp_lagrange::*;
pub use line_search::*;
pub use linear_fitting::*;
pub use min_bracketing::*;
pub use min_solver::*;
pub use newton_solver::*;
pub use num_jacobian::*;
pub use quadrature::*;
pub use root_finder::*;
pub use stats::*;
pub use testing::*;
