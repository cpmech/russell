//! This module contains functions for calculations with matrices and vectors

mod complex_mat_vec_mul;
mod complex_solve_lin_sys;
mod complex_vec_mat_mul;
mod mat_sum_cols;
mod mat_sum_rows;
mod mat_vec_mul;
mod solve_lin_sys;
mod vec_mat_mul;
mod vec_outer;
pub use crate::matvec::complex_mat_vec_mul::*;
pub use crate::matvec::complex_solve_lin_sys::*;
pub use crate::matvec::complex_vec_mat_mul::*;
pub use crate::matvec::mat_sum_cols::*;
pub use crate::matvec::mat_sum_rows::*;
pub use crate::matvec::mat_vec_mul::*;
pub use crate::matvec::solve_lin_sys::*;
pub use crate::matvec::vec_mat_mul::*;
pub use crate::matvec::vec_outer::*;
