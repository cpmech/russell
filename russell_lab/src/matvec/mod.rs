//! This module contains functions for calculations with matrices and vectors

mod mat_sum_cols;
mod mat_sum_rows;
mod mat_vec_mul;
mod outer;
mod solve_lin_sys;
mod vec_mat_mul;
pub use crate::matvec::mat_sum_cols::*;
pub use crate::matvec::mat_sum_rows::*;
pub use crate::matvec::mat_vec_mul::*;
pub use crate::matvec::outer::*;
pub use crate::matvec::solve_lin_sys::*;
pub use crate::matvec::vec_mat_mul::*;
