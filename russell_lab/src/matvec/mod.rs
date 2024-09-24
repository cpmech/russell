//! This module contains functions for calculations with matrices and vectors

mod complex_mat_vec_mul;
mod complex_solve_lin_sys;
mod complex_vec_mat_mul;
mod mat_sum_cols;
mod mat_sum_rows;
mod mat_vec_mul;
mod mat_vec_mul_update;
mod solve_lin_sys;
mod vec_mat_mul;
mod vec_outer;
mod vec_outer_update;

pub use complex_mat_vec_mul::*;
pub use complex_solve_lin_sys::*;
pub use complex_vec_mat_mul::*;
pub use mat_sum_cols::*;
pub use mat_sum_rows::*;
pub use mat_vec_mul::*;
pub use mat_vec_mul_update::*;
pub use solve_lin_sys::*;
pub use vec_mat_mul::*;
pub use vec_outer::*;
pub use vec_outer_update::*;
