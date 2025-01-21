//! This module contains functions for calculations with matrices

mod aliases;
mod complex_mat_add;
mod complex_mat_approx_eq;
mod complex_mat_cholesky;
mod complex_mat_copy;
mod complex_mat_eigen;
mod complex_mat_eigen_herm;
mod complex_mat_gen_eigen;
mod complex_mat_herm_rank_op;
mod complex_mat_inverse;
mod complex_mat_mat_mul;
mod complex_mat_norm;
mod complex_mat_scale;
mod complex_mat_svd;
mod complex_mat_sym_rank_op;
mod complex_mat_t_mat_mul;
mod complex_mat_unzip;
mod complex_mat_update;
mod complex_mat_zip;
mod mat_add;
mod mat_approx_eq;
mod mat_cholesky;
mod mat_convert_to_blas_band;
mod mat_copy;
mod mat_eigen;
mod mat_eigen_sym;
mod mat_eigen_sym_jacobi;
mod mat_eigenvalues;
mod mat_gen_eigen;
mod mat_inverse;
mod mat_mat_mul;
mod mat_max_abs_diff;
mod mat_norm;
mod mat_pseudo_inverse;
mod mat_scale;
mod mat_svd;
mod mat_sym_rank_op;
mod mat_t_mat_mul;
mod mat_to_static_array;
mod mat_update;
mod mat_write_vismatrix;
mod num_matrix;
mod testing;

pub use aliases::*;
pub use complex_mat_add::*;
pub use complex_mat_approx_eq::*;
pub use complex_mat_cholesky::*;
pub use complex_mat_copy::*;
pub use complex_mat_eigen::*;
pub use complex_mat_eigen_herm::*;
pub use complex_mat_gen_eigen::*;
pub use complex_mat_herm_rank_op::*;
pub use complex_mat_inverse::*;
pub use complex_mat_mat_mul::*;
pub use complex_mat_norm::*;
pub use complex_mat_scale::*;
pub use complex_mat_svd::*;
pub use complex_mat_sym_rank_op::*;
pub use complex_mat_t_mat_mul::*;
pub use complex_mat_unzip::*;
pub use complex_mat_update::*;
pub use complex_mat_zip::*;
pub use mat_add::*;
pub use mat_approx_eq::*;
pub use mat_cholesky::*;
pub use mat_convert_to_blas_band::*;
pub use mat_copy::*;
pub use mat_eigen::*;
pub use mat_eigen_sym::*;
pub use mat_eigen_sym_jacobi::*;
pub use mat_eigenvalues::*;
pub use mat_gen_eigen::*;
pub use mat_inverse::*;
pub use mat_mat_mul::*;
pub use mat_max_abs_diff::*;
pub use mat_norm::*;
pub use mat_pseudo_inverse::*;
pub use mat_scale::*;
pub use mat_svd::*;
pub use mat_sym_rank_op::*;
pub use mat_t_mat_mul::*;
pub use mat_to_static_array::*;
pub use mat_update::*;
pub use mat_write_vismatrix::*;
pub use num_matrix::*;
