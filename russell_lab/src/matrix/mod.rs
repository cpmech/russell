//! This module contains functions for calculations with matrices

mod aliases;
mod complex_mat_add;
mod complex_mat_approx_eq;
mod complex_mat_copy;
mod complex_mat_eigen;
mod complex_mat_inverse;
mod complex_mat_mat_mul;
mod complex_mat_norm;
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
mod mat_inverse;
mod mat_mat_mul;
mod mat_max_abs_diff;
mod mat_norm;
mod mat_pseudo_inverse;
mod mat_scale;
mod mat_svd;
mod mat_t_mat_mul;
mod mat_update;
mod mat_write_vismatrix;
mod num_matrix;
mod testing;
pub use crate::matrix::aliases::*;
pub use crate::matrix::complex_mat_add::*;
pub use crate::matrix::complex_mat_approx_eq::*;
pub use crate::matrix::complex_mat_copy::*;
pub use crate::matrix::complex_mat_eigen::*;
pub use crate::matrix::complex_mat_inverse::*;
pub use crate::matrix::complex_mat_mat_mul::*;
pub use crate::matrix::complex_mat_norm::*;
pub use crate::matrix::complex_mat_unzip::*;
pub use crate::matrix::complex_mat_update::*;
pub use crate::matrix::complex_mat_zip::*;
pub use crate::matrix::mat_add::*;
pub use crate::matrix::mat_approx_eq::*;
pub use crate::matrix::mat_cholesky::*;
pub use crate::matrix::mat_convert_to_blas_band::*;
pub use crate::matrix::mat_copy::*;
pub use crate::matrix::mat_eigen::*;
pub use crate::matrix::mat_eigen_sym::*;
pub use crate::matrix::mat_eigen_sym_jacobi::*;
pub use crate::matrix::mat_inverse::*;
pub use crate::matrix::mat_mat_mul::*;
pub use crate::matrix::mat_max_abs_diff::*;
pub use crate::matrix::mat_norm::*;
pub use crate::matrix::mat_pseudo_inverse::*;
pub use crate::matrix::mat_scale::*;
pub use crate::matrix::mat_svd::*;
pub use crate::matrix::mat_t_mat_mul::*;
pub use crate::matrix::mat_update::*;
pub use crate::matrix::mat_write_vismatrix::*;
pub use crate::matrix::num_matrix::*;
