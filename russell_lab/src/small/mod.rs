//! This module implements calculations with small matrices and vectors that are
//! allocated on the stack.
//!
//! # Conventions
//!
//! * The types are defined by the [`SmallMatrix`] and [`SmallVector`] aliases,
//!   which are fixed-size arrays allocated on the stack (row-major, direct
//!   `a[i][j]` / `v[i]` access, no heap allocation).
//! * The arithmetic functions operating on square matrices take an `n` parameter
//!   specifying the *active* dimension (the top-left `n×n` block) such that
//!   `n ≤ N`. Passing `n > N` causes a panic. Other functions operate on the full
//!   `N×N` matrix.
//! * The basic arithmetic operations ([`small_mat_add`], [`small_mat_update`],
//!   [`small_mat_mat_mul`], [`small_mat_t_mat_mul`], [`small_vec_add`], [`small_vec_update`]) are generic
//!   over the element type `T`, which must implement `Num` (from `num_traits`)
//!   and `Copy`.
//! * The Gauss-Jordan inversion and solver routines ([`small_mat_inv`],
//!   [`num_recipes_gaussj_inv`], [`num_recipes_gaussj_sol`],
//!   [`small_solve_lin_sys`]) are generic over `T: Float` and return a `Result`
//!   because they can fail on a singular matrix.
//! * The singular value decomposition ([`small_mat_svd`]) wraps the LAPACK
//!   `dgesvd` routine and operates on `f64` data (rectangular `M×N` matrices).
//! * The symmetric eigen-decomposition ([`small_mat_eigen_sym_jacobi`]) operates
//!   on the full `N×N` matrix and returns the (unsorted) eigenvalues and
//!   eigenvectors.
//! * [`small_mat_approx_eq`] compares a small matrix against any
//!   [`crate::AsArray2D`] (e.g. another small matrix) within a given tolerance.

mod num_recipes_gaussj;
mod small_mat_add;
mod small_mat_approx_eq;
mod small_mat_eigen_sym_jacobi;
mod small_mat_inv;
mod small_mat_mat_mul;
mod small_mat_svd;
mod small_mat_t_mat_mul;
mod small_mat_update;
mod small_matrix;
mod small_solve_lin_sys;
mod small_vec_add;
mod small_vec_update;
mod small_vector;

pub use num_recipes_gaussj::*;
pub use small_mat_add::*;
pub use small_mat_approx_eq::*;
pub use small_mat_eigen_sym_jacobi::*;
pub use small_mat_inv::*;
pub use small_mat_mat_mul::*;
pub use small_mat_svd::*;
pub use small_mat_t_mat_mul::*;
pub use small_mat_update::*;
pub use small_matrix::*;
pub use small_solve_lin_sys::*;
pub use small_vec_add::*;
pub use small_vec_update::*;
pub use small_vector::*;
