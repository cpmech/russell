//! Russell - Rust Scientific Library
//!
//! **sparse**: Sparse matrix tools and solvers

/// Returns package description
pub fn desc() -> String {
    "Sparse matrix tools and solvers".to_string()
}

mod enums;
mod read_matrix_market;
mod solver_mmp;
mod sparse_triplet;
mod to_i32;
pub use crate::enums::*;
pub use crate::read_matrix_market::*;
pub use crate::solver_mmp::*;
pub use crate::sparse_triplet::*;
pub use crate::to_i32::*;
