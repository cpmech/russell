//! Russell - Rust Scientific Library
//!
//! **sparse**: Sparse matrix tools and solvers

/// Returns package description
pub fn desc() -> String {
    "Sparse matrix tools and solvers".to_string()
}

mod config_solver;
mod enums;
mod read_matrix_market;
mod solver;
mod sparse_triplet;
mod to_i32;
pub use crate::config_solver::*;
pub use crate::enums::*;
pub use crate::read_matrix_market::*;
pub use crate::solver::*;
pub use crate::sparse_triplet::*;
pub use crate::to_i32::*;
