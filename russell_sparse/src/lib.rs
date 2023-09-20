//! Russell - Rust Scientific Library
//!
//! **sparse**: Sparse matrix tools and solvers
//!
//! # Example - solving a sparse linear system
//!
//! ```
//! use russell_lab::{Matrix, Vector};
//! use russell_sparse::prelude::*;
//! use russell_sparse::StrError;
//!
//! fn main() -> Result<(), StrError> {
//!     // allocate a square matrix
//!     let (nrow, ncol, nnz) = (5, 5, 13);
//!     let mut coo = CooMatrix::new(Layout::Full, nrow, ncol, nnz)?;
//!     coo.put(0, 0, 1.0)?; // << (0, 0, a00/2)
//!     coo.put(0, 0, 1.0)?; // << (0, 0, a00/2)
//!     coo.put(1, 0, 3.0)?;
//!     coo.put(0, 1, 3.0)?;
//!     coo.put(2, 1, -1.0)?;
//!     coo.put(4, 1, 4.0)?;
//!     coo.put(1, 2, 4.0)?;
//!     coo.put(2, 2, -3.0)?;
//!     coo.put(3, 2, 1.0)?;
//!     coo.put(4, 2, 2.0)?;
//!     coo.put(2, 3, 2.0)?;
//!     coo.put(1, 4, 6.0)?;
//!     coo.put(4, 4, 1.0)?;
//!
//!     // print matrix
//!     let mut a = Matrix::new(nrow, ncol);
//!     coo.to_matrix(&mut a)?;
//!     let correct = "┌                ┐\n\
//!                    │  2  3  0  0  0 │\n\
//!                    │  3  0  4  0  6 │\n\
//!                    │  0 -1 -3  2  0 │\n\
//!                    │  0  0  1  0  0 │\n\
//!                    │  0  4  2  0  1 │\n\
//!                    └                ┘";
//!     assert_eq!(format!("{}", a), correct);
//!
//!     // allocate x and rhs
//!     let mut x = Vector::new(nrow);
//!     let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
//!
//!     // initialize, factorize, and solve
//!     let config = ConfigSolver::new();
//!     let mut solver = Solver::new(config, nrow, nnz, None)?;
//!     solver.factorize(&coo)?;
//!     solver.solve(&mut x, &rhs)?;
//!     let correct = "┌          ┐\n\
//!                    │ 1.000000 │\n\
//!                    │ 2.000000 │\n\
//!                    │ 3.000000 │\n\
//!                    │ 4.000000 │\n\
//!                    │ 5.000000 │\n\
//!                    └          ┘";
//!     assert_eq!(format!("{:.6}", x), correct);
//!     Ok(())
//! }
//! ```

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod config_solver;
mod coo_matrix;
mod csr_matrix;
mod enums;
pub mod prelude;
mod read_matrix_market;
mod solver;
mod verify_lin_sys;
mod write_matrix_market;
pub use crate::config_solver::*;
pub use crate::coo_matrix::*;
pub use crate::csr_matrix::*;
pub use crate::enums::*;
pub use crate::read_matrix_market::*;
pub use crate::solver::*;
pub use crate::verify_lin_sys::*;
pub use crate::write_matrix_market::*;

// run code from README file
#[cfg(doctest)]
mod test_readme {
    macro_rules! external_doc_test {
        ($x:expr) => {
            #[doc = $x]
            extern "C" {}
        };
    }
    external_doc_test!(include_str!("../README.md"));
    external_doc_test!(include_str!("../../README.md"));
}
