//! Russell - Rust Scientific Library
//!
//! **sparse**: Sparse matrix tools and solvers
//!
//! # Example - solving a sparse linear system
//!
//! ```
//! use russell_lab::*;
//! use russell_sparse::*;
//!
//! fn main() -> Result<(), &'static str> {
//!
//!     // allocate a square matrix
//!     let mut trip = SparseTriplet::new(5, 5, 13, Symmetry::No)?;
//!     trip.put(0, 0, 1.0); // << (0, 0, a00/2)
//!     trip.put(0, 0, 1.0); // << (0, 0, a00/2)
//!     trip.put(1, 0, 3.0);
//!     trip.put(0, 1, 3.0);
//!     trip.put(2, 1, -1.0);
//!     trip.put(4, 1, 4.0);
//!     trip.put(1, 2, 4.0);
//!     trip.put(2, 2, -3.0);
//!     trip.put(3, 2, 1.0);
//!     trip.put(4, 2, 2.0);
//!     trip.put(2, 3, 2.0);
//!     trip.put(1, 4, 6.0);
//!     trip.put(4, 4, 1.0);
//!
//!     // print matrix
//!     let (m, n) = trip.dims();
//!     let mut a = Matrix::new(m, n);
//!     trip.to_matrix(&mut a)?;
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
//!     let mut x = Vector::new(5);
//!     let rhs = Vector::from(&[8.0, 45.0, -3.0, 3.0, 19.0]);
//!
//!     // initialize, factorize, and solve
//!     let config = ConfigSolver::new();
//!     let mut solver = Solver::new(config)?;
//!     solver.initialize(&trip)?;
//!     solver.factorize()?;
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

/// Returns package description
pub fn desc() -> String {
    "Sparse matrix tools and solvers".to_string()
}

mod config_solver;
mod enums;
mod read_matrix_market;
mod solver;
mod sparse_triplet;
mod verify_lin_sys;
pub use crate::config_solver::*;
pub use crate::enums::*;
pub use crate::read_matrix_market::*;
pub use crate::solver::*;
pub use crate::sparse_triplet::*;
pub use crate::verify_lin_sys::*;

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
