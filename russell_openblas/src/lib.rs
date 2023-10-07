//! Russell - Rust Scientific Library
//!
//! **openblas**: Thin wrapper to some OpenBLAS routines
//!
//! **NOTE**: Only the COL-MAJOR representation is considered here.
//!
//! ```text
//!     ┌     ┐  row_major = {0, 3,
//!     │ 0 3 │               1, 4,
//! A = │ 1 4 │               2, 5};
//!     │ 2 5 │
//!     └     ┘  col_major = {0, 1, 2,
//!     (m × n)               3, 4, 5}
//!
//! Aᵢⱼ = col_major[i + j·m] = row_major[i·n + j]
//!         ↑
//! COL-MAJOR IS ADOPTED HERE
//! ```
//!
//! The main reason to use the **col-major** representation is to make the code work
//! better with BLAS/LAPACK written in Fortran. Although those libraries have functions
//! to handle row-major data, they usually add an overhead due to temporary memory
//! allocation and copies, including transposing matrices. Moreover, the row-major
//! versions of some BLAS/LAPACK libraries produce incorrect results (notably the DSYEV).
//!
//! # Example - dnrm2
//!
//! ```
//! use russell_lab::approx_eq;
//! use russell_openblas::{dnrm2, to_i32};
//! let x = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0];
//! let (n, incx) = (to_i32(x.len()), 1_i32);
//! approx_eq(dnrm2(n, &x, incx), 5.0, 1e-15);
//! ```

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod config;
mod constants;
mod conversions;
mod highlevel;
mod matrix;
mod matvec;
mod to_i32;
mod vector;
pub use crate::config::*;
use crate::constants::*;
pub use crate::conversions::*;
pub use crate::highlevel::*;
pub use crate::matrix::*;
pub use crate::matvec::*;
pub use crate::to_i32::*;
pub use crate::vector::*;

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
}
