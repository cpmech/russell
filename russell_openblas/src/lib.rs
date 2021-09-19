//! Russell - Rust Scientific Library
//!
//! **openblas**: Thin wrapper to some OpenBLAS routines
//!
//! # Example - dnrm2
//!
//! ```
//! use russell_chk::*;
//! use russell_openblas::*;
//! let x = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0];
//! let (n, incx) = (to_i32(x.len()), 1_i32);
//! assert_approx_eq!(dnrm2(n, &x, incx), 5.0, 1e-15);
//! ```

/// Returns package description
pub fn desc() -> String {
    "Thin wrapper to some OpenBLAS routines".to_string()
}

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
