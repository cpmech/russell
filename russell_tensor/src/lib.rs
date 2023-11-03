//! Russell - Rust Scientific Library
//!
//! `russell_tensor`: Tensor analysis structures and functions for continuum mechanics
//!
//! **Important:** This crate depends on external libraries (non-Rust). Thus, please check the [Installation Instructions on the GitHub Repository](https://github.com/cpmech/russell).
//! # Introduction
//!
//! TODO
//!
//! # Examples
//!
//! TODO

/// Defines a type alias for the error type as a static string
pub type StrError = &'static str;

mod as_matrix_3x3;
mod constants;
mod derivatives_t2;
mod derivatives_t4;
mod enums;
mod lin_elasticity;
mod operations;
mod samples_tensor2;
mod samples_tensor4;
mod spectral2;
mod stress_strain_path;
mod tensor2;
mod tensor4;
pub use crate::as_matrix_3x3::*;
pub use crate::constants::*;
pub use crate::derivatives_t2::*;
pub use crate::derivatives_t4::*;
pub use crate::enums::*;
pub use crate::lin_elasticity::*;
pub use crate::operations::*;
pub use crate::samples_tensor2::*;
pub use crate::samples_tensor4::*;
pub use crate::spectral2::*;
pub use crate::stress_strain_path::*;
pub use crate::tensor2::*;
pub use crate::tensor4::*;

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
