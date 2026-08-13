//! Russell - Rust Scientific Library
//!
//! `russell_tensor`: Tensor analysis, calculus, and functions for continuum mechanics
//!
//! **Important:** This crate depends on external libraries (non-Rust). Thus, please check the [Installation Instructions on the GitHub Repository](https://github.com/cpmech/russell).
//!
//! # Introduction
//!
//! This library implements structures and functions for tensor analysis and calculus, with focus on applications in engineering and [Continuum Mechanics](https://en.wikipedia.org/wiki/Continuum_mechanics). The essential functionality for the targeted applications includes second-order and fourth-order tensors, scalar "invariants," and derivatives.
//!
//! # Capabilities
//!
//! * [Tensor2] — second-order tensors (symmetric or not) with functions such as the determinant, inverse, norm, and invariants (principal, deviatoric, Lode, octahedral, ...)
//! * [Tensor4] — fourth-order tensors (minor-symmetric or not)
//! * Operations between tensors — addition, single and double contractions (dot and ddot), and dyadic products
//! * Analytical derivatives — first and second derivatives of invariants and tensor functions (e.g., the inverse and squared tensors) with respect to tensors
//! * [Spectral2] — the spectral (eigen) representation of symmetric second-order tensors
//! * [LinElasticity] — the linear elasticity equations for small-strain problems (Hooke's law)
//! * Constants — identity, transposition, and projector tensors
//!
//! # Kelvin notation
//!
//! Internally, tensors are stored in the Kelvin basis (Kelvin notation), an isometric (norm-preserving) alternative to [Voigt notation](https://en.wikipedia.org/wiki/Voigt_notation).
//!
//! In the Kelvin basis, a second-order tensor is mapped to a column matrix (vector) and a fourth-order tensor is mapped to a square matrix. The `√2` factors make the mapping isometric; thus the tensor norm is preserved and standard matrix/vector operations can be used directly.
//!
//! The [Rep] enum specifies the available representations:
//!
//! * [Rep::General] — 9×1 / 9×9 (all components)
//! * [Rep::Symmetric] — 6×1 / 6×6 (symmetric tensors in 3D)
//! * [Rep::Symmetric2D] — 4×1 / 4×4 (symmetric tensors in 2D)
//!
//! # Examples
//!
//! ```
//! use russell_tensor::*;
//!
//! fn main() -> Result<(), StrError> {
//!     // allocate a symmetric second-order tensor
//!     let sigma = Tensor2::from_matrix(
//!         &[
//!             [1.0, 2.0, 3.0],
//!             [2.0, 2.0, 4.0],
//!             [3.0, 4.0, 3.0],
//!         ],
//!         Rep::Symmetric,
//!     )?;
//!
//!     // compute the principal invariants
//!     let ii1 = sigma.invariant_ii1();
//!     let ii2 = sigma.invariant_ii2();
//!     let ii3 = sigma.invariant_ii3();
//!
//!     println!("I1 = {:.6}", ii1);
//!     println!("I2 = {:.6}", ii2);
//!     println!("I3 = {:.6}", ii3);
//!     Ok(())
//! }
//! ```

/// Defines the error output as a static string
pub type StrError = &'static str;

mod as_matrix_3x3;
mod as_matrix_9x9;
mod constants;
mod derivatives_t2;
mod derivatives_t4;
mod enums;
mod lin_elasticity;
mod operations_mix1;
mod operations_mix2;
mod operations_t2;
mod operations_t4;
mod samples_tensor2;
mod samples_tensor4;
mod spectral2;
mod tensor2;
mod tensor4;

pub use as_matrix_3x3::*;
pub use as_matrix_9x9::*;
pub use constants::*;
pub use derivatives_t2::*;
pub use derivatives_t4::*;
pub use enums::*;
pub use lin_elasticity::*;
pub use operations_mix1::*;
pub use operations_mix2::*;
pub use operations_t2::*;
pub use operations_t4::*;
pub use samples_tensor2::*;
pub use samples_tensor4::*;
pub use spectral2::*;
pub use tensor2::*;
pub use tensor4::*;

// run code from README file
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctest;
