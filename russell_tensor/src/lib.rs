//! Russell - Rust Scientific Library
//!
//! `russell_tensor`: Tensor analysis, calculus, and functions for continuum mechanics
//!
//! **Important:** This crate depends on external libraries (non-Rust). Thus, please check the [Installation Instructions on the GitHub Repository](https://github.com/cpmech/russell).
//!
//! # Introduction
//!
//! This library implements structures and functions for tensor analysis and calculus, with focus on applications in engineering and [Continuum Mechanics](https://en.wikipedia.org/wiki/Continuum_mechanics). The essential functionality for the targeted applications includes second-order, third-order, and fourth-order tensors, scalar "invariants," and derivatives.
//!
//! # Capabilities
//!
//! * [Tensor2] — second-order tensors (symmetric or not) with functions such as the determinant, inverse, norm, and invariants (principal, deviatoric, Lode, octahedral, ...)
//! * [Tensor3] — third-order tensors (minor-symmetric or not)
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
//! In the Kelvin basis, a second-order tensor is mapped to a column matrix (vector), a third-order tensor is mapped to a rectangular matrix, and a fourth-order tensor is mapped to a square matrix. The `√2` factors make the mapping isometric; thus the tensor norm is preserved and standard matrix/vector operations can be used directly.
//!
//! The [Rep] enum specifies the available representations:
//!
//! * [Rep::General] — 9×1 / 9×9 (all components)
//! * [Rep::Symmetric] — 6×1 / 6×6 (symmetric tensors in 3D)
//! * [Rep::Symmetric2D] — 4×1 / 4×4 (symmetric tensors in 2D)
//!
//! # Standard vs Kelvin components
//!
//! The tensor accessors follow a naming convention that distinguishes the **standard** (Cartesian) components `Tᵢⱼ` / `Dᵢⱼₖ` / `Dᵢⱼₖₗ` from the **Kelvin** components stored internally:
//!
//! * Accessors dealing with **standard components** carry the `std` qualifier
//!   in their names:
//!   * [Tensor2] — [`Tensor2::set_std_matrix`], [`Tensor2::from_std_matrix`],
//!     [`Tensor2::get_std`], [`Tensor2::as_std_matrix`], [`Tensor2::to_std_matrix`],
//!     [`Tensor2::as_std_matrix_2d`], [`Tensor2::sym_set_std`], [`Tensor2::sym_add_std`]
//!   * [Tensor3] — [`Tensor3::from_std_array`], [`Tensor3::from_std_matrix`],
//!     [`Tensor3::get_std`], [`Tensor3::as_std_array`], [`Tensor3::to_std_array`],
//!     [`Tensor3::as_std_matrix`], [`Tensor3::to_std_matrix`], [`Tensor3::sym_set_std`]
//!   * [Tensor4] — [`Tensor4::from_std_array`], [`Tensor4::from_std_matrix`],
//!     [`Tensor4::get_std`], [`Tensor4::as_std_array`], [`Tensor4::to_std_array`],
//!     [`Tensor4::as_std_matrix`], [`Tensor4::to_std_matrix`], [`Tensor4::sym_set_std`]
//! * Accessors dealing directly with the **Kelvin components** carry no qualifier:
//!   * [Tensor2] — [`Tensor2::vector`], [`Tensor2::vector_mut`], [`Tensor2::set_vector`],
//!     [`Tensor2::set_tensor`], [`Tensor2::update`], [`Tensor2::clear`]
//!   * [Tensor3] — [`Tensor3::matrix`], [`Tensor3::matrix_mut`], [`Tensor3::get_mn`],
//!     [`Tensor3::set`], [`Tensor3::set_tensor`], [`Tensor3::update`]
//!   * [Tensor4] — [`Tensor4::matrix`], [`Tensor4::matrix_mut`], [`Tensor4::get_mn`],
//!     [`Tensor4::set`], [`Tensor4::set_tensor`], [`Tensor4::update`]
//!
//! # Examples
//!
//! ```
//! use russell_tensor::*;
//!
//! fn main() -> Result<(), StrError> {
//!     // Allocate a symmetric second-order tensor given the standard components
//!     let sigma = Tensor2::from_std_matrix(
//!         &[
//!             [1.0, 2.0, 3.0],
//!             [2.0, 2.0, 4.0],
//!             [3.0, 4.0, 3.0],
//!         ],
//!         Rep::Symmetric,
//!     )?;
//!
//!     // Compute the principal invariants
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
mod as_matrix_9x3;
mod as_matrix_9x9;
mod constants;
mod derivatives_t2;
mod derivatives_t4;
mod enums;
mod lin_elasticity;
mod operations_mix1;
mod operations_mix2;
mod operations_t2;
mod operations_t3;
mod operations_t4;
mod samples_tensor2;
mod samples_tensor3;
mod samples_tensor4;
mod spectral2;
mod tensor2;
mod tensor3;
mod tensor4;

pub use as_matrix_3x3::*;
pub use as_matrix_9x3::*;
pub use as_matrix_9x9::*;
pub use constants::*;
pub use derivatives_t2::*;
pub use derivatives_t4::*;
pub use enums::*;
pub use lin_elasticity::*;
pub use operations_mix1::*;
pub use operations_mix2::*;
pub use operations_t2::*;
pub use operations_t3::*;
pub use operations_t4::*;
pub use samples_tensor2::*;
pub use samples_tensor3::*;
pub use samples_tensor4::*;
pub use spectral2::*;
pub use tensor2::*;
pub use tensor3::*;
pub use tensor4::*;

// run code from README file
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctest;
