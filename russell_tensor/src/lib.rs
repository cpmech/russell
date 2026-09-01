//! Russell - Rust Scientific Library
//!
//! `russell_tensor`: Tensor analysis, calculus, and functions for continuum mechanics
//!
//! **Important:** This crate depends on external libraries (non-Rust). Thus, please check the [Installation Instructions on the GitHub Repository](https://github.com/cpmech/russell).
//!
//! # Introduction
//!
//! This library implements structures and functions for tensor analysis and calculus, with focus on applications in engineering and [Continuum Mechanics](https://en.wikipedia.org/wiki/Continuum_mechanics). The essential functionality for the targeted applications includes first-order, second-order, third-order, and fourth-order tensors, scalar "invariants," and derivatives.
//!
//! # Capabilities
//!
//! * [Tensor1] — First-order tensors (vectors) in R³. Includes operations such as the dot and cross products
//! * [Tensor2] — Second-order tensors in R³×R³. Allows symmetric specialization. Includes functions such as the determinant, inverse, norm, and invariants (principal, deviatoric, Lode, octahedral, ...)
//! * [Tensor3] — Third-order tensors R³×R³×R³. Allows minor-symmetric specialization. Includes functions such as permutation (Levi-Civita) tensor
//! * [Tensor4] — Fourth-order tensors R³×R³×R³×R³. Allows minor-symmetric specialization. Includes functions to generate isotropic tensors.
//! * [Spectral2] — The spectral (eigen) representation of symmetric second-order tensors.
//! * [LinElasticity] — The linear elasticity equations for small-strain problems (Generalized Hooke's law)
//! * Polar decomposition — Computes the polar decomposition `F = R U = V R` of a general [Tensor2] using the classic Eigen/SVD algorithms, the iterative Brannon algorithm, the closed-form in-plane Brannon algorithm, or the quaternion-based Higham & Noferini algorithm (see [PolarAlgo] and [polar_decomp]).
//! * Constants — Includes Identity, transposition, and other projector tensors.
//! * Operations between tensors — Includes addition, single and double contractions (dot and ddot), and dyadic products.
//! * Derivatives — Implements first and second derivatives of invariants and tensor functions (e.g., the inverse and squared tensors)
//!
//! # Kelvin-Mandel notation
//!
//! Internally, tensors are stored in the Kelvin-Mandel basis (Kelvin-Mandel notation).
//!
//! In the Kelvin-Mandel basis, a second-order tensor is mapped to a column matrix (vector), a third-order tensor is mapped to a rectangular matrix, and a fourth-order tensor is mapped to a square matrix. The `√2` factors make the mapping isometric; thus the tensor norm is preserved and standard matrix/vector operations can be used directly.
//!
//! The dimension — the const generic `N` of [Tensor2]/[Tensor4], and `M`/`N` of
//! [Tensor3] — selects the representation:
//!
//! * `9` — all components (general): 9×1 / 9×3 / 3×9 / 9×9
//! * `6` — symmetric [Tensor2] / minor-symmetric [Tensor3]/[Tensor4] (3D): 6×1 / 6×3 / 3×6 / 6×6
//! * `4` — symmetric [Tensor2] / minor-symmetric [Tensor3]/[Tensor4] (2D): 4×1 / 4×3 / 3×4 / 4×4
//!
//! The dimensions above correspond to [Tensor2] (vector), [Tensor3] (Case A / Case B rectangular matrix), and [Tensor4] (square matrix), respectively.
//!
//! A [Tensor3] is stored as a rectangular Kelvin-Mandel matrix with dimensions `(M, N)`
//! set by const generics. Two cases are considered, where `DIM` (the leading dimension)
//! is one of 4, 6, or 9:
//!
//! * **Case A** `(DIM, 3)` — `M = DIM`, `N = 3`: the Tensor3 acts on a [Tensor1] (vector)
//!   yielding a [Tensor2] (`T = H · u`)
//! * **Case B** `(3, DIM)` — `M = 3`, `N = DIM`: the Tensor3 acts on a [Tensor2] yielding
//!   a [Tensor1] (vector) (`v = M : S`)
//!
//! # Standard vs Kelvin-Mandel components
//!
//! The tensor accessors follow a naming convention that distinguishes the **standard** (Cartesian) components `Tᵢⱼ` / `Hᵢⱼₖ` / `Dᵢⱼₖₗ` from the **Kelvin-Mandel** components stored internally:
//!
//! * Accessors dealing with **standard components** carry the `std` qualifier
//!   in their names:
//!   * [Tensor2] — [Tensor2::set_std_matrix], [Tensor2::from_std_matrix],
//!     [Tensor2::get_std], [Tensor2::as_std_matrix], [Tensor2::to_std_matrix],
//!     [Tensor2::as_std_matrix_2d], [Tensor2::sym_set_std], [Tensor2::sym_add_std]
//!   * [Tensor3] — [Tensor3::from_std_array], [Tensor3::from_std_matrix],
//!     [Tensor3::get_std], [Tensor3::as_std_array], [Tensor3::to_std_array],
//!     [Tensor3::as_std_matrix], [Tensor3::to_std_matrix], [Tensor3::sym_set_std]
//!   * [Tensor4] — [Tensor4::from_std_array], [Tensor4::from_std_matrix],
//!     [Tensor4::get_std], [Tensor4::as_std_array], [Tensor4::to_std_array],
//!     [Tensor4::as_std_matrix], [Tensor4::to_std_matrix], [Tensor4::sym_set_std]
//! * Accessors dealing directly with the **Kelvin-Mandel components** carry no qualifier:
//!   * [Tensor2] — [Tensor2::get], [Tensor2::set], [Tensor2::set_vector],
//!     [Tensor2::set_tensor], [Tensor2::update], [Tensor2::clear]
//!   * [Tensor3] — [Tensor3::get], [Tensor3::set], [Tensor3::set_tensor], [Tensor3::update]
//!   * [Tensor4] — [Tensor4::get], [Tensor4::set], [Tensor4::set_tensor], [Tensor4::update]
//!
//! **Note:** [Tensor1] stores the three **standard** components directly (there is no Kelvin-Mandel mapping for first-order tensors); access them with [Tensor1::get] and [Tensor1::set].
//!
//! # Optional features
//!
//! The following (Rust) features are available:
//!
//! * `intel_mkl` — use Intel MKL instead of OpenBLAS
//! * `heap` — use heap-allocated (dynamically allocated) storage for the tensor
//!   components instead of the default stack-allocated (fixed-size) storage
//!
//! # Examples
//!
//! ```
//! use russell_tensor::*;
//!
//! fn main() -> Result<(), StrError> {
//!     // Allocate a symmetric second-order tensor given the standard components
//!     let sigma = Tensor2::<6>::from_std_matrix(&[
//!         [1.0, 2.0, 3.0],
//!         [2.0, 2.0, 4.0],
//!         [3.0, 4.0, 3.0],
//!     ])?;
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

/// Defines the error type as a static string
pub type StrError = &'static str;

pub mod analysis;
mod constants;
mod derivatives_t2;
mod derivatives_t4;
mod lin_elasticity;
mod operations_mix1;
mod operations_mix2;
mod operations_t2;
mod operations_t2x;
mod operations_t3;
mod operations_t4;
mod polar_brannon;
mod polar_classic;
mod polar_decomp;
mod polar_higham;
mod samples_tensor2;
mod samples_tensor3;
mod samples_tensor4;
mod spectral2;
mod tensor1;
mod tensor2;
mod tensor3;
mod tensor4;

#[cfg(test)]
mod test_common;

pub mod z_reference_loop_fns;

pub use constants::*;
pub use derivatives_t2::*;
pub use derivatives_t4::*;
pub use lin_elasticity::*;
pub use operations_mix1::*;
pub use operations_mix2::*;
pub use operations_t2::*;
pub use operations_t2x::*;
pub use operations_t3::*;
pub use operations_t4::*;
pub use polar_decomp::*;
pub use samples_tensor2::*;
pub use samples_tensor3::*;
pub use samples_tensor4::*;
pub use spectral2::*;
pub use tensor1::*;
pub use tensor2::*;
pub use tensor3::*;
pub use tensor4::*;

// run code from README file
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctest;
