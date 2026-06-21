//! Russell - Rust Scientific Library
//!
//! `russell_pde`: Solvers for partial differential equations using elliptic operators
//!
//! **Important:** This crate depends on external libraries (non-Rust). Thus, please check the [Installation Instructions on the GitHub Repository](https://github.com/cpmech/russell).
//!
//! To account for the EBCs, two approaches are possible:
//!
//! 1. Use the system partitioning strategy (SPS)
//! 2. Use the Lagrange multipliers method (LMM)
//!
//! ## Approach 1: System partitioning strategy (SPS)
//!
//! Consider the following partitioning of the vectors `a` and `f` and the matrix `K`:
//!
//! ```text
//! ┌       ┐ ┌   ┐   ┌   ┐
//! │ K̄   Ǩ │ │ ̄a │   │ f̄ │
//! │       │ │   │ = │   │
//! │ Ḵ   ̰K │ │ ǎ │   │ f̌ │
//! └       ┘ └   ┘   └   ┘
//!     K       a       f
//! ```
//!
//! where `ā` (a-bar) is a reduced vector containing only the unknown values (i.e., non-EBC nodes), and `ǎ` (a-check)
//! is a reduced vector containing only the prescribed values (i.e., EBC nodes). `f̄` and `f̌` are the associated reduced
//! right-hand side vectors. The `K̄` (K-bar) matrix is the reduced discrete Laplacian operator and `Ǩ` (K-check) is a
//! *correction* matrix. The `Ḵ` (K-underline) and `K̰` (K-under-tilde) matrices are often not needed.
//!
//! Thus, the linear system to be solved is:
//!
//! ```text
//! K̄ ā = f̄ - Ǩ ǎ
//! ```
//!
//! ## Approach 2: Lagrange multipliers method (LMM)
//!
//! The LMM consists of augmenting the original linear system with additional equations:
//!
//! ```text
//! ┌       ┐ ┌   ┐   ┌   ┐
//! │ K  Cᵀ │ │ a │   │ f │
//! │       │ │   │ = │   │
//! │ C  0  │ │ ℓ │   │ ǎ │
//! └       ┘ └   ┘   └   ┘
//!     M       A       F
//! ```
//!
//! where `ℓ` is the vector of Lagrange multipliers, `C` is the constraints matrix, and `ǎ` is the vector of
//! prescribed values at EBC nodes. The constraints matrix `C` has a row for each EBC (prescribed) node and a column
//! for every node. Each row in `C` has a single `1` at the column corresponding to the EBC node, and `0`s elsewhere.
//!
//! # Examples
//!
//! Solve the Poisson equation in 1D with homogeneous Dirichlet boundary conditions:
//!
//! ```text
//! -d²ϕ/dx² = 1   on  x ∈ [0, 1]
//!
//! ϕ(0) = 0
//! ϕ(1) = 0
//! ```
//!
//! The analytical solution is `ϕ(x) = (x - x²) / 2`.
//!
//! ```
//! use russell_lab::approx_eq;
//! use russell_pde::{EssentialBcs1d, Fdm1d, Grid1d, NaturalBcs1d, Side, StrError};
//!
//! fn main() -> Result<(), StrError> {
//!     // grid
//!     let xmin = 0.0;
//!     let xmax = 1.0;
//!     let nx = 4;
//!     let mut grid = Grid1d::new_uniform(xmin, xmax, nx)?;
//!
//!     // Essential BCs
//!     let mut ebcs = EssentialBcs1d::new();
//!     ebcs.set(Side::Xmin, |_| 0.0);
//!     ebcs.set(Side::Xmax, |_| 0.0);
//!
//!     // Natural BCs (none)
//!     let nbcs = NaturalBcs1d::new();
//!
//!     // FDM solver
//!     let kx = 1.0;
//!     let fdm = Fdm1d::new(grid, ebcs, nbcs, kx)?;
//!
//!     // Solve system
//!     let alpha = 0.0; // Poisson
//!     let source = |_| 1.0;
//!     let phi = fdm.solve_sps(alpha, source)?;
//!
//!     // Check
//!     fdm.for_each_coord(|m, x| {
//!         let analytical = x * (1.0 - x) / 2.0;
//!         approx_eq(phi[m], analytical, 1e-14);
//!     });
//!     Ok(())
//! }
//! ```

/// Defines the error output as a static string
pub type StrError = &'static str;

mod constants;
mod enums;
mod equation_handler;
mod essential_bcs_1d;
mod essential_bcs_2d;
mod fdm_1d;
mod fdm_2d;
mod grid_1d;
mod grid_2d;
mod metrics;
mod natural_bcs_1d;
mod natural_bcs_2d;
mod problem_samples;
mod spc_1d;
mod spc_2d;
mod spc_map_2d;
mod transfinite_2d;
mod transfinite_3d;
mod transfinite_samples;
mod util;

pub use constants::*;
pub use enums::*;
pub use equation_handler::*;
pub use essential_bcs_1d::*;
pub use essential_bcs_2d::*;
pub use fdm_1d::*;
pub use fdm_2d::*;
pub use grid_1d::*;
pub use grid_2d::*;
pub use metrics::*;
pub use natural_bcs_1d::*;
pub use natural_bcs_2d::*;
pub use problem_samples::*;
pub use spc_1d::*;
pub use spc_2d::*;
pub use spc_map_2d::*;
pub use transfinite_2d::*;
pub use transfinite_3d::*;
pub use transfinite_samples::*;

// run code from README file
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctest;
