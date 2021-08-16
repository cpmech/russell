//! Russell - Rust Scientific Library
//!
//! **openblas**: Thin wrapper to (some) OpenBLAS routines

// modules //////////////////////////////////////
mod constants;
mod conversions;
mod part1;
use crate::constants::*;
pub use crate::conversions::*;
pub use crate::part1::*;
