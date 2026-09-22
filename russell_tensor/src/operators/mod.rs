//! This module contains operators and functions to perform algebraic calculations with tensors

mod dsd;
mod odyad;
mod qsd;
mod ssd;
mod t2_and_t4;
mod t2_essential;
mod t2_matmul;
mod t3_essential;
mod t4_essential;
mod udyad;

pub use dsd::*;
pub use odyad::*;
pub use qsd::*;
pub use ssd::*;
pub use t2_and_t4::*;
pub use t2_essential::*;
pub use t2_matmul::*;
pub use t3_essential::*;
pub use t4_essential::*;
pub use udyad::*;
