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

// Per-dimension kernels of the sum-dyadic operators.
//
// They exist so the dimension dispatch of `dsd`, `qsd` and `ssd` is static: each
// implementation is monomorphic, hence fully reachable, while a single
// const-generic function with `if N == 4 ... else if N == 6 ... else` leaves two
// thirds of every instantiation unreachable (`N` is a const parameter).
pub(crate) trait DsdDim {
    fn dsd(&mut self, op: u8, s: f64, a: &[f64], b: &[f64]);
}

pub(crate) trait QsdDim {
    fn qsd(&mut self, op: u8, s: f64, a: &[f64], b: &[f64]);
}

pub(crate) trait SsdDim {
    fn ssd(&mut self, op: u8, s: f64, a: &[f64]);
}
