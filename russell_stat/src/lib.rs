//! Russell - Rust Scientific Library
//!
//! **stat**: Statistics, Probabilities, and Random Numbers

// errors ///////////////////////////////////////
#[macro_use]
extern crate error_chain;
mod err {
    error_chain! {}
}

// modules //////////////////////////////////////
mod histogram;
mod stat;
pub use crate::histogram::*;
pub use crate::stat::*;
